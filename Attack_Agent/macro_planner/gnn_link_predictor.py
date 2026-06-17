# macro_planner/gnn_link_predictor.py
"""
GraphSAGE-based link predictor cho technique graph.
Dự đoán P(edge | technique_i, technique_j) dựa trên SCF + graph structure.
Viết bằng PyTorch thuần (không cần torch-geometric).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from pathlib import Path


class GraphSAGEEncoder(nn.Module):
    """Encode mỗi technique node thành embedding vector."""
    def __init__(self, input_dim=25, hidden_dim=64, output_dim=32):
        super().__init__()
        self.sage1 = nn.Linear(input_dim * 2, hidden_dim)
        self.sage2 = nn.Linear(hidden_dim * 2, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def aggregate_neighbors(self, node_feats, adj_matrix):
        """Mean aggregation của neighbor features."""
        neighbor_mean = torch.mm(adj_matrix, node_feats)
        count = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        neighbor_mean = neighbor_mean / count
        return neighbor_mean

    def forward(self, node_feats, adj_matrix):
        # Layer 1
        agg = self.aggregate_neighbors(node_feats, adj_matrix)
        h1 = F.relu(self.bn1(self.sage1(torch.cat([node_feats, agg], dim=1))))
        # Layer 2
        agg2 = self.aggregate_neighbors(h1, adj_matrix)
        h2 = self.sage2(torch.cat([h1, agg2], dim=1))
        return F.normalize(h2, p=2, dim=1)


class LinkPredictor(nn.Module):
    """Dự đoán link probability từ hai node embeddings."""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, emb_i, emb_j):
        return self.fc(torch.cat([emb_i, emb_j], dim=1)).squeeze(-1)


class GNNLinkPredictor:
    """Wrapper train và inference."""
    def __init__(self, scf_dict: dict, tech_graph: nx.DiGraph,
                 model_path="macro_planner/data/link_pred_model.pt"):
        self.scf_dict = scf_dict
        self.graph = tech_graph
        self.model_path = model_path

        self.nodes = sorted(list(tech_graph.nodes()))
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        N = len(self.nodes)

        # Feature matrix (N, 25)
        self.X = np.zeros((N, 25), dtype=np.float32)
        for i, tid in enumerate(self.nodes):
            self.X[i] = scf_dict.get(tid, np.zeros(25))
        self.X_tensor = torch.FloatTensor(self.X)

        # Adjacency matrix (N, N) - normalized
        adj = np.zeros((N, N), dtype=np.float32)
        for u, v in tech_graph.edges():
            if u in self.node_idx and v in self.node_idx:
                adj[self.node_idx[u], self.node_idx[v]] = 1.0
        row_sum = adj.sum(axis=1, keepdims=True).clip(min=1)
        self.adj_tensor = torch.FloatTensor(adj / row_sum)

        self.encoder = GraphSAGEEncoder(25, 64, 32)
        self.predictor = LinkPredictor(32)

        if Path(model_path).exists():
            self.load_model()
        else:
            print("[GNN] No pretrained model found. Run train() first.")

    def train(self, epochs=100, lr=0.001):
        """Train link predictor trên technique graph."""
        import random
        print(f"[GNN] Training on {len(self.nodes)} nodes...")
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=lr)
        criterion = nn.BCELoss()

        pos_edges = [(self.node_idx[u], self.node_idx[v])
                     for u, v in self.graph.edges()
                     if u in self.node_idx and v in self.node_idx]

        for epoch in range(epochs):
            self.encoder.train()
            self.predictor.train()
            opt.zero_grad()

            embeddings = self.encoder(self.X_tensor, self.adj_tensor)

            batch_pos = random.sample(pos_edges, min(256, len(pos_edges)))
            N = len(self.nodes)
            batch_neg = [(random.randint(0, N-1), random.randint(0, N-1))
                         for _ in range(len(batch_pos))]

            edges = batch_pos + batch_neg
            labels = [1.0] * len(batch_pos) + [0.0] * len(batch_neg)

            i_idx = torch.LongTensor([e[0] for e in edges])
            j_idx = torch.LongTensor([e[1] for e in edges])
            y = torch.FloatTensor(labels)

            pred = self.predictor(embeddings[i_idx], embeddings[j_idx])
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

        self.save_model()
        print("[GNN] Training complete.")

    def predict_next_techniques(self, current_tid: str, top_k: int = 5,
                                tactic_filter: str = None) -> list:
        """
        Trả về top_k techniques có khả năng theo sau current_tid.
        Returns: [(technique_id, probability, technique_name), ...]
        """
        self.encoder.eval()
        self.predictor.eval()
        if current_tid not in self.node_idx:
            return []

        with torch.no_grad():
            embeddings = self.encoder(self.X_tensor, self.adj_tensor)
            i = self.node_idx[current_tid]
            emb_i = embeddings[i].unsqueeze(0).expand(len(self.nodes), -1)
            probs = self.predictor(emb_i, embeddings).numpy()

        results = []
        for j, tid in enumerate(self.nodes):
            if j == i:
                continue
            if tactic_filter:
                tactics = self.graph.nodes[tid].get("tactics", [])
                if tactic_filter not in tactics:
                    continue
            results.append((tid, float(probs[j]),
                           self.graph.nodes[tid].get("name", "")))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save_model(self):
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict()
        }, self.model_path)

    def load_model(self):
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        print(f"[GNN] Loaded model from {self.model_path}")
