"""
networks.py
===========
Network definitions cho MARL-VariantHunter.

Kiến trúc:
  - GAT  : Graph Attention Network (thay thế GCN) — Detection Agent backbone.
            Trả về node embeddings + attention weights để giải thích.
  - DetectionMLP : MLP phân loại + EWC Continual Learning.
  - CoordinatorNetwork : PPO policy/value head.
  - PositionalEncoder  : Sinusoidal positional encoding cho node features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# Original Graph Convolutional Network (GCN) — SAGEConv
# ─────────────────────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Network (GATs)
# ─────────────────────────────────────────────────────────────────────────────

class GAT(torch.nn.Module):
    """
    Graph Attention Network (GATs) — Detection Agent backbone.

    Kiến trúc 2 lớp:
      conv1: in_channels → hidden_dim (8 heads × 8 dim = 64)
      conv2: hidden_dim  → out_channels (1 head)

    Ưu điểm so với GCN (SAGEConv):
      - Attention mechanism học tự động để chú ý vào các edge bất thường.
      - Trả về attention_weights (alpha) cho mỗi edge — hỗ trợ explainability.
      - Phù hợp hơn với provenance graph dạng thưa (sparse), không đồng nhất.

    Tham khảo:
      Veličković et al., 2018 — "Graph Attention Networks"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Lớp 1: Multi-head attention (heads × head_dim)
        self.conv1 = GATConv(
            in_channels,
            hidden_dim // heads,   # mỗi head output = hidden_dim / heads
            heads=heads,
            dropout=dropout,
            concat=True,           # concat → output: heads * (hidden_dim/heads) = hidden_dim
        )

        # Lớp 2: Single-head attention → out_channels
        self.conv2 = GATConv(
            hidden_dim,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,          # out shape: [N, out_channels]
        )

        # Batch Norm sau conv1 để ổn định training
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass không trả về attention weights (inference nhanh).

        Args:
            x           : Node feature matrix [N, in_channels]
            edge_index  : Edge list [2, E]

        Returns:
            out : Node embeddings [N, out_channels]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:          # BatchNorm cần ít nhất 2 samples
            x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward_with_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass trả về kèm attention weights từ lớp conv2.
        Dùng để tính Gradient-weighted Attention Score và Confidence Score.

        Returns:
            out              : Node embeddings [N, out_channels]
            attention_weights: Tensor attention [E, 1] từ lớp 2
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_out, alpha) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        return x, alpha   # alpha: [E, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoder
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoder:
    """Sinusoidal positional encoding cho node features (cùng kích thước d_model)."""

    def __init__(self, d_model: int, max_len: int = 100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


# ─────────────────────────────────────────────────────────────────────────────
# PPO Coordinator Network
# ─────────────────────────────────────────────────────────────────────────────

class CoordinatorNetwork(nn.Module):
    """
    High-level PPO policy network (Coordinator).

    Input : Graph State Representation [state_dim]
    Output:
      - action_logits : [num_actions] — chọn Agent nào chạy tiếp
      - state_value   : [1]           — V(s) cho PPO advantage
    """

    def __init__(self, state_dim: int = 128, num_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, num_actions)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.action_head(x), self.value_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Detection MLP với EWC Continual Learning
# ─────────────────────────────────────────────────────────────────────────────

class DetectionMLP(nn.Module):
    """
    DetectionMLP phân loại graph latent → Benign / Malicious.

    Bổ sung Elastic Weight Consolidation (EWC) để tránh Catastrophic Forgetting
    khi học liên tục từ các Hard Adversarial Samples mới.

    Input  : graph_latent [B, input_dim] (mean-pooled từ GAT output)
    Output : probabilities [B, 2] — Softmax (0: Benign, 1: Malicious)

    Ref: Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in neural networks"
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 32,
        ewc_lambda: float = 400.0,
    ):
        super().__init__()
        # Sequential khop chinh xac voi checkpoint mlp.pth:
        #   net.0 = Linear(input_dim, hidden_dim)
        #   net.1 = ReLU
        #   net.2 = Linear(hidden_dim, 2)
        # Dropout KHONG nam trong Sequential (tranh shift index -> key mismatch)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # net.0
            nn.ReLU(),                          # net.1
            nn.Linear(hidden_dim, 2),           # net.2
        )
        self.dropout = nn.Dropout(0.2)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: dict = {}     # param_name -> Fisher diagonal tensor
        self.optimal_params: dict = {}  # param_name -> W* tensor (anchor snapshot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)          # Linear
        x = self.net[1](x)          # ReLU
        x = self.dropout(x)         # Dropout (manual, khong trong Sequential)
        x = self.net[2](x)          # Linear -> 2 logits
        return F.softmax(x, dim=-1)

    def compute_fisher(
        self,
        anchor_data_loader,
        device: torch.device,
        num_samples: int = 200,
    ):
        """
        Tính Fisher Information matrix (diagonal) trên tập anchor.

        Args:
            anchor_data_loader : iterable của (x_tensor, y_tensor)
            device             : torch.device
            num_samples        : số mẫu tối đa
        """
        self.train()
        # Snapshot trọng số hiện tại làm W* (anchor sau task cũ)
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        criterion = nn.CrossEntropyLoss()
        count = 0

        for x_batch, y_batch in anchor_data_loader:
            if count >= num_samples:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            self.zero_grad()
            output = self.forward(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            count += x_batch.size(0)

        n = max(count, 1)
        self.fisher_dict = {name: f / n for name, f in fisher.items()}
        self.eval()
        print(f"[EWC] Computed Fisher Information on {count} anchor samples.")

    def ewc_loss(self) -> torch.Tensor:
        """
        EWC regularization loss:
            L_EWC = λ/2 * Σ_i F_i * (θ_i − θ*_i)²
        """
        if not self.fisher_dict or not self.optimal_params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher    = self.fisher_dict[name].to(param.device)
                opt_param = self.optimal_params[name].to(param.device)
                loss = loss + (fisher * (param - opt_param) ** 2).sum()

        return (self.ewc_lambda / 2.0) * loss
