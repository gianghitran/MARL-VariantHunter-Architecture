import os
import sys
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import global_mean_pool

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from networks import GAT, DetectionMLP, PositionalEncoder
from graph_utils import prepare_graph, infer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Word2Vec and Encoder
    w2v_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn.model")
    encoder = PositionalEncoder(30)
    w2vmodel = Word2Vec.load(w2v_model_path)
    
    # 1. Load GAT model
    gat_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn0_gat.pth")
    # Must match train_mlp.py architecture exactly!
    gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(device)
    gat.load_state_dict(torch.load(gat_path, map_location=device, weights_only=True))
    gat.eval()
    print("Loaded GAT model.")
    
    # 2. Load MLP model
    mlp_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "mlp.pth")
    mlp = DetectionMLP(input_dim=20).to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp.eval()
    print("Loaded DetectionMLP model.")
    
    # 3. Load dataset
    print("\nPreparing dataset from 'unicorn' folder...")
    dataset_graphs = []
    benign_count = 0
    attack_count = 0
    
    for i in range(150):
        fpath = os.path.join(BASE_DIR, "unicorn", f"{i}.txt")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_csv(fpath, sep='\t', names=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
            label = 0 if i < 125 else 1
            
            phrases, _, edges, _ = prepare_graph(df)
            if len(phrases) == 0: continue
            
            nodes = [infer(x, w2vmodel, encoder) for x in phrases]
            x_tensor = torch.tensor(np.array(nodes), dtype=torch.float32)
            edge_index_tensor = torch.tensor(edges, dtype=torch.long)
            
            from torch_geometric.data import Data
            data = Data(x=x_tensor, edge_index=edge_index_tensor, y=torch.tensor([label], dtype=torch.float32))
            dataset_graphs.append(data)
            if label == 1: attack_count += 1
            else: benign_count += 1
        except Exception as e:
            continue
            
    print(f"Extracted {len(dataset_graphs)} samples. Attack (1): {attack_count}, Benign (0): {benign_count}")
    
    # Do not oversample for evaluation
    loader = DataLoader(dataset_graphs, batch_size=4, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # End-to-end forward pass
            latent_features = gat(batch.x, batch.edge_index)
            graph_latent = global_mean_pool(latent_features, batch.batch)
            probs = mlp(graph_latent)
            
            # Predict
            preds = probs.argmax(dim=1)
            prob_mal = probs[:, 1]
            
            all_preds.extend(preds.cpu().tolist())
            y_target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            all_labels.extend(y_target.cpu().tolist())
            all_probs.extend(prob_mal.cpu().tolist())
            
    print("\n[Eval Post-EWC] === Detection MLP Evaluation (Real Test Set) ===")
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    TP = sum((p == 1 and y == 1) for p, y in zip(all_preds, all_labels))
    FP = sum((p == 1 and y == 0) for p, y in zip(all_preds, all_labels))
    FN = sum((p == 0 and y == 1) for p, y in zip(all_preds, all_labels))
    TN = sum((p == 0 and y == 0) for p, y in zip(all_preds, all_labels))
    
    print(f"  Samples : {len(all_labels)} (mal={sum(all_labels)}, ben={len(all_labels) - sum(all_labels)})")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Confusion: TP={TP} FP={FP} FN={FN} TN={TN}")
    
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Benign", "Malicious"], zero_division=0))

if __name__ == "__main__":
    main()
