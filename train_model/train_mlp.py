import sys
import random
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import necessary classes and functions from the main module
from networks import GAT, DetectionMLP, PositionalEncoder
from graph_utils import prepare_graph, infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_mlp():
    w2v_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn.model")
    encoder = PositionalEncoder(30)
    try:
        w2vmodel = Word2Vec.load(w2v_model_path)
    except:
        print(f"Error: Word2Vec model not found at {w2v_model_path}")
        return
    
    gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(device)
    
    print("\nPreparing dataset from 'unicorn' folder...")
    dataset_graphs = []
    
    for i in range(150):
        fpath = os.path.join(BASE_DIR, "unicorn", f"{i}.txt")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_csv(fpath, sep='\t', names=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
            
            # 0 -> 124: Benign (label 0)
            # 125 -> 149: Attack (label 1)
            label = 0 if i < 125 else 1
            
            phrases, _, edges, _ = prepare_graph(df)
            if len(phrases) == 0: continue
            
            nodes = [infer(x, w2vmodel, encoder) for x in phrases]
            x_tensor = torch.tensor(np.array(nodes), dtype=torch.float32)
            edge_index_tensor = torch.tensor(edges, dtype=torch.long)
            
            # Create PyG Data object
            data = Data(x=x_tensor, edge_index=edge_index_tensor, y=torch.tensor([label], dtype=torch.float32))
            dataset_graphs.append(data)
        except Exception as e:
            continue
        
    num_attack = sum(1 for data in dataset_graphs if data.y.item() == 1)
    num_benign = sum(1 for data in dataset_graphs if data.y.item() == 0)
    print(f"Extracted {len(dataset_graphs)} samples. Attack (1): {num_attack}, Benign (0): {num_benign}")
    
    # Oversampling
    if num_attack > 0 and num_benign > num_attack:
        num_to_duplicate = num_benign - num_attack
        print(f"Applying Oversampling: Adding {num_to_duplicate} Attack samples.")
        attack_graphs = [g for g in dataset_graphs if g.y.item() == 1]
        oversampled = random.choices(attack_graphs, k=num_to_duplicate)
        dataset_graphs.extend(oversampled)
        
    gat_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn0_gat.pth")
    if os.path.exists(gat_model_path):
        gat.load_state_dict(torch.load(gat_model_path, map_location=device))
        print(f"Loaded pre-trained GAT from {gat_model_path}")
    else:
        print(f"Warning: Pre-trained GAT not found at {gat_model_path}. You should run pretrain_gat.py first.")

    # Train GAT and MLP end-to-end
    for param in gat.parameters():
        param.requires_grad = True
    gat.train()

    # MLP to evaluate
    mlp = DetectionMLP(input_dim=20, hidden_dim=32).to(device)
    
    criterion = nn.BCELoss()
    
    # Optimizer for BOTH GAT and MLP
    optimizer = optim.Adam(list(gat.parameters()) + list(mlp.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Create DataLoader for batching (reduced batch size to prevent CUDA OOM)
    loader = DataLoader(dataset_graphs, batch_size=4, shuffle=True)
    
    epochs = 30
    print(f"\nStarting End-to-End Supervised Training (GAT + MLP) for {epochs} epochs...")
    for epoch in range(epochs):
        gat.train()
        mlp.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # GAT pass (end-to-end)
            latent_features = gat(batch.x, batch.edge_index)
            
            # Global mean pool across the batch
            from torch_geometric.nn import global_mean_pool
            graph_latent = global_mean_pool(latent_features, batch.batch)
            
            # MLP pass
            out = mlp(graph_latent)
            attack_prob = out[:, 1]
            
            # Fix dimension mismatch: batch.y is [batch_size, 1], attack_prob is [batch_size]
            y_target = batch.y.squeeze(-1) if batch.y.dim() > 1 else batch.y
            
            loss = criterion(attack_prob, y_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch.num_graphs
            
            preds = (attack_prob > 0.5).float()
            correct += (preds == y_target).sum().item()
            total += batch.num_graphs
            
        scheduler.step()
        
        epoch_loss /= total
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if epoch_acc >= 0.98 and epoch >= 5:
            print(f"Early stopping triggered at epoch {epoch+1}! Accuracy reached {epoch_acc:.4f} >= 0.98")
            break
            
    # Save weights
    mlp_save_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "mlp.pth")
    gat_save_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn0_gat.pth")
    os.makedirs(os.path.dirname(mlp_save_path), exist_ok=True)
    
    torch.save(mlp.state_dict(), mlp_save_path)
    torch.save(gat.state_dict(), gat_save_path)
    print(f"\nTraining complete! Saved MLP to '{mlp_save_path}' and updated GAT to '{gat_save_path}'")

if __name__ == "__main__":
    train_mlp()
