import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

# Import necessary classes and functions from the main module
from marl_provenance_ppo import GCN, DetectionMLP, prepare_graph, infer, PositionalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# ==========================================
# Cấu hình đường dẫn gốc (Base Directory)
# ==========================================
BASE_DIR = r"D:\\notebook_UITNam3\\Nam3_ki2\\dacn\\reinforcement-learning-in-llm"

def extract_features():
    gcn_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn0.pth")
    w2v_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn.model")
    
    # Load pre-trained models
    gcn = GCN(30, 14).to(device)
    if os.path.exists(gcn_model_path):
        gcn.load_state_dict(torch.load(gcn_model_path, map_location=device))
    else:
        print(f"Error: GCN model not found at {gcn_model_path}")
        return [], []
    gcn.eval()
    
    encoder = PositionalEncoder(30)
    try:
        w2vmodel = Word2Vec.load(w2v_model_path)
    except:
        print(f"Error: Word2Vec model not found at {w2v_model_path}")
        return [], []
    
    latent_vectors = []
    labels = []
    
    print("Extracting Latent Features from 150 graph files...")
    # Use tqdm to display progress bar
    for i in tqdm(range(150), desc="Processing Graphs"):
        file_path = os.path.join(BASE_DIR, "unicorn", f"{i}.txt")
        if not os.path.exists(file_path):   
            continue
            
        try:
            # Read dataset file
            df = pd.read_csv(file_path, sep='\t', names=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
            
            # Label assignment: 0 to 124 are Benign (0), 125 to 149 are Attack (1)
            label = 0 if i < 125 else 1
            
            # Convert to Node and Edge lists
            phrases, node_labels, edges, _ = prepare_graph(df)
            if len(phrases) == 0:
                continue
                
            # Word2Vec Embedding
            nodes = [infer(x, w2vmodel, encoder) for x in phrases]
            x_tensor = torch.tensor(np.array(nodes), dtype=torch.float32).to(device)
            edge_index_tensor = torch.tensor(edges, dtype=torch.long).to(device)
            
            # Pass through GCN (frozen weights) to extract features
            with torch.no_grad():
                latent_features = gcn.conv1(x_tensor, edge_index_tensor).relu()
                latent_features = gcn.conv2(latent_features, edge_index_tensor)
                # Mean Pooling to aggregate into a single Vector (1, 20) representing the graph
                graph_latent = latent_features.mean(dim=0)
                
            latent_vectors.append(graph_latent.cpu().numpy())
            labels.append(label)
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            
    return np.array(latent_vectors), np.array(labels)

def train_mlp():
    X, y = extract_features()
    
    if len(X) == 0:
        print("No valid data. Exiting.")
        return
        
    print(f"\nExtracted {len(X)} samples. Attack (1): {np.sum(y)}, Benign (0): {len(y) - np.sum(y)}")
    
    # ---------------------------------------------------------
    # Oversampling: Handling data imbalance
    # ---------------------------------------------------------
    attack_indices = np.where(y == 1)[0]
    benign_indices = np.where(y == 0)[0]
    
    # Calculate the required number of duplicates
    num_to_duplicate = len(benign_indices) - len(attack_indices)
    if num_to_duplicate > 0 and len(attack_indices) > 0:
        print(f"Applying Oversampling: Adding {num_to_duplicate} Attack samples via sampling with replacement.")
        oversampled_indices = np.random.choice(attack_indices, num_to_duplicate, replace=True)
        X = np.concatenate([X, X[oversampled_indices]], axis=0)
        y = np.concatenate([y, y[oversampled_indices]], axis=0)
        
    print(f"After Oversampling -> Attack (1): {np.sum(y)}, Benign (0): {len(y) - np.sum(y)}")
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device) 
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize the model
    mlp = DetectionMLP(20, 32).to(device)
    
    # Using BCELoss. MLP output is Softmax(dim=-1) so we take the Attack column: out[:, 1]
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.005)
    
    epochs = 50
    print(f"\nStarting Training for {epochs} epochs...")
    mlp.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            out = mlp(batch_x)
            attack_prob = out[:, 1] # Column 1 is the probability of the Attack class
            
            loss = criterion(attack_prob, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
            # Calculate accuracy (Threshold = 0.5)
            preds = (attack_prob > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
        epoch_loss /= total
        epoch_acc = correct / total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
            
    # Save weights to file
    save_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "mlp.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mlp.state_dict(), save_path)
    print(f"\nTraining complete! Pre-trained weights saved to '{save_path}'")

if __name__ == "__main__":
    train_mlp()
