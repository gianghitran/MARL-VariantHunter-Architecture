import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

from networks import GAT, PositionalEncoder
from graph_utils import prepare_graph, infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def pretrain_gat():
    w2v_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn.model")
    encoder = PositionalEncoder(30)
    try:
        w2vmodel = Word2Vec.load(w2v_model_path)
    except:
        print(f"Error: Word2Vec model not found at {w2v_model_path}")
        return
    
    gat = GAT(in_channels=30, out_channels=20).to(device)
    optimizer = optim.Adam(gat.parameters(), lr=0.005)
    
    print("Preparing graphs for unsupervised link prediction...")
    dataset = []
    for i in range(150):
        fpath = os.path.join(BASE_DIR, "unicorn", f"{i}.txt")
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath, sep='\t', names=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        phrases, _, edge_idx, _ = prepare_graph(df)
        if len(phrases) == 0 or len(edge_idx[0]) == 0: continue
        
        nodes = [infer(x, w2vmodel, encoder) for x in phrases]
        x_t = torch.tensor(np.array(nodes), dtype=torch.float32).to(device)
        e_t = torch.tensor(edge_idx, dtype=torch.long).to(device)
        dataset.append((x_t, e_t))
        
    print(f"Loaded {len(dataset)} graphs.")
    
    epochs = 30
    for epoch in range(epochs):
        gat.train()
        total_loss = 0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x_t, e_t in pbar:
            optimizer.zero_grad()
            z = gat(x_t, e_t) # [N, 20]
            
            # Positive edges
            pos_src, pos_dst = e_t[0], e_t[1]
            pos_score = (z[pos_src] * z[pos_dst]).sum(dim=-1)
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            
            # Negative edges (random sampling)
            num_neg = e_t.size(1)
            neg_src = torch.randint(0, x_t.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, x_t.size(0), (num_neg,), device=device)
            neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
            
    gat_model_path = os.path.join(BASE_DIR, "trained_weights", "unicorn", "unicorn0_gat.pth")
    torch.save(gat.state_dict(), gat_model_path)
    print(f"Saved Pre-trained GAT to {gat_model_path}")

if __name__ == '__main__':
    pretrain_gat()
