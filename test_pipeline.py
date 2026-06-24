"""
Script kiem tra pipeline: GAT -> MLP
Kiểm tra xem tại sao accuracy=0 trong hệ thống
"""
import torch
import os
import numpy as np
import sys

sys.path.insert(0, 'd:/notebook_UITNam3/Nam3_ki2/dacn/TAGAPT/TAGAPT_APTDetection')

from networks import GAT, DetectionMLP, PositionalEncoder
from graph_utils import infer, prepare_graph
from gensim.models import Word2Vec
import pandas as pd

base     = 'd:/notebook_UITNam3/Nam3_ki2/dacn/TAGAPT/TAGAPT_APTDetection'
w2v_path = os.path.join(base, 'trained_weights', 'unicorn', 'unicorn.model')
gat_path = os.path.join(base, 'trained_weights', 'unicorn', 'unicorn0.pth')
mlp_path = os.path.join(base, 'trained_weights', 'unicorn', 'mlp.pth')

print("=" * 60)
print("DIAGNOSTIC: GAT -> MLP Pipeline Check")
print("=" * 60)

# 1. Check weight files
for label, path in [("GAT", gat_path), ("MLP", mlp_path), ("W2V", w2v_path)]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"[{label}] Exists={exists} | Size={size/1024:.1f} KB | Path={path}")

print()

# 2. Load MLP and inspect weights
print("--- MLP Weight Inspection ---")
mlp = DetectionMLP(input_dim=20, hidden_dim=32)
if os.path.exists(mlp_path):
    state = torch.load(mlp_path, map_location='cpu')
    print(f"Keys in mlp.pth: {list(state.keys())}")
    for k, v in state.items():
        print(f"  {k}: shape={tuple(v.shape)}, mean={v.float().mean():.4f}, std={v.float().std():.4f}")
    result = mlp.load_state_dict(state, strict=False)
    print(f"load_state_dict result: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
else:
    print("mlp.pth NOT FOUND!")

mlp.eval()

# 3. Load GAT
print()
print("--- GAT Weight Inspection ---")
gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3)
if os.path.exists(gat_path):
    state = torch.load(gat_path, map_location='cpu')
    print(f"Keys in gat.pth: {list(state.keys())[:5]} ...")
    result = gat.load_state_dict(state, strict=False)
    print(f"load_state_dict result: missing={result.missing_keys[:3]}, unexpected={result.unexpected_keys[:3]}")
else:
    print("GAT NOT FOUND, using random init!")

gat.eval()

# 4. Load W2V
print()
print("--- W2V Inspection ---")
try:
    w2v = Word2Vec.load(w2v_path)
    print(f"W2V vocab size = {len(w2v.wv)}")
    print(f"W2V vector size = {w2v.wv.vector_size}")
    # Sample words
    sample_words = list(w2v.wv.key_to_index.keys())[:5]
    print(f"Sample words: {sample_words}")
except Exception as e:
    print(f"W2V load error: {e}")
    w2v = None

encoder = PositionalEncoder(30)

# 5. Build dummy graph and test forward
print()
print("--- Forward Pass Test ---")
dummy = pd.DataFrame({
    'actorID':    ['P1', 'P1', 'P2', 'P3'],
    'actor_type': ['MP', 'MP', 'TP', 'TP'],
    'objectID':   ['F1', 'S1', 'F1', 'S1'],
    'object':     ['MF', 'SO', 'MF', 'SO'],
    'action':     ['WR', 'ST', 'RD', 'RCV'],
    'timestamp':  [1,    2,    3,    4],
})

phrases, labels, edges, node_names = prepare_graph(dummy)
print(f"Nodes: {len(phrases)} | Edge pairs: {len(edges[0])} | Node names: {node_names}")
print(f"Phrases (actions per node): {phrases}")

nodes = [infer(x, w2v, encoder) for x in phrases]
print(f"Node embeddings (first): shape={np.array(nodes[0]).shape}, mean={np.array(nodes[0]).mean():.4f}")

x_tensor    = torch.tensor(np.array(nodes), dtype=torch.float32)
edge_index  = torch.tensor(edges, dtype=torch.long)
print(f"x_tensor: {x_tensor.shape} | edge_index: {edge_index.shape}")

with torch.no_grad():
    node_emb = gat(x_tensor, edge_index)
    print(f"GAT output shape: {node_emb.shape}")
    print(f"GAT output mean: {node_emb.mean():.4f}, std: {node_emb.std():.4f}")

    graph_latent = node_emb.mean(dim=0).unsqueeze(0)
    print(f"graph_latent shape: {graph_latent.shape}")
    print(f"graph_latent values: {graph_latent[0].tolist()[:5]} ...")

    probs = mlp(graph_latent)
    print(f"MLP output (probs): {probs}")
    print(f"Benign prob : {probs[0][0].item():.4f}")
    print(f"Malicious prob: {probs[0][1].item():.4f}")
    predicted = probs.argmax(dim=1).item()
    print(f"Predicted class: {'Malicious' if predicted == 1 else 'Benign'}")
