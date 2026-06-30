"""
Kiểm chứng đòn bẩy evasion: max_stages (độ dài kill-chain) -> causal depth -> conf.
Nếu ít stage => conf thấp hơn (tiến tới <0.5) thì variant-selection bandit theo
max_stages sẽ tạo được FN -> kích hoạt EWC. KHÔNG sửa Attack Agent.
"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)
import numpy as np, torch, networkx as nx
from networks import GAT, DetectionMLP
from graph_utils import prepare_graph, infer
sys.path.insert(0, os.path.join(BASE, "Attack_Agent"))
from generation_interface import generate_apt_subgraph

dev = "cpu"
gat = GAT(30, 20, 64, 8, 0.3); gat.load_state_dict(torch.load("trained_weights/darpa/gat.pth", map_location=dev)); gat.eval()
mlp = DetectionMLP(20, 32); mlp.load_state_dict(torch.load("trained_weights/darpa/mlp.pth", map_location=dev), strict=False); mlp.eval()


def score(edges_df):
    phrases, _, edges, _ = prepare_graph(edges_df)
    if len(phrases) == 0 or len(edges[0]) == 0:
        return None, None
    x = torch.tensor(np.array([infer(p) for p in phrases]), dtype=torch.float32)
    e = torch.tensor(edges, dtype=torch.long)
    with torch.no_grad():
        conf = mlp(gat(x, e).mean(0, keepdim=True))[0, 1].item()
    G = nx.DiGraph(); G.add_edges_from(zip(edges[0], edges[1]))
    depth = nx.dag_longest_path_length(nx.condensation(G))
    return conf, depth


print("TESTLOG stages | edges | depth | conf | pred")
for s in [2, 2, 3, 3, 4, 5, 6]:
    out = generate_apt_subgraph(max_stages=s)
    df = out["edges_df"]
    if df.empty:
        print(f"TESTLOG s={s} EMPTY"); continue
    conf, depth = score(df)
    if conf is None:
        print(f"TESTLOG s={s} no-edges"); continue
    print(f"TESTLOG s={s} | {len(df):3d} | {depth:2d} | {conf:.3f} | {'EVADE(FN)' if conf < 0.5 else 'detected'} | {out.get('variant')}")
print("TESTLOG PROBE DONE")
