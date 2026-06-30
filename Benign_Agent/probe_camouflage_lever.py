"""
Kiểm chứng đòn bẩy evasion env-level: CAMOUFLAGE (trộn benign-gen quanh APT subgraph).
Nếu tăng camouflage => conf giảm và vượt <0.5 => attacker có đòn bẩy evasion thật
(benign mimicry) -> tạo FN -> kích hoạt EWC. Dùng benign-gen của ta + APT từ JSON.
KHÔNG sửa Attack Agent.
"""
import os, sys, glob, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)
import numpy as np, torch, pandas as pd
from networks import GAT, DetectionMLP
from graph_utils import prepare_graph, infer
from benign_interface import build_benign_corpus, bundle_to_edges_df

dev = "cpu"
gat = GAT(30, 20, 64, 8, 0.3); gat.load_state_dict(torch.load("trained_weights/darpa/gat.pth", map_location=dev)); gat.eval()
mlp = DetectionMLP(20, 32); mlp.load_state_dict(torch.load("trained_weights/darpa/mlp.pth", map_location=dev), strict=False); mlp.eval()


def conf_of(df):
    phrases, _, edges, _ = prepare_graph(df)
    if len(phrases) == 0 or len(edges[0]) == 0:
        return None
    x = torch.tensor(np.array([infer(p) for p in phrases]), dtype=torch.float32)
    e = torch.tensor(edges, dtype=torch.long)
    with torch.no_grad():
        return mlp(gat.graph_latent(x, e))[0, 1].item()   # attention-weighted pooling


# APT subgraphs (malicious) từ detection JSON
apts = []
for f in sorted(glob.glob(os.path.join(BASE, "Attack_Agent", "result_handoff", "*", "*_detection.json")))[:5]:
    d = json.load(open(f, encoding="utf-8"))
    apts.append(bundle_to_edges_df({"nodes": d["nodes"], "edges": d["edges"]}))

# camouflage pool = benign-gen edges
bcorpus = build_benign_corpus(40, source="hybrid", mimicry_level=0.4, seed=7)
benign_edges = pd.concat([bundle_to_edges_df(b) for b in bcorpus], ignore_index=True)

print("TESTLOG camo_ratio | apt_edges | total_edges | mean_conf | n_evade")
for ratio in [0.0, 0.5, 1.0, 2.0, 4.0]:
    confs = []; nev = 0
    for apt in apts:
        k = int(len(apt) * ratio)
        camo = benign_edges.sample(n=min(k, len(benign_edges)), replace=(k > len(benign_edges)), random_state=1) if k > 0 else benign_edges.iloc[:0]
        combined = pd.concat([apt, camo], ignore_index=True)
        c = conf_of(combined)
        if c is not None:
            confs.append(c); nev += int(c < 0.5)
    print(f"TESTLOG {ratio:.1f} | ~{int(np.mean([len(a) for a in apts]))} | {int(np.mean([len(a) for a in apts])*(1+ratio))} | {np.mean(confs):.3f} | {nev}/{len(confs)}")
print("TESTLOG PROBE DONE")
