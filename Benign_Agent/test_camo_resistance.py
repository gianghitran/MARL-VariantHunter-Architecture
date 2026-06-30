"""
Test dứt khoát: với ATTENTION POOLING, detector có HỌC ĐƯỢC bắt camouflaged APT
sau khi train trên chúng không? (mean-pool không học được vì không có focus).

Đo recall trên camouflaged HELD-OUT trước/sau khi train vài bước trên camo+clean+benign.
"""
import os, sys, glob, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)
import numpy as np, torch, pandas as pd
from networks import GAT, DetectionMLP
from graph_utils import prepare_graph, infer
from benign_interface import build_benign_corpus, bundle_to_edges_df

dev = "cpu"
gat = GAT(30, 20, 64, 8, 0.3); gat.load_state_dict(torch.load("trained_weights/darpa/gat.pth", map_location=dev))
mlp = DetectionMLP(20, 32); mlp.load_state_dict(torch.load("trained_weights/darpa/mlp.pth", map_location=dev), strict=False)


def tensors(df):
    ph, _, e, _ = prepare_graph(df)
    if len(ph) == 0 or len(e[0]) == 0:
        return None
    return (torch.tensor(np.array([infer(p) for p in ph]), dtype=torch.float32),
            torch.tensor(e, dtype=torch.long))


# APT từ JSON
apts = []
for f in sorted(glob.glob(os.path.join(BASE, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
    apts.append(bundle_to_edges_df({"nodes": (d := json.load(open(f, encoding="utf-8")))["nodes"], "edges": d["edges"]}))
# benign pool cho camouflage + benign class
bcorpus = build_benign_corpus(60, source="hybrid", mimicry_level=0.4, seed=7)
benign_pool = pd.concat([bundle_to_edges_df(b) for b in bcorpus], ignore_index=True)


def camouflage(apt, ratio=1.0, rng=None):
    k = int(len(apt) * ratio)
    camo = benign_pool.sample(n=k, replace=(k > len(benign_pool)), random_state=rng)
    return pd.concat([apt, camo], ignore_index=True)


rng = np.random.RandomState(0)
# Dataset: benign(0), clean-mal(1), camo-mal(1). Split train/test theo apt index.
n = len(apts); idx = list(range(n)); rng.shuffle(idx)
tr_apt, te_apt = idx[: n * 2 // 3], idx[n * 2 // 3:]
def build(split, ratio):
    data = []
    for i in split:
        t = tensors(camouflage(apts[i], ratio, rng))
        if t: data.append((t[0], t[1], 1))
    return data
train_camo = build(tr_apt, 1.0)
test_camo  = build(te_apt, 1.0)
train_clean = [(*tensors(apts[i]), 1) for i in tr_apt if tensors(apts[i])]
benign_data = [(*t, 0) for b in bcorpus[:40] if (t := tensors(bundle_to_edges_df(b)))]


def camo_recall():
    gat.eval(); mlp.eval()
    confs = []
    with torch.no_grad():
        for x, e, _ in test_camo:
            confs.append(mlp(gat.graph_latent(x, e))[0, 1].item())
    confs = np.array(confs)
    return float((confs >= 0.5).mean()), float(confs.mean())


r0, c0 = camo_recall()
print(f"TESTLOG BEFORE train: camo recall(held-out)={r0:.2f} avg_conf={c0:.3f}")

# Train vài bước trên {benign(0), clean-mal(1), camo-mal(1)} — attention pooling, BN-freeze
opt = torch.optim.Adam(list(gat.parameters()) + list(mlp.parameters()), lr=1e-3, weight_decay=1e-4)
ds = benign_data + train_clean + train_camo
nb = len(benign_data); nm = len(train_clean) + len(train_camo)
for ep in range(25):
    gat.train(); mlp.train()
    for m in gat.modules():
        if isinstance(m, torch.nn.BatchNorm1d): m.eval()
    opt.zero_grad(); loss = 0.0
    for x, e, y in ds:
        p = mlp(gat.graph_latent(x, e))[0, 1]
        w = (nb / nm) if y == 1 else 1.0
        loss = loss - w * (y * torch.log(p + 1e-9) + (1 - y) * torch.log(1 - p + 1e-9))
    (loss / len(ds)).backward(); opt.step()

r1, c1 = camo_recall()
print(f"TESTLOG AFTER  train: camo recall(held-out)={r1:.2f} avg_conf={c1:.3f}")
print(f"TESTLOG => attention pooling {'HỌC ĐƯỢC bắt camo' if r1 > r0 + 0.3 else 'chưa cải thiện rõ'} (Δrecall={r1-r0:+.2f})")
print("TESTLOG DONE")
