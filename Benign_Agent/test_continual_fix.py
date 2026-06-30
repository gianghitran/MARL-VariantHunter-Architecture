"""
Test dut khoat cho FIX continual-learning:
  Truoc fix: TP/FP/TN dong bang, FN tang -> model KHONG hoc mau FN camo moi.
  Sau  fix: chay NHIEU EWC update -> recall tren camo HELD-OUT phai TANG,
            dong thoi benign precision (TN) phai GIU (khong sup FP).

Cach lam: populate env.replay_buffer bang camouflaged APT (label=1), do held-out
camo recall + benign acc TRUOC, chay K EWC update, do lai SAU. KHONG dung Attack
Agent (chi can benign-gen + APT JSON co san) -> nhanh, khong dung internals attack.
"""
import os, sys, glob, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)
import numpy as np, torch, pandas as pd
from graph_utils import prepare_graph, infer
from benign_interface import build_benign_corpus, bundle_to_edges_df
from environment import ProvenanceGraphEnv

dev = "cpu"


def tensors(df):
    ph, _, e, _ = prepare_graph(df)
    if len(ph) == 0 or len(e[0]) == 0:
        return None
    return (torch.tensor(np.array([infer(p) for p in ph]), dtype=torch.float32),
            torch.tensor(e, dtype=torch.long))


# ── Du lieu: APT tu JSON + benign-gen pool (camouflage) ──────────────────────
apts = []
for f in sorted(glob.glob(os.path.join(BASE, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
    d = json.load(open(f, encoding="utf-8"))
    apts.append(bundle_to_edges_df({"nodes": d["nodes"], "edges": d["edges"]}))
bcorpus = build_benign_corpus(60, source="hybrid", mimicry_level=0.4, seed=7)
benign_pool = pd.concat([bundle_to_edges_df(b) for b in bcorpus], ignore_index=True)


def camouflage(apt, ratio=1.0, rng=None):
    k = int(len(apt) * ratio)
    camo = benign_pool.sample(n=k, replace=(k > len(benign_pool)), random_state=rng)
    return pd.concat([apt, camo], ignore_index=True)


rng = np.random.RandomState(0)
idx = list(range(len(apts))); rng.shuffle(idx)
tr_apt, te_apt = idx[: len(apts) * 2 // 3], idx[len(apts) * 2 // 3:]

# camo HELD-OUT (de do hoc THAT su, khong phai memorize) — ratio cao = kho
heldout = [tensors(camouflage(apts[i], 1.0, rng)) for i in te_apt]
heldout = [t for t in heldout if t]
# clean APT held-out (tran cua separation: neu clean cung khong tach => feature yeu)
clean_heldout = [tensors(apts[i]) for i in te_apt]
clean_heldout = [t for t in clean_heldout if t]
# benign held-out (kiem tra khong sup precision)
ben_heldout = [tensors(bundle_to_edges_df(b)) for b in bcorpus[40:]]
ben_heldout = [t for t in ben_heldout if t]

# ── Env (dung GAT+MLP + optimizer + EWC THAT cua he thong) ───────────────────
os.makedirs("runs/_test_cl", exist_ok=True)
env = ProvenanceGraphEnv(gcn_model_path="trained_weights/darpa/gat.pth",
                         w2v_model_path="trained_weights/darpa/w2v.model",
                         run_dir="runs/_test_cl")


def measure():
    env.gat.eval(); env.detection_mlp.eval()
    with torch.no_grad():
        mc = [env.detection_mlp(env.gat.graph_latent(x, e))[0, 1].item() for x, e in heldout]
        bc = [env.detection_mlp(env.gat.graph_latent(x, e))[0, 1].item() for x, e in ben_heldout]
        cc = [env.detection_mlp(env.gat.graph_latent(x, e))[0, 1].item() for x, e in clean_heldout]
    mc, bc, cc = np.array(mc), np.array(bc), np.array(cc)
    camo_recall  = float((mc >= 0.5).mean())
    clean_recall = float((cc >= 0.5).mean())
    ben_acc      = float((bc < 0.5).mean())   # benign duoc giu dung
    return camo_recall, mc.mean(), ben_acc, bc.mean(), clean_recall


# Populate buffer bang camo TRAIN samples (label=1) — chinh la cac Hard Sample FN
for i in tr_apt:
    t = tensors(camouflage(apts[i], 1.0, rng))
    if not t:
        continue
    env.replay_buffer.add({
        "x_tensor": t[0], "edge_index": t[1], "label": 1,
        "mlp_prob": 0.3, "anomaly_ratio": 0.2, "grad_norms": None, "attention_max": 0.0,
    }, priority=0.9)

r0, m0, b0, bm0, cr0 = measure()
print(f"TESTLOG BEFORE: camo_recall={r0:.2f} (mal_conf={m0:.3f}) | clean_recall={cr0:.2f} | "
      f"benign_acc={b0:.2f} (ben_conf={bm0:.3f}) | gap={m0-bm0:+.3f} | buffer={len(env.replay_buffer)}")

print("TESTLOG === chay 24 EWC update (mo phong 24 episode continual, do drift dai han) ===")
for u in range(24):
    env.train_detection_agent_ewc(run_dir="runs/_test_cl", batch_size=8)
    r, m, b, bm, cr = measure()
    print(f"TESTLOG  update{u+1:02d}: camo_recall={r:.2f} (mal={m:.3f}) clean={cr:.2f} | "
          f"benign_acc={b:.2f} (ben={bm:.3f}) | gap={m-bm:+.3f}")

r1, m1, b1, bm1, cr1 = measure()
print(f"TESTLOG AFTER : camo_recall={r1:.2f} clean_recall={cr1:.2f} benign_acc={b1:.2f} | gap={m1-bm1:+.3f}")
# Tieu chi: hoc duoc camo (recall tang) VA giu precision (benign_acc khong sup)
learned      = r1 > r0 + 0.2
precision_ok = b1 >= 0.6
balanced     = learned and precision_ok
print(f"TESTLOG => continual {'CAN BANG: hoc FN moi + giu precision' if balanced else ('HOC nhung SUP precision' if learned else 'CHUA hoc')} "
      f"(Δcamo_recall={r1-r0:+.2f}, benign_acc={b1:.2f})")
print("TESTLOG DONE")
