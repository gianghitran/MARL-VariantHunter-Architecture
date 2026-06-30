"""
pretrain_on_generated.py
========================
Bước 4 — pretrain GAT + DetectionMLP trên dataset CÙNG MỨC:
  benign  = Benign Generator (hybrid: UNICORN-derived + synthetic)  -> label 0
  malicious = Attack Agent (*_detection.json, KHÔNG sửa Attack Agent) -> label 1

Hai pha (mirror cách đã sửa thành công cho UNICORN):
  Pha 1: GAT link-prediction pretrain (BN train -> set running stats) trên tất cả graph.
  Pha 2: GAT+MLP supervised (BN-FREEZE -> khớp inference) với class-weight.

Lưu vào trained_weights/darpa/{gat.pth, mlp.pth} (env load sẵn các path này).
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_DIR)
for p in (BASE_DIR, _DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from networks import GAT, DetectionMLP
from graph_utils import prepare_graph, infer
from benign_interface import build_benign_corpus, build_malicious_corpus, bundle_to_edges_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nguồn dữ liệu pretrain (đổi tại đây hoặc qua biến môi trường PRETRAIN_SOURCE):
#   'darpa'        : DARPA TC E3 thật (khuyến nghị — subgraph quanh node attack có nhãn)
#   'unicorn'      : UNICORN 0-124 benign / 125-149 attack (KHÔNG tách được, F1≈0)
#   'attack_agent' : Attack Agent kill-chains (synthetic, tách tốt nhưng gắn result_handoff)
PRETRAIN_SOURCE = os.environ.get("PRETRAIN_SOURCE", "darpa")
DARPA_DATASET   = os.environ.get("DARPA_DATASET", "cadets")


def graph_to_tensors(edges_df):
    phrases, _, edges, _ = prepare_graph(edges_df)
    if len(phrases) == 0 or len(edges[0]) == 0:
        return None
    x = torch.tensor(np.array([infer(p) for p in phrases]), dtype=torch.float32)
    e = torch.tensor(edges, dtype=torch.long)
    return x.to(device), e.to(device)


def _corpus_to_data(benign_bundles, malicious_bundles):
    """Bundle (benign/malicious) -> list (x, edge_index, label) qua build_node_feature."""
    data, n_mal = [], 0
    for b in benign_bundles:
        t = graph_to_tensors(bundle_to_edges_df(b))
        if t:
            data.append((t[0], t[1], 0))
    for m in malicious_bundles:
        t = graph_to_tensors(bundle_to_edges_df(m))
        if t:
            data.append((t[0], t[1], 1))
            n_mal += 1
    return data, len(data) - n_mal, n_mal


def _build_camo_pool(benign_bundles):
    """Pool cạnh benign (giống env _get_camouflage_pool) để trộn camouflage."""
    dfs = [bundle_to_edges_df(b) for b in benign_bundles]
    dfs = [d for d in dfs if len(d) > 0]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _corpus_to_data_camo(benign_bundles, malicious_bundles, camo_pool, ratios, seed=0):
    """
    Như _corpus_to_data nhưng AUGMENT mỗi malicious bằng các biến thể CAMOUFLAGE
    (APT-core + benign edges ở nhiều ratio) — vẫn nhãn malicious. Dạy detector bắt
    APT-core dù bị trộn benign (khớp đối thủ vòng kín: camouflage = LOTL/benign mimicry).
    Lưu ý: class-weight w_mal = n_ben/n_mal tự chuẩn hoá nên thêm biến thể malicious
    không phá cân bằng tổng gradient.
    """
    rng = np.random.default_rng(seed)
    data, n_ben, n_mal = [], 0, 0
    for b in benign_bundles:
        t = graph_to_tensors(bundle_to_edges_df(b))
        if t:
            data.append((t[0], t[1], 0)); n_ben += 1
    for m in malicious_bundles:
        apt = bundle_to_edges_df(m)
        variants = [apt]                       # clean APT
        if camo_pool is not None and len(camo_pool) > 0:
            for r in ratios:
                k = int(len(apt) * r)
                if k <= 0:
                    continue
                camo = camo_pool.sample(n=k, replace=(k > len(camo_pool)),
                                        random_state=int(rng.integers(0, 2**31)))[apt.columns]
                variants.append(pd.concat([apt, camo], ignore_index=True))
        for v in variants:
            t = graph_to_tensors(v)
            if t:
                data.append((t[0], t[1], 1)); n_mal += 1
    return data, n_ben, n_mal


def load_dataset(n_benign=125, n_malicious=125, source=None, mimicry_level=0.5, seed=0):
    """
    Nạp tập pretrain {benign(0), malicious(1)} qua CÙNG pipeline build_node_feature
    (30-dim, KHÔNG hash w2v). `source` (mặc định PRETRAIN_SOURCE):
      - 'darpa'        : DARPA TC E3 — malicious = subgraph quanh node attack có nhãn,
                         benign = vùng benign. type/action THẬT -> tách lớp thật.
      - 'unicorn'      : UNICORN benign 0-124 / attack 125-149 (không tách được).
      - 'attack_agent' : benign UNICORN-derived + malicious = Attack Agent JSON.
    """
    src = source or PRETRAIN_SOURCE

    def _attack_agent_bundles():
        # EXP_PRETRAIN_HELDOUT_APTS (CSV): loai han cac ho APT nay khoi tap pretrain
        # malicious -> de RQ1 test tren ho UNSEEN (model chua tung gap). Mac dinh: khong loai.
        heldout = {a.strip() for a in os.environ.get("EXP_PRETRAIN_HELDOUT_APTS", "").split(",") if a.strip()}
        bundles = []
        skipped = 0
        for f in sorted(glob.glob(os.path.join(BASE_DIR, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
            try:
                d = json.load(open(f, encoding="utf-8"))
                if d.get("apt") in heldout:
                    skipped += 1
                    continue
                bundles.append({"nodes": d["nodes"], "edges": d["edges"]})
            except Exception:
                pass
        if heldout:
            print(f"[Pretrain] HELD-OUT APTs {sorted(heldout)} -> excluded {skipped} variant bundles from training")
        return bundles

    if src == "mixed":
        # MIXED: detector học CẢ HAI miền để Global F1 (DARPA) cao VÀ bắt được biến thể
        # sống của Attack Agent (FN_rate thấp).
        #   benign    = DARPA benign  + Benign Generator (đa dạng hoá benign)
        #   malicious = DARPA malicious + Attack Agent JSON (khớp đối thủ vòng kín)
        from darpa_tc_sampler import build_darpa_corpus
        d_ben, d_mal = build_darpa_corpus(n_benign, n_malicious, dataset=DARPA_DATASET, seed=seed)
        g_ben = build_benign_corpus(n_benign, source="hybrid", mimicry_level=mimicry_level, seed=seed)
        a_mal = _attack_agent_bundles()
        # CÂN BẰNG MIỀN malicious (Fix balance): DARPA (125) >> Attack (27) khiến detector
        # nghiêng hẳn về DARPA -> conf≈0 trên biến thể sống -> attacker áp đảo tức thì.
        # Oversample Attack JSON lên ngang DARPA để detector nhận ra CẢ HAI miền mạnh
        # như nhau (giống attack_agent-only đạt conf 0.8-0.96 trên biến thể sống).
        a_mal_bal = list(a_mal)
        if a_mal and len(a_mal) < len(d_mal):
            reps = (len(d_mal) + len(a_mal) - 1) // len(a_mal)   # ceil
            a_mal_bal = (list(a_mal) * reps)[:len(d_mal)]
        benign    = list(d_ben) + list(g_ben)
        malicious = list(d_mal) + a_mal_bal
        # CAMOUFLAGE-AUGMENT (Fix balance gốc rễ): detector pretrain trên malicious SẠCH
        # -> chỉ cần trộn 25% benign là conf sụp 0.999->0.006 (camouflage né tầm thường,
        # attacker áp đảo tức thì, không có arms race). Train thêm biến thể APT+camo ở
        # nhiều ratio để detector học bắt APT-core dù bị trộn benign.
        camo_pool = _build_camo_pool(benign)
        data, n_ben, n_mal = _corpus_to_data_camo(
            benign, malicious, camo_pool, ratios=[0.5, 1.0, 2.0], seed=seed
        )
        print(f"[Dataset] (MIXED-balanced+CAMO) benign={n_ben} (DARPA {len(d_ben)} + Gen {len(g_ben)}) | "
              f"malicious={n_mal} (clean {len(d_mal)+len(a_mal_bal)} + camo-augmented, ratios 0.5/1/2)")
        return data, n_ben, n_mal

    if src == "darpa":
        from darpa_tc_sampler import build_darpa_corpus
        benign, malicious = build_darpa_corpus(n_benign, n_malicious, dataset=DARPA_DATASET, seed=seed)
        data, n_ben, n_mal = _corpus_to_data(benign, malicious)
        print(f"[Dataset] (DARPA TC/{DARPA_DATASET}) benign={n_ben}  malicious={n_mal}")
        return data, n_ben, n_mal

    if src == "unicorn":
        benign = build_benign_corpus(n_benign, source="hybrid", mimicry_level=mimicry_level, seed=seed)
        malicious = build_malicious_corpus(n_malicious, seed=seed)
        data, n_ben, n_mal = _corpus_to_data(benign, malicious)
        print(f"[Dataset] (UNICORN) benign={n_ben}  malicious={n_mal}")
        return data, n_ben, n_mal

    # 'attack_agent': benign UNICORN-derived + malicious từ Attack Agent JSON
    benign = build_benign_corpus(n_benign, source="hybrid", mimicry_level=mimicry_level, seed=seed)
    mal_bundles = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
            mal_bundles.append({"nodes": d["nodes"], "edges": d["edges"]})
        except Exception:
            pass
    data, n_ben, n_mal = _corpus_to_data(benign, mal_bundles)
    print(f"[Dataset] (Attack Agent) benign={n_ben}  malicious={n_mal}")
    return data, n_ben, n_mal


def pretrain_gat(gat, data, epochs=30):
    opt = torch.optim.Adam(gat.parameters(), lr=0.005)
    for ep in range(epochs):
        gat.train()
        tot = 0.0
        for x, e, _ in data:
            opt.zero_grad()
            z = gat(x, e)
            ps = (z[e[0]] * z[e[1]]).sum(-1)
            pos = -torch.log(torch.sigmoid(ps) + 1e-15).mean()
            ns, nd = torch.randint(0, x.size(0), (e.size(1),), device=device), torch.randint(0, x.size(0), (e.size(1),), device=device)
            neg = -torch.log(1 - torch.sigmoid((z[ns] * z[nd]).sum(-1)) + 1e-15).mean()
            loss = pos + neg
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"[GAT link-pretrain] epoch {ep+1}/{epochs} loss={tot/len(data):.4f}")


@torch.no_grad()
def _eval_f1(gat, mlp, data, thresh=0.5):
    """F1 o eval mode (dung nhu env: gat.eval()+mlp.eval())."""
    gat.eval(); mlp.eval()
    TP = FP = FN = TN = 0
    for x, e, y in data:
        p = mlp(gat.graph_latent(x, e))[0, 1].item()
        pred = int(p >= thresh)
        if   y == 1 and pred == 1: TP += 1
        elif y == 1 and pred == 0: FN += 1
        elif y == 0 and pred == 1: FP += 1
        else:                      TN += 1
    prec = TP / (TP + FP + 1e-9); rec = TP / (TP + FN + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def train_mlp(gat, mlp, data, n_ben, n_mal, epochs=120):
    # FIX: bản cũ accumulate loss qua toàn bộ data rồi chỉ step() 1 lần/epoch
    # => 40 epoch = 40 gradient steps (quá ít, loss gần như không hội tụ, F1=0).
    # Chuyển sang SGD theo từng graph (mini-batch=1, shuffle mỗi epoch) giống Phase 1
    # => 250*epochs steps. Tín hiệu tách lớp ~tuyến tính (LogReg F1≈0.90) nên hội tụ tốt.
    import copy
    opt = torch.optim.Adam(list(gat.parameters()) + list(mlp.parameters()), lr=1e-3, weight_decay=1e-4)
    w_mal = min(n_ben / max(n_mal, 1), 3.0)  # class weight (đã cân khi n_ben≈n_mal -> 1.0)
    n = len(data)
    order = np.arange(n)
    # FIX 2: SGD per-graph dao động mạnh ở epoch cuối -> mức confidence tuyệt đối có thể
    # trượt xuống dưới 0.5 (eval_f1=0 dù tách lớp tốt). Lưu BEST-checkpoint theo eval_f1
    # (eval mode = đúng như env) và khôi phục cuối cùng, thay vì giữ epoch cuối.
    best_f1 = -1.0
    best_gat = copy.deepcopy(gat.state_dict())
    best_mlp = copy.deepcopy(mlp.state_dict())
    best_ep = 0
    for ep in range(epochs):
        gat.train(); mlp.train()
        for m in gat.modules():              # BN-FREEZE (khớp inference gat.eval())
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
        np.random.shuffle(order)
        ep_loss = 0.0; correct = 0
        for i in order:
            x, e, y = data[i]
            opt.zero_grad()
            lat = gat.graph_latent(x, e)             # attention-weighted pooling
            p = mlp(lat)[0, 1]
            w = w_mal if y == 1 else 1.0
            loss = - w * (y * torch.log(p + 1e-9) + (1 - y) * torch.log(1 - p + 1e-9))
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            correct += int((p.item() >= 0.5) == y)
        # Eval mỗi epoch (rẻ) để chọn best-checkpoint
        f1 = _eval_f1(gat, mlp, data)
        if f1 > best_f1:
            best_f1 = f1
            best_gat = copy.deepcopy(gat.state_dict())
            best_mlp = copy.deepcopy(mlp.state_dict())
            best_ep = ep + 1
        if (ep + 1) % 10 == 0:
            print(f"[MLP supervised] epoch {ep+1}/{epochs} loss={ep_loss/n:.4f} "
                  f"train_acc={correct/n:.3f} eval_f1={f1:.3f} (best={best_f1:.3f}@ep{best_ep})")
    # Khôi phục BEST checkpoint trước khi lưu/VERIFY
    gat.load_state_dict(best_gat)
    mlp.load_state_dict(best_mlp)
    print(f"[MLP supervised] >> restored BEST checkpoint: eval_f1={best_f1:.3f} @ epoch {best_ep}")


def main():
    data, n_ben, n_mal = load_dataset()
    if n_mal == 0 or n_ben == 0:
        raise SystemExit(
            f"[Pretrain] Dữ liệu thiếu lớp (benign={n_ben}, malicious={n_mal}). "
            f"Nguồn '{PRETRAIN_SOURCE}' chưa sẵn sàng — kiểm tra data/groundtruth. "
            f"KHÔNG ghi đè weights để tránh hỏng checkpoint."
        )
    gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(device)
    mlp = DetectionMLP(input_dim=20, hidden_dim=32).to(device)

    print("\n=== Phase 1: GAT link-prediction pretrain ===")
    pretrain_gat(gat, data, epochs=30)
    print("\n=== Phase 2: GAT+MLP supervised (BN-freeze) ===")
    train_mlp(gat, mlp, data, n_ben, n_mal, epochs=120)

    # EXP_PRETRAIN_OUTDIR: ghi weights ra thu muc khac (vd trained_weights/darpa_ho cho
    # baseline hold-out RQ1) -> KHONG ghi de trained_weights/darpa dang dung.
    outdir = os.environ.get("EXP_PRETRAIN_OUTDIR",
                            os.path.join(BASE_DIR, "trained_weights", "darpa"))
    os.makedirs(outdir, exist_ok=True)
    gp = os.path.join(outdir, "gat.pth")
    mp = os.path.join(outdir, "mlp.pth")
    torch.save(gat.state_dict(), gp)
    torch.save(mlp.state_dict(), mp)
    print(f"\nSaved -> {gp}\n         {mp}")

    # ── Verify: conf BIẾN THIÊN (eval mode, đúng như env) ──
    gat.eval(); mlp.eval()
    bcon, mcon = [], []
    for x, e, y in data:
        with torch.no_grad():
            c = mlp(gat.graph_latent(x, e))[0, 1].item()
        (mcon if y == 1 else bcon).append(c)
    bcon, mcon = np.array(bcon), np.array(mcon)
    TP = int((mcon >= 0.5).sum()); FN = int((mcon < 0.5).sum())
    TN = int((bcon < 0.5).sum()); FP = int((bcon >= 0.5).sum())
    prec = TP / (TP + FP + 1e-9); rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    print("\n=== VERIFY (eval mode, đúng như env) ===")
    print(f"  avg_ben_conf={bcon.mean():.4f} (std {bcon.std():.3f})  avg_mal_conf={mcon.mean():.4f} (std {mcon.std():.3f})")
    print(f"  TP={TP} FP={FP} FN={FN} TN={TN} | prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
    print(f"  conf biến thiên? ben[{bcon.min():.2f},{bcon.max():.2f}] mal[{mcon.min():.2f},{mcon.max():.2f}]")


if __name__ == "__main__":
    main()
