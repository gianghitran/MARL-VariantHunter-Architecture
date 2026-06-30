"""
experiments/common.py — shared utilities cho RQ1/RQ2/RQ3.

Mot diem chung duy nhat de moi RQ dung CHUNG:
  - set_global_seed      : reproducibility
  - load_detector        : load GAT + classifier (mlp/linear/deep) tu checkpoint
  - eval_detector        : danh gia detector tren mot corpus co dinh -> full metrics
                           + per-sample preds + tap FN (de tinh Recovery Rate)
  - cac helper metric     : recovery_rate, retention, summarize

KHONG tai phat minh: tai dung prepare_graph/infer (graph_utils), GAT.graph_latent,
DetectionMLP/make_classifier (networks) — dung dung path end-to-end nhu
environment.py::_detection_agent va eval_mlp.py.
"""
import os
import sys
import time
import json

import numpy as np
import torch

# ── Dam bao repo root nam tren sys.path du common.py duoc import tu dau ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from networks import GAT, make_classifier                     # noqa: E402
from graph_utils import prepare_graph, infer                  # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_DIR = os.path.join(_REPO_ROOT, "trained_weights", "darpa")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seed(seed: int):
    """Dat seed toan cuc (random/numpy/torch) + bat cudnn deterministic."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_detector(classifier: str = "mlp",
                  gat_path: str = None,
                  mlp_path: str = None):
    """
    Load detector = GAT encoder + classifier head.

    Args:
        classifier : "mlp" | "linear" | "deep" (kien truc head).
        gat_path   : path gat.pth. None -> trained_weights/darpa/gat.pth.
        mlp_path   : path classifier weights (vd best_mlp.pth tu mot MARL run).
                     None -> trained_weights/darpa/mlp.pth (pretrained = B0).

    Returns:
        (gat, clf) ca hai o eval() tren DEVICE.
    """
    gat_path = gat_path or os.path.join(TRAINED_DIR, "gat.pth")
    mlp_path = mlp_path or os.path.join(TRAINED_DIR, "mlp.pth")

    gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(DEVICE)
    gat.load_state_dict(torch.load(gat_path, map_location=DEVICE, weights_only=True))
    gat.eval()

    clf = make_classifier(classifier, input_dim=20).to(DEVICE)
    # Load SHAPE-SAFE: strict=False chi bo qua key thieu/thua, KHONG bao ve shape mismatch.
    # Khi fallback mlp.pth (shape head mlp) vao classifier linear/deep -> shape khac ->
    # se raise neu load thang. Vi vay loc chi giu key trung shape (con lai = random init).
    state = torch.load(mlp_path, map_location=DEVICE, weights_only=True)
    model_sd = clf.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    clf.load_state_dict(compatible, strict=False)
    n_skip = len(set(state) - set(compatible))
    n_miss = len(set(model_sd) - set(compatible))
    if n_skip or n_miss:
        print(f"[load_detector] clf={classifier}: loaded {len(compatible)}/{len(model_sd)} "
              f"tensors from {os.path.basename(mlp_path)} "
              f"(skipped shape-mismatch={n_skip}, random-init={n_miss})")
    clf.eval()
    return gat, clf


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample classification (dung path end-to-end nhu _detection_agent)
# ─────────────────────────────────────────────────────────────────────────────
def _classify_one(gat, clf, edges_df):
    """
    edges_df -> prob_mal in [0,1]. Tra ve None neu graph rong/khong build duoc.
    """
    if edges_df is None or len(edges_df) == 0:
        return None
    phrases, _labels, edges, _names = prepare_graph(edges_df)
    if len(phrases) == 0 or len(edges[0]) == 0:
        return None
    nodes = [infer(x) for x in phrases]
    x = torch.tensor(np.array(nodes), dtype=torch.float32).to(DEVICE)
    ei = torch.tensor(edges, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        latent = gat.graph_latent(x, ei)         # [1, 20] attention pooling
        probs = clf(latent)                      # [1, 2] softmax
    return float(probs[0, 1].item())


# ─────────────────────────────────────────────────────────────────────────────
# Corpus evaluation
# ─────────────────────────────────────────────────────────────────────────────
def eval_detector(gat, clf, corpus, threshold: float = 0.5):
    """
    Danh gia detector tren corpus.

    Args:
        corpus : list[dict] moi item co {sample_id, label (0/1), edges_df, ...}.
        threshold : nguong P(malicious) -> du doan 1 (khop _detection_agent = 0.5).

    Returns dict:
        n, n_mal, n_ben, n_skipped,
        accuracy, precision, recall, f1, fn_rate, auc,
        TP, FP, FN, TN,
        fn_ids        : sample_id cua malicious bi miss (FN)
        correct_ids   : sample_id phan loai dung (cho retention)
        per_sample    : list {sample_id, label, pred, prob_mal}
        mean_latency_ms
    """
    per_sample = []
    n_skipped = 0
    lat_acc = 0.0
    n_timed = 0

    for item in corpus:
        sid = item["sample_id"]
        label = int(item["label"])
        t0 = time.perf_counter()
        prob = _classify_one(gat, clf, item.get("edges_df"))
        if prob is None:
            n_skipped += 1
            continue
        lat_acc += (time.perf_counter() - t0) * 1000.0
        n_timed += 1
        pred = 1 if prob >= threshold else 0
        per_sample.append({"sample_id": sid, "label": label, "pred": pred, "prob_mal": prob})

    TP = sum(1 for s in per_sample if s["pred"] == 1 and s["label"] == 1)
    FP = sum(1 for s in per_sample if s["pred"] == 1 and s["label"] == 0)
    FN = sum(1 for s in per_sample if s["pred"] == 0 and s["label"] == 1)
    TN = sum(1 for s in per_sample if s["pred"] == 0 and s["label"] == 0)

    n = len(per_sample)
    n_mal = TP + FN
    n_ben = TN + FP
    eps = 1e-9
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (TP + TN) / (n + eps)
    fn_rate = FN / (n_mal + eps)

    # AUC chi tinh duoc khi co ca 2 lop
    auc = None
    labels = [s["label"] for s in per_sample]
    if len(set(labels)) == 2:
        try:
            auc = float(roc_auc_score(labels, [s["prob_mal"] for s in per_sample]))
        except Exception:
            auc = None

    return {
        "n": n, "n_mal": n_mal, "n_ben": n_ben, "n_skipped": n_skipped,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fn_rate": fn_rate, "auc": auc,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "fn_ids": [s["sample_id"] for s in per_sample if s["pred"] == 0 and s["label"] == 1],
        "correct_ids": [s["sample_id"] for s in per_sample if s["pred"] == s["label"]],
        "per_sample": per_sample,
        "mean_latency_ms": (lat_acc / n_timed) if n_timed else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-condition metrics
# ─────────────────────────────────────────────────────────────────────────────
def recovery_rate(fn0_ids, after_eval):
    """
    Recovery Rate = ty le mau bi miss boi B0 (fn0_ids) ma model sau (after_eval)
    da phat hien dung. fn0_ids: list sample_id. after_eval: ket qua eval_detector.
    """
    fn0 = set(fn0_ids)
    if not fn0:
        return None  # khong co mau nao bi miss ban dau
    after_fn = set(after_eval["fn_ids"])
    recovered = len(fn0 - after_fn)  # truoc miss, gio khong con miss
    return recovered / len(fn0)


def retention(b0_eval, b1_eval):
    """
    Retention = trong cac mau B0 phan loai DUNG, ty le B1 van phan loai dung
    (kiem tra catastrophic forgetting).
    """
    b0_correct = set(b0_eval["correct_ids"])
    if not b0_correct:
        return None
    b1_correct = set(b1_eval["correct_ids"])
    return len(b0_correct & b1_correct) / len(b0_correct)


def param_count(clf):
    return int(sum(p.numel() for p in clf.parameters()))


# ─────────────────────────────────────────────────────────────────────────────
# Result IO (raw per-condition JSON -> consolidate.py gop lai)
# ─────────────────────────────────────────────────────────────────────────────
# EXP_RESULTS_DIR cho phep tach ket qua sang thu muc khac (khong ghi de run cu).
RESULTS_DIR = os.environ.get("EXP_RESULTS_DIR") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

# Cac key scalar trich tu eval_detector de luu (bo per_sample/*_ids cho gon).
_SCALAR_KEYS = ["accuracy", "precision", "recall", "f1", "fn_rate", "auc",
                "TP", "FP", "FN", "TN", "n", "n_mal", "n_ben", "n_skipped",
                "mean_latency_ms"]


def scalar_metrics(eval_res: dict) -> dict:
    """Trich cac metric scalar tu ket qua eval_detector."""
    return {k: eval_res.get(k) for k in _SCALAR_KEYS}


def write_raw_result(rq: str, condition: str, seed: int,
                     metrics: dict, meta: dict = None):
    """Ghi results/raw/<rq>_<condition>_s<seed>.json (1 file / dieu kien / seed).

    Seed nam trong ten file -> chay nhieu seed KHONG ghi de len nhau; consolidate.py
    gom tat ca *.json va tinh mean+-std (n_seeds > 1).
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    rec = {"rq": rq, "condition": condition, "seed": seed,
           "metrics": metrics, "meta": meta or {}}
    path = os.path.join(RAW_DIR, f"{rq}_{condition}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[result] wrote {path}")
    return path
