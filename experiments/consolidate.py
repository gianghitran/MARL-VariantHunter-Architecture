"""
experiments/consolidate.py — gop tat ca results/raw/*.json thanh MOT file de so sanh.

Xuat:
  results/all_results.csv  — long format: rq, condition, seed, metric, value
  results/summary.md       — 1 bang / RQ (mean[±std neu nhieu seed]) cho de doc

Chay:  python experiments/consolidate.py
"""
import os
import csv
import glob
import json
from collections import defaultdict
from statistics import mean, pstdev

RESULTS_DIR = os.environ.get("EXP_RESULTS_DIR") or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

# Thu tu metric hien thi cho moi RQ (chi cac key co mat moi hien).
DISPLAY = {
    "RQ1": ["f1", "precision", "recall", "fn_rate", "auc",
            "recovery_rate", "retention", "delta_f1", "episodes_to_nash"],
    "RQ2": ["f1", "recall", "fn_rate", "precision", "recovery_rate", "episodes_to_nash"],
    "RQ3": ["f1", "fn_rate", "recall", "param_count", "episodes_to_nash", "mean_latency_ms"],
}
# Thu tu condition mong muon (cac condition khac xep sau theo alphabet).
COND_ORDER = ["B0_noMARL", "B1_MARL",
              "C0_static", "C1_ewc_only", "C2_full",
              "A0_mlp", "A1_linear", "A3_deep"]


def _load_raw():
    recs = []
    for fp in sorted(glob.glob(os.path.join(RAW_DIR, "*.json"))):
        try:
            with open(fp) as f:
                recs.append(json.load(f))
        except Exception as e:
            print(f"[consolidate] skip {fp}: {e}")
    return recs


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_csv(recs):
    path = os.path.join(RESULTS_DIR, "all_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rq", "condition", "seed", "metric", "value"])
        for r in recs:
            for k, v in r.get("metrics", {}).items():
                w.writerow([r["rq"], r["condition"], r["seed"], k,
                            "" if v is None else v])
    print(f"[consolidate] wrote {path}")
    return path


def write_summary(recs):
    # group: rq -> condition -> metric -> list[value]
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                grouped[r["rq"]][r["condition"]][k].append(v)

    lines = ["# MARL-VariantHunter — Experiment Summary", ""]
    for rq in sorted(grouped):
        lines.append(f"## {rq}")
        lines.append("")
        conds = list(grouped[rq])
        conds.sort(key=lambda c: (COND_ORDER.index(c) if c in COND_ORDER else 999, c))
        # cot metric: theo DISPLAY, chi giu metric co du lieu. Cac metric tho khac
        # (FN/FP/TN/TP/n/...) van nam day du trong all_results.csv, khong lam roi bang.
        metrics = [m for m in DISPLAY.get(rq, []) if any(m in grouped[rq][c] for c in conds)]
        if not metrics:  # RQ khong nam trong DISPLAY -> hien tat ca metric co
            metrics = sorted({m for c in conds for m in grouped[rq][c]})

        header = "| condition | " + " | ".join(metrics) + " |"
        sep = "|" + "---|" * (len(metrics) + 1)
        lines += [header, sep]
        for c in conds:
            cells = []
            for m in metrics:
                vals = grouped[rq][c].get(m, [])
                if not vals:
                    cells.append("-")
                elif len(vals) == 1:
                    cells.append(_fmt(vals[0]))
                else:
                    cells.append(f"{mean(vals):.3f}±{pstdev(vals):.3f}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        # so seed
        nseed = max((len(grouped[rq][c].get(metrics[0], [])) for c in conds), default=0) if metrics else 0
        lines += ["", f"_n_seeds = {nseed}_", ""]

    path = os.path.join(RESULTS_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[consolidate] wrote {path}")
    return path


def main():
    recs = _load_raw()
    if not recs:
        print(f"[consolidate] no raw results in {RAW_DIR}. Run rq*.py first.")
        return
    write_csv(recs)
    write_summary(recs)
    print(f"[consolidate] consolidated {len(recs)} condition-records.")


if __name__ == "__main__":
    main()
