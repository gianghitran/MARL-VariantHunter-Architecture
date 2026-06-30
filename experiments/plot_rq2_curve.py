import os
import re
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(_REPO, "experiments", "results", "marl_runs")
IMG_DIR = os.path.join(_REPO, "Images")
os.makedirs(IMG_DIR, exist_ok=True)

_F1 = re.compile(r"Global F1\s*:\s*([0-9.]+)")


def f1_curve(run_dir):
    log = os.path.join(run_dir, "run.log")
    ys = []
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _F1.search(line)
                if m:
                    ys.append(float(m.group(1)))
    return ys


def curves_for(prefix):
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, f"{prefix}_s*_e50"))):
        ys = f1_curve(d)
        if ys:
            out.append((os.path.basename(d), ys))
    return out


def main():
    c1 = curves_for("rq2c1")       # C1 (chi EWC)
    c2 = curves_for("canonical")   # C2 (day du)

    plt.figure(figsize=(8, 4.5))
    # C2 (day du) - cam
    for i, (_name, ys) in enumerate(c2):
        plt.plot(range(1, len(ys) + 1), ys, color="tab:orange", alpha=0.75,
                 marker="o", markersize=3, linewidth=1.4,
                 label="$C_2$ (Gen + EWC)" if i == 0 else None)
    # C1 (chi EWC) - xanh
    for i, (_name, ys) in enumerate(c1):
        plt.plot(range(1, len(ys) + 1), ys, color="tab:blue", alpha=0.75,
                 marker="s", markersize=3, linewidth=1.4,
                 label="$C_1$ (only EWC)" if i == 0 else None)

    plt.xlabel("Episode")
    plt.ylabel("Global F1 (đánh giá nội bộ)")
    plt.title("Đường cong Global F1 theo episode trong quá trình đồng tiến hóa (RQ2)")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "rq2-f1-curve.png")
    plt.savefig(out, dpi=150)
    print(f"[plot] saved {out}")
    print(f"[plot] C1 runs: {[(n, len(y)) for n, y in c1]}")
    print(f"[plot] C2 runs: {[(n, len(y)) for n, y in c2]}")


if __name__ == "__main__":
    main()
