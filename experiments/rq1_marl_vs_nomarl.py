"""
RQ1 — MARL co hieu qua hon khong dung MARL khong, tap trung vao cac mau B0 BO SOT?

  B0 (no-MARL) = detector pretrained (trained_weights/darpa) — chi inference.
  B1 (MARL)    = canonical run (gen ON, EWC ON, clf=mlp, variants=aligned).

Metric trong tam: Recovery Rate = ty le mau B0 miss (FN0) ma B1 bat duoc.
Ngoai ra: dF1/dRecall/dFN-rate, Retention (chong catastrophic forgetting).

Chay:
  python experiments/rq1_marl_vs_nomarl.py --episodes 20      # smoke
  python experiments/rq1_marl_vs_nomarl.py                    # full (150 ep)
"""
import os
import argparse

import common as C
import datasets as D
import run_marl as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EXP_SEED", 42)))
    ap.add_argument("--episodes", type=int, default=None, help="EXP_NUM_EPISODES (smoke)")
    args = ap.parse_args()
    C.set_global_seed(args.seed)

    print("=== RQ1: building frozen test corpus ===")
    corpus = D.build_rq1_corpus()
    print(f"    corpus: {len(corpus)} samples "
          f"(mal={sum(c['label'] for c in corpus)}, ben={sum(1-c['label'] for c in corpus)})")

    # ── B0: pretrained, inference only ───────────────────────────────
    print("=== RQ1: evaluating B0 (no-MARL, pretrained) ===")
    gat0, clf0 = C.load_detector("mlp")            # mlp_path=None -> pretrained
    b0 = C.eval_detector(gat0, clf0, corpus)
    print(f"    B0: F1={b0['f1']:.3f} recall={b0['recall']:.3f} FN0={len(b0['fn_ids'])}")
    C.write_raw_result("RQ1", "B0_noMARL", args.seed, C.scalar_metrics(b0),
                       meta={"n_FN0": len(b0["fn_ids"])})

    # ── B1: canonical MARL run ───────────────────────────────────────
    print("=== RQ1: running/loading B1 (canonical MARL) ===")
    run = R.canonical_run(seed=args.seed, episodes=args.episodes)
    gat1, clf1 = C.load_detector("mlp", mlp_path=run["checkpoint"],
                                 gat_path=run.get("gat_checkpoint"))
    b1 = C.eval_detector(gat1, clf1, corpus)

    m = C.scalar_metrics(b1)
    m["recovery_rate"] = C.recovery_rate(b0["fn_ids"], b1)
    m["retention"] = C.retention(b0, b1)
    m["delta_f1"] = b1["f1"] - b0["f1"]
    m["delta_recall"] = b1["recall"] - b0["recall"]
    m["delta_fn_rate"] = b1["fn_rate"] - b0["fn_rate"]
    m["episodes_to_nash"] = run.get("episodes_to_nash")
    print(f"    B1: F1={b1['f1']:.3f} recall={b1['recall']:.3f} "
          f"recovery={m['recovery_rate']} retention={m['retention']}")
    C.write_raw_result("RQ1", "B1_MARL", args.seed, m,
                       meta={"run_tag": run["tag"], "reused": run["reused"],
                             "stop_reason": run.get("stop_reason"),
                             "checkpoint": run["checkpoint"]})
    print("=== RQ1 done ===")


if __name__ == "__main__":
    main()
