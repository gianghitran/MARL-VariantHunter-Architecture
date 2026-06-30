"""
RQ2 — Generation Agent co giup Detection duy tri hieu qua tren BIEN THE CHUA THAY?

Leave-Variant-Out: co-evolution chi train tren variant 'aligned' (EXP_ALLOWED_VARIANTS),
test tren 'evolved' + 'composite' (unseen). Ba dieu kien:

  C0 (static)  = gen OFF + EWC OFF  == detector pretrained (B0) -> KHONG chay MARL.
  C1 (EWC-only)= gen OFF + EWC ON                              -> co lap dong gop EWC.
  C2 (full)    = gen ON  + EWC ON   == canonical run           -> he thong de xuat.

Metric: F1 / Recall / FN-rate tren unseen variants; them recovery vs C0 va
F1-vs-episode (luu trong meta) cho C1/C2.

Chay:
  python experiments/rq2_unseen_variants.py --episodes 20     # smoke
"""
import os
import argparse

import common as C
import datasets as D
import run_marl as R


def _eval_and_write(condition, gat, clf, corpus, seed, c0=None, run=None):
    res = C.eval_detector(gat, clf, corpus)
    m = C.scalar_metrics(res)
    if c0 is not None:
        m["recovery_rate"] = C.recovery_rate(c0["fn_ids"], res)
        m["retention"] = C.retention(c0, res)
    meta = {}
    if run is not None:
        m["episodes_to_nash"] = run.get("episodes_to_nash")
        meta = {"run_tag": run["tag"], "reused": run["reused"],
                "stop_reason": run.get("stop_reason"),
                "f1_curve": run.get("f1_curve"), "checkpoint": run["checkpoint"]}
    print(f"    {condition}: F1={res['f1']:.3f} recall={res['recall']:.3f} "
          f"fn_rate={res['fn_rate']:.3f}")
    C.write_raw_result("RQ2", condition, seed, m, meta=meta)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EXP_SEED", 42)))
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()
    C.set_global_seed(args.seed)

    print("=== RQ2: building unseen-variant (evolved+composite) test corpus ===")
    corpus = D.build_rq2_corpus()
    print(f"    corpus: {len(corpus)} samples "
          f"(mal={sum(c['label'] for c in corpus)}, ben={sum(1-c['label'] for c in corpus)})")

    # ── C0 (static) == B0 pretrained, inference only ─────────────────
    print("=== RQ2: C0 (static, pretrained) ===")
    gat0, clf0 = C.load_detector("mlp")
    c0 = _eval_and_write("C0_static", gat0, clf0, corpus, args.seed)

    # ── C1: gen OFF, EWC ON ──────────────────────────────────────────
    print("=== RQ2: C1 (EWC-only, generation OFF) ===")
    run1 = R.c1_run(seed=args.seed, episodes=args.episodes)
    gat1, clf1 = C.load_detector("mlp", mlp_path=run1["checkpoint"],
                                 gat_path=run1.get("gat_checkpoint"))
    _eval_and_write("C1_ewc_only", gat1, clf1, corpus, args.seed, c0=c0, run=run1)

    # ── C2: full (canonical) ─────────────────────────────────────────
    print("=== RQ2: C2 (full: generation + EWC) ===")
    run2 = R.canonical_run(seed=args.seed, episodes=args.episodes)
    gat2, clf2 = C.load_detector("mlp", mlp_path=run2["checkpoint"],
                                 gat_path=run2.get("gat_checkpoint"))
    _eval_and_write("C2_full", gat2, clf2, corpus, args.seed, c0=c0, run=run2)
    print("=== RQ2 done ===")


if __name__ == "__main__":
    main()
