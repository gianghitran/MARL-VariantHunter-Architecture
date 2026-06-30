"""
RQ1 (HOLD-OUT) — phien ban co y nghia: test tren ho APT model pretrain CHUA TUNG GAP.

Baseline darpa_ho duoc pretrain (source=mixed) NHUNG loai han cac ho test
(APT29/APT33/Lazarus). Vi vay tap test malicious = chinh cac ho do -> accuracy
khong con bi thoi phong boi data leakage.

  B0_heldout = darpa_ho pretrained (chi inference).
  B1_heldout = co-evolution xuat phat tu darpa_ho, Generation chi sinh ho TRAIN
               (ho test van unseen xuyen suot) -> do MARL co phuc hoi duoc ho unseen.

Yeu cau: chay pretrain hold-out truoc (tao trained_weights/darpa_ho/):
  CUDA_VISIBLE_DEVICES=1 PRETRAIN_SOURCE=mixed \
    EXP_PRETRAIN_HELDOUT_APTS="APT29,APT33,Lazarus" \
    EXP_PRETRAIN_OUTDIR="trained_weights/darpa_ho" \
    python Benign_Agent/pretrain_on_generated.py

Chay:
  python experiments/rq1_heldout.py --episodes 50
"""
import os
import argparse

import common as C
import datasets as D
import run_marl as R

HO_DIR = os.path.join(C._REPO_ROOT, "trained_weights", "darpa_ho")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EXP_SEED", 42)))
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()
    C.set_global_seed(args.seed)

    gat_ho = os.path.join(HO_DIR, "gat.pth")
    mlp_ho = os.path.join(HO_DIR, "mlp.pth")
    if not (os.path.exists(gat_ho) and os.path.exists(mlp_ho)):
        raise SystemExit(f"[RQ1-HO] Missing held-out baseline in {HO_DIR}. "
                         f"Run pretrain hold-out first (see docstring).")

    print("=== RQ1-HO: building held-out test corpus (unseen APT families) ===")
    corpus = D.build_rq1_heldout_corpus()
    print(f"    corpus: {len(corpus)} samples "
          f"(mal={sum(c['label'] for c in corpus)}, ben={sum(1-c['label'] for c in corpus)}); "
          f"heldout={D.RQ1_HELDOUT_APTS}")

    # ── B0: held-out baseline, inference only ────────────────────────
    print("=== RQ1-HO: B0 (held-out baseline, no-MARL) ===")
    gat0, clf0 = C.load_detector("mlp", gat_path=gat_ho, mlp_path=mlp_ho)
    b0 = C.eval_detector(gat0, clf0, corpus)
    print(f"    B0: F1={b0['f1']:.3f} recall={b0['recall']:.3f} FN0={len(b0['fn_ids'])}")
    C.write_raw_result("RQ1", "B0_heldout", args.seed, C.scalar_metrics(b0),
                       meta={"n_FN0": len(b0["fn_ids"]), "heldout_apts": list(D.RQ1_HELDOUT_APTS),
                             "baseline": "darpa_ho"})

    # ── B1: MARL from held-out baseline (gen only train families) ────
    print("=== RQ1-HO: B1 (MARL co-evolution from held-out baseline) ===")
    run = R.rq1_heldout_run(seed=args.seed, episodes=args.episodes)
    gat1, clf1 = C.load_detector("mlp", mlp_path=run["checkpoint"],
                                 gat_path=run.get("gat_checkpoint"))
    b1 = C.eval_detector(gat1, clf1, corpus)
    m = C.scalar_metrics(b1)
    m["recovery_rate"] = C.recovery_rate(b0["fn_ids"], b1)
    m["retention"] = C.retention(b0, b1)
    m["delta_f1"] = b1["f1"] - b0["f1"]
    m["delta_recall"] = b1["recall"] - b0["recall"]
    m["episodes_to_nash"] = run.get("episodes_to_nash")
    print(f"    B1: F1={b1['f1']:.3f} recall={b1['recall']:.3f} "
          f"recovery={m['recovery_rate']} retention={m['retention']}")
    C.write_raw_result("RQ1", "B1_heldout_MARL", args.seed, m,
                       meta={"run_tag": run["tag"], "reused": run["reused"],
                             "stop_reason": run.get("stop_reason"),
                             "checkpoint": run["checkpoint"]})
    print("=== RQ1-HO done ===")


if __name__ == "__main__":
    main()
