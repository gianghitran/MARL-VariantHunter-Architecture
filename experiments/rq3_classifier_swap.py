"""
RQ3 — Doi classifier head (MLP -> Linear / Deep) anh huong gi?

Giu nguyen GAT encoder + Generation + Pruning + EWC; chi thay head.
  A0 = mlp    (canonical run, da co)
  A1 = linear (run #3)
  A3 = deep   (run #4)
Tat ca EXP_ALLOWED_VARIANTS=aligned (giong canonical) -> so sanh fair.

Metric: F1, FN-rate, Recall, #episodes-to-Nash, param_count, inference latency.

Chay:
  python experiments/rq3_classifier_swap.py --episodes 20     # smoke
"""
import os
import argparse

import common as C
import datasets as D
import run_marl as R

_COND = {"mlp": "A0_mlp", "linear": "A1_linear", "deep": "A3_deep"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EXP_SEED", 42)))
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--classifiers", nargs="+", default=["mlp", "linear", "deep"])
    args = ap.parse_args()
    C.set_global_seed(args.seed)

    print("=== RQ3: building test corpus (same as RQ1) ===")
    corpus = D.build_rq1_corpus()

    for name in args.classifiers:
        print(f"=== RQ3: classifier='{name}' ===")
        run = R.classifier_run(name, seed=args.seed, episodes=args.episodes)
        gat, clf = C.load_detector(name, mlp_path=run["checkpoint"],
                                   gat_path=run.get("gat_checkpoint"))
        res = C.eval_detector(gat, clf, corpus)
        m = C.scalar_metrics(res)
        m["episodes_to_nash"] = run.get("episodes_to_nash")
        m["param_count"] = C.param_count(clf)
        print(f"    {name}: F1={res['f1']:.3f} fn_rate={res['fn_rate']:.3f} "
              f"params={m['param_count']} nash={m['episodes_to_nash']} "
              f"lat={res['mean_latency_ms']:.2f}ms")
        C.write_raw_result("RQ3", _COND.get(name, name), args.seed, m,
                           meta={"run_tag": run["tag"], "reused": run["reused"],
                                 "stop_reason": run.get("stop_reason"),
                                 "checkpoint": run["checkpoint"]})
    print("=== RQ3 done ===")


if __name__ == "__main__":
    main()
