#!/bin/bash
# Multi-seed parallel orchestrator (lambda=500).
# Train 5 configs x N seeds DONG THOI tren GPU1 -> dung het core ranh, n_seeds>1.
# Wall-clock ~ = 1 run dai nhat (rq3deep 50 ep) vi tat ca chay song song.
#
# Moi run cap thread (OMP/MKL=2): models tini (738-3490 params), hot-path la Python
# single-core -> capping tranh 15x53 thread thrash, KHONG lam cham tung run.
#
# 5 config (khop tag run_marl.py, episodes=50, lambda=500):
#   canonical_s<seed>_e50  gen ON,  EWC ON, mlp            -> RQ1-B1 / RQ2-C2 / RQ3-A0
#   rq2c1_s<seed>_e50      gen OFF, EWC ON, mlp            -> RQ2-C1
#   rq3linear_s<seed>_e50  gen ON,  EWC ON, linear         -> RQ3-A1
#   rq3deep_s<seed>_e50    gen ON,  EWC ON, deep           -> RQ3-A3
#   rq1ho_s<seed>_e50      gen ON(train-apts), baseline darpa_ho -> RQ1-HO B1
set -u
cd /home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection
PY=/home/tmkhiem/.conda/envs/nmuggle_APT/bin/python
REPO=$(pwd)
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUNS=experiments/results/marl_runs

SEEDS="42 43 44"          # n_seeds=3 -> mean+-std
EPISODES=50
LAMBDA=500

echo "=== MULTISEED e50 (lambda=$LAMBDA, seeds=$SEEDS) START $(date) ==="
# Sach: xoa marl_runs + raw cu (cache DARPA pkl GIU lai -> eval nhanh).
rm -rf experiments/results/marl_runs/* experiments/results/raw/*.json \
       experiments/results/all_results.csv experiments/results/summary.md
mkdir -p "$RUNS"

train() {  # $1=tag  $2=seed  $3=extra-env
  local tag="$1"; local seed="$2"; local extra="$3"
  mkdir -p "$RUNS/$tag"
  env EXP_SEED="$seed" EXP_NUM_EPISODES="$EPISODES" EXP_EWC_LAMBDA="$LAMBDA" \
      EXP_RUN_DIR="$RUNS/$tag" $extra \
      "$PY" -u main.py > "$RUNS/$tag.train.log" 2>&1
  echo "[done] $tag (exit=$?) $(date)"
}

# Fan-out: 5 config x moi seed, tat ca chay nen song song.
for s in $SEEDS; do
  train "canonical_s${s}_e50" "$s" "EXP_ALLOWED_VARIANTS=aligned"                         & sleep 3
  train "rq2c1_s${s}_e50"     "$s" "EXP_ALLOWED_VARIANTS=aligned EXP_DISABLE_GENERATION=1" & sleep 3
  train "rq3linear_s${s}_e50" "$s" "EXP_ALLOWED_VARIANTS=aligned EXP_CLASSIFIER=linear"    & sleep 3
  train "rq3deep_s${s}_e50"   "$s" "EXP_ALLOWED_VARIANTS=aligned EXP_CLASSIFIER=deep"      & sleep 3
  train "rq1ho_s${s}_e50"     "$s" "EXP_PRETRAIN_DIR=$REPO/trained_weights/darpa_ho EXP_ALLOWED_APTS=APT28,APT32,APT38,APT41,FIN7,Carbanak" & sleep 3
done

wait
echo "=== ALL TRAIN RUNS DONE $(date) ==="
for s in $SEEDS; do
  for t in canonical rq2c1 rq3linear rq3deep rq1ho; do
    tag="${t}_s${s}_e50"
    echo "  $tag: best_mlp=$([ -f $RUNS/$tag/best_mlp.pth ] && echo Y || echo n) final_mlp=$([ -f $RUNS/$tag/final_mlp.pth ] && echo Y || echo n)"
  done
done

# EVAL: moi RQ script chay 1 lan / seed (REUSE checkpoint -> khong train lai).
# Cac eval cua 1 seed chay tuan tu; cac seed song song de nhanh.
echo "=== EVAL (reuse checkpoints, per-seed parallel) $(date) ==="
eval_seed() {  # $1=seed
  local s="$1"
  $PY -u experiments/rq1_marl_vs_nomarl.py  --seed "$s" --episodes "$EPISODES"
  $PY -u experiments/rq1_heldout.py         --seed "$s" --episodes "$EPISODES"
  $PY -u experiments/rq2_unseen_variants.py --seed "$s" --episodes "$EPISODES"
  $PY -u experiments/rq3_classifier_swap.py --seed "$s" --episodes "$EPISODES"
  echo "[eval done] seed=$s $(date)"
}
for s in $SEEDS; do eval_seed "$s" > "experiments/results/eval_s${s}.log" 2>&1 & sleep 2; done
wait
echo "=== EVAL ALL DONE $(date) ==="
for s in $SEEDS; do echo "--- eval seed $s tail ---"; tail -4 "experiments/results/eval_s${s}.log"; done

echo "=== CONSOLIDATE ==="
$PY -u experiments/consolidate.py
echo "=== MULTISEED e50 (lambda=$LAMBDA) ALL DONE $(date) ==="
