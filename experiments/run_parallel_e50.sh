#!/bin/bash
# Parallel orchestrator (lambda=500): 5 MARL training runs DOC LAP song song tren GPU1,
# sau khi xong moi eval + consolidate (RQ scripts REUSE checkpoint, KHONG train lai).
#
# 5 run (tag khop run_marl.py: seed=42, episodes=50):
#   canonical_s42_e50  gen ON,  EWC ON, mlp            -> RQ1-B1(leaky) / RQ2-C2 / RQ3-A0
#   rq2c1_s42_e50      gen OFF, EWC ON, mlp            -> RQ2-C1
#   rq3linear_s42_e50  gen ON,  EWC ON, linear         -> RQ3-A1
#   rq3deep_s42_e50    gen ON,  EWC ON, deep           -> RQ3-A3
#   rq1ho_s42_e50      gen ON(train-apts), EWC ON, mlp -> RQ1-HO B1 (baseline darpa_ho)
set -u
cd /home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection
PY=/home/tmkhiem/.conda/envs/nmuggle_APT/bin/python
REPO=$(pwd)
export CUDA_VISIBLE_DEVICES=1
export EXP_EWC_LAMBDA=500                      # lambda da chot
RUNS=experiments/results/marl_runs
COMMON="EXP_SEED=42 EXP_NUM_EPISODES=50 EXP_EWC_LAMBDA=500"

echo "=== PARALLEL e50 (lambda=500) START $(date) ==="
# Cache cu (lambda=8/30) -> STALE, xoa de train lai sach.
rm -rf experiments/results/marl_runs/* experiments/results/raw/*.json \
       experiments/results/all_results.csv experiments/results/summary.md
mkdir -p "$RUNS"

train() {  # $1=tag  $2=extra-env
  local tag="$1"; local extra="$2"
  mkdir -p "$RUNS/$tag"
  env $COMMON EXP_RUN_DIR="$RUNS/$tag" $extra \
      "$PY" -u main.py > "$RUNS/$tag.train.log" 2>&1
  echo "[done] $tag (exit=$?) $(date)"
}

train canonical_s42_e50 "EXP_ALLOWED_VARIANTS=aligned"                          & sleep 5
train rq2c1_s42_e50     "EXP_ALLOWED_VARIANTS=aligned EXP_DISABLE_GENERATION=1"  & sleep 5
train rq3linear_s42_e50 "EXP_ALLOWED_VARIANTS=aligned EXP_CLASSIFIER=linear"     & sleep 5
train rq3deep_s42_e50   "EXP_ALLOWED_VARIANTS=aligned EXP_CLASSIFIER=deep"       & sleep 5
train rq1ho_s42_e50     "EXP_PRETRAIN_DIR=$REPO/trained_weights/darpa_ho EXP_ALLOWED_APTS=APT28,APT32,APT38,APT41,FIN7,Carbanak" &

wait
echo "=== ALL 5 TRAIN RUNS DONE $(date) ==="
for t in canonical_s42_e50 rq2c1_s42_e50 rq3linear_s42_e50 rq3deep_s42_e50 rq1ho_s42_e50; do
  echo "  $t: best_mlp=$([ -f $RUNS/$t/best_mlp.pth ] && echo Y || echo n) final_mlp=$([ -f $RUNS/$t/final_mlp.pth ] && echo Y || echo n)"
done

echo "=== EVAL (reuse checkpoints) $(date) ==="
$PY -u experiments/rq1_marl_vs_nomarl.py  --episodes 50
$PY -u experiments/rq1_heldout.py         --episodes 50
$PY -u experiments/rq2_unseen_variants.py --episodes 50
$PY -u experiments/rq3_classifier_swap.py --episodes 50
echo "=== CONSOLIDATE ==="
$PY -u experiments/consolidate.py
echo "=== PARALLEL e50 (lambda=500) ALL DONE $(date) ==="
