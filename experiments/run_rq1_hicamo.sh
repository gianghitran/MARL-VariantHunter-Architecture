#!/bin/bash
# RQ1 hold-out voi CAMO CAO (r>=3) de baseline lo diem yeu (FN0>0) -> do Recovery cua MARL.
# KHONG train lai: tai dung checkpoint rq1ho_s{seed}_e50 da co (eval-only, 3 seed song song).
# Ket qua ghi sang THU MUC RIENG (EXP_RESULTS_DIR) -> KHONG ghi de run goc.
set -u
cd /home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection
PY=/home/tmkhiem/.conda/envs/nmuggle_APT/bin/python
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export EXP_RESULTS_DIR="$(pwd)/experiments/results_rq1_hicamo"   # tach rieng
export EXP_RQ1_HO_CAMO="0.0,3.0,5.0,7.0"                         # camo cao
SEEDS="42 43 44"
EPISODES=50

echo "=== RQ1 HI-CAMO (camo=$EXP_RQ1_HO_CAMO) START $(date) ==="
echo "    out dir: $EXP_RESULTS_DIR (run goc KHONG bi dong toi)"
mkdir -p "$EXP_RESULTS_DIR/raw"

# 3 seed eval song song (REUSE checkpoint -> nhanh, ~1-2 phut).
for s in $SEEDS; do
  $PY -u experiments/rq1_heldout.py --seed "$s" --episodes "$EPISODES" \
      > "$EXP_RESULTS_DIR/rq1ho_eval_s${s}.log" 2>&1 &
  sleep 2
done
wait
echo "=== EVAL DONE $(date) ==="
for s in $SEEDS; do
  echo "--- seed $s ---"; grep -E "corpus:|B0:|B1:|recovery" "$EXP_RESULTS_DIR/rq1ho_eval_s${s}.log"
done

echo "=== CONSOLIDATE (-> $EXP_RESULTS_DIR/summary.md) ==="
$PY -u experiments/consolidate.py
echo "=== RQ1 HI-CAMO ALL DONE $(date) ==="
