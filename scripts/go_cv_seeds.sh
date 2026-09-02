#!/usr/bin/env bash
# 3-seed CV for the FEM-vs-GT training comparison.  The 1-seed probes established direction;
# nothing is quotable without seed variance.  Both arms use caches built from CURRENT code.
set -u
LOG=outputs/logs/overnight_20260901
STATUS="$LOG/STATUS.txt"
stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$STATUS"
}
stage 16_cv_gt_s3  python scripts/run_phase9_cv.py --tag gt_s3  --cache v42     --folds 5 --seeds 3
stage 17_cv_fem_s3 python scripts/run_phase9_cv.py --tag fem_s3 --cache v4_fem  --folds 5 --seeds 3
stage 18_readout_gt_s3  python scripts/eval_expected_score_readout.py --tags gt_s3  --cache v42
stage 19_readout_fem_s3 python scripts/eval_expected_score_readout.py --tags fem_s3 --cache v4_fem
echo "[$(date +%H:%M:%S)] CV SEEDS DONE" | tee -a "$STATUS"
