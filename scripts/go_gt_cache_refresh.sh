#!/usr/bin/env bash
# Rebuild the GT feature caches with CURRENT code into NEW directories, so the FEM-vs-GT
# training comparison is not confounded by cache vintage (the shipped v4 cache is 2026-08-17,
# older than the v3 cache it was derived from).  The originals are the only record of what the
# shipped model was trained against and are left untouched.
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

stage 12_cache_gt_fresh python scripts/build_clot_ml_cache.py --flow gt \
  --out outputs/clot_ml_cache_gt2 --force
stage 13_cache_v4_fresh python scripts/build_clot_ml_cache_v4.py --flow gt \
  --src outputs/clot_ml_cache_gt2 --out outputs/clot_ml_cache_v42 --force
stage 14_cv_gt_fresh python scripts/run_phase9_cv.py --tag gt_fresh_probe \
  --cache v42 --folds 5 --seeds 1
stage 15_readout_gt_fresh python scripts/eval_expected_score_readout.py \
  --tags gt_fresh_probe --cache v42
echo "[$(date +%H:%M:%S)] GT REFRESH DONE" | tee -a "$STATUS"
