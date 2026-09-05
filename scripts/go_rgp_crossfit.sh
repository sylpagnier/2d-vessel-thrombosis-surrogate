#!/usr/bin/env bash
# The honest form of the combined-model measurement: train, calibrate and score two half-pool
# arms so that EVERY vessel's t=0 flow comes from a model that never saw it.
#
#   1. train  E7_crossfit_A on half A, E7_crossfit_B on half B  (disjoint)
#   2. calibrate each arm's residual scale on ITS OWN training half -- the amplitude the
#      objective leaves free (RGP_DEQ_REPAIR_PLAN.md s18.5).  Fitting alpha on the half the arm
#      trained on keeps the other half clean.
#   3. precache half B with arm A, half A with arm B, and everything outside the pool with the
#      full-pool arm (which never saw those either)
#   4. rebuild the clot caches, rerun the biochem CV, pair against plain FEM
#
#   FULL_CKPT=outputs/runs/E5_band_gateup/kinematics_best_calibrated.pth \
#     bash scripts/go_rgp_crossfit.sh
#
# `SKIP_TRAIN=1` reuses checkpoints already on disk.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.
FULL_CKPT=${FULL_CKPT:?set FULL_CKPT to the full-pool calibrated checkpoint}
EPOCHS=${EPOCHS:-25}
TAG=${TAG:-dc_rgpxf}
L=outputs/logs/rgp_arm; mkdir -p "$L"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/${TAG}.status"; }

for HALF in A B; do
  ARM=E7_crossfit_${HALF}
  CK=outputs/runs/${ARM}/kinematics_best.pth
  if [ "${SKIP_TRAIN:-0}" != "1" ] || [ ! -f "$CK" ]; then
    say "TRAIN ${ARM}"
    HALF=${HALF} EPOCHS=${EPOCHS} bash scripts/stage_a/run_E7_crossfit.sh \
        > "$L/${ARM}.log" 2>&1 || { say "  !! ${ARM} failed, see $L/${ARM}.log"; exit 1; }
  fi
  say "CALIBRATE ${ARM} on its own training half"
  STEMS=$(python -c "
from scripts.stage_a.crossfit_halves import halves
a, b = halves()
print(','.join(a if '${HALF}' == 'A' else b))")
  python -u scripts/calibrate_residual_scale.py --ckpt "$CK" \
      --stems "$STEMS" \
      --out "outputs/runs/${ARM}/kinematics_best_calibrated.pth" \
      --report "outputs/diag_residual_calibration_${ARM}.json" \
      > "$L/${ARM}_calib.log" 2>&1
  say "  $(grep 'best alpha' "$L/${ARM}_calib.log" || echo 'no alpha line')"
done

say "SCORE cross-fit"
CROSSFIT=1 TAG=${TAG} \
  CKPT_A=outputs/runs/E7_crossfit_A/kinematics_best_calibrated.pth \
  CKPT_B=outputs/runs/E7_crossfit_B/kinematics_best_calibrated.pth \
  CKPT="${FULL_CKPT}" \
  bash scripts/go_rgp_deploy_score.sh
say "DONE"
