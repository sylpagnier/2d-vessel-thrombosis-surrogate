#!/usr/bin/env bash
# Common Stage-A launch environment.  An arm sets ARM_NAME + whatever it varies, then sources
# this.  Everything here is either the s11.5 launch config or an iteration-speed setting.
set -e
export SPECIES_PRIOR_SOURCE=analytic
export KINEMATICS_ELEVATE_P2=1
export KINEMATICS_COORD_MODE=centered
export KINEMATICS_NORMALIZE_SHEAR_GRAD=1
# In data/reference/ (tracked), NOT outputs/ (gitignored).  A missing file here is not an
# error -- `_resolve_loss_weights` prints one WARN line and falls back to the historical
# recipe -- so from outputs/ this ran s12's calibrated weights on the workstation and the
# pre-calibration defaults on the COMSOL box, and only one line of a long log said so.
export KINEMATICS_LOSS_WEIGHTS=${KINEMATICS_LOSS_WEIGHTS:-data/reference/kine_loss_weights_20260828.json}
if [ ! -f "${KINEMATICS_LOSS_WEIGHTS}" ]; then
  echo "[arm] FATAL: loss weights not found at ${KINEMATICS_LOSS_WEIGHTS}" >&2
  echo "[arm]        the run would silently fall back to the pre-calibration defaults." >&2
  exit 1
fi
export KINEMATICS_MAX_NODES=${KINEMATICS_MAX_NODES:-26000}
export KINEMATICS_SELECT_MAX_GRAPHS=${KINEMATICS_SELECT_MAX_GRAPHS:-8}
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-8}
export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
# Prepared graphs (P2 elevation + legal priors + PDE floors) survive between runs.
export KINEMATICS_PREPARED_CACHE=${KINEMATICS_PREPARED_CACHE:-outputs/cache/kine_prepared}
# The MLS gradient operator is a per-node Python loop costing 1.7 s of a 2.28 s step, and the
# default LRU holds 12 entries against 225 training graphs -- so it missed on essentially every
# step.  Holding the whole cohort is 2.1 GB and makes a step 0.61 s.
export BIOCHEM_GRAD_CACHE_CPU=${BIOCHEM_GRAD_CACHE_CPU:-300}
# Trades cohort size for turnaround WITHOUT re-preparing: applied after the prepared cache and
# left out of its key, stratified by geometry level.
# Full cohort is requested as SUBSAMPLE=0 (or "all"), NOT as an empty string.  Two reasons:
#   * PowerShell's `$env:SUBSAMPLE = ""` DELETES the variable rather than setting it empty, so
#     an empty-string protocol arrives at bash as *unset* and silently falls back to the arm's
#     120-vessel default -- a subsampled run wearing a full-cohort run's name.
#   * `unset` in the else branch also stops a KINEMATICS_TRAIN_SUBSAMPLE exported by a previous
#     iteration arm from surviving into this one.
# Either way the resolved choice is echoed, because "how many vessels did this run see" is not
# a question anyone should have to reconstruct from shell history.
case "${SUBSAMPLE:-}" in
  ""|0|all|ALL|full|FULL)
    unset KINEMATICS_TRAIN_SUBSAMPLE
    COHORT_NOTE="FULL (no subsample)"
    ;;
  *)
    export KINEMATICS_TRAIN_SUBSAMPLE=${SUBSAMPLE}
    COHORT_NOTE="SUBSAMPLED to ${SUBSAMPLE} vessels"
    ;;
esac
export KINEMATICS_OUTPUT_DIR=outputs/runs/${ARM_NAME}
mkdir -p "outputs/runs"
echo "[arm] ${ARM_NAME}"
echo "[arm]   cohort        ${COHORT_NOTE}"
env | grep -E "^(KINEMATICS_|SPECIES_|BIOCHEM_)" | sort | sed 's/^/[arm]   /'
python -u -m src.training.train_kinematics_predictor \
  --epochs ${EPOCHS:-14} --adam-epochs ${EPOCHS:-14} \
  --stage1-end-epoch 0 --stage2-end-epoch 0 --no-prompt --quiet
