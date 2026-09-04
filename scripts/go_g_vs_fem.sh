#!/usr/bin/env bash
# The one comparison the manifests cannot answer: DeployClotG (trained on the GT feature
# cache v5) against DeployClot2 (trained on v5_fem), both deployed on FEM flow, same cohort,
# same protocol.  The promoted manifests carry an IDENTICAL hardcoded `scores_strict_cv`
# block for both, so nothing on disk distinguishes them held-out.
set -u
LOG=outputs/logs/gvsfem_20260904
STATUS="$LOG/STATUS.txt"
: > "$STATUS"
stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$STATUS"
}
stage 01_eval_DeployClot2_0 python scripts/eval_clot_ml_0.py --v0 DeployClot2_0 \
  --cohort --flow fem --out outputs/deployclot/cmp_DeployClot2_0.json
stage 02_eval_DeployClotG_0 python scripts/eval_clot_ml_0.py --v0 DeployClotG_0 \
  --cohort --flow fem --out outputs/deployclot/cmp_DeployClotG_0.json
echo "[$(date +%H:%M:%S)] DONE" | tee -a "$STATUS"
