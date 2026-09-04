#!/usr/bin/env bash
# Overnight compute for the 2026-09-01 FEM repair.  Sequential so a crash in one stage does
# not poison the next; every stage logs to its own file and records an exit code.
set -u
LOG=outputs/logs/overnight_20260901
mkdir -p "$LOG"
STATUS="$LOG/STATUS.txt"
: > "$STATUS"

stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$STATUS"
  return $rc
}

# 1. Smoke: the Reynolds sweep.  Pre-fix this returned bit-identical arms for Re 150-600.
stage 01_smoke_re python scripts/run_research_sweep.py --sweep 03_inlet_re

# 2. COMSOL anchor-side FEM accuracy arm table (n=33).  Unaffected by the mesh-scale bug, so this is
#    a regression check on the dsrx_gain refactor: it should reproduce the 2026-09-01 numbers.
stage 02_arm_table python -m src.tools.diagnostics.local_fem_accuracy --cohort \
  --hops 3,6 --gains 1,2.18,3 --out outputs/diag_fem_arm_table_refit.json

# 2b. Paired deltas on the refreshed table (difference-of-medians is not a paired test).
stage 02b_paired python -m src.tools.diagnostics.fem_arm_paired   --table outputs/diag_fem_arm_table_refit.json   --out outputs/diag_fem_arm_paired_refit.json

# 3. All 20 research sweeps with the repaired solver.  The figure inputs for tomorrow.
stage 03_sweeps_all python scripts/run_research_sweep.py --all

# 4. Three-arm clot table on the full cohort, one arm per flow source.
for f in gt pred fem; do
  stage "04_clot_arm_$f" python scripts/eval_clot_ml_0.py --cohort --flow "$f" \
    --out "outputs/fem_arm_${f}_cohort.json"
done

# 5. Publication data derived from the sweeps.
stage 05_pub_sweepdata python scripts/publication/generate_research_sweep_fig_data.py

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$STATUS"
