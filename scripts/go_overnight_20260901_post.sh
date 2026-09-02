#!/usr/bin/env bash
# Second half of the 2026-09-01 overnight run.  Kept separate because the timing table must
# be measured on a QUIET machine -- running it beside the sweeps would time CPU contention.
set -u
LOG=outputs/logs/overnight_20260901
mkdir -p "$LOG"
STATUS="$LOG/STATUS.txt"

stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$STATUS"
  return $rc
}

# Fig 1 solves the FEM itself, so regenerate it against the repaired solver.  The patient
# anchors were never mis-scaled, so this is expected to reproduce -- run it to KNOW that.
stage 10_fig1_data python scripts/publication/generate_fig1_data.py

# The paper's speedup claim, measured rather than recalled.  Needs a quiet machine.
stage 11_timing python scripts/publication/generate_timing_data.py

echo "[$(date +%H:%M:%S)] POST DONE" | tee -a "$STATUS"
