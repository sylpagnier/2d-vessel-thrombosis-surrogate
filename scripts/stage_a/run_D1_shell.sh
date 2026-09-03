#!/usr/bin/env bash
# D1: C2's recipe PLUS the two dynamic-range flags (RGP_DEQ_REPAIR_PLAN.md s17).
#
# `RGPBlock` ends in `nn.LayerNorm`, so `z*` sits on one fixed shell -- cv(||z_i||) is 1e-03
# across nodes on four deploy packs.  Per-node amplitude is not something the DEQ output can
# express, and the residual it emits is ~3x short of the labels on the p99/p50 TAIL that wall
# `dsrx` spread is made of.  DECODER_SKIP hands the decoder `x_enc` (un-normalised, per-node)
# alongside `z*`; RESIDUAL_GAIN puts a bounded per-node multiplier on the hard-BC residual.
# Both are exact no-ops at initialisation, so this arm differs from D0 only in capacity.
#
#   dsrxS moves  -> the shell was a real ceiling and s17 is the fix
#   dsrxS pinned -> the ceiling is elsewhere; s17.3 is wrong and must be withdrawn like D1 was
set -u
export ARM_NAME=D1_shell_off
export EPOCHS=${EPOCHS:-25}
export SUBSAMPLE=${SUBSAMPLE:-120}
export KINEMATICS_VAL_EVERY=4
export KINEMATICS_SELECT_MAX_GRAPHS=8
export KINEMATICS_SELECT_PATIENCE=0
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
export KINEMATICS_DECODER_SKIP=1
export KINEMATICS_RESIDUAL_GAIN=1
bash scripts/stage_a/launch.sh
