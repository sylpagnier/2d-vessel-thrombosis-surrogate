#!/usr/bin/env bash
# D0: the ABLATION for D1 -- byte-identical except the two s17 flags are off.
#
# This arm exists because a win in D1 is otherwise unattributable.  The corpus on disk
# (2026-08-29) passes preflight's wall-shear regime check that the 2026-08-28 cohort FAILED
# (`wall_dsrx_sd` 1349 vs deploy 717.7, `sep-only` 0.818 vs 0.9146), so a D1 that beats the
# analytic prior's 32.5 could be the corpus, the flags, or both.  D0 is the only thing that
# separates them, and s16 had to issue six corrections for exactly this class of mistake.
#
# RUN THIS FIRST: if D0 alone clears the prior, s17's premise still holds but its urgency does
# not, and the honest headline is the corpus, not the architecture.
set -u
export ARM_NAME=D0_shell_on
export EPOCHS=${EPOCHS:-25}
export SUBSAMPLE=${SUBSAMPLE:-120}
export KINEMATICS_VAL_EVERY=4
export KINEMATICS_SELECT_MAX_GRAPHS=8
export KINEMATICS_SELECT_PATIENCE=0
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
export KINEMATICS_DECODER_SKIP=0
export KINEMATICS_RESIDUAL_GAIN=0
bash scripts/stage_a/launch.sh
