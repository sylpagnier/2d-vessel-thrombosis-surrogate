#!/usr/bin/env bash
# E2: E1 plus ReZero on the velocity residual -- the fix for why E1's residual head HURT.
#
# E1 scored 89.1 gateJ% against the 95.4 of the FEM prior it was handed: the model made the
# field it started from worse.  Audited (`python -m src.tools.diagnostics residual-head-audit`),
# the head's output is not mis-scaled signal, it is noise -- corr(delta, prior_error) = +0.03,
# and the best possible rescaling of it removes 0.1% of the prior's error.
#
# The cause is the initialisation, not the capacity.  The decoder is a random map on a LayerNorm
# shell, so it emits an O(1) field whatever the prior needs.  Against the analytic prior, whose
# error is O(0.4), that is the right order and training refines it.  Against the FEM prior, whose
# error is O(0.01), a fresh model overshoots by ~20x (measured 19-24x) and lands at relL2 0.59
# where the prior alone is 0.020.  The run then spends itself suppressing its own initialisation
# noise, reaches 1.4-3.4x, and stops -- and what survives is uncorrelated.  It also leaves the
# field 50x less divergence-free than the FEM it started from (|div| 0.214 against 0.0043).
#
# KINEMATICS_RESIDUAL_REZERO puts one learnable scalar, initialised to 0, on the velocity
# residual, so a fresh model predicts `u = uv_prior` EXACTLY and every departure has to be earned
# by the objective.  The same trick is already used twice in `ginodeq.py` for the same reason.
#
#   E2 > 95.4  -> the head has real signal once it is not fighting its own initialisation
#   E2 ~ 95.4  -> the scalar stayed near zero; the model correctly fell back to the prior, and
#                 the honest deploy answer is the FEM solve used directly
#   E2 < 95.4  -> the objective itself is pulling the field off COMSOL; look at the loss weights
set -u
export ARM_NAME=E2_prior_fem_rezero
export EPOCHS=${EPOCHS:-25}

# --- what makes this an E-series arm at all ------------------------------------------------
# Train on the 23 deploy packs only.  The 250-vessel synthetic corpus has no meshes, so it
# cannot carry a FEM prior; including it would force E1 to mix two different base points under
# one decoder, which is not an ablation of anything.
export KINEMATICS_DEPLOY_PACKS_ONLY=1
# These packs are ALREADY P2 and carry COMSOL's own mid-side velocities.  Elevation exists to
# fabricate mid-side labels by interpolation on the P1 synthetic corpus; running it here would
# overwrite real labels with interpolated ones.
export KINEMATICS_ELEVATE_P2=0
export SUBSAMPLE=${SUBSAMPLE:-0}

# --- the prior under test (the ONLY line that differs from E0) ------------------------------
export SPECIES_PRIOR_SOURCE=fem

# --- residual parameterisation --------------------------------------------------------------
# `u = prior + sdf * r` needs the decoder to span |r| p99/p50 ~ 24-30; it can emit ~9
# (ginodeq.py s17).  Under the envelope `1 - exp(-40*sdf)` the same labels need 4.8 (analytic)
# / 7.6 (fem) -- inside range, for BOTH arms, which is why this is set here and not only in E1.
# Measured by `python -m src.tools.diagnostics fem-prior-headroom --cohort`.
export KINEMATICS_BC_ENVELOPE=1
export KINEMATICS_BC_LAMBDA=40
# s17's two dynamic-range flags, carried over from D1.  No-ops at initialisation.
export KINEMATICS_DECODER_SKIP=1
export KINEMATICS_RESIDUAL_GAIN=1
# The one line that differs from E1.
export KINEMATICS_RESIDUAL_REZERO=1

export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
export KINEMATICS_SELECT_MAX_GRAPHS=6
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-0}
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
bash scripts/stage_a/launch.sh
