#!/usr/bin/env bash
# E0: the ABLATION for E1 -- byte-identical except the prior block is the analytic Poiseuille
# field instead of the local FEM solve.
#
# E1 changes TWO things at once against every previous arm: it trains on the deploy packs alone
# (no synthetic corpus), and it takes the FEM field as the hard BC's base point.  Either could
# carry a win.  The deploy packs are the wall-shear regime the synthetic corpus does not contain
# (RGP_DEQ_REPAIR_PLAN.md s16.5), so "trained on 23 real vessels" is a live alternative
# explanation and has to be measured, not argued away -- the same mistake s16 had to issue six
# corrections for, and the reason D0 exists beside D1.
#
# RUN THIS FIRST.  If E0 alone clears the analytic prior's gateJ%, the headline is the cohort,
# not the prior.
set -u
export ARM_NAME=E0_prior_analytic
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

# --- the prior under test (the ONLY line that differs from E1) ------------------------------
export SPECIES_PRIOR_SOURCE=analytic

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

export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
export KINEMATICS_SELECT_MAX_GRAPHS=6
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-0}
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
bash scripts/stage_a/launch.sh
