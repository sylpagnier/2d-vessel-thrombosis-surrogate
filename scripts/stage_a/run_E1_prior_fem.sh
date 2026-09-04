#!/usr/bin/env bash
# E1: the local FEM solve as the RGP-DEQ prior block -- the model's job becomes the FEM's own
# error rather than the flow.
#
# The hard BC reads the prior as its BASE POINT (`u = prior + envelope * r`), so swapping the
# prior changes what `r` has to be without touching a single weight shape.  Measured over the
# 37-vessel cohort (`python -m src.tools.diagnostics fem-prior-headroom --cohort`):
#
#     prior                      relL2    relL2 wall band   gate Jaccard
#     analytic (shipped)         0.155         0.622            0.101
#     fem, analytic inlet        0.0166        0.043            0.917
#
# and the residual left over is smooth (1-hop error autocorrelation 0.99) and concentrated
# (top 10% of nodes carry 64% of squared error) -- learnable, and exactly the "spend capacity
# where the FEM was wrong" shape this arm is for.
#
# The prior is deploy-legal: `build_fem_priors` solves under an ANALYTIC inlet, never COMSOL's
# (pinned by `src/tests/test_fem_deq_coupling.py`).  Pre-solve the cache with
# `python scripts/build_fem_prior_cache.py` or the first epoch pays ~3 minutes of CPU.
#
# KNOWN TAIL: comsol045 and comsol046 solve to relL2 0.53 / 0.67 against COMSOL where the
# other 27 vessels sit at 0.007-0.057 -- a genuine flow disagreement, not a registration or
# inlet artefact (dir_cos 0.66-0.76, both inlets give the identical field). Both are in the
# TRAIN pool and neither is in the selection set, so they cannot move the checkpoint choice;
# they are 2 of 23 training vessels and worth watching if the run underperforms E0.
#
#   E1 > E0 on gateJ%  -> the FEM prior is the win, and it is the base point that did it
#   E1 ~ E0            -> the cohort did it; the prior swap bought nothing on top
set -u
export ARM_NAME=E1_prior_fem
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

export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
export KINEMATICS_SELECT_MAX_GRAPHS=6
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-0}
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
bash scripts/stage_a/launch.sh
