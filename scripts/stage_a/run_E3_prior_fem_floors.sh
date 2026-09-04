#!/usr/bin/env bash
# E3: E2 plus both PRIOR FLOORS turned up -- "only move where you strictly improve".
#
# E2 (ReZero) stopped the residual head hurting, but it buys gate agreement and pays for it in
# global field accuracy.  Held out on the VIZ packs against the FEM prior it was handed:
#
#     metric        FEM prior   E2      verdict
#     gateJ%          97.9      98.3    better
#     dsrxCorr        0.990     0.990   tie
#     relL2           0.0168    0.0186  WORSE
#
# Nothing in the objective forbids that trade.  Two one-sided hinges already exist for exactly
# this and both are effectively dormant: `prior_floor_loss` (T6) hinges on |u - y| against
# |prior - y| and is weighted 0.1004*500 = 50, which measured 0.3% of the loss; `_band_shear_floor`
# (s16.4) is the same hinge in the sr/dsrx channel and ships at weight 0, disabled pending
# exactly this measurement.  Both are ZERO wherever the model beats the prior, so raising them
# cannot fight the model where it is right -- it only removes its licence to be wrong.
#
# The pairing with ReZero is what makes a large weight well-conditioned: a fresh E2/E3 model
# predicts `u = uv_prior` exactly, so BOTH hinges start at exactly 0.  The run begins on the
# constraint boundary instead of having to climb back to it.
#
# Also tried and rejected on the way: giving the head an explicit FEM-error indicator channel
# (`art_frac`, `cell_re`, |div u|).  They correlate with |e| in the median (+0.27, +0.31) but the
# sign FLIPS per vessel, and leave-one-vessel-out ridge R^2 for |e| from all of them together is
# 0.0175 -- a pooled model cannot use them (`src.tools.diagnostics.fem-error-indicators`).
#
#   all three metrics beat the prior -> the floors were the missing constraint
#   relL2 fixed, gate falls back    -> the two objectives genuinely trade; report the frontier
set -u
# Overridable so the floor weight can be swept into distinct run dirs.
export ARM_NAME=${ARM_NAME:-E3_prior_fem_floors}
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

# --- the lines that differ from E2 ----------------------------------------------------------
# Make the FEM prior a FLOOR, in velocity and in the wall-shear channel both.
# SWEEP THIS.  Both hinges are PER-NODE -- `mean(relu(|pred-y|^2 - |prior-y|^2))` -- so at a
# large weight the only feasible point is `pred == prior`: any global improvement has to worsen
# SOME node, and each one is charged.  Measured at 2000 the run pinned `residual_scale` to
# +0.0001 and reproduced the prior exactly (gateJ% 95.4, depL2 0.021, dsrxS 0.978, to three
# decimals).  E2's unweighted-floor value was 50.  The useful regime is between.
export KINEMATICS_PRIOR_FLOOR_WEIGHT=${KINEMATICS_PRIOR_FLOOR_WEIGHT:-300}
export KINEMATICS_BAND_SHEAR_FLOOR=1
export KINEMATICS_BAND_FLOOR_WEIGHT=${KINEMATICS_BAND_FLOOR_WEIGHT:-300}

export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
export KINEMATICS_SELECT_MAX_GRAPHS=6
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-0}
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
export KINEMATICS_WALL_SHEAR_WEIGHT=150
export KINEMATICS_GATE_WEIGHT=100
bash scripts/stage_a/launch.sh
