#!/usr/bin/env bash
# E4: E2 (ReZero) plus a BAND-LOCALISED residual envelope.
#
# The hard BC multiplies the residual by `1 - exp(-bc_lambda*sdf)`, which is ~1 everywhere except
# a thin layer at the wall.  Against an already-accurate prior that is backwards.  Measured on
# the deploy packs against the FEM prior:
#
#     wall-distance decile     1     2     3     4     5     6     7     8     9    10
#     share of |e|^2        0.049 0.145 0.179 0.179 0.157 0.116 0.076 0.048 0.030 0.020
#     envelope (lambda=40)  0.591 0.950 0.994 0.999 1.000 1.000 1.000 1.000 1.000 1.000
#
# The outer 40% of the domain carries 17% of the prior's error and gets FULL authority; the
# near-wall decile the gate metric reads is damped to 0.59.  And that is where the head's signal
# actually is -- corr(delta, prior_error) is +0.25 in the wall band and -0.03 globally.  So the
# plain envelope licenses the head to add noise in the core, which is exactly how E2 improves
# gateJ% and dsrxScale while LOSING rel-L2.
#
#     env(sdf) = (1 - exp(-bc_lambda*sdf)) * exp(-bc_envelope_decay*sdf)
#
# still exactly zero at the wall, peaks in the near-wall band, decays in the core.  At decay=6 the
# envelope reads 0.52/0.61/0.47/0.34 over deciles 1-4 and 0.06-0.14 over 7-10.
#
# Tried and rejected first: raising the two prior-floor hinges (E3).  They are PER-NODE, so at a
# large weight the only feasible point is `pred == prior` -- at weight 2000 the run pinned
# `residual_scale` to +0.0001 and reproduced the prior exactly; at 300 likewise; at 100 it merely
# interpolates between E2 and the prior.  A constraint on the head is not the same as a better
# head, which is what all three metrics need.
set -u
export ARM_NAME=${ARM_NAME:-E4_band_residual}
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
# The one line that differs from E2.
export KINEMATICS_BC_ENVELOPE_DECAY=${KINEMATICS_BC_ENVELOPE_DECAY:-6}

export KINEMATICS_VAL_EVERY=${KINEMATICS_VAL_EVERY:-2}
export KINEMATICS_SELECT_MAX_GRAPHS=6
export KINEMATICS_SELECT_PATIENCE=${KINEMATICS_SELECT_PATIENCE:-0}
export KINEMATICS_BAND_ON_CORNERS=1
export KINEMATICS_BAND_DSRX_ABS=1
# Overridable: the alpha sweep shows the trained residual scale is ~half the gate
# optimum, and these are the only terms whose gradient pushes it up (the data term
# is 91.6% of the loss and is flat in the residual scale).
export KINEMATICS_WALL_SHEAR_WEIGHT=${KINEMATICS_WALL_SHEAR_WEIGHT:-150}
export KINEMATICS_GATE_WEIGHT=${KINEMATICS_GATE_WEIGHT:-100}
bash scripts/stage_a/launch.sh
