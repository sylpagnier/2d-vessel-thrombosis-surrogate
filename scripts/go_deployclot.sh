#!/usr/bin/env bash
# DeployClot -- the final clot_ml_0 training, on the local FEM solver's t=0 flow.
#
# WHAT MAKES THIS THE DEPLOY ARM.  Every clot number in Phases 6-10 was measured on COMSOL's
# own t=0 velocity (`flow="gt"`), and `clot_ml_v0`'s manifest carries `cold_deploy: blocked`
# for exactly that reason.  Here the t=0 field comes from `src/core_physics/local_fem_solver`
# -- a steady Carreau Navier-Stokes solve on the vessel's own mesh, given the mesh, the
# inlet/outlet/wall boundary tags and the inlet velocity profile, and nothing else.  No COMSOL
# field enters any input; GT appears only in the labels.
#
# Stages, in dependency order.  Each writes its own log under outputs/logs/deployclot.
#
#   0  fem_flow_audit     what the solver costs, per vessel, in the four quantities the
#                         deposition gate consumes
#   1  caches             55-col + 68-col feature caches, GT arm and FEM arm, over the full
#                         2026-09-02 corpus (FIT 29 / DEV 5 / CLOT_FREE 9 / SEALED 4)
#   2  cv                 geometry-stratified 5-fold x 3 seeds, three arms:
#                           fem_c0   FEM flow  + C0 shape constraint   <- DeployClot
#                           gt_c0    GT  flow  + C0                    <- upper bound
#                           fem_noc0 FEM flow, no C0                   <- the C0 ablation
#   3  readout            strictly-nested readout selection per arm
#   4  wound              two-regime (G_pre, G_post), leave-one-vessel-out on all 6 wounds
#   5  promote            base ensemble -> temporal head -> wound complement -> clot_ml_0
#   6  eval               deploy metric, the A/B counterfactual, and the one SEALED read
#
set -u
LOG=outputs/logs/deployclot
STATUS="$LOG/STATUS.txt"
mkdir -p "$LOG"
stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$STATUS"
  local t0=$SECONDS
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$STATUS"
  return $rc
}

FLOW=fem
NAME=DeployClot          # base ensemble + temporal head
WNAME=DeployClot_w       # + wound complement
UNAME=DeployClot         # unified clot_ml_0 artifact name is set by --name at stage 5d

# --- 0  what the deploy-legal flow costs -------------------------------------------------
stage 01_fem_flow_audit_fixed \
  python scripts/diag_fem_flow_audit.py --out outputs/deployclot/fem_flow_audit.json

# --- 1  caches over the full corpus ------------------------------------------------------
# `--include-sealed` caches 007/013/031/043 so the ONE final read has features to run on.
# Caching is not spending: `run_phase9_cv.py` and `promote_clot_gnn_v4.py` drop SEALED from
# every training pool themselves.
stage 02_cache_gt      python scripts/build_clot_ml_cache.py --flow gt  --force --include-sealed
stage 03_cache_v5_gt   python scripts/build_clot_ml_cache_v4.py --flow gt \
                              --out outputs/clot_ml_cache_v5 --force --include-sealed
stage 04_cache_fem     python scripts/build_clot_ml_cache.py --flow fem --force --include-sealed
stage 05_cache_v5_fem  python scripts/build_clot_ml_cache_v4.py --flow fem \
                              --src outputs/clot_ml_cache_fem \
                              --out outputs/clot_ml_cache_v5_fem --force --include-sealed

# --- 2  cross-validation -----------------------------------------------------------------
stage 06_cv_gt_c0   python scripts/run_phase9_cv.py --tag dc_gt_c0  --cache v5     \
                           --folds 5 --seeds 3 --shape-w 2.0
stage 08_cv_fem_c0  python scripts/run_phase9_cv.py --tag dc_fem_c0 --cache v5_fem \
                           --folds 5 --seeds 3 --shape-w 2.0

# --- 3  strictly-nested readout ----------------------------------------------------------
stage 09_readout_gt   python scripts/eval_expected_score_readout.py --tags dc_gt_c0  --cache v5
stage 10_readout_fem  python scripts/eval_expected_score_readout.py --tags dc_fem_c0 --cache v5_fem

# --- 4  wound complement, leave-one-vessel-out on all six wounds -------------------------
stage 07_wound_rate_fem  python scripts/train_wound_rate.py --flow $FLOW \
                                --out outputs/clot_ml/wound_rate_fem

# --- 5  promotion chain ------------------------------------------------------------------
stage 11_transport_fem  python scripts/build_temporal_transport.py --flow $FLOW --force
stage 12_promote_base   python scripts/promote_clot_gnn_v4.py --name $NAME \
                               --cache v5_fem --shape-w 2.0
stage 13_promote_temporal python scripts/promote_clot_gnn_v4_temporal.py --name $NAME --flow $FLOW
stage 14_promote_wound  python scripts/promote_clot_gnn_v4_wound.py --base $NAME --name $WNAME \
                               --flow $FLOW
stage 15_promote_v0     python scripts/promote_clot_ml_0.py --base $WNAME --name $UNAME \
                               --flow $FLOW

# --- 6  evaluation -----------------------------------------------------------------------
stage 16_eval_v0    python scripts/eval_clot_ml_0.py --v0 $UNAME --baseline $WNAME \
                           --flow $FLOW --cohort --out outputs/deployclot/eval_v0.json
stage 17_ab_pair    python scripts/eval_wound_ab_pair.py --model $UNAME --flow $FLOW \
                           --out outputs/deployclot/ab_pair.json
echo "[$(date +%H:%M:%S)] DEPLOYCLOT DONE" | tee -a "$STATUS"
