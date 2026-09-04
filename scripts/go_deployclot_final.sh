#!/usr/bin/env bash
# The final DeployClot build. Overnight, one frozen configuration, five blocks.
#
# BLOCK 1  VALIDATED family  (DeployClot2 / _w / _0)   trains on the 36 non-SEALED vessels.
#          SEALED is held out, the strictly-nested CV is a real generalisation estimate, and
#          EVERY published number comes from here.
# BLOCK 2  PRODUCTION family (DeployClotP / _w / _0)   trains on all 40, SEALED included.
#          Maximum data for the deployed product; NO valid metric of any kind.  Stamped
#          `metrics_invalid`, and `eval_clot_ml_0.py` refuses to score it.
# BLOCK 3  GT-FLOW arm       (DeployClotG / _w / _0)   same recipe on COMSOL t=0 flow, then
#          evaluated against the in-house solver.  Answers: is it better to train on GT flow
#          and accept a train/deploy skew, or to train matched on FEM as we ship today?
# BLOCK 4  Threshold refit on FEM features for the GT-trained model.  Tells us whether the
#          GT-vs-FEM gap is CALIBRATION (a refit recovers it) or INFORMATION (it does not).
# BLOCK 5  Research sweeps, re-run.  The 2026-09-01 sweeps resolved `clot_ml_0` through the
#          pointer bug (DEPLOYCLOT 21) to the legacy `clot_ml_v0`, whose `replace_scope`
#          defaulted to `all_lumen` -- which ERASES the GNN's whole lumen verdict on a wound
#          pack.  All 15 wound arms reported exactly lumen=0 / occlusion=0 / open=100.  Both
#          causes are fixed; the sweeps must be re-run before any wound figure is trusted.
#
# The two artifact families are identical in every other respect -- same configuration, same
# code path, same wound complement -- which is why the boundary is a machine-checkable stamp
# and not a naming convention.  docs/PUBLICATION_PLAN.md 12.
#
# The LOCKED POINTER goes to the VALIDATED artifact.  The product fetches production by
# explicit name, so a default load can never silently return an unscoreable model.
#
#   CFW=0.25 bash scripts/go_deployclot_final.sh
#   CFW=0.25 BLOCKS="1 2" bash scripts/go_deployclot_final.sh     # subset
#
set -u
cd "$(dirname "$0")/.."
L=outputs/logs/deployclot_final; mkdir -p "$L"
S="$L/STATUS.txt"
FLOW=fem
CACHE=v5_fem
GT_CACHE=v5
CFW="${CFW:-0.25}"                  # clot-free node-loss weight (DEPLOYCLOT.md 25)
BLOCKS="${BLOCKS:-1 2 3 4 5}"
RATE_DIR=outputs/clot_ml/wound_rate_fem_v2
RATE_ARM=const_noapc
# the wound off-wall depth rule, leave-one-vessel-out over all six wounds (DEPLOYCLOT.md 18)
V0_ARGS="--replace-scope wound_region --replace-depth 1 --att-beta 0.5"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$S"; }
stage () { local n="$1"; shift; say "START $n"; "$@" > "$L/$n.log" 2>&1;
           local rc=$?; say "END   $n rc=$rc"; [ $rc -eq 0 ] || say "  !! see $L/$n.log"; }
has () { case " $BLOCKS " in *" $1 "*) return 0;; *) return 1;; esac; }

say "DEPLOYCLOT FINAL  flow=$FLOW cache=$CACHE clot_free_w=$CFW blocks=[$BLOCKS]"
say "  feature fingerprint: $(python -c 'import sys;sys.path.insert(0,".");from src.clot_ml.feature_fingerprint import feature_fingerprint as f;print(f())' 2>/dev/null)"

# ---------------------------------------------------------------- 1  VALIDATED family
if has 1; then
stage 01_val_base     python scripts/promote_clot_gnn_v4.py --name DeployClot2 \
                             --cache $CACHE --shape-w 2.0 --clot-free-w "$CFW"
stage 02_val_temporal python scripts/promote_clot_gnn_v4_temporal.py --name DeployClot2 \
                             --flow $FLOW
stage 03_val_wound    python scripts/promote_clot_gnn_v4_wound.py --base DeployClot2 \
                             --name DeployClot2_w --flow $FLOW \
                             --rate-dir $RATE_DIR --rate-arm $RATE_ARM
stage 04_val_v0       python scripts/promote_clot_ml_0.py --base DeployClot2_w \
                             --name DeployClot2_0 --flow $FLOW $V0_ARGS --repoint
stage 05_eval_val     python scripts/eval_clot_ml_0.py --v0 DeployClot2_0 \
                             --baseline DeployClot2_w --flow $FLOW --cohort \
                             --out outputs/deployclot/final_eval_validated.json
stage 06_ab_pair      python scripts/eval_wound_ab_pair.py --model DeployClot2_0 \
                             --flow $FLOW --out outputs/deployclot/final_ab_pair.json
fi

# ---------------------------------------------------------------- 2  PRODUCTION family
if has 2; then
stage 10_prod_base     python scripts/promote_clot_gnn_v4.py --name DeployClotP \
                              --cache $CACHE --shape-w 2.0 --clot-free-w "$CFW" \
                              --include-sealed
stage 11_prod_temporal python scripts/promote_clot_gnn_v4_temporal.py --name DeployClotP \
                              --flow $FLOW
stage 12_prod_wound    python scripts/promote_clot_gnn_v4_wound.py --base DeployClotP \
                              --name DeployClotP_w --flow $FLOW \
                              --rate-dir $RATE_DIR --rate-arm $RATE_ARM
stage 13_prod_v0       python scripts/promote_clot_ml_0.py --base DeployClotP_w \
                              --name DeployClotP_0 --flow $FLOW $V0_ARGS
# the refusal is a GATE, not a formality: it must fail (rc=2) or the stamp is not working
stage 14_refusal_gate  python scripts/eval_clot_ml_0.py --v0 DeployClotP_0 \
                              --baseline DeployClot2_w --flow $FLOW --stems patient012
fi

# ---------------------------------------------------------------- 3  GT-FLOW arm
# Trains on COMSOL t=0 flow, deploys against the in-house solver.  `promote_*_temporal --flow
# fem` is deliberate: the head and readout are fitted on the flow the PRODUCT will see, so the
# only skew is in the frozen GNN weights, which is the question being asked.
if has 3; then
stage 20_gt_cv       python scripts/run_phase9_cv.py --tag dc_gt_cfw --cache $GT_CACHE \
                            --folds 5 --seeds 3 --shape-w 2.0 --clot-free-w "$CFW"
stage 21_gt_readout  python scripts/eval_expected_score_readout.py --tags dc_gt_cfw \
                            --cache $GT_CACHE --save outputs/deployclot/readout_arms_gt.json
stage 22_gt_base     python scripts/promote_clot_gnn_v4.py --name DeployClotG \
                            --cache $GT_CACHE --shape-w 2.0 --clot-free-w "$CFW"
stage 23_gt_temporal python scripts/promote_clot_gnn_v4_temporal.py --name DeployClotG \
                            --flow $FLOW
stage 24_gt_wound    python scripts/promote_clot_gnn_v4_wound.py --base DeployClotG \
                            --name DeployClotG_w --flow $FLOW \
                            --rate-dir $RATE_DIR --rate-arm $RATE_ARM
stage 25_gt_v0       python scripts/promote_clot_ml_0.py --base DeployClotG_w \
                            --name DeployClotG_0 --flow $FLOW $V0_ARGS
# THE comparison: GT-trained weights, deployed on solved flow, against the matched-FEM family
stage 26_gt_eval     python scripts/eval_clot_ml_0.py --v0 DeployClotG_0 \
                            --baseline DeployClot2_w --flow $FLOW --cohort \
                            --out outputs/deployclot/final_eval_gtflow.json
fi

# ---------------------------------------------------------------- 4  threshold refit
# Calibration or information?  Both arms are scored on FEM features; only the readout scalars
# differ, so a recovered gap is calibration and a persistent one is information.
if has 4; then
# Each arm is scored on ITS OWN cache, thresholds refitted in-fold on that flow, so both
# sit at their own best operating point.  A gap surviving two independent refits is
# INFORMATION, not calibration.  (An earlier draft scored GT-derived OOF scores against the
# FEM cache: that pairs a score vector from one flow with masks from another and measures
# nothing, since the fold models are never re-run on FEM features by that path.)
stage 30_flow_paired python scripts/eval_flow_source_paired.py \
                            --a dc_gt_cfw     --a-cache $GT_CACHE \
                            --b dc_fem_cfw025 --b-cache $CACHE \
                            --out outputs/deployclot/flow_source_paired_final.json
stage 31_rank_cmp    python scripts/diag_offwall_ranking.py \
                            --tags dc_fem_cfw025 dc_gt_cfw \
                            --out outputs/deployclot/ranking_fem_vs_gt.json
fi

# ---------------------------------------------------------------- 5  research sweeps
# Against the CORRECT artifact this time.  `CustomerDeployPipeline` asks for `clot_ml_0`,
# which now resolves through the pointer to DeployClot2_0 rather than the legacy stub.
if has 5; then
stage 40_sweeps      python scripts/run_research_sweep.py --all
stage 41_sweep_check python scripts/diag_sweep_lumen_audit.py \
                            --out outputs/deployclot/sweep_lumen_audit.json
fi

say "DEPLOYCLOT FINAL DONE"
say "  validated  -> DeployClot2_0   (pointer; publish these numbers)"
say "  production -> DeployClotP_0   (product; unscoreable by construction)"
say "  gt-flow    -> DeployClotG_0   (comparison arm only)"
