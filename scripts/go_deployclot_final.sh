#!/usr/bin/env bash
# The final DeployClot build: TWO artifact families from one frozen configuration.
#
#   VALIDATED  (DeployClot2 / _w / _0)   trains on the 36 non-SEALED vessels.  SEALED is held
#                                        out, the strictly-nested CV is a real generalisation
#                                        estimate, and EVERY published number comes from here.
#   PRODUCTION (DeployClotP / _w / _0)   trains on all 40, SEALED included.  Maximum data for
#                                        the deployed product; NO valid metric of any kind.
#                                        Stamped `metrics_invalid` on the manifest, and
#                                        `eval_clot_ml_0.py` refuses to score it.
#
# The two are identical in every other respect -- same configuration, same code path, same
# wound complement -- which is exactly why the boundary is a machine-checkable stamp and not a
# naming convention.  docs/PUBLICATION_PLAN.md 12.
#
# The LOCKED POINTER goes to the VALIDATED artifact.  The product fetches the production one
# by explicit name, so a default load can never silently return an unscoreable model.
#
#   CFW=0.25 bash scripts/go_deployclot_final.sh
#
set -u
cd "$(dirname "$0")/.."
L=outputs/logs/deployclot_final; mkdir -p "$L"
S="$L/STATUS.txt"
FLOW=fem
CACHE=v5_fem
CFW="${CFW:-1.0}"                   # clot-free node-loss weight, chosen by CV
RATE_DIR=outputs/clot_ml/wound_rate_fem_v2
RATE_ARM=const_noapc
# the wound off-wall depth rule, leave-one-vessel-out over all six wounds (DEPLOYCLOT.md 18)
V0_ARGS="--replace-scope wound_region --replace-depth 1 --att-beta 0.5"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$S"; }
stage () { local n="$1"; shift; say "START $n"; "$@" > "$L/$n.log" 2>&1;
           local rc=$?; say "END   $n rc=$rc"; [ $rc -eq 0 ] || say "  !! see $L/$n.log"; }

say "DEPLOYCLOT FINAL  flow=$FLOW cache=$CACHE clot_free_w=$CFW"

# ---------------------------------------------------------------- 1  VALIDATED family
stage 01_val_base     python scripts/promote_clot_gnn_v4.py --name DeployClot2 \
                             --cache $CACHE --shape-w 2.0 --clot-free-w "$CFW"
stage 02_val_temporal python scripts/promote_clot_gnn_v4_temporal.py --name DeployClot2 \
                             --flow $FLOW
stage 03_val_wound    python scripts/promote_clot_gnn_v4_wound.py --base DeployClot2 \
                             --name DeployClot2_w --flow $FLOW \
                             --rate-dir $RATE_DIR --rate-arm $RATE_ARM
stage 04_val_v0       python scripts/promote_clot_ml_0.py --base DeployClot2_w \
                             --name DeployClot2_0 --flow $FLOW $V0_ARGS --repoint

# ---------------------------------------------------------------- 2  PRODUCTION family
stage 05_prod_base     python scripts/promote_clot_gnn_v4.py --name DeployClotP \
                              --cache $CACHE --shape-w 2.0 --clot-free-w "$CFW" \
                              --include-sealed
stage 06_prod_temporal python scripts/promote_clot_gnn_v4_temporal.py --name DeployClotP \
                              --flow $FLOW
stage 07_prod_wound    python scripts/promote_clot_gnn_v4_wound.py --base DeployClotP \
                              --name DeployClotP_w --flow $FLOW \
                              --rate-dir $RATE_DIR --rate-arm $RATE_ARM
stage 08_prod_v0       python scripts/promote_clot_ml_0.py --base DeployClotP_w \
                              --name DeployClotP_0 --flow $FLOW $V0_ARGS

# ---------------------------------------------------------------- 3  the honest numbers
# Scored on the VALIDATED artifact only.  The production one is refused by the scorer.
stage 09_eval_val   python scripts/eval_clot_ml_0.py --v0 DeployClot2_0 \
                           --baseline DeployClot2_w --flow $FLOW --cohort \
                           --out outputs/deployclot/final_eval_validated.json
stage 10_ab_pair    python scripts/eval_wound_ab_pair.py --model DeployClot2_0 --flow $FLOW \
                           --out outputs/deployclot/final_ab_pair.json
stage 11_refuses    python scripts/eval_clot_ml_0.py --v0 DeployClotP_0 \
                           --baseline DeployClot2_w --flow $FLOW --stems patient012

say "DEPLOYCLOT FINAL DONE"
say "  validated  -> DeployClot2_0   (pointer; publish these numbers)"
say "  production -> DeployClotP_0   (product; unscoreable by construction)"
