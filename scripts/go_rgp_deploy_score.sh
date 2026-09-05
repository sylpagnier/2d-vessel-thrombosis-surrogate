#!/usr/bin/env bash
# Score an RGP-DEQ-on-FEM flow arm on the metric that ships: the biochem GNN's strictly-nested
# wall / off-wall deploy score, paired vessel-for-vessel against the plain-FEM arm.
#
# Everything upstream of the biochem CV is a function of the t=0 velocity field, so the whole
# chain has to be rebuilt per arm:
#
#   precache  RGP-DEQ(prior=fem) -> pack.u0_pred        (~6 s/vessel)
#   v3 cache  build_clot_ml_cache.py    --flow rgp      (~20 s/vessel)
#   v4 cache  build_clot_ml_cache_v4.py --flow rgp      (~1 s/vessel)
#   CV        run_phase9_cv.py, 5 folds x 3 seeds       (~50 min)
#   paired    against dc_fem_cfw025 / v5_fem, with the flow-holdout panel
#
#   CKPT=outputs/runs/E6_envelope_floor/kinematics_best.pth TAG=dc_e6 bash scripts/go_rgp_deploy_score.sh
#
# CROSS-FIT.  Pass `CROSSFIT=1` and the three checkpoints CKPT_A / CKPT_B / CKPT (full pool)
# instead of a single CKPT: each vessel is then precached by the arm that never trained on it
# (scripts/stage_a/crossfit_halves.py).  This is the only form of the measurement in which the
# RGP-DEQ arm and the FEM baseline are on equal footing, because the FEM solver has no
# training set to leak from.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=.
TAG=${TAG:?set TAG, e.g. dc_e6}
CACHE=${CACHE:-v5_${TAG}}
CROSSFIT=${CROSSFIT:-0}
L=outputs/logs/rgp_arm; mkdir -p "$L"
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/${TAG}.status"; }

ANCHORS=$(python -c "
from src.core_physics.wall_cohort_splits import FIT,DEV,CLOT_FREE
print(','.join(list(FIT)+list(DEV)+list(CLOT_FREE)))")

say "START ${TAG}  crossfit=${CROSSFIT}"

if [ "${CROSSFIT}" = "1" ]; then
  : "${CKPT_A:?}" "${CKPT_B:?}" "${CKPT:?}"
  # `assignment` maps every anchor to the ONE arm that did not train on it.
  for ARM in A B FULL; do
    STEMS=$(python -c "
from scripts.stage_a.crossfit_halves import assignment
print(','.join(assignment('${ANCHORS}'.split(','))['${ARM}']))")
    [ -n "${STEMS}" ] || continue
    case "${ARM}" in A) W=$CKPT_A;; B) W=$CKPT_B;; *) W=$CKPT;; esac
    say "precache half ${ARM} with $(basename "$(dirname "$W")")/$(basename "$W")"
    python -u scripts/precache_rgp_deq.py --only "${STEMS}" --prior-source fem \
           --checkpoint "$W" --force >> "$L/${TAG}_precache.log" 2>&1
  done
else
  : "${CKPT:?}"
  say "precache all with $(basename "$(dirname "$CKPT")")/$(basename "$CKPT")"
  python -u scripts/precache_rgp_deq.py --only "${ANCHORS}" --prior-source fem \
         --checkpoint "$CKPT" --force > "$L/${TAG}_precache.log" 2>&1
fi
say "  worst rel-L2: $(grep '^\[OK\]' "$L/${TAG}_precache.log" | awk '{print $2, $4}' | sort -k2 -r | head -3 | tr '\n' ' ')"

say "v3 cache"
python -u scripts/build_clot_ml_cache.py --flow rgp --out "outputs/clot_ml_cache_${TAG}" \
       --force > "$L/${TAG}_v3.log" 2>&1
say "v4 cache"
python -u scripts/build_clot_ml_cache_v4.py --flow rgp --src "outputs/clot_ml_cache_${TAG}" \
       --out "outputs/clot_ml_cache_${CACHE}" --force > "$L/${TAG}_v4.log" 2>&1
say "  cached $(ls "outputs/clot_ml_cache_${CACHE}" | wc -l) vessels"

say "biochem CV (5 folds x 3 seeds)"
python -u scripts/run_phase9_cv.py --tag "${TAG}" --cache "${CACHE}" --folds 5 --seeds 3 \
       --shape-w 2.0 --clot-free-w 0.25 > "$L/${TAG}_cv.log" 2>&1

say "paired vs plain FEM"
python -u scripts/eval_flow_source_paired.py \
       --a dc_fem_cfw025 --a-cache v5_fem --b "${TAG}" --b-cache "${CACHE}" \
       --flow-holdout-panel --out "outputs/deployclot/flow_paired_${TAG}.json" \
       2>&1 | tee "$L/${TAG}_paired.log"
say "DONE ${TAG} -> outputs/deployclot/flow_paired_${TAG}.json"
