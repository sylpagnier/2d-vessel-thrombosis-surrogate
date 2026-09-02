#!/usr/bin/env bash
# Sequential (NOT parallel -- 4 GB GPU) generation of the flow-requirement inputs.
# Regenerates what commit b2eebb9 deleted; see docs/PUBLICATION_PLAN.md s9.
set -u
cd "$(dirname "$0")/../.."
log() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

log "1/3 cohort eval, flow=gt"
python scripts/eval_clot_ml_0.py --cohort --flow gt --out outputs/runs/eval_gt.json
echo "exit=$?"

log "2/3 cohort eval, flow=pred"
python scripts/eval_clot_ml_0.py --cohort --flow pred --out outputs/runs/pred_all.json
echo "exit=$?"

log "3/3 tolerance curve, 3 vessels"
python scripts/diag_flow_sensitivity.py patient010 patient005 patient020 \
    --source pred --out outputs/runs/flow_sensitivity.json
echo "exit=$?"

log "collecting"
python scripts/publication/generate_flow_requirement_data.py
python scripts/publication/plot_flow_requirement.py
log "DONE"
