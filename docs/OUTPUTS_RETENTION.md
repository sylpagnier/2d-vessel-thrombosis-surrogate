# Local outputs retention (not in git)

`outputs/` is gitignored. This doc is the **keep list** for active stacks; everything else
is disposable experiment cruft.

## Keep (do not bulk-delete)

| Path | Stack |
|------|--------|
| `outputs/kinematics/` | RGP-DEQ production + comsol finetune checkpoints |
| `outputs/clot_ml/locked/` | Shipped deploy-clot (`clot_ml_0`) artifact |
| `outputs/clot_ml/wound_rate/` | Wound complement LOVO constants |
| `outputs/phase9_scores/` | CV caches for `eval_strict*.py` |
| `outputs/phase9_log.jsonl` | Phase 9 training log (if present) |
| `outputs/clot_ml_cache_*/` | Feature caches for CV / promote — kept by PREFIX, one per flow source and generation |
| `outputs/temporal_transport*/` | `eval_strict_temporal.py` transport NPZ — one directory per flow source |
| `outputs/runs/` | **Stage-A flow checkpoints.** `runs/E5_band_gateup/kinematics_best_calibrated.pth` is what the `split` recipe consumes; losing it costs a full retrain |
| `outputs/archive/` | **Snapshotted model cohorts** (`scripts/archive_model_cohort.py`) — the weights a paper's numbers are attached to |
| `outputs/deployclot/` | Deploy-score evaluations; the published numbers live here |
| `outputs/publication/` | Figure data and PDFs (`scripts/publication/make_all.ps1`) |
| `outputs/research_sweeps/` | FEM mesh cache + sweep results |
| `outputs/customer_predict/` | Customer UI session outputs |
| `outputs/cache/` | Kinematics prepared cache (`KINEMATICS_PREPARED_CACHE`) |
| `outputs/kine_loss_weights*.json` | Calibrated Stage-A loss weights |
| `outputs/pi_corpus/` | Pi-flux wall-shear training corpus |
| `outputs/reports/` | Kinematics / occlusion figures |
| `outputs/clot_ml_0_*temporal*.json` | OOF viz payloads |
| `outputs/offwall_temporal_data.json` | Zero-param viz template (optional) |
| `outputs/biochem/biochem_gnn/locked/` | Legacy biochem checkpoints (`legacy_species` only) |

## Safe to delete

Pre-Aug 2026 experiment trees, smoke dirs, and one-off viz under `outputs/` that are **not**
in the table above. Examples removed in the 2026-09-01 cleanup:

- `outputs/biochem/*` except `biochem_gnn/locked/`
- `outputs/ml_ladder`, `opt_ladder`, `ml_clean_protocol`, `temporal_only`
- `outputs/wall_species_cache`, `ap_closure`, `onset_*`, `rollout_trackA`
- `outputs/mesh_test`, `temp_vessels`, `viz_*`, `*_smoke`
- `outputs/mat_field_cache*`

Re-run training / CV to regenerate caches if you delete a kept path by mistake — except
`outputs/runs/` and `outputs/archive/`, which hold trained weights and cannot be regenerated
from anything else on disk.

> **The keep list drifted, and it was one command from costing a retrain.**  Checked
> 2026-09-05: the previous list named 17 directories and `outputs/` held 45, so
> `cleanup_stale_outputs.ps1` would have deleted 28 of them — including `runs/` (the flow
> checkpoint the whole split recipe consumes), `archive/`, `deployclot/`, `publication/`, and
> every feature cache and transport directory added since the list was written.  The two
> families that legitimately grow — `clot_ml_cache_*` and `temporal_transport*` — are now kept
> by PREFIX rather than by name, so a new flow source is protected the day it is created
> rather than the day someone notices.

## Cleanup command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\cleanup_stale_outputs.ps1
```

Use `-WhatIf` first to preview.
