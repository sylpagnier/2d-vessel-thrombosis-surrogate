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
| `outputs/clot_ml_cache_{gt,pred,v4,v4_pred,v5,v5_pred}/` | Feature caches for CV / promote |
| `outputs/temporal_transport/` | `eval_strict_temporal.py` transport NPZ |
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
- `outputs/mesh_test`, `temp_vessels`, `viz_*`, `*_smoke`, `runs/`, `publication/`
- `outputs/mat_field_cache*`, `ap_field_cache`

Re-run training / CV to regenerate caches if you delete a kept path by mistake.

## Cleanup command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\cleanup_stale_outputs.ps1
```

Use `-WhatIf` first to preview.
