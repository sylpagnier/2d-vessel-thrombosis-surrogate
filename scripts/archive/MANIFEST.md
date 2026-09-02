# Archived scripts manifest (deleted from tree 2026-09-01)

**~583 launcher/script files** lived here before cleanup. They are **not in the working tree**
anymore. Recover any file from git history:

```bash
git log --oneline -- scripts/archive/
git show <commit>:scripts/archive/mat_growth_era/eval_mat_growth_simple.py
```

Do not restore bulk archives into `scripts/` without updating `scripts/README.md` and
`docs/ARCHIVED_STACKS.md`.

## Inventory by bucket

| Bucket | .py | .ps1 | Era / purpose | Superseded by |
|--------|-----|------|---------------|---------------|
| *(root)* | 105 | 77 | GNODE, T0, graybox, early biochem launchers | RGP-DEQ + `clot_ml_v0` |
| `mat_growth_era/` | 86 | 41 | Wall-gen `go_wg_*`, phase1, `eval_mat_growth_simple` | `clot_ml_v0`, `deploy_ckpt_recipe.py` |
| `tier1_retired/` | 188 | 0 | One-off diags, eval_misc, EDA, step0 | Active diags removed; forensics via git |
| `ml_ladder_era/` | 35 | 5 | Differentiable-wall ODE, Optuna ladders | `eval_strict*.py` / `clot_ml_v0` |
| `clot_gnn_v4_era/` | 21 | 0 | v4 promote + temporal viz builders | `promote_clot_ml_v0.py`, OOF viz chain |
| `kinematics_sweeps/` | 1 | 9 | Retired Stage-A sweep launchers | `go_kinematics_production_allfix.ps1` |
| `wall_physics_era/` | 15 | 0 | Phase 3-6 zero-param wall readout + diags | `clot_ml_v0` wall arm |

## Notable entry points (git only)

| Script | Bucket | Notes |
|--------|--------|-------|
| `eval_mat_growth_simple.py` | mat_growth_era | Cold deploy eval; logic in `src/evaluation/deploy_ckpt_recipe.py` |
| `promote_biochem_gnn.py` | mat_growth_era | Locked WC_v7 customer biochem |
| `go_wg_prec_*.ps1` | mat_growth_era | Precision frontier ladders |
| `promote_clot_gnn_v4*.py` | clot_gnn_v4_era | Pre-unified v4/v4w promote |
| `predict_wall_clot.py` | wall_physics_era | Zero-param Phase 3 wall model |
| `diag_regime_gate_sweep.py` | tier1_retired | Scoring parity: call `canonical_grade_series` only |
| `validate_comsol_calibration.py` | root | COMSOL validator (see `docs/COMSOL_PHYSICS_VALIDATION.md`) |

## Disposition tags

- **reference** — may rerun from git for paper / table repro (`eval_mat_growth_simple`, promote scripts)
- **forensic** — read saved JSON under `outputs/` instead of rerunning diags
- **superseded** — alternate path; do not port forward (`ml_ladder_era`, most `go_wg_*`)
