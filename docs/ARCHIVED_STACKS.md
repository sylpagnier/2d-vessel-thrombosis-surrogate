# Archived stacks and reference code

This repo keeps four **active** surfaces. Retired launchers are inventoried in
`scripts/archive/MANIFEST.md`; retired model eras are kept out of the published tree
(see [PUBLISHING.md](PUBLISHING.md)) and recoverable from git history.

Recover a deleted script: `git show <commit>:scripts/archive/<path>`.

## Active stacks (use these)

| Stack | Train / promote | Deploy / eval | Docs |
|-------|-----------------|---------------|------|
| **deploy-clot** (`clot_ml_0`) | `train_clot_gnn.py`, `run_phase9_cv.py`, `promote_clot_ml_0.py` | `eval_clot_ml_0.py`, `eval_strict*.py`, customer Predict | [SEALED_SPLIT.md](SEALED_SPLIT.md), [BIOCHEM_GNN.md](BIOCHEM_GNN.md) |
| **RGP-DEQ** (Stage A) | `python -m src.bin.main train rgp-deq-kine`, `go_kinematics_production_allfix.ps1` | `precache_rgp_deq.py`, gate-J selection packs | [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md) |
| **Customer** | (locked artifacts only) | `go_customer_predict*.ps1`, `CustomerDeployPipeline` | [CUSTOMER_INSTALLER.md](CUSTOMER_INSTALLER.md), [VIZ_STANDARD.md](VIZ_STANDARD.md) |
| **Research sweeps** | configs in `configs/research_sweeps/` | `go_research_sweep.ps1` / `run_research_sweep.py` | [RESEARCH_SWEEPS.md](RESEARCH_SWEEPS.md) |

Shipped default: **local FEM at t=0** + **deploy-clot** (`clot_ml_0`) C0-tail rollout — no GT
velocity and no local corrector on the default path.

---

## Retired eras

| Era | What it was for | Status |
|-----|-----------------|--------|
| `mat_growth` | Species pushforward, off-wall growth, biochem GNN trainer | The three modules still reachable were promoted into `src/training/`; the rest is out of tree |
| `corrector_era` | Local kinematic corrector training and verification tools | Deprecated 2026-09-01, not for publication. The two modules the legacy biochem path still imports live at `src/core_physics/coupled_shear_gnn.py` and `src/inference/corrector_coupling.py` |
| `differentiable_wall_model` | ML-ladder survival head and wall-ODE experiments | Out of tree; nothing imports it |

Cohort vessel lists for wall-cohort scoring live in
`src/biochem_gnn/wall_cohort_constants.py` (not the mat-growth leg registry).

`src/biochem_gnn/mat_growth_simple.py` remains as a **legacy leg registry** for explicit
`legacy_species` / `locked_canonical` research comparisons only.

---

## Extending

Do **not** add new env knobs or trainers to a retired era; extend the active stacks and
typed configs (`PushforwardConfig`, `BiochemRuntimeConfig`) instead.
