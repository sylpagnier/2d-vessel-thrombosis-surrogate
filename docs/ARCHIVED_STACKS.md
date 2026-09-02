# Archived stacks and reference code

This repo keeps four **active** surfaces. Retired code is documented under
`scripts/archive/MANIFEST.md` and `src/archive/` (import shims only where legacy paths remain).

Recover deleted scripts from git: `git show <commit>:scripts/archive/<path>`.

## Active stacks (use these)

| Stack | Train / promote | Deploy / eval | Docs |
|-------|-----------------|---------------|------|
| **deploy-clot** (`clot_ml_0`) | `train_clot_gnn.py`, `run_phase9_cv.py`, `promote_clot_ml_0.py` | `eval_clot_ml_0.py`, `eval_strict*.py`, customer Predict | `docs/WOUND_PROGRESS.md`, `docs/PHASE10_V4.md` |
| **RGP-DEQ** (Stage A) | `python -m src.bin.main train rgp-deq-kine`, `go_kinematics_production_allfix.ps1` | `precache_rgp_deq.py`, gate-J selection packs | `docs/KINEMATICS_BEST_ARCHITECTURE.md` |
| **Customer** | (locked artifacts only) | `go_customer_predict*.ps1`, `CustomerDeployPipeline` | `docs/VIZ_STANDARD.md` |
| **Research sweeps** | configs in `configs/research_sweeps/` | `go_research_sweep.ps1` / `run_research_sweep.py` | `docs/RESEARCH_SWEEPS.md` |

Customer default: **RGP-DEQ once at t=0** + **deploy-clot** (`clot_ml_0`) C0-tail rollout (no GT
velocity, no local corrector on the default path).

---

## Script archives (deleted from tree 2026-09-01)

Runnable files removed; inventory in [`scripts/archive/MANIFEST.md`](../scripts/archive/MANIFEST.md).
Recover via `git log -- scripts/archive/`.

---

## Source archives (`src/archive/`)

| Folder | What it was for | Still reachable via |
|--------|-----------------|---------------------|
| `mat_growth/` | Species pushforward, offwall growth, `train_biochem_gnn`, deploy coupled forward | Deprecation shims in `src/training/`, `python -m src.bin.main train biochem-gnn` |
| `corrector_era/` | Local kinematic corrector training and verification tools — **deprecated, deleted 2026-09-01, not for publication** ([docs/LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md)) | Shims at `src/core_physics/coupled_shear_gnn.py`, `src/inference/corrector_coupling.py` (legacy biochem only) |
| `differentiable_wall_model/` | ML-ladder survival head + wall ODE experiments | Import `src.archive.differentiable_wall_model.*` |

Cohort vessel lists for wall-cohort scoring now live in
`src/biochem_gnn/wall_cohort_constants.py` (not the mat-growth leg registry).

`src/biochem_gnn/mat_growth_simple.py` remains as a **legacy leg registry** for explicit
`legacy_species` / `locked_canonical` research comparisons only.

---

## When to open an archive

- Reproducing a number quoted in `docs/archive/` or `docs/PHASE*.md`
- Comparing a new clot leg against an old mat-growth checkpoint recipe (`deploy_ckpt_recipe.py`)
- Forensics on a retired diagnostic (gate support, corrector characterization)

Do **not** add new env knobs or trainers there; extend the active stacks and typed configs
(`PushforwardConfig`, `BiochemRuntimeConfig`) instead.
