# Archived reference implementations

Python modules moved out of the active import graph. See [`docs/ARCHIVED_STACKS.md`](../../docs/ARCHIVED_STACKS.md).

| Folder | Era | Purpose |
|--------|-----|---------|
| `mat_growth/` | 2025-26 | Species pushforward, biochem GNN trainers, wall-gen deploy coupling |
| `corrector_era/` | Stage-A+B | Local kinematic corrector training and verification tools |
| `differentiable_wall_model/` | Phase 7-8 | ML-ladder survival head and differentiable wall ODE experiments |

Deprecation shims at `src/training/train_biochem_gnn.py` (etc.) re-export `mat_growth/` for
one release cycle. New work belongs in `clot_ml/`, `src/training/train_kinematics_predictor.py`,
or customer/research runners.

Cohort constants: `src/biochem_gnn/wall_cohort_constants.py` (not `mat_growth_simple.py`).
