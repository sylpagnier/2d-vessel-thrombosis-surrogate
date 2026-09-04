# Documentation index

Active design and operator docs for **Local FEM Solver**. Lab notebooks, sweep logs, and retired ladders live under [`archive/`](archive/).

## Start here

| Doc | Purpose |
|-----|---------|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Goals, stages, source map, CLI entry points |
| [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md) | RGP-DEQ, biochem_gnn, local corrector IDs |
| [PUBLISHING.md](PUBLISHING.md) | What is git-tracked vs local-only |
| [PUBLICATION_PLAN.md](PUBLICATION_PLAN.md) | Which results can carry a paper, in what order, and what each is missing |

## Stage A — flow (RGP-DEQ)

| Doc | Purpose |
|-----|---------|
| [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md) | Locked architecture + training recipe |
| **[RGP_DEQ_REPAIR_PLAN.md](RGP_DEQ_REPAIR_PLAN.md)** | **Why the deployable flow arm fails: prior leak, wall-band blindness, bug list + redesign** |
| [PILOT_COHORT_RUNBOOK.md](PILOT_COHORT_RUNBOOK.md) | Generate on the COMSOL box, slim, preflight, workshop — the step before a full retrain |
| [LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md) | k-hop clot diversion GNN — **deprecated, deleted 2026-09-01, not for publication** |
| [COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md) | Flow / physics parity vs COMSOL |
| [COMSOL_MU_RHEOLOGY_CHECKLIST.md](COMSOL_MU_RHEOLOGY_CHECKLIST.md) | Viscosity / rheology checklist |

## Stage B — biochemistry / clot

| Doc | Purpose |
|-----|---------|
| [BIOCHEM_GNN.md](BIOCHEM_GNN.md) | Deploy stack (`biochem_gnn`) |
| [MAT_GROWTH.md](MAT_GROWTH.md) | Canonical mat-growth baseline and how to extend it |
| [BIOCHEM_LEGACY_LESSONS.md](BIOCHEM_LEGACY_LESSONS.md) | Condensed takeaways from retired ladders |
| [PHASE10_V4.md](PHASE10_V4.md) | `clot_gnn_v4`: strict protocol, readout, noise floor |
| [WOUND_PROGRESS.md](WOUND_PROGRESS.md) | Injured wall: the ungated law, `clot_gnn_v4w`, unified deploy-clot (`clot_ml_0`) |
| **[MODEL_REVIEW_2026-08-22.md](MODEL_REVIEW_2026-08-22.md)** | **Full-stack review: where the remaining score is, and the current to-do list** |
| [SEALED_SPLIT.md](SEALED_SPLIT.md) | Which vessels are held out, and the 2026-08-22 release |

## Operators

| Doc | Purpose |
|-----|---------|
| [../scripts/README.md](../scripts/README.md) | Supported launchers |
| [../AGENTS.md](../AGENTS.md) | Short agent / contributor cheat sheet |
| [../data/reference/README.md](../data/reference/README.md) | Tracked baseline manifests |
| [CUSTOMER_INSTALLER.md](CUSTOMER_INSTALLER.md) | Building and releasing the self-contained Predict app bundle |
| [assets/](assets/) | README figures |

## Archive

Historical chronicles, baseline leaderboards, decision dumps, and cleanup notes:

- [archive/README.md](archive/README.md)
