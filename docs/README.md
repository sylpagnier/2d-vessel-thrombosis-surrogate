# Documentation index

Design and validation docs for **Local FEM Solver**. Start with the
[project README](../README.md) for what the project is and how to run it.

## Orientation

| Doc | Purpose |
|-----|---------|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Goals, stages, source map, CLI entry points |
| [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md) | Canonical model IDs and legacy aliases |
| [PUBLISHING.md](PUBLISHING.md) | What is git-tracked vs. generated locally |
| [RELATED_WORK.md](RELATED_WORK.md) | Where this sits in the literature |

## Flow

| Doc | Purpose |
|-----|---------|
| [COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md) | Flow and physics parity against COMSOL |
| [COMSOL_MU_RHEOLOGY_CHECKLIST.md](COMSOL_MU_RHEOLOGY_CHECKLIST.md) | Viscosity / rheology setup checklist |
| [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md) | RGP-DEQ architecture and training recipe (research arm) |

## Clot

| Doc | Purpose |
|-----|---------|
| [BIOCHEM_GNN.md](BIOCHEM_GNN.md) | Biochemistry / species deploy stack |
| [SEALED_SPLIT.md](SEALED_SPLIT.md) | Which vessels are held out, and why |
| [RESEARCH_SWEEPS.md](RESEARCH_SWEEPS.md) | Parametric geometry sweeps (FEM + `clot_ml_0`) |

## Operating

| Doc | Purpose |
|-----|---------|
| [CUSTOMER_INSTALLER.md](CUSTOMER_INSTALLER.md) | Building and releasing the self-contained Predict app |
| [VIZ_STANDARD.md](VIZ_STANDARD.md) | Figure and colour conventions |
| [OUTPUTS_RETENTION.md](OUTPUTS_RETENTION.md) | What to keep and prune under `outputs/` |
| [ARCHIVED_STACKS.md](ARCHIVED_STACKS.md) | Retired model stacks and where they went |
| [../scripts/README.md](../scripts/README.md) | Supported launchers |
| [../data/reference/README.md](../data/reference/README.md) | Tracked baseline manifests |
| [assets/](assets/) | Figures used by the README |

## A note on internal history

This project kept detailed working logs during development — per-phase result
chronicles, ablation ladders, and decision records. Those are working notes rather
than documentation, so they are not published here, and some code comments cite them
by filename (`WALL_MODEL_PLAN.md`, `PHASE*.md`, and similar). Treat such a reference
as provenance for why a line exists, not as a document you are missing. The results
those notes led to are summarised in the docs above and in the project README.
