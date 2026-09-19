# 2D Thrombosis Surrogate — project context

Orientation for contributors: goals, stages, layout, and safe places to change behavior.

## Goal

**2D Thrombosis Surrogate** is a mesh-agnostic deployable thrombosis surrogate for parametric vessel studies: **local FEM at t=0** + unified **deploy-clot** (`clot_ml_0`) clot rollout on vessel graphs. `rgp_deq_kine` (RGP-DEQ) remains a live research/ablation flow arm — see publication/STORY.md (local only) Leg 2. `biochem_gnn` is a **retired** architecture (species pushforward + gelation rule), superseded by `clot_ml_0`'s own solved-physics + learned `Mat`-field design — see [BIOCHEM_GNN.md](BIOCHEM_GNN.md).

Canonical deploy path: in-house Carreau FEM + **deploy-clot** (`clot_ml_0`). Injured vessels run through the same artifact's wound branch (publication/STORY.md 0.2, local only). Sweeps: [RESEARCH_SWEEPS.md](RESEARCH_SWEEPS.md).

## Stages

| Stage | What | Entry |
|-------|------|-------|
| **A — Kinematics** | Steady non-Newtonian flow on vessel graphs | `python -m src.bin.main train rgp-deq-kine` |
| **B — Biochemistry** *(retired)* | Species pushforward + clot readout on frozen flow | none -- see [BIOCHEM_GNN.md](BIOCHEM_GNN.md); the shipped clot model is `clot_ml_0`, trained via `scripts/train_clot_gnn.py` |

Orchestration: `python -m src.bin.orchestrate {a|b|all}`.

Config “phase” names (`kinematics`, `biochem`, `biochem_anchors`, …) select **physics terms and datasets**, not a separate product line.

## Architecture snapshot

- **`RGP_DEQ`** (`src/architecture/ginodeq.py`) — Stage-A RGP-DEQ; id `rgp_deq_kine`. Legacy class alias: `GINO_DEQ`.
- **`PhysicsKernels`** (`src/core_physics/physics_kernels.py`) — residual / BC / rheology shared by train and tests.
- **`ClotGNN`** (`src/clot_ml/gnn.py`) — shipped clot model backbone (`clot_ml_0`); solved deposition ODE + AP closure + transport (`src/core_physics/`) feed it, plus a learned off-wall `Mat` field head (`src/clot_ml/mat_field.py`).
- **Deprecated** `LocalKinematicCorrector` — local `[dU, dV]` on frozen UV; deleted 2026-09-01, not for publication.

### Clot semantics

A “clot” here is not a discrete classifier. It is a region where continuous `mu_eff` is elevated by platelet/fibrin gelation. Primary targets are viscosity / species regressions; threshold Dice (`HighMuDice@thr`) is diagnostic only.

`BiochemConfig.mu_ratio_max` caps **dimensionless gelation step heights**, not physical clot `mu_eff`. Prefer `mu_log_mae`, wall / high-mu subsets, and deploy clot F1 for decisions.

## Entry points

| Action | Command |
|--------|---------|
| Chain A then B | `python -m src.bin.orchestrate all` |
| Train RGP-DEQ | `python -m src.bin.main train rgp-deq-kine` (alias: `train kinematics`) |
| Train deploy-clot (`clot_ml_0`) | `scripts/go_deployclot.sh` -- see [BIOCHEM_GNN.md](BIOCHEM_GNN.md) "Retraining `clot_ml_0` today" |
| Kinematics datagen | `python -m src.data_gen.pipeline_kinematics` |
| Biochem datagen | `python -m src.data_gen.pipeline_biochem` |
| Customer UI | `scripts/go_customer_predict.ps1` (same browser app the release zips run) |
| Flow demo | `python -m src.bin.main inspect flow -- --rheology carreau` |

Checkpoints: `outputs/clot_ml/locked/` (shipped clot model), `outputs/kinematics/` (local). Reference manifests: `data/reference/`. Path helpers: `data_root()`, `outputs_root()`, `kinematics_dir()`, `biochem_dir()`, `reports_dir()`, `comsol_models_dir()`, `resolve_checkpoint()`.

## Simulation BC assumptions

- `Re` from **inlet diameter**; held fixed across sims in current recipes
- Inlet: fully developed flow; outlet: zero pressure
- Biochem adds species BCs (inlet concentrations, outlet flux) on top of flow

## Anchor vs non-anchor

- **Anchors** — full COMSOL labels on the graph
- **Non-anchors** — geometry + physics residuals only
- Kinematics preference: **~50/50** mix for generalization and continuity

## Source map

| Path | Role |
|------|------|
| `src/core_physics/` | PDE kernels, rheology, biochem closures |
| `src/architecture/` | RGP-DEQ, DEQ loop, spectral layers |
| `src/biochem_gnn/` | Retired-model package; live parts are wound-cohort constants + RGP-DEQ deploy-env config -- [BIOCHEM_GNN.md](BIOCHEM_GNN.md) |
| `src/data_gen/` | Pipelines + `lib/` mesh/COMSOL/graph builders |
| `src/training/` | Trainers (kinematics; local-corrector and biochem/mat-growth trainers archived, deprecated) |
| `src/evaluation/` | Benchmarks and viz |
| `src/inference/` | Deploy / customer pipelines |
| `src/utils/` | Paths, metrics, batching, kinematics helpers |
| `src/bin/` | CLI router + orchestrator |
| `src/tools/` | Interactive inspectors and GUIs |
| `src/tests/` | Pytest (fast, deterministic where possible) |

## Artifacts (mostly local)

| Path | Tracked? |
|------|----------|
| `data/reference/*.json` | Yes — small manifests |
| `data/raw/`, `data/processed/` | No — bulk CFD / graphs |
| `outputs/` | No — checkpoints, logs, figures |
| `comsol_models/` | No — `.mph` sources |

Policy: [PUBLISHING.md](PUBLISHING.md).

## Interactive tools

| Module | Purpose |
|--------|---------|
| `src.tools.customer_predict_web` | Predict UI (local browser app; also what both release zips launch) |
| `src.tools.demo_kinematics_flow` | Drag walls → Gmsh → RGP-DEQ |
| `src.tools.inspect_kinematics_data` | Kinematics anchors / graph health |
| `src.tools.inspect_biochem_data` | Biochem domain txt + graphs |
| `src.tools.extract_biochem_comsol` | Pull solved `.mph` → graphs |
| `src.tools.verify_deq_convergence` | Picard vs Anderson residual curves |

Routed also via `python -m src.bin.main inspect <target>`.

## Edit conventions

- Mesh-agnostic: no fixed node/edge counts or regular grids
- Prefer physical names: `u`, `v`, `p`, `phi`, `mu`, `gamma_dot`
- Small local changes over broad refactors unless requested
- Meaningful logic changes need focused pytest coverage

## Tests

- Physics / index safety: `src/tests/`
- CLI / graph / trainer wiring: `test_cli_routing_regression.py`, `test_mesh_to_graph_regression.py`, `test_predictor_trainer_*.py`
- Optional strict BC checks vs graph wall masks: see `KINEMATICS_PHYSICS_CHECK_BC` in kinematics physics tests
