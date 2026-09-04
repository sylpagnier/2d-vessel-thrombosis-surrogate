# Scripts

Supported launchers for **Local FEM Solver** (four active stacks). Retired ladders live under
[`archive/`](archive/). Full archive index: [`docs/ARCHIVED_STACKS.md`](../docs/ARCHIVED_STACKS.md).

## Active surface (~30 scripts)

| Stack | Entry | Docs |
|-------|-------|------|
| **deploy-clot** (`clot_ml_0`) | `promote_clot_ml_0.py`, `eval_clot_ml_0.py`, `run_phase9_cv.py`, `eval_strict*.py` | `docs/WOUND_PROGRESS.md` |
| **RGP-DEQ** (Stage A flow) | `go_kinematics_production_allfix.ps1`, `run_kinematics_production.py`, `precache_rgp_deq.py` | `docs/KINEMATICS_BEST_ARCHITECTURE.md` |
| **Research sweeps** | `go_research_sweep.ps1` / `run_research_sweep.py` | `docs/RESEARCH_SWEEPS.md` |
| **Diagnostics** | `go_diag.ps1` / `python -m src.tools.diagnostics` | `src/tools/diagnostics/registry.py` |
| **Customer / viz** | `go_customer_predict.ps1`, `go_customer_predict_web.ps1` | `docs/VIZ_STANDARD.md` |

Utility `.ps1`: `_launcher_common.ps1`, `go_diag.ps1`, `kill_stale_python.ps1`, `install_torch_cuda.ps1`,
`promote_kinematics_checkpoint.ps1`.

## DeployClot -- the whole deploy-flow pipeline, end to end

`go_deployclot.sh` runs every stage below in dependency order against `flow="fem"`, the local
Carreau solve; each stage is idempotent and skips work already on disk, so a rerun resumes.
See [`docs/DEPLOYCLOT.md`](../docs/DEPLOYCLOT.md).

- `go_deployclot.sh` -- the runbook.
- `diag_fem_flow_audit.py` -- what the local FEM solve costs per vessel, in the four
  quantities the deposition gate consumes.
- `build_temporal_transport.py` -- per-(node, time) transport channels, one directory per
  flow source; the timing head is fitted against these and must not read another flow's.
- `promote_clot_gnn_v4.py` -> `promote_clot_gnn_v4_temporal.py` ->
  `promote_clot_gnn_v4_wound.py` -> `promote_clot_ml_0.py` -- the promotion chain: base
  ensemble, timing head, wound complement, unified artifact.
- `eval_wound_ab_pair.py` -- the matched A/B counterfactual, which scores the DIFFERENCE the
  injury makes on a fixed geometry rather than the two vessels.
- `build_deployclot_report.py` -- renders whatever is on disk into the validation report.

## deploy-clot (`clot_ml_0`) -- train, promote, eval

- `promote_clot_ml_0.py` -- lock unified wounded/non-wounded stack (`kind: unified_v0`).
- `eval_clot_ml_0.py` -- compare against pinned baseline (`clot_gnn_v5w` default).
- `eval_wound_complement.py` -- wound complement A/B vs pinned `clot_gnn_v4w`.
- `train_wound_rate.py` -- wound-rate constants (LOVO).
- `train_clot_gnn.py` -- C0 GNN member training (used by `run_phase9_cv.py`).
- `build_clot_ml_cache.py`, `build_clot_ml_cache_v4.py` -- feature caches for CV / promote.
- `eval_significance.py` -- paired bootstrap noise floor for CV arms.
- **OOF viz:** `run_phase9_cv.py` -> `eval_strict_temporal.py` ->
  `gen_clot_ml_0_oof_viz_data.py` -> `build_clot_oof_temporal_artifact.py`.
  Wound LOVO: `gen_clot_ml_0_oof_viz_data.py --wound`. Combine: `combine_clot_ml_0_viz_data.py`.
- `eval_by_class.py` -- geometry-class breakdown for CV tags.

## RGP-DEQ kinematics (Stage A)

- `go_kinematics_production_allfix.ps1` / `run_kinematics_production.py` -- production allfix baseline (orchestrator)
- Config + runner: `src/training/kinematics_production_config.py`, `kinematics_production_runner.py`
- `go_kinematics_stage_a_ladder.ps1` -- ladder-only entry (delegates to `run_kinematics_production.py ladder`)
- `go_kinematics_data_gen.ps1` -- mesh / pack generation helper
- `precompute_kinematics_t0.py`, `precache_rgp_deq.py`, `check_kinematics_promotion_gates.py`
- `preflight_kine_cohort.py`, `slim_kine_packs.py`, `finetune_kine_patient_anchors.py`,
  `calibrate_kine_loss_weights.py`
- `train_pi_wall_shear.py` -- pi-flux wall-shear calibration
- `viz_occlusion_flow_sweep.py` -- FEM occlusion oracle figures
- Archived sweep launchers: `archive/kinematics_sweeps/`

## Research geometry sweeps

- `go_research_sweep.ps1` / `run_research_sweep.py` -- FEM t=0 + deploy-clot (`clot_ml_0`) (default)
- Config loader: `src/evaluation/research_sweep_config.py` (shared control defaults)
- Presets / axis grids: `src/evaluation/research_sweep_presets.py`
- Configs: `configs/research_sweeps/*.json` (legacy biochem: `configs/research_sweeps/legacy/`)
- Shared launcher helpers: `_launcher_common.ps1` (all `go_*.ps1` delegate here)

## Ad-hoc diagnostics

- `go_diag.ps1` / `python -m src.tools.diagnostics` / `scripts/diag.py` (thin shim)
- Registry: `src/tools/diagnostics/registry.py`

## Customer + visualization

- `go_customer_predict.ps1` -- desktop GUI
- `go_customer_predict_web.ps1` -- local browser UI (CUDA inference server)
- `build_customer_bundle.ps1` -- package the web UI into a self-contained Windows zip (embedded Python + CPU torch + checkpoints + demo vessel)
- `release_bundle.ps1` -- build + tag + push + publish that zip as a GitHub Release (see [`docs/PUBLISHING.md`](../docs/PUBLISHING.md) for why the release, not `git clone`, ships the bundle)
- `gen_offwall_temporal_data.py`, `build_offwall_temporal_artifact.py` -- zero-param physics viz template
- `gen_clot_ml_0_oof_viz_data.py`, `build_clot_oof_temporal_artifact.py` -- learned-model OOF viz
- `python -m src.evaluation.visualize_pipeline` -- steady-kin + deploy smoke

## Pack repair

- `repair_wound_pack_geometry.py`, `repair_pack_wall_normals.py` -- see `src/data_gen/lib/pack_repair.py`

## Compatibility / optional viz

- `predict_wall_clot.py` -- zero-param Phase 3 wall readout (optional viz template only; git history for other wall_physics scripts)

Deploy ckpt recipe (mat-growth forensics): `src/evaluation/deploy_ckpt_recipe.py`.

## Archived (documentation only)

Deleted 2026-09-01. Inventory: [`archive/MANIFEST.md`](archive/MANIFEST.md) (recover via git).

| Folder README | Era |
|---------------|-----|
| `archive/mat_growth_era/README.md` | Wall-gen / biochem mat-growth |
| `archive/wall_physics_era/README.md` | Phase 3-6 zero-param wall |
| `archive/ml_ladder_era/README.md` | ML ladder / Optuna |

Stale local outputs: `docs/OUTPUTS_RETENTION.md`, `scripts/cleanup_stale_outputs.ps1`.

See: `AGENTS.md`, `docs/ARCHIVED_STACKS.md`, `docs/PUBLISHING.md`.
