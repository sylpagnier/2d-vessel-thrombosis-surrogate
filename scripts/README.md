# Scripts

Supported launchers for **HemoRGP** (active stacks only). Retired ladders live under [`archive/`](archive/).

## Active surface (44 scripts)

| Stack | Entry | Docs |
|-------|-------|------|
| **clot_ml_v0** (deploy clot) | `promote_clot_ml_v0.py`, `eval_clot_ml_v0.py`, `run_phase9_cv.py`, `eval_strict*.py` | `docs/WOUND_PROGRESS.md` |
| **RGP-DEQ** (Stage-A flow) | `go_kinematics_production_allfix.ps1`, `go_kinematics_data_gen.ps1`, `precache_rgp_deq.py`, `python -m src.bin.main train rgp-deq-kine` | `docs/KINEMATICS_BEST_ARCHITECTURE.md` |
| **Research sweeps** (FEM + clot_ml_v0) | `go_research_sweep.ps1` / `run_research_sweep.py` | `docs/RESEARCH_SWEEPS.md` |
| **Customer / viz** | `go_customer_predict.ps1`, `go_customer_predict_web.ps1` | `docs/VIZ_STANDARD.md` |

Utility `.ps1`: `_python_rc.ps1`, `kill_stale_python.ps1`, `install_torch_cuda.ps1`, `promote_kinematics_checkpoint.ps1`.

## clot_ml_v0 -- train, promote, eval

- `promote_clot_ml_v0.py` -- lock unified wounded/non-wounded stack (`kind: unified_v0`).
- `eval_clot_ml_v0.py` -- compare against pinned baseline (`clot_gnn_v5w` default).
- `eval_wound_complement.py` -- wound complement A/B vs pinned `clot_gnn_v4w`.
- `train_wound_rate.py` -- wound-rate constants (LOVO).
- `train_clot_gnn.py` -- C0 GNN member training (used by `run_phase9_cv.py`).
- `build_clot_ml_cache.py`, `build_clot_ml_cache_v4.py` -- feature caches for CV / promote.
- **OOF viz:** `run_phase9_cv.py` (three C0 arms) -> `eval_strict_temporal.py` ->
  `gen_clot_ml_v0_oof_viz_data.py` -> `build_clot_oof_temporal_artifact.py`.
  Wound LOVO: `gen_clot_ml_v0_oof_viz_data.py --wound`. Combine: `combine_clot_ml_v0_viz_data.py`.
- `eval_by_class.py` -- geometry-class breakdown for CV tags.
- Historical v4 promote pins: `archive/clot_gnn_v4_era/promote_clot_gnn_v4*.py`.

## RGP-DEQ kinematics (Stage A)

- `go_kinematics_production_allfix.ps1` -- production allfix baseline
- `go_kinematics_data_gen.ps1` -- mesh / pack generation helper
- `precompute_kinematics_t0.py`, `precache_rgp_deq.py`, `check_kinematics_promotion_gates.py`
- `preflight_kine_cohort.py`, `slim_kine_packs.py`, `finetune_kine_patient_anchors.py`, `calibrate_kine_loss_weights.py`
- `train_pi_wall_shear.py` -- pi-flux wall-shear calibration
- `viz_occlusion_flow_sweep.py` -- FEM occlusion oracle figures
- Archived sweep launchers: `archive/kinematics_sweeps/`

## Research geometry sweeps

- `go_research_sweep.ps1` / `run_research_sweep.py` -- FEM t=0 + `clot_ml_v0` (default)
- Configs: `configs/research_sweeps/*.json`

## Customer + visualization

- `go_customer_predict.ps1` -- desktop GUI
- `go_customer_predict_web.ps1` -- local browser UI (CUDA inference server)
- `gen_offwall_temporal_data.py`, `build_offwall_temporal_artifact.py` -- standard viz template
- `python -m src.evaluation.visualize_pipeline` -- steady-kin + deploy smoke

## Pack repair

- `repair_wound_pack_geometry.py`, `repair_pack_wall_normals.py` -- see `src/data_gen/lib/pack_repair.py`

## Diagnostics (kept workhorses)

- `diag_field_calibration.py`, `diag_geometry_class_recal.py`, `diag_clot_free_headroom.py`
- `diag_wound_composition.py`, `diag_wound_p003_causes.py`
- `diag_local_fem_accuracy.py`, `diag_pi_flux_interaction.py`, `diag_physics_gate_support.py`
- `diagnose_crack_001_root.py`, `diagnose_001_signs_of_life.py`, `diagnose_lumen_001_vs_007.py`
- `fit_gelation_wake_kernel.py`
- `eval_wall_protocol.py`, `eval_significance.py`
- `predict_wall_clot.py` -- shipped wall readout (physics gate + growth)

Deploy ckpt recipe: `src/evaluation/deploy_ckpt_recipe.py`.

## Archived (reference only)

| Folder | What |
|--------|------|
| `archive/tier1_retired/` | One-off diag probes, eval_misc, eda, step0, viz one-offs |
| `archive/ml_ladder_era/` | Differentiable-wall ODE + Optuna / temporal ML ladders |
| `archive/mat_growth_era/` | Wall-gen `go_wg_*`, `eval_mat_growth_simple`, phase1 sweeps |
| `archive/kinematics_sweeps/` | Retired Stage-A sweep launchers |
| `archive/clot_gnn_v4_era/` | Phase 9 v4 promote + legacy temporal viz builders |
| `archive/` (root) | GNODE, T0, legacy graybox |

See also: `AGENTS.md`, `docs/PUBLISHING.md`, `docs/MAT_GROWTH.md` (historical mat-growth notes).
