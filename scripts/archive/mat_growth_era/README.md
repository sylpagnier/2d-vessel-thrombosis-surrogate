# Mat-growth / wall-gen launcher era (retired 2026-09-01)

Pre-`clot_ml_v0` biochem mat-growth research: `go_wg_*` precision ladders, phase1
wall-gen sweeps, `eval_mat_growth_simple.py`, compound deploy promotion, and related
probes/sweeps.

**Superseded by:** `clot_ml_v0` + in-house FEM research sweeps (`go_research_sweep.ps1`)
and RGP-DEQ kinematics (`go_kinematics_*.ps1`).

Deploy recipe binding lives in `src/evaluation/deploy_ckpt_recipe.py` (extracted from
`eval_mat_growth_simple.py`). The `src/biochem_gnn/mat_growth_simple.py` leg registry
remains in `src/` only for `locked_canonical` research-sweep compatibility.

Do not add these launchers back to `scripts/README.md`.
