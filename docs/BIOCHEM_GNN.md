# `biochem_gnn` -- retired architecture, naming record only

**This is not the current biochem architecture.** If you are looking for how the shipped
model (`clot_ml_0`) actually turns flow into a clot prediction, read
[`docs/PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) and the root [`README.md`](../README.md)'s
"How it works" table instead, and see "What replaced it" below. This document exists so a
reader who finds `src/biochem_gnn/` in the tree, or the `biochem_gnn` model id in
[`MODEL_NOMENCLATURE.md`](MODEL_NOMENCLATURE.md), understands what it was and why it is no
longer live -- not as a guide to retraining or using it.

## What replaced it

The shipped clot model (`clot_ml_0`) does not learn all twelve chemical species and then
apply a gelation rule. It **discretises** the deposition ODE, the wall activated-platelet
closure, and the advective transport equation directly (`src/core_physics/`), and learns only
one thing off-wall: a threshold-crossing `Mat` field head, `src/clot_ml/mat_field.py`, trained
via `scripts/go_mat_field_v6.py`. This project's own academic report (since deleted as
outdated) recorded the species-pushforward approach documented below as **superseded**:
"error in the species compounds through the threshold." That is a documented design
decision, not an oversight -- do not re-wire `clot_ml_0` to depend on `biochem_gnn`'s model
code.

## What is still live in `src/biochem_gnn/`

The package was not deleted wholesale because two of its three files are still load-bearing,
for reasons unrelated to the retired model:

| File | Status | Used by |
|------|--------|---------|
| `wall_cohort_constants.py` | **Live** | Wound scripts (`promote_clot_gnn_v4_wound.py`, `train_wound_rate.py`, `eval_wound_ab_pair.py`, `eval_wound_complement.py`) import `WOUND_COHORT` etc. -- vessel-name lists, not the model |
| `config.py` | **Live** | `scripts/precompute_kinematics_t0.py` calls `apply_deploy_env`; `src/tests/conftest.py` uses it for fixtures. Holds checkpoint-path helpers and env-application logic for the RGP-DEQ deploy path, not a model definition |
| `__init__.py` | Re-exports `config.py` only | Nothing in it defines or exports a model class |

**There is no `BiochemGNN` class in the tracked tree.** A repo-wide search for
`class BiochemGNN` and for actual `BiochemGNN(...)` / `.from_manifest(` call sites found none
-- not in `src/biochem_gnn/`, not anywhere else. `from src.biochem_deploy import BiochemGNN`
(the legacy alias shim) would raise `ImportError` today. Earlier revisions of this doc, and
`src/biochem_deploy/__init__.py`'s own docstring, describe a Python API that no longer exists;
both have been corrected in this pass. Treat any doc or comment elsewhere in the repo that
still shows `from src.biochem_gnn import BiochemGNN` as stale.

## The retired architecture (history only)

```text
rgp_deq_kine              [frozen RGP-DEQ, Stage A]
  -> species_graphsage    [trained]  wall-band GraphSAGE (FI / Mat)   -- REMOVED
  -> gelation_beta        [trained]  global Mat scale                 -- REMOVED
  -> clot_trigger_physics [equations] Carreau + gelation + nucleation -- REMOVED
  -> local_kinematic_corrector [deprecated, deleted 2026-09-01] k-hop [dU, dV]
```

`species_graphsage`/`gelation_beta`'s model code, and the two `src/training/` modules that
fed it (`biochem_species_scope.py`, `clot_trigger_stack.py`), were confirmed to have zero
callers anywhere in the tracked `clot_ml_0` build/train/promote/eval chain and were removed
in this cleanup pass. The dedicated trainer (`train_biochem_gnn.py`, `go_biochem_gnn.ps1`,
`promote_biochem_gnn.py`) was already retired earlier (commit `2161caa`) and never replaced.
An orphaned local checkpoint may still exist at
`outputs/biochem/biochem_gnn/locked/species_gnn_best.pth` (canonical `WC_v7_clot_phi_mse`,
2026-07-19) with nothing left in the tracked tree that loads it.

Naming: [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md). Baseline metrics: `MAT_GROWTH.md`
(internal working note, not tracked in git). GNODE teacher/corrector path
(`train_biochem_corrector`) is separately retired; condensed lessons and leaderboards are in
internal working notes (`BIOCHEM_LEGACY_LESSONS.md`, `archive/BIOCHEM_GNN_BASELINES.md`), not
tracked in git.

## Retraining `clot_ml_0` today

This does **not** depend on any of the retired model code above -- only on the live
`wall_cohort_constants.py`:

```bash
# the whole deploy-flow pipeline, end to end -- see scripts/README.md "DeployClot"
scripts/go_deployclot.sh

# or the individual stages it runs, in order:
python scripts/train_clot_gnn.py --lovo --epochs 300 --seeds 3 --tag final
python scripts/run_phase9_cv.py
python scripts/promote_clot_gnn_v4.py
python scripts/promote_clot_gnn_v4_temporal.py
python scripts/promote_clot_gnn_v4_wound.py   # uses WOUND_COHORT vessel names only
python scripts/promote_clot_ml_0.py
python scripts/eval_clot_ml_0.py --cohort

# the learned off-wall Mat field head (wound cohort's off-wall component):
python scripts/go_mat_field_v6.py cache --flow fem
python scripts/go_mat_field_v6.py lovo  --flow fem --epochs 30
python scripts/go_mat_field_v6.py eval  --flow fem --lovo
```
