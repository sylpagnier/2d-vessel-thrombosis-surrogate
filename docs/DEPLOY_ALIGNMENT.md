# Deploy alignment: does the Predict app feed the model what it was trained on?

2026-09-14. Audit of every path by which the Predict app builds a vessel for `clot_ml_0`
(parametric sliders, mesh upload, `.pt` upload) against the cohort the deployed model was fit on:
the 36 COMSOL vessels in `clot_ml_final`'s `training_pool` and the 6 wound vessels of
`clot_ml_final_w`.

## How it was checked

1. **Geometry and conventions**, measured on the training data itself (raw `.nas` meshes in
   `data/raw/biochem_anchors`, packs in `data/processed/graphs_biochem_anchors`) with the same
   outline code the app uses for uploads. The results are the `TRAINED_*` constants in
   `src/data_gen/lib/customer_geometry_import.py`; `src/tests/test_customer_training_envelope.py`
   re-measures them whenever the data is present.
2. **End to end against COMSOL ground truth.** A training vessel with a COMSOL solution is run
   (a) exactly as in training -- its own pack, native 201-step grid -- and (b) through the app's
   own input path, starting from the raw mesh. Predictions are mapped to the pack's nodes by
   physical position (median match distance 0 um) and scored with the deploy metric
   (`src.clot_ml.evaluate.domain_score`, final time and mean over time). The two runs differ only
   in how the app builds the graph, so any gap is a train/deploy mismatch.
3. **Deploy-legal reference.** The training pack itself run with only what a customer run can have
   -- the analytic inlet profile and FEM t=0 pressure instead of COMSOL's fields (`pack_deploy`).
   The app path cannot be expected to beat this: it is the model's own ceiling without COMSOL.
4. **Ablation.** Where (b) fell short, pieces of the training pack were transplanted into the
   app-built graph one at a time to find which input carried the gap.

## Mismatches found, and what was done

| # | Input | Training | App before | Effect / fix |
|---|---|---|---|---|
| 1 | Inlet / outlet node masks | every node on the cut, both corners shared with the wall | corners carved out by `gmsh_line_boundary_masks` (and the P2 mid-sides beside them): 41 inlet nodes vs 45 on comsol041 | **Largest effect.** Transplanting only the pack's masks: wall 0.60 -> 0.85. Fixed: `_keep_opening_corners`. Also affects slider-built vessels, which go through the same tag reader. |
| 2 | t=0 pressure (`p_nd`, one of the 69 features) | COMSOL's solved t=0 pressure | 0 (no COMSOL field) | Second effect. Fixed deploy-legally: filled from the app's own FEM solve (`solve_fem_into_pack`), same units and outlet gauge; matches COMSOL's field to 2% rel-L2 on comsol041. Training packs keep COMSOL pressure. |
| 3 | Length scale `d_bar` | inlet width, on all 42 vessels | mean width (generator sidecar and new outline code) | Same on a plain vessel; ~10% on a 77% stenosis, moving inlet velocity and every non-dimensional feature with it. Fixed: `_inlet_width_length_scale`. Alone it did not close the gap. |
| 4 | Prothrombin / antithrombin / fibrinogen at t=0 | bulk level (log1p(1)) everywhere | 0 | No measurable effect on a non-wound vessel (identical scores with and without); corrected anyway, since the wound chemistry reads these species. |
| 5 | Open-end detection on uploads | -- | cap stopped one edge short of the corner | Fixed (`_extend_straight`). Inferred masks now match COMSOL's tags on 40/42 vessels; the other two are vessels whose COMSOL tags are themselves incomplete (comsol038, comsol048, already documented in `local_fem_solver.py`). |
| 6 | Node channels the model reads (`wall_normal`, `width_d1`, `width_d2`) | built by `ComsolAnchorDataExtractor`: graph-derived normals on the final P2 nodes, WLS sparse operators for the width derivatives | built by `MeshToGraph`: line-segment normals on the P1 mesh, interpolated through P2 elevation; different derivative operators | Off-wall effect once 1-2 were fixed: comsol032 off-wall 0.823 -> 0.881. Fixed: `_training_recipe_node_x` rebuilds every customer graph's channels with the extractor's own functions (`mesh_wls.wls_sparse_operators`, now shared). |
| 8 | Time grid the temporal head is queried on | the `n_times` = 11 evenly spaced indices of each vessel's own pack axis, as `eval_strict_temporal.py` queries them | every index of a 120-step linspace (app), every index of the sweep axis, every 4th pack frame (`eval_clot_ml_0.py`) | The off-wall lag regressor predicts in WHOLE GRID STEPS, so a denser axis rescales the clock: a lag of 4 means 4/10 of the run on the trained grid and 4/119 on the app's. Final-time scores are unaffected (both grids end on the same frame), which is why the step-count check above cleared it; mean-over-time is not. App on comsol041: off-wall 0.57 -> 0.78. Cohort-wide, `flow=gt`: mean-over-time off-wall 0.539 -> 0.697, wall 0.721 -> 0.728 (40 vessels). Fixed: `trained_time_axis` / `trained_query_grid` in `src/clot_ml/locked.py`, used by the app (2026-09-14), and by the research sweeps and `eval_clot_ml_0.py` (2026-09-18). |
| 7 | Wall at a stenosis throat | COMSOL's selection misses 6 nodes per side on comsol041 | tagged as wall | Kept: the app's topology-derived wall is physically right; the pack's gap is a COMSOL selection artefact. |

### Control ranges (the geometry a user can build)

| Control | Trained cohort | App before | App now |
|---|---|---|---|
| Vessel width (inlet) | 8.3 - 20.0 mm, median ~15 | 4 - 12 mm, default 8 | 8.5 - 20 mm, default 15 |
| S-curve amplitude | at most +/-4.7 mm | 0 - 12 mm | 0 - 5 mm |
| Bend | 0 - 115 deg | 0 - 90 deg | unchanged |
| Stenosis | 77% narrowing, centred, FWHM 0.12 L | up to 80%, location 0.2-0.8, sharpness 0.1-3 | severity up to 77%, fixed trained position and taper |
| Aneurysm | 1.99x inlet, centred, FWHM 0.20 L | up to 3x, free position and taper | severity up to 1.99x, fixed trained position and taper |
| Wound position / width | centre 32-69% of length, 6-15% wide | 5-95% / 2-60% | 32-69% / 6-15% |
| Simulated time with a wound | 6,136 - 11,975 s (one outlier at 19,118) | up to 30,000 s | up to 12,000 s |
| Reynolds number | 450 on every vessel | 450 fixed | unchanged |
| Vessel length | ~100 mm on every vessel | 100 mm fixed (sliders); anything (uploads) | uploads outside the range are flagged in the preview |

Uploads are not blocked outside the envelope; the preview lists each measurement that is outside
it (`training_envelope_warnings`).

### Checked and found aligned

- **Reynolds number and inlet profile.** Re = 450 on every pack. The analytic Carreau inlet the
  app imposes matches COMSOL's inlet velocity to 1-5% rel-L2 with the same mean flow.
- ~~**Time step.**~~ **Withdrawn 2026-09-18 -- this was mismatch #8, not an alignment.** See the
  row below: the check that cleared it looked only at final scores, which are the one thing the
  query grid does not move.
- **Inlet side.** Every trained vessel flows left to right from its minimum-x end, which is the
  app's automatic choice.
- **Flow-prior channels (`u_prior`, `wss_prior`, `mu_prior`).** Still differ from the packs, and
  are not model inputs (none is among the 69 features), so they were left alone. (An early
  transplant of all 18 node channels "changed nothing", but it was run before the mask fix; once
  the masks were right the geometry channels mattered -- row 6.)

## End-to-end result

Final time, deploy metric, `wall / off-wall`. "App" columns are the same raw COMSOL mesh uploaded
through the Predict app's own path at each stage of the fix.

| vessel | training pack, COMSOL inputs | training pack, deploy-legal inputs | app before | app + masks & d_bar | app + FEM pressure | **app, final** |
|---|---|---|---|---|---|---|
| comsol041 (stenosis) | 0.948 / 0.902 | 0.951 / 0.893 | 0.621 / 0.611 | 0.850 / 0.823 | 0.944 / 0.851 | **0.948 / 0.871** |
| comsol044 (stenosis) | 0.965 / 0.923 | 0.965 / 0.925 | 0.581 / 0.532 | 0.849 / 0.812 | 0.954 / 0.912 | **0.962 / 0.915** |
| comsol012 (baseline) | 0.901 / 0.883 | 0.887 / 0.883 | -- | 0.885 / 0.833 | 0.864 / 0.830 | **0.854 / 0.881** |
| comsol032 (115 deg bend) | 0.996 / 0.907 | 0.984 / 0.899 | -- | 0.984 / 0.823 | 0.984 / 0.823 | **0.984 / 0.881** |
| comsol020 (widest) | 0.956 / 0.543 | 0.635 / 0.219 | -- | 0.628 / 0.233 | 0.628 / 0.230 | **0.626 / 0.223** |

The app path now sits at the deploy-legal ceiling on every vessel to within the pipeline's noise
(same cache, disjoint seeds: wall +/-0.005, off-wall +/-0.045), except comsol012's wall
(-0.033). "App before" is not measured for 012/032/020: the fixes were found on 041/044 first.

### Two residuals that are the model's, not the app's

**comsol012 wall, -0.033: the model keys on numerical noise.** Every input now matches except the
width derivatives at the stenosis throat. There the training pack's `width_d2` is +135 to -35,354
while any other mesh gives -3 to -292: second derivatives through WLS stencils on COMSOL's slightly
curved P2 mid-side nodes (<=16 um off the edge midpoint) are dominated by that curvature, and 10 of
the 11 wall nodes the two paths disagree on sit exactly there. A customer mesh -- and every
slider-built vessel -- cannot reproduce that noise, and should not. Recommendation for the next
training round: build `width_d1/d2` with `mls_gradient` (stable) or drop them.

**comsol020, -0.32 wall: sensitivity to the inlet profile.** With deploy-legal inputs the training
pack itself falls from 0.956 to 0.635, so this is not the app path. The trigger is COMSOL's inlet
profile, which on this vessel's 12.5-deg tilted inlet is skewed (0.078 vs 0.088 m/s at mirror
positions) where the analytic profile is symmetric; same flux, and the t=0 FEM speed field differs by
only 0.9% on average. Restoring COMSOL's pressure does nothing (0.635); only its inlet velocity
recovers the score. The model is on a knife edge here; the analytic inlet is the correct deploy
input and the gap belongs in the model's validation, not its UI.

## Limits of this check

- It scores vessels the model was trained on, so it measures alignment of the input path, not
  generalisation. Held-out performance is the strict-CV table in the `clot_ml_final` manifest.
- The t=0 inlet velocity in the reference arm is COMSOL's; the app's is analytic. They agree to a
  few percent (above), but the deploy-legal path can never use COMSOL's.
- Slider-built vessels have no COMSOL solution, so they are covered by the shared fixes (1-4) and
  the control ranges, not by a direct score.
- Draft meshing (the default for slider-built vessels, 2x coarser than training density) was not
  re-scored here.
