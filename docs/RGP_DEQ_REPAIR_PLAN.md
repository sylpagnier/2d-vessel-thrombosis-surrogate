# RGP-DEQ repair plan — making the deployable flow arm real

`clot_ml_0` scores wall 0.952 / off-wall 0.828 on COMSOL's own `t=0` velocity and
0.586 / 0.350 on RGP-DEQ's `u0_pred` (`docs/DEPLOY_FLOW_PLAN.md` §2).  This document is the
audit of *why*, and the ordered repair.  Everything here is measured; the measurement is
quoted next to the claim.

**The one-line finding.**  The pred-flow arm was never deployable and its collapse was never
the leak everyone was looking for.  Both arms run on GT-derived priors, so there is no
train/deploy prior shift in the numbers above.  The real cause is that **RGP-DEQ never learned
near-wall shear structure** — the only thing `clot_ml_0` reads — because it was scored on
velocity rel-L2, a metric the fast core owns and the wall band barely enters.  The prior leak
*hid* this: with `uv_prior = GT` the wall band was already correct for free.

---

## 1. What was measured

### 1a. The prior leak is total, and it is in the deploy path too

`data/processed/graphs_kinematics_anchors/carreau` (the Stage-A clinical training set),
all 43 packs:

```
rel-L2( x[:,11:13], y[:,0:2] )  = 0.000000   on 43/43
rel-L2( x[:,13],    y[:,3]    )  = 0.000000   on 43/43     (mu_prior)
max |x[:,14]|                    = 0.0        on 43/43     (wss_prior, dead channel)
```

Bit-identical.  This is `WALL_MODEL_PLAN.md` §16.1 / §17 Z2, already documented, and
`src/data_gen/lib/legal_priors.py` already exists to fix it.  **What is new is that nothing
calls it.**  `data/processed/graphs_biochem_anchors`, all 52 packs:

| pack class | prior rel-L2 vs GT | note |
|---|---|---|
| 43 plain `patientNNN` | 0.012 – 0.049 | CFD-derived |
| 7 `*_mirror_y` | 0.06 – 0.45 | stale / unmirrored CFD |
| 3 `wound_patient00*` | 0.26 – 0.49 | closest to legal, still not analytic |

For scale, a genuine analytic prior on the same packs reads **0.72 – 0.80**.

`scripts/precache_rgp_deq.py` never calls `apply_prior_source`, so `resolve_prior_source()`
falls to its `"stored"` default; `src/clot_ml/features.py:153` and
`src/core_physics/physics_wall_model.py:642` then consume the resulting `u0_pred`.
**Every predicted-flow number in the project is leak-assisted.**

### 1b. RGP-DEQ is a net destroyer of information

Cached `u0_pred` against GT, alongside the prior it was handed as input, all 52 packs:

```
median  RGP-DEQ output       0.141
median  its own input prior  0.025          <- 5.4x better than the model's answer
worse than its own prior:    45 / 52 packs
```

The 7 exceptions are exactly the 7 packs whose priors are *not* near-GT (the mirrors and the
wound packs) — i.e. the only packs where the model is doing genuine work.

### 1c. Under deploy-legal priors the field collapses

`scripts/diag_rgp_deq_flow_audit.py`, 8 vessels, promoted checkpoint:

```
source     relL2_u   cos     AUC(pred)   AUC(gt)
stored      0.124   +0.995     0.754      0.749     <- leaked, not deployable
analytic    0.385   +0.938     0.699      0.749     <- the only legal row
zero        0.900   +0.616     0.680      0.749
```

`relL2_v` under `analytic` runs 1.4 – 4.6, i.e. worse than predicting zero.  Transverse
velocity is where recirculation lives, and recirculation (`-u`) is the top single feature for
aneurysm clot (`WALL_MODEL_PLAN.md` §16.2).

### 1d. The model recovers little of what the wall band needs

The hard BC is `u = uv_prior + sdf * uvp` (`src/architecture/ginodeq.py:500`) — additive over
the **whole** domain, not just the boundary, so the model's own contribution is exactly
`pred - uv_prior` and it is multiplied by `sdf`, which goes to zero at the wall.  Measured in
the 3-hop wall band (`scratch/diag_prior_authority.py`), `analytic` rows using the **repaired**
prior builder of §1f:

```
vessel      priors      model/need   wall-band rel residual
patient020  stored         1.003          0.203
patient020  analytic       0.393          0.503
patient039  stored         0.854          0.222
patient039  analytic       0.326          0.466
patient041  analytic       0.171          0.642
patient044  analytic       0.207          0.665
patient001  analytic       0.590          0.446
```

Under legal priors the model supplies **17–59% of the correction the band needs**, leaving
**45–67% relative error in the only region the clot gate reads** — against 13–26% when it is
handed the answer.

> **Correction — "the BC denies the model authority at the wall" is WRONG.**  An earlier draft
> of this section read the `sdf` factor as nulling the model's influence where the consumer
> reads.  It does not.  `d/dn (uv_prior + sdf * uvp)` at `sdf = 0` is
> `d(uv_prior)/dn + uvp`: the *value* is pinned (correct no-slip) but wall shear has **unit
> first-order sensitivity to `uvp`**, and §1j shows the model exercising it.  What the table
> above measures is that the model does not *use* that authority well, which is an objective
> problem, not a capacity one.  The reparameterisation this originally motivated (D1) is
> therefore not the fix; D2 is.

> **Correction (superseded numbers).**  The first pass of §1d and §1e used a hand-rolled band
> dilation, `acc[row] = band[col]`, which is *last-write-wins* on duplicate indices: a node
> entered the band only if its last incident edge happened to point into it.  That returned a
> random subsample roughly half the true size, and reported the residual as 0.72–0.90 and the
> gradient share as 0.7–4.3%.  Both tables here are re-measured with
> `kinematics_physics_terms.wall_band_mask`, which dilates by `index_add`.  The direction of
> every conclusion is unchanged; the magnitudes are not.

### 1e. And the loss under-weights that region

`l_data_kine` is an **absolute** MSE on `(u, v, p)` (`src/utils/kinematics_physics_terms.py`).
The 3-hop wall band's share:

```
vessel      % nodes   % field energy   % of actual squared error
patient001    22.9        5.21             20.88
patient020    11.1        0.81              2.41
patient039    18.0        3.38             10.48
patient041    15.9        6.48              3.06
patient044    14.1        5.03              2.10
```

11–23% of the nodes carrying **2–21% of the gradient**, and on the three vessels where the
surrogate is worst (020, 041, 044) it is 2–3%.  And `boundary_data_weight = 2.0` was applied to
`mask_wall` — where GT `u` is exactly 0 and the hard BC already pins `pred = uv_prior`.
It doubled a term that is zero by construction.

### 1f. The deploy-legal prior builder is itself broken

100% of degree-2 nodes on every biochem mesh have `cos(edge1, edge2) = -1.0000` exactly: they
are P2 mid-side nodes lying exactly between two corners, and they are **74.5% of the mesh**.
`legal_priors._lsq_gradient` fits a 2x2 normal matrix that is therefore rank-1, and the
`scale * 1e-6` ridge lifts the null direction just enough to be inverted rather than
truncated.  `potential_flow_direction` against GT:

```
vessel      deg-2 (74.5% of nodes)   deg-6 corners (21%)
patient001         +0.66                   +1.00
patient020         +0.65                   +1.00
patient041         +0.65                   +0.99
patient044         +0.64                   +0.99
```

**Near-perfect on P1 corners, ~50 degrees off on three-quarters of the mesh.**  This is the
same collinear-stencil pathology as the `width_d1/d2` bug (`DEPLOY_FLOW_PLAN.md` §1b), in a
second location.  Every "how much does the surrogate add over the prior" number in this
project is measured against a baseline that carries it.

The magnitude is wrong too.  `mass_conserving_umax_nd` uses fixed `U_MAX_BASE_ND = 1.5`,
`R_REF_ND = 0.5` and `U_PRIOR_PEAK_CAP_ND = 2.0` and never reads the vessel's own inlet BC:

```
vessel            relL2   best global scale   relL2 @ best   % nodes at the cap
patient001        0.730         0.717            0.669            0.2
patient020        0.747         0.704            0.681            0.0
patient041        0.783         0.864            0.777            3.2
patient044        0.798         0.892            0.795            3.4
wound_patient001  0.720         0.810            0.700            7.7
```

28% over-scaled, and the cap clips exactly the stenoses and wound packs the clot model cares
about.  But note the residual after rescaling is still 0.67–0.80 — **the magnitude is the
small half; the direction is the large half.**

### 1g. After the repair: the analytic prior matches the surrogate

Rank-aware gradient (B3) plus an inlet-anchored, relatively-capped magnitude (B4).  Direction
first — the same table as above, re-measured:

```
vessel            deg-2 cos   deg-6 cos   before (deg-2)
patient001          1.000       1.000         0.66
patient020          0.999       0.999         0.65
patient041          0.994       0.994         0.65
patient044          0.988       0.988         0.64
wound_patient001    1.000       1.000         --
```

The degree split is gone: every node class now reads the same near-unit cosine.  Prior rel-L2
against COMSOL, whole cohort, against the cached surrogate:

```
                            analytic (repaired)   RGP-DEQ (leaked priors)
median over 52 packs               0.140                  0.141
analytic beats the surrogate on 19 / 52 vessels
```

**A closed-form prior computable from geometry and the inlet BC alone now matches a trained
surrogate that was handed COMSOL's own velocity field as an input.**  It was 0.72–0.80 before
the repair, which is the only reason the surrogate ever looked like it was adding something.

Where the analytic prior is genuinely weak is narrow and identifiable — the stenoses
(`patient041` 0.509, `patient042` 0.500, `patient044` 0.552), `patient012` (0.387) and
`wound_patient003` (0.396).  That, and nothing else, is the region a flow surrogate has to earn.

### 1h. The WLS derivative operator is blind on three quarters of every mesh

The same collinear-stencil fact as §1f, one order higher.  `patient020`, per-node rank of the
5-term WLS normal matrix, and the relative error recovering a field whose derivatives are known
exactly (`f = x^2/2 + xy`):

```
stencil rank   nodes            degrees      quadratic recovery (rel err, median)
     2         14699  (74.6%)   [2]                    7.2e-01
     3             4  ( 0.0%)   [3]                    1.9e-03
     4           326  ( 1.7%)   [4, 5]                 3.7e-04
     5          4679  (23.7%)   [5, 6, 7, 8]           4.2e-16
```

**Full-rank rows are exact to machine precision; three quarters of the mesh carries no usable
derivative at all.**  Every downstream shear, continuity and momentum term reads this operator,
and so does `l_shear_grad`, which is masked to wall vertices -- many of them mid-side.

Filling the deficient rows from well-conditioned neighbours (a mid-side node is the exact
midpoint of its two corners, so the average is 2nd-order exact) takes mid-side recovery from
**0.72 to 4.0e-07 median**, and it works on the packs' existing stored `M_inv` -- no pack
rewrite is needed.

### 1i. The stored WLS operators are stale, and that is what `clamped_width_priors` patches

Rebuilding `V`/`W`/`M_inv` from each pack's **own** `edge_index` and node positions, against
what the pack actually carries:

```
pack          |M_inv| stored   |M_inv| rebuilt      width_d2 stored   width_d2 rebuilt
patient001         1.000e+06          5.6e+03              6.406e+01              64.2
patient012         6.658e+05          2.1e+04              1.046e+05             183.8
patient020         7.001e+05          2.8e+04              4.784e+04              21.8
patient041         7.243e+05          1.7e+04              1.019e+05             273.1
patient044         6.256e+05          2.9e+04              1.773e+05             258.2
```

`patient001`'s `1.000e+06` is exactly `1/epsilon` for the old `M + 1e-6*I` — the signature of a
node whose `M` was **empty**, i.e. assembled from an edge list that did not contain it.  The
rebuild is 20-60x smaller and drops `width_d2` from 1.8e5 to **22-273**, against a training p95
of 73.8.

**This reframes `DEPLOY_FLOW_PLAN.md` §1b.**  The `|width_d2| = 1.8e5` blow-up is not caused by
the collinear P2 stencils, and not by the `1e-6` ridge (§1h and the D5 note): it is caused by
**the stored operator not matching the stored graph**.  `clamped_width_priors` forces the
symptom back into range at every call site; a rebuild removes the cause.  The rebuild is
implemented (`mesh_wls.rebuild_wls_operators_from_graph`) and deliberately **not** wired in:
it moves every flow-derived quantity downstream, and by this plan's own rule that has to be
judged on wall `dsrx` correlation, gate union Jaccard and oracle-F1 before it becomes a default.

### 1j. The surrogate moves shear amplitude and damages shear structure

The quantity `clot_ml_0` reads, measured on wall nodes under repaired analytic priors, at the
`hops=6` stencil `build_features` uses (`scratch/diag_wall_shear_authority.py`):

```
vessel      field                   sr corr  sr scale  dsrx corr  dsrx scale
patient020  analytic prior alone      0.568     0.104      0.612       0.060
patient020  RGP-DEQ on analytic       0.703     0.506      0.228       0.387
patient039  analytic prior alone      0.897     0.464      0.573       0.204
patient039  RGP-DEQ on analytic       0.766     0.642      0.552       0.299
patient041  analytic prior alone      0.681     0.280      0.384       0.156
patient041  RGP-DEQ on analytic       0.804     0.304      0.806       0.193
patient001  analytic prior alone      0.806     0.410      0.962       0.146
patient001  RGP-DEQ on analytic       0.552     1.165      0.826       0.332
```

Three readings, and they are the ones that should drive the retrain:

* **The model has wall-shear authority and uses it.**  `sr scale` moves 0.10 → 0.51 (020) and
  0.41 → 1.17 (001).  This is what disproves the D1 premise above.
* **It uses it badly.**  It *lowers* the correlation on 2 of 4 vessels — `patient020` `dsrx`
  0.612 → 0.228, `patient001` `sr` 0.806 → 0.552.  It adds magnitude without preserving
  structure, which is exactly what a velocity-L2 objective rewards and nothing penalises.
* **Everything is under-scaled**: `sr scale` 0.10–1.17, `dsrx scale` 0.06–0.39.  This is the
  deficit `PRED_DSRX_GAIN = 3.0` exists to patch downstream, visible here at its source.

---

## 2. The bug list

Ordered by blast radius.  Status is updated in place as each lands.

| # | Bug | Where | Status |
|---|---|---|---|
| B1 | Deploy inference never applies a legal prior source | `scripts/precache_rgp_deq.py` | **DONE** — `--prior-source`, default `analytic`, applied before the solve |
| B2 | Stage-A training never applies one either; `assert_train_deploy_prior_parity` is called from nowhere | `finetune_kine_patient_anchors.py`, `train_kinematics_predictor.py` | **DONE** — `_apply_prior_source_to_dataset` at load; finetune sets `SPECIES_PRIOR_SOURCE=analytic` |
| B3 | `_lsq_gradient` is rank-deficient on 74.5% of every biochem mesh | `src/data_gen/lib/legal_priors.py` | **DONE** — eigen-truncation + neighbour fill; deg-2 cos 0.65 -> 0.99+ |
| B4 | Analytic prior magnitude ignores `u_inlet_bc` and is hard-capped at 2.0 ND | `src/data_gen/lib/graph_velocity_priors.py` | **DONE** — `inlet_anchored_umax_nd`, cap now 6x the vessel's own inlet peak |
| B5 | `kinematics_ckpt_latest.pth` is always saved with NaN metrics | `train_kinematics_predictor.py:1066` | **DONE** — periodic/latest saves carry the last validation + `run_id` + `prior_source` |
| B6 | `save_best` needs finite `val_comp`, so a NaN-metric run silently never promotes | `train_kinematics_predictor.py:1162` | **DONE** — the silent `latest -> best` copy is removed; an unpromoted run says so |
| B7 | `resolve_kinematics_checkpoint` cannot see checkpoints in run subdirectories | `src/utils/kinematics_inference.py` | **DONE** — newer run-subdir checkpoints are named; `assert_promotable_checkpoint` added |
| B8 | Holdout is one env-var stem; 013/031/043 are SEALED and were trained on | `finetune_kine_patient_anchors.py:37` | **DONE** — holdout derived from `wall_cohort_splits` (SEALED + DEV) |
| B9 | `training_manifest` is stale in the saved checkpoints (`epochs: 40` at epoch 75) | `train_kinematics_predictor.py` | **DONE** — a resumed manifest is recorded as `resumed_from`, not merged over the live run |
| B10 | Wall data weighting is a no-op (weights nodes where GT `u == 0` and pred is pinned) | `src/utils/kinematics_physics_terms.py` | **DONE** — the data term weights the 3-hop band, not the bare wall vertices |
| B11 | Cached `u0_pred` carries no provenance (no ckpt hash, no prior_source, no timestamp) | `scripts/precache_rgp_deq.py` | **DONE** — `u0_pred_provenance` (prior source, ckpt md5 + role + metrics, timestamp) |
| B12 | `apply_prior_source("stored")` returns the caller's object while other branches clone | `src/data_gen/lib/legal_priors.py` | **DONE** — `_shallow_view`: fresh store, shared tensors, no caller aliasing |
| B13 | The stored `M_inv` / `G` operators do not correspond to the packs' own `edge_index`; `width_d2` is 700x too large as a result | `data/processed/graphs_biochem_anchors/*.pt` | **PARTIAL** — `rebuild_wls_operators_from_graph` added; NOT wired in by default, needs the end-to-end measurement first |
| B14 | The Stage-A **training** copy of each patient carries pre-repair `width_nd` and `wall_normal`; the deploy copy was repaired and they now disagree | `graphs_kinematics_anchors/carreau` vs `graphs_biochem_anchors` | **DONE** — §7.1; 8 of 18 channels differed (five identically zero in training); `sync_geometry_from_deploy_pack` |
| B15 | The stored synthetic corpus contains no vessel from the failing regime: stenosis ratio p95 **1.37** against a deploy tail of 3.3-4.6 | `data/processed/graphs_kinematics/carreau` | **STALE CORPUS, NOT A GENERATOR BUG** — §9.1. The live sampler already spans the deploy tail; regeneration alone closes it |
| B16 | `l_shear_grad` is an unnormalised MSE on d(shear)/dx at weight 50 — **99.99% of the total loss**, leaving 10 of 11 terms inert including the data term at weight 500 | `src/core_physics/physics_kernels.py` | **DONE** — §10; `KINEMATICS_NORMALIZE_SHEAR_GRAD=1` |
| B17 | On an elevated graph `anchor_node_mask` broadcasts a graph-level flag to every node, so 74.5% of the data term would be computed on **interpolated** mid-side labels | `src/data_gen/lib/p2_elevation.py` | **DONE** — §10.2; per-node `is_anchor`, True on corners only |
| B18 | `l_shear` was gated on `hasattr(data, 'G_x')`, but `compute_gt_shear_rate` builds MLS operators from positions+connectivity and never reads `G_x` outside legacy mode | `train_kinematics_predictor.py` | **DONE** — §11.1. Latent: the term is separately inert because `shear_head=False` |
| B19 | `rank_aware_pinv_sym` / `_deficient_rows` ran batched 5x5 `eigh` on CUDA; cuSOLVER asked for **5.76 GiB** on a 15.7k-node graph and OOM'd a 4 GB card | `mesh_wls.py`, `math_operators.py` | **DONE** — §11.2; both offloaded to CPU |
| B20 | `_DEFICIENT_CACHE` keyed on `data_ptr()`, which is reused after free — a stale mask could be served for a different mesh | `src/utils/math_operators.py` | **DONE** — value fingerprint added to the key |
| B21 | The SEALED holdout default lived in the launcher, so invoking `train_kinematics_predictor` directly still trained on 013/031/043 | `src/utils/kinematics_geometry.py` | **DONE** — §11.6; default now derived from `wall_cohort_splits` at the point of use |
| B22 | `node_type_0..3` is identically **zero on both training corpora** and live at deploy — the same staleness as the stenosis tail | `graphs_kinematics/carreau`, `graphs_kinematics_anchors` | **Regeneration fixes synthetic; B14's sync fixes the clinical anchors.** Caught by `preflight_kine_cohort.py` |
| B23 | 7 deploy packs are from an older extractor revision: `patient002` + all six `*_mirror_y` have dead `node_type` and anomalous priors | `graphs_biochem_anchors` | **OPEN** — `patient002` is in the training pool; consider excluding or re-extracting |
| B24 | The synthetic builder wrote `node_type` as a literal `torch.zeros((N, 4))  # Placeholder` — regenerating the corpus would NOT have fixed the dead channel | `src/data_gen/lib/mesh_to_graph.py` | **DONE** — real one-hot; supersedes B22's claim that regeneration was sufficient |
| B25 | Generation silently clobbered a populated cohort: `MeshToGraph.run()` clears every `*.pt` in its output dir and intent was never required | `src/data_gen/pipeline_kinematics.py` | **DONE** — refuses without `--overwrite`/`--append`; `--dry-run` added |

**Evidence of B5/B6 in the current artifacts.**  Both checkpoints trained on 2026-08-27:

```
outputs/kinematics/production_allfix/kinematics_best.pth
outputs/kinematics/clinical_anchor_finetune/kinematics_best.pth
  checkpoint_role = kinematics_ckpt_latest       <- manually copied from `latest`
  rel_l2 = nan   continuity = nan   composite = nan
  run_id = ''    training_manifest.epochs = 40   (file is at epoch 50 / 75)
```

No model selection happened.  And `resolve_kinematics_checkpoint()` returns
`outputs/kinematics/kinematics_best.pth` — the 2026-08-12 ep-208 model, `rel_l2 0.0712` —
so **neither new checkpoint is in the deploy path at all** (B7).

---

## 3. The redesign

The framing that decides every item below: **`clot_ml_0` never reads velocity.**  From
`build_features` it reads `sr`, `dsrx`, `dsry`, `vort`, `div`, `spd`, and the hard gate
`(sr < lss) + (dsrx < sgt)` — all in the wall band.  Velocity rel-L2 is a proxy for a quantity
the consumer never touches, which is why the width fix halved rel-L2 and made the frozen clot
model *worse* (`DEPLOY_FLOW_PLAN.md` §3).

| # | Change | Rationale | Status |
|---|---|---|---|
| D1 | Reparameterize the output so the BC does not gate authority: predict the wall-normal profile in the local wall frame, `u = tau(x)*sdf + c2(x)*sdf^2 + ...`, making no-slip exact at `sdf=0` **and** wall shear a first-class predicted output | §1j: the model moves `sr scale` 0.10 -> 0.51; capacity was never the constraint | **WITHDRAWN** — premise disproved by §1j: `d/dn(sdf*uvp) = uvp` at the wall, so the BC never denied wall-shear authority. Folded into D2. |
| D2 | Supervise what is consumed, where it is consumed: wall-band `sr`, along-wall `dsrx`, and the gate indicator, on the **same** MLS operator and `hops=6` stencil `features.py` uses | §1e: the band is 2–21% of the gradient and 2–3% on the worst vessels; `l_shear_grad` and `l_wss` are masked to wall vertices only | **DONE** — `wall_band_shear_losses`; GT-spread-normalised `sr` + `dsrx` on the band, `KINEMATICS_WALL_SHEAR_WEIGHT` |
| D3 | Make the data term relative rather than absolute (per-vessel or per-band normalisation) | §1e: the fast core owns the gradient | **DONE** — `relative=` on `boundary_weighted_mse` |
| D4 | Fix the prior builder before training against it (B3 + B4) | §1f: the legal baseline every comparison uses is broken on 74.5% of nodes | **DONE** — B3 + B4; §1g records the effect |
| D5 | Make the WLS operator rank-aware once, globally, and **fill the rows it cannot resolve** from well-conditioned neighbours | 74.6% of biochem nodes are rank-2 of 5; quadratic recovery 0.72 there vs 4.2e-16 on full-rank rows | **DONE** — `rank_aware_pinv_sym` at all 3 sites + `_fill_rank_deficient_rows` in `wls_derivatives`; mid-side recovery 0.72 -> 4.0e-07. See §1h: the *fill* is the fix, the truncation is hygiene |
| D6 | Prior parity as a hard gate: call `assert_train_deploy_prior_parity` in trainer and precache, stamp `prior_source` + ckpt hash on the pack, refuse a clot cache whose stamp disagrees | B1/B2/B11 | **DONE** — parity assert in precache, `prior_source` on checkpoints, provenance on packs |
| D7 | Promotion hygiene: reject `role != kinematics_best`, NaN metrics, empty `run_id`, and a `best` predating the run; derive holdout from `wall_cohort_splits` and exclude SEALED | B5/B6/B8 | **DONE** — B5/B6/B7/B8; `assert_promotable_checkpoint` |
| D8 | Only then rebuild the pred cache and retrain / augment `clot_ml_0` | D1+D2 change what `u0_pred` means | TODO — the retrain and cache rebuild; unblocked by everything above |

**How to judge any of it.**  Not by velocity rel-L2.  In order: wall `dsrx` correlation, gate
union Jaccard, oracle-F1, then the domain deploy score — as `DEPLOY_FLOW_PLAN.md` §3 already
says.  Plus one new mandatory metric: the **stored → analytic rel-L2 drop**, which is the
anti-leak tripwire.  Today it reads 0.124 → 0.385 and would fail any sane threshold.

**Order of work.**  B1-B12 and D2-D7 are **landed** (2026-08-27), with 14 regression tests in
`src/tests/test_rgp_deq_repair.py`.  What is left, in order:

1. **Re-run §1c end to end.**  Its `analytic` row (rel-L2 0.385) was measured against the
   broken prior builder; §1g fixed the prior but the DEQ has not been re-audited on top of it.
   `python scripts/diag_rgp_deq_flow_audit.py --all --sources stored,analytic`
2. **Decide B13 on evidence.**  `rebuild_wls_operators_from_graph` is implemented and unwired.
   Judge it on wall `dsrx` correlation, gate union Jaccard and oracle-F1 -- not on the
   operator norm, and not on velocity rel-L2.
3. **Retrain Stage-A** with `SPECIES_PRIOR_SOURCE=analytic` and
   `KINEMATICS_WALL_SHEAR_WEIGHT` set.  Both default to off, so nothing changes until this is
   deliberate.  The promotion path will now refuse to mint a `best` it did not select.
4. **Rebuild the pred cache** (`precache_rgp_deq.py --force`) and only then revisit
   `clot_ml_0` (D8).  The stale-cache check means a plain re-run will now recompute every pack
   whose provenance does not match, rather than skipping them.

---

## 4. Caveats on the measurements

* The authority (§1d) and band-share (§1e) tables are 5 vessels, chosen to span FIT/DEV and
  aneurysm/stenosis.  Treat the magnitudes as indicative and the sign as settled.
* The leak (§1a), the prior-degradation ratio (§1b) and the collinearity (§1f) are the full
  43 / 52 / 52.
* §1c is 8 vessels and was measured against the BROKEN prior builder; §1g supersedes its
  `analytic` row for the prior itself, but the end-to-end DEQ row still needs re-running.

---

## 5. Corrections made while implementing

*(Six, as of 2026-08-28.  Kept in full rather than edited out: each was load-bearing when written.)*

Three claims in the first draft of this document did not survive being tested.  They are left
here rather than quietly edited out, because each was load-bearing when it was written.

1. **"The BC denies the model authority at the wall" — wrong.**  `d/dn(uv_prior + sdf*uvp)` at
   `sdf = 0` is `d(uv_prior)/dn + uvp`, so wall shear has unit sensitivity to the model's own
   output.  §1j shows it exercising that sensitivity (`sr scale` 0.10 → 0.51 on `patient020`).
   D1 — the profile reparameterisation — was withdrawn on this basis.  The model's problem is
   that it moves shear *amplitude* while degrading shear *structure*, which is an objective
   failure, and D2 is the fix.

2. **The wall-band statistics were measured with a broken band.**  `acc[row] = band[col]` is
   last-write-wins on duplicate indices, so it returned roughly half the true band.  Corrected
   in §1d/§1e; `wall_band_mask` now dilates by `index_add` and a test pins it.

3. **"The `1e-6` ridge is what breaks the WLS operator" — not on these meshes.**  Truncated and
   ridged operators produce identical per-node ranks at every scale tested, and agree to
   within 3% in norm on `patient020`.  The rank deficiency itself is the defect and the
   neighbour fill is the repair; the truncation is kept as hygiene, not as a measured win.

4. **"The synthetic generator cannot produce severe stenosis" — wrong** (§9.1).  The stored
   corpus cannot, and the live sampler already spans the deployment tail (`frac >= 2.0` of
   0.16-0.20 against deployment's 0.140).  The corpus is stale; the generator is fine.  T3's
   request for new generation ranges was withdrawn.

5. **"Select on wall `dsrx` correlation" — wrong, and it was the plan's own headline rule**
   (§10.3).  Measured against the locked clot ensemble, gate union Jaccard tracks oracle-F1 at
   +0.918 while `dsrx` correlation reads **-0.073 within a single flow arm**.  The ordering in
   T7 — and in `DEPLOY_FLOW_PLAN.md` §3 — was backwards.

6. **"The analytic prior ties the surrogate" — wrong** (§10.4).  True on velocity rel-L2,
   cosine and AUC-of-speed; false on the only metric that matters.  On the real clot task the
   surrogate scores 0.675 against the prior's 0.370.

The pattern in all six: a mechanism or a metric that was plausible from reading the code,
asserted before it was measured.  Three of them (4, 5, 6) were only caught because the proxy
was eventually checked against the thing it stood for — which is the lesson worth keeping.
The measurements that did survive — the leak (43/43), the prior-degradation ratio (45/52), the
collinearity (100% of degree-2 nodes), the rank split (0.72 vs 4.2e-16), the loss-term
imbalance (99.99%), and the gate-Jaccard correlation (+0.918) — are the ones this plan rests
on.

---

## 6. The Stage-A retrain specification

The goal is not a better velocity field.  It is that **`clot_ml_0` scores the same under
`flow="pred"` as under `flow="gt"`** — today the gap is `-0.366 wall / -0.478 off-wall`
(`DEPLOY_FLOW_PLAN.md` §2).  Everything below is chosen against that.

### 6.0 What the audit actually established

Full 35-vessel cohort, promoted checkpoint, after the §1f/§1g prior repair:

```
                          relL2_u     cos      AUC      dsrx corr (mean/med)   dsrx scale   gate J
analytic PRIOR ALONE        0.188   +0.998    0.761        +0.644 / +0.631        0.205      0.213
RGP-DEQ on analytic         0.197   +0.993    0.766        +0.316 / +0.256        0.494      0.222
RGP-DEQ on stored (leak)    0.113   +0.994    0.787              -                  -          -
GT                             -        -     0.789            1.000              1.000      1.000
```

**The surrogate trades shear structure for shear amplitude.**  It halves the wall `dsrx`
correlation (0.644 → 0.316) and roughly doubles the amplitude (0.205 → 0.494).  That is the
exact trade a velocity-L2 objective rewards, and it is the wrong one: amplitude is already
patched downstream by a single fitted scalar (`PRED_DSRX_GAIN = 3.0`), and correlation is not
patchable by any scalar.  This is the finding the retrain has to answer.

### 6.1 Where the training distribution actually differs from deployment

Sampled from each corpus (medians, p5-p95):

```
                              SYNTHETIC (train)      BIOCHEM (deploy)
nodes N                    2983   [1957-4323]     14830  [9443-18570]
degree-2 (P2 mid-side)      0.0%  [0.0-0.06]       74.5%  [74.2-74.6]
degree>=5 (P1 corner)      89.6%  [85.4-92.8]      23.4%  [22.7-23.7]
stenosis ratio              1.13  [1.06-1.37]       1.17  [1.05-4.33]
u_ref                      0.112  [0.076-0.172]    0.095  [0.076-0.154]
d_bar                     0.0129  [0.0084-0.0197]  0.0151 [0.0092-0.0188]
```

Two gaps, and only two:

* **Topology.**  Stage-A has literally **never seen a P2 mesh** — not "mostly P1", zero
  degree-2 nodes — and deploys on meshes that are 74.5% degree-2 and 5x larger.
* **The shape tail.**  Stenosis ratio p95 is **1.37** in training against **4.33** in deploy,
  and severity tracks failure: ratio ≤ 1.6 → analytic rel-L2 0.10-0.20; ratio 3.3-4.6 →
  0.30-0.50 (`patient012/041/042/044`).  The failing regime is simply not in the corpus.

**The BC ranges are already fine.**  `u_ref` and `d_bar` overlap almost exactly.  Do not spend
generation effort widening them.

### 6.2 The plan

| | Change | Why |
|---|---|---|
| **T1** | **One copy of the geometry.**  Either apply `pack_repair` to `graphs_kinematics_anchors/carreau`, or have Stage-A read the `x` block straight from the biochem packs.  Prefer the latter: a single source of truth retires the whole bug class. | B14 — training and deploy disagree on `width_nd` (0.13-0.17) and `wall_normal` (0.17-0.24) for the *same mesh*.  `wall_normal` drives `mod_adv`/`mod_rheo`/`mod_curve`, so the model trains with one attention geometry and deploys with another.  Note `pack_repair`'s own warning: write rows or deltas, never a wholesale rebuild. |
| **T2** | **Close the topology gap — try the cheap direction first.**  *Route B:* run the surrogate on the **corner-only P1 subgraph** of the deploy mesh (corner-corner adjacency through the mid-sides is exactly the original P1 triangulation), then interpolate the *prediction* onto the mid-side nodes.  *Route A:* emit P2 synthetic meshes and re-solve. | Route B lands deployment *inside* the training distribution rather than making training match an odd discretisation: the corner subgraph of a deploy mesh is `14830 x 23.4% ~= 3470` nodes against a synthetic corpus median of **2983 [1957-4323]**. It needs **no new CFD and no retrain to evaluate**, and mid-side interpolation of our own output is legitimate where interpolating *labels* would not be — a P2 CFD solution's mid-side value is not the corner average, so Route A cannot fake its labels. Measure Route B before committing to Route A. |
| **T3** | **Generate the missing shape tail**: stenosis ratio 1.0-5.0, matching the aneurysm/stenosis classes the cohort contains. | B15.  The three worst vessels are ratio 4.2-4.6 and the corpus stops at 1.37. |
| **T4** | **Objective**: enable `KINEMATICS_WALL_SHEAR_WEIGHT` (§D2) and the relative data term (§D3). | §6.0: the model's measured failure is structure-for-amplitude, which is exactly what these two terms penalise and what velocity-L2 rewards. |
| **T5** | **Priors**: `SPECIES_PRIOR_SOURCE=analytic`, enforced by `assert_train_deploy_prior_parity`. | B1/B2.  Already wired. |
| **T6** | **Make the prior the floor, not zero.**  The DEQ is already residual (`u = uv_prior + sdf*uvp`); add shrinkage on `uvp` so that where the model has no signal it returns the prior instead of noise. | This is the robustness property the ask actually needs: it makes deploy performance **>= the analytic prior's** by construction.  Today the surrogate is free to be worse than its own input, and on 45 of 52 packs it is (§1b). |
| **T7** | **Selection**: wall `dsrx` correlation, then gate union Jaccard, then oracle-F1.  Never velocity rel-L2. | The width fix halved rel-L2 and made the frozen clot model *worse* (`DEPLOY_FLOW_PLAN.md` §3).  §6.0 is the same lesson again. |
| **T8** | **Acceptance test is downstream, not Stage-A.**  Accept only if the `clot_ml_0` pred-vs-GT deploy delta improves against `-0.366 wall / -0.478 off-wall`, fitted on FIT and read on DEV, with the GT arm re-scored for regression. | The stated goal is eliminating the drop.  A Stage-A metric that improves without moving this has not done the job. |

### 6.3 Order

1. **T2 Route B** — no training, no new data, and it either closes most of the topology gap or
   rules it out.  Cheapest decisive experiment on the board.
2. **T1** — a correctness fix that must land before any retrain, or the retrain re-learns the
   wrong geometry.
3. **Refit `PRED_DSRX_GAIN` per flow source** and re-score the gate.  Both sources are badly
   miscalibrated (scale 0.205 and 0.494 against 1.0) so the current gate Jaccard tie
   (0.222 vs 0.213) is measured at the wrong operating point for both.
4. **T3** generation, then the retrain with **T4/T5/T6**, selected by **T7**, accepted by **T8**.

---

## 7. Results: T1, T2 Route B, and the gain refit  (2026-08-27)

### 7.1 T1 — the training/deploy geometry mismatch is bigger than B14 first read

Per-channel rel-L2 between `graphs_kinematics_anchors/carreau` (what Stage-A trains on) and
`graphs_biochem_anchors` (what it deploys on), **all 43 shared patients**.  `edge_index`,
`mask_wall`, node positions and `sdf` are bit-identical, so these are the same meshes:

```
 [4,5]  wall_normal      0.178 / 0.199        [11-13] u/v/mu_prior  0.023 / 0.028 / 0.027
 [6-9]  node_type_0..3   1.000 (all four)     [14]    wss_prior     1.000
 [15]   width_nd         0.149                [16,17] width_d1/d2   8.68 / 9.03
```

**Eight of eighteen input channels differ, and five are identically zero in training.**  A
rel-L2 of exactly 1.000 on `node_type_0..3` and `wss_prior` means Stage-A has never seen those
channels as anything but zero, and the encoder consumes all five.  `wall_normal` is the worst
of them in effect: `mod_adv`/`mod_rheo`/`mod_curve`, the GAT's three attention biases, are
built entirely from it, so the model trains with one attention geometry and deploys with
another.  Only the biochem copy ever received `repair_pack_wall_normals` and the width fix.

**Fixed** — `kinematics_paths.sync_geometry_from_deploy_pack`, called from
`load_patient_kine_anchor_graphs`.  It copies the nine *mesh* channels
(`GEOMETRY_SYNC_CHANNELS`) and nothing else: the prior block is excluded because
`apply_prior_source` rewrites it anyway, and `pack_repair`'s warning against wholesale
rebuilds is respected.  Verified: all nine go to rel-L2 0.0000 and every other channel stays
bit-identical.  Nothing is written to disk.

### 7.2 T2 Route B — the corner graph fixes amplitude, not structure

`p1_corner_graph.build_corner_graph` collapses a P2 pack onto its corner subgraph
(corner-to-corner adjacency through the mid-sides = the original P1 triangulation).  Mid-side
detection is **anti-parallelism, not an exact midpoint**: COMSOL places boundary mid-sides on
the true curved geometry, and a 1e-5 midpoint tolerance misclassifies every one of them
(off-wall deviation 0.0e+00, wall deviation median 1.9e-03, max 1.4e-02).

Deployment lands inside the training distribution:

```
                    P2 deploy      ->  P1 corner graph      training corpus
nodes                14830              2447 - 5009         2983 [1957-4323]
degree-2             74.5%              0.00%               0.0%
median degree        2                  6.0                 6 (89.6% are >=5)
|width_d2| max       1.0e+05            17 - 232            p95 73.8
```

`width_d2` falls from 1e5 to 17-232 **with no clamp** — independent confirmation of §1i, since
nothing about the geometry changed, only the connectivity the operator is built on.

Four arms, 35 vessels, all evaluated on the full P2 mesh so the numbers are comparable:

```
              relL2_u   cos     AUC   dsrx corr (mean/med)  dsrx scale  gateJ
analytic P2    0.188  +0.998  0.761     +0.644 / +0.631       0.205     0.213
RGP-DEQ P2     0.197  +0.993  0.766     +0.316 / +0.256       0.494     0.222
analytic P1    0.183  +0.998  0.779     +0.453 / +0.614       0.353     0.212
RGP-DEQ P1     0.198  +0.997  0.750     +0.302 / +0.358       0.926     0.237
```

**Partial win, and worth being precise about which part.**

* **Amplitude is fixed by topology alone.**  `dsrx scale` 0.494 -> **0.926** (1.0 = GT).  The
  amplitude deficit that `PRED_DSRX_GAIN = 3.0` exists to patch was substantially a P1/P2
  artifact, not a model deficit.
* **Gate agreement improves** to 0.237, the best of the four arms.
* **Correlation does not move** (mean 0.316 -> 0.302, median 0.256 -> 0.358), and AUC drops
  0.766 -> 0.750.  So the `dsrx` *structure* deficit is **not** a topology artifact.  It is the
  objective, which is what T4/T6 exist for, and Route B does not substitute for them.
* `analytic P1` posts the best AUC of any arm (**0.779** against GT's 0.789).

### 7.3 The gain refit — and why it should not be shipped

Fitted on FIT (n=25) to maximise mean gate Jaccard, read on DEV (n=5), the protocol
`PRED_DSRX_GAIN = 3.00` itself used:

```
arm            best gain (FIT)  J FIT@3.0  J FIT@fit  J DEV@3.0  J DEV@fit
analytic P2              28.00      0.168      0.451      0.348      0.325
RGP-DEQ P2               13.00      0.216      0.285      0.207      0.258
analytic P1              30.00      0.172      0.258      0.372      0.348
RGP-DEQ P1                8.00      0.250      0.261      0.162      0.197
```

The shipped 3.00 is far too small for every arm — but **do not ship a refit**.  Two reasons,
both measured:

1. **It does not transfer for the arms that gain most.**  `analytic P2` goes 0.168 -> 0.451 on
   FIT and 0.348 -> **0.325** on DEV.  Same for `analytic P1`.  Only the DEQ arms transfer.
2. **The optimum is bought by over-firing.**  GT fires on 19.6% of wall nodes; at gain 28 the
   analytic gate fires on 42.6%, and `19.6 / 42.6 = 0.46` against a measured J of 0.451.

   **The analytic gate is essentially a superset of the GT gate** — recall ~100%, precision
   ~46% — so its Jaccard is capped by its fire rate, not by its ranking.  The surrogate's gate
   is not a superset: its ceiling at fire 58% is `19.6/58 = 0.34` and it reaches only 0.261, so
   it misses GT nodes *while* over-firing.

**The conclusion is that the gain is the wrong knob.**  One global scalar cannot fix a
one-sided hard threshold whose problem is precision, not scale.  A per-vessel threshold, or
making the gate soft, is worth more than any refit of this constant — and the fact that the
analytic prior's gate already contains the GT gate says its *ranking* is sound and only its
cut is loose.  That is the cheapest remaining lever in the whole document.

---

## 8. Alignment, not patching  (2026-08-27, direction set by the project owner)

Three constraints reframe §6 and §7:

1. **The biochem anchor pipeline is expensive and does not change.**  It therefore *defines*
   the deployment domain, and training has to meet it.
2. **Graph topology is ours to choose** on the training side.
3. **Prefer alignment over patching.**

This retires T2 Route B as a *plan*.  Collapsing deployment onto its corner subgraph was the
"make deploy look like training" direction; it only fixed shear amplitude (§7.2) and it edits
the deploy path to suit a training artefact.  `p1_corner_graph` stays as a diagnostic and as
the machinery for reasoning about P1/P2 correspondence — it is not the deploy path.

### A1. Elevate the synthetic corpus to the deployment mesh order — DONE

`p2_elevation.elevate_to_p2` inserts a mid-side node on every edge and rewires to COMSOL's
convention.  That convention was measured, not assumed (`patient020`):

```
corners 5009   midsides 14699   midside fraction 0.7458   midsides/corners 2.935
directed edges 58796:  corner-midside 100.00%   corner-corner 0.00%   midside-midside 0.00%
```

**A mid-side node connects only to its two parent corners; there are no corner-corner edges at
all.**  Elevation is therefore "replace every P1 edge with two half-edges", not "add nodes and
keep the old edges" — getting that wrong reproduces the wrong degree distribution.

Result, median over 12 synthetic graphs:

```
              N     midside%  mid/corner  corner-midside edge%  corner deg  |d1|max  |d2|max
P1 (before)  2983     0.00       0.000            0.00             6.0       1.13     15.66
elevated    14738    74.45       2.914          100.00             6.0       1.59     37.04
deploy      14830    74.58       2.935          100.00             6.0      11.05   2099.00
```

**Labels are interpolated, and that is defensible here because it was measured.**  A true P2
mid-side value against the mean of its two corners, on the deploy packs:

```
channel   mean rel err   p95
u             0.2-1.0%   0.7-6.3%
v             0.3-2.2%   1.1-5.3%
mu_eff        1.3-3.7%   6.3-10.8%
```

1-2% against a model whose own error is ~15-20%, so **no new CFD is needed**.  This is a claim
about these fields on these meshes; re-measure if mesh density or physics changes materially.

Wired into `load_dataset` behind `KINEMATICS_ELEVATE_P2`, off by default.  Native-P2 clinical
anchors are detected and skipped.  Corner rows and labels stay bit-identical; only `width_d1`
and `width_d2` are re-derived, because they are *derivatives* and a value computed on P1
connectivity is meaningless once the graph is P2.

### A2. Absolute coordinates are a memorisation shortcut — DONE (behind a flag)

**The question was right.**  `_apply_fourier_encoding` puts **absolute** `x, y` through 16
sin/cos frequencies — the NeRF construction, whose purpose is to let a network memorise what
happens at a location — and every synthetic vessel lives in the same box (`x` in [0, 5.54],
`y` in [-0.63, 0.56], inlet at `x = 0`), so absolute `x` is a near-perfect proxy for "fraction
along the vessel".

Translating a vessel changes nothing physical.  Measured on the shipped checkpoint:

```
                        10% of span      full span
patient020  absolute       0.390           0.552
patient041  absolute       0.265           0.286
```

The model's *total* error against COMSOL is ~0.20.  **A 10% translation moves the answer by
more than the entire error budget.**

Fixing the encoder alone was not enough, and the measurement caught it: centring the Fourier
block left `patient041` at **0.715** under a full-span shift, *worse* than the 0.286 it
started at.  The reason is that the **SIREN decoder is itself a coordinate network** and was
still receiving the absolute frame — a stronger memorisation path than the encoder's.  With
both canonicalised through `_canonical_coords`:

```
                        10% of span      full span
patient020  centered      0.00088          0.0019
patient041  centered      0.0568           0.0741
```

The analytic prior is invariant to 1e-6 under the same shifts, so `patient041`'s residual is
the *model*, not its input — expected, because these weights were **trained** on absolute
coordinates and centring is off-distribution for them.  What the flag does is close the
mechanism; only a retrain makes the model itself invariant.

`KINEMATICS_COORD_MODE=centered`, default `absolute`.  Centring gives exact translation
invariance.  It does **not** give rotation invariance (`wall_normal` is covariant), and it does
not remove streamwise position as a feature — position along a vessel is genuine physics for a
developing flow.  What it removes is the shared absolute frame that makes that position
*memorisable* across a corpus of similarly-placed vessels.

**The deeper version, not implemented:** replace `(x, y)` with a curvilinear pair — the
inlet-to-outlet potential `phi` from `potential_flow_direction` (already computed for the
analytic prior, legal at deploy, translation *and* rotation invariant) paired with a signed
wall-normal coordinate.  That is a genuinely physical parameterisation rather than a
canonicalised Cartesian one.  It needs `phi` stored as a channel and a sign convention for the
transverse coordinate, and it should be decided by measurement against `centered`.

### A3. The retrain configuration

```bash
SPECIES_PRIOR_SOURCE=analytic          # A5/B2 -- deploy-legal priors, parity-asserted
KINEMATICS_ELEVATE_P2=1                # A1   -- training mesh order matches deployment
KINEMATICS_COORD_MODE=centered         # A2   -- no absolute frame to memorise
KINEMATICS_WALL_SHEAR_WEIGHT=<w>       # D2   -- supervise sr and dsrx in the wall band
```

plus T1's geometry sync, which is on by default because a mismatched mesh is never wanted.

Still open, in order: **T3** (the stenosis tail — synthetic p95 1.37 against a deploy tail of
4.6, and the worst vessels are all in that tail), **T6** (shrinkage on `uvp` so the analytic
prior is the performance floor), **T7/T8** (select on `dsrx` correlation and gate Jaccard;
accept only on the `clot_ml_0` pred-vs-GT delta).

---

## 9. Everything ready before the relaunch  (2026-08-27)

### 9.1 T3 — CORRECTION: the generator is fine; the corpus is stale

§6.1 said the corpus "contains no vessel from the regime that fails" and §6.2 T3 asked for new
generation ranges.  **The first half is true and the second is wrong.**  Exercising
`_sample_params` + `compute_geometry_from_params` directly, 300 draws per level, measuring the
true geometric width ratio from the wall polylines:

```
        ratio median   p95    max    frac>=2.0  frac>=3.0
L0          1.07      4.99   5.00      0.163      0.130
L1          1.08      4.97   5.00      0.197      0.115
L2          1.13      5.00   5.42      0.198      0.142
DEPLOY      1.27      4.32     -       0.140      0.116
```

**The live generator already matches the deployment stenosis tail, slightly exceeding it.**  The
stored corpus (`max 1.85`, `frac>=2.0` exactly 0.000 over all 370 graphs) predates the current
configuration.  `VesselConfig` permits 80% diameter occlusion — ratio 5.0 — and the sampler
reaches it.

**No generator change is required.**  Regenerating the corpus with today's parameters closes
B15 by itself, and that regeneration is happening anyway.  Do not widen the ranges: the L2 tail
already runs slightly *past* deployment, and pushing further would trade a real gap for an
imaginary one.

*(This is the third mechanism in this document that was plausible from reading the code and
did not survive measurement.  See §5.)*

### 9.2 T6 — the analytic prior as a performance floor — DONE

`kinematics_physics_terms.prior_floor_loss`, weight `KINEMATICS_PRIOR_FLOOR_WEIGHT`:

```
L = mean( relu( |pred - y|^2 - |prior - y|^2 ) ) / mean(y^2)
```

A one-sided hinge, **not** shrinkage on `uvp`.  Shrinkage fights the model everywhere including
where it is right; the hinge is exactly zero wherever the model beats the prior and grows only
where it does not.  The DEQ is already residual (`u = uv_prior + sdf * uvp`) but `uvp` is
unconstrained, and measured, the model is worse than its own input on **45 of 52 packs**.  This
makes the closed-form prior a floor rather than a starting point — the robustness property the
deployable arm needs, given the prior now reaches rel-L2 0.188 and `dsrx` corr 0.644 on its own.

### 9.3 T7 — selection on what the clot stack consumes — DONE

`src/utils/kinematics_selection.py`, computed with the **same** operator and stencils
`build_features` uses (`hops=6` on predicted flow, `hops=3` on GT) — a selection metric on a
different operator is measuring a different quantity.

Ordered: **wall `dsrx` correlation, then gate union Jaccard, then rel-L2 as a tie-break only.**
The two are complementary, which is the point — verified on `patient020`:

```
field            dsrx_corr   gate_jaccard
exact                0.996          0.948
0.4x under-scaled    0.996          0.114
```

Correlation is scale-blind, so it cannot see an amplitude failure; the gate term can.  Neither
alone is sufficient.

Wired into validation (logged as `SELECT dsrx_corr=... gate_J=...`, recorded in
`kinematics_validation.jsonl`) and into the promotion gates via `KINEMATICS_MIN_DSRX_CORR` /
`KINEMATICS_MIN_GATE_JACCARD`.  **Unset means historical behaviour; a threshold that is set but
cannot be evaluated FAILS**, so a metric that did not compute can never look like one that
passed.

### 9.4 T8 — the acceptance test — DONE

`scripts/eval_deploy_flow_acceptance.py`.  Refuses a checkpoint that
`assert_promotable_checkpoint` rejects, evaluates on FIT and DEV separately, and prints the
baselines a retrain has to beat:

```
                       dsrx corr   gate J
analytic prior alone      0.644      0.213    <- the bar.  No network, no checkpoint.
current surrogate         0.316      0.222
```

**A retrain that does not beat the analytic prior on `dsrx` correlation has not earned its place
in the stack.**

### 9.5 The launch configuration

```bash
SPECIES_PRIOR_SOURCE=analytic          # B1/B2  deploy-legal priors, parity-asserted
KINEMATICS_ELEVATE_P2=1                # A1     training mesh order == deployment
KINEMATICS_COORD_MODE=centered         # A2     no absolute frame to memorise
KINEMATICS_WALL_SHEAR_WEIGHT=<w>       # D2     supervise sr and dsrx in the wall band
KINEMATICS_PRIOR_FLOOR_WEIGHT=<w>      # T6     the prior becomes the floor
KINEMATICS_MIN_DSRX_CORR=0.65          # T7     must beat the closed-form prior (0.644)
KINEMATICS_MIN_GATE_JACCARD=0.25       # T7     must beat both current arms (0.213 / 0.222)
KINEMATICS_NORMALIZE_SHEAR_GRAD=1      # B16    WITHOUT THIS every other term is inert
```

T1's geometry sync is on by default — a mismatched mesh is never wanted.  Everything else
defaults off, so an unset environment reproduces historical runs exactly.

**Regenerate the synthetic corpus first** (§9.1); the elevation and every distribution match
above assume a corpus built with the current generator.  The two loss weights are the only
unfitted numbers in the list and want a short sweep, judged on §9.3's metrics, not on rel-L2.

---

## 10. Pre-launch sweep: two more bugs, and the selection metric was wrong

### 10.1 B16 — one loss term owned the entire objective

Measured on an elevated P2 graph with a **spatially smooth** 15% velocity error (i.e. a
realistically-trained model, not random init), each term times its configured weight:

```
                                     NORMALIZE=0        NORMALIZE=1
l_shear_grad (w 50)                    99.99%              6.23%
l_cont       (w 50)                     0.01%             73.18%
l_wss        (w 10)                     0.00%             11.67%
l_data_kine  (w 500)                    0.00%              5.39%
l_io         (w 5)                      0.00%              2.68%
terms above 1%                          1 of 11            5 of 11
```

`wall_shear_gradient_loss` is an **absolute** MSE on `d(shear rate)/dx` -- units `(1/s)/m`, so
raw values ~1e4 where every other term is O(1).  At weight 50 it is **99.99% of the loss**, and
the supervised data term at weight 500 contributes **0.00%**.

**This is the mechanism behind the central finding of this document.**  §1b measured that the
surrogate is farther from COMSOL than its own input prior on 45 of 52 packs.  It is, because it
was never meaningfully trained to match velocity: the data term never reached the optimiser.
Every configured weight in the recipe was decorative.

Fixed by `KINEMATICS_NORMALIZE_SHEAR_GRAD=1` (normalise by the GT's own spread on the
supervised nodes).  Confirmed sound in both directions: the term is exactly 0 for a perfect
prediction, before and after.

### 10.2 Elevated-graph supervision — mid-side nodes get no fabricated labels

`anchor_node_mask` broadcasts a graph-level `is_anchor=[1]` to every node, so an elevated graph
would have had its data term computed on **interpolated** mid-side values at 74.5% of the mesh
-- teaching the model that a mid-side value *is* the mean of its corners.  `elevate_to_p2` now
emits a per-node `is_anchor` that is True on corners and False on mid-sides.  Mid-side nodes
still receive continuity, momentum, BC and the wall-band shear terms; they just do not receive
a fabricated data label.

### 10.3 The selection metric was the wrong one — CORRECTED

T7 and `DEPLOY_FLOW_PLAN.md` §3 both say to judge by "wall `dsrx` correlation, gate union
Jaccard, and oracle-F1, **in that order**".  That order is backwards.  Measured against the
locked clot ensemble's own per-node oracle-F1 (`clot_gnn_v6`, 12 vessels x 2 flow arms,
`scratch/diag_selection_vs_clot.py`):

```
                                    pearson   spearman
gate_jaccard vs oracle-F1            +0.918     +0.904
dsrx_corr    vs oracle-F1            +0.431     +0.555
gate_jaccard vs F1 drop vs GT        +0.905     +0.889
dsrx_corr    vs F1 drop vs GT        +0.304     +0.392

within the ANALYTIC arm alone (removes the between-arm effect):
gate_jaccard vs oracle-F1            +0.765     +0.797
dsrx_corr    vs oracle-F1            -0.073     -0.126     <- no relationship at all
```

**Gate union Jaccard is what to select on.**  `dsrx` correlation is a useful diagnostic -- the
gate is built from it, so it explains *why* a gate fails -- but within a single flow arm it
carries no information about the downstream outcome.  `selection_score` now weights gate
Jaccard 2.0, `dsrx` correlation 0.3, rel-L2 0.05.

### 10.4 And the surrogate DOES earn its place

The same run settles §1g's open question.  Mean oracle-F1 over 12 vessels:

```
GT flow                    0.882
RGP-DEQ (leak-assisted)    0.675      drop -0.207
analytic prior alone       0.370      drop -0.512
```

**The surrogate is worth +0.305 oracle-F1 over the closed-form prior.**  §1g suggested the
opposite -- that a repaired analytic prior "ties the surrogate" -- and that was wrong because
it was read off velocity rel-L2 (0.188 vs 0.197), cosine (0.998 vs 0.993) and AUC-of-speed
(0.761 vs 0.766), **none of which predict the clot outcome**.  Directly measured, `dsrx_corr`
vs the AUC-of-speed gap reads +0.018: that proxy has no predictive value either, which also
undermines the interpretation guidance printed by `diag_rgp_deq_flow_audit.py`.

The 0.675 is leak-assisted (the cached `u0_pred` was built with stored priors), so the true
deployable value sits somewhere between 0.370 and 0.675.  **Closing that interval is exactly
what the retrain is for**, and 0.675 is the number to beat on legal priors.

### 10.5 Launch configuration, final

```bash
SPECIES_PRIOR_SOURCE=analytic
KINEMATICS_ELEVATE_P2=1
KINEMATICS_COORD_MODE=centered
KINEMATICS_NORMALIZE_SHEAR_GRAD=1      # WITHOUT THIS every other term is inert (§10.1)
KINEMATICS_WALL_SHEAR_WEIGHT=<w>
KINEMATICS_PRIOR_FLOOR_WEIGHT=<w>
KINEMATICS_MIN_GATE_JACCARD=0.25       # the PRIMARY gate (§10.3)
KINEMATICS_MIN_DSRX_CORR=0.65          # secondary; diagnostic value
```

Regenerate the synthetic corpus first (§9.1).  With B16 fixed the existing weights
(`data 500 / cont 50 / wss 10 / shear_grad 50`) have never actually been exercised against each
other, so treat every one of them as unfitted and sweep -- judged on gate Jaccard.

---

## 11. Final sweep: efficiency, four more bugs, and closing the training/objective gap

### 11.1 B18 — a loss term gated on the wrong thing

`l_shear` required `hasattr(data, 'G_x')`.  But `compute_gt_shear_rate` routes through
`graph_gradient_operators`, which defaults to **MLS** mode and builds its operators from
positions and connectivity — it only touches `data.G_x` under `BIOCHEM_GRAD_OPERATOR=legacy`.
The guard therefore disabled the term on any graph without the packs' `[N, N]` operators, which
is every P2-elevated graph.  Now gated on what is actually required.

**Latent, not active**: the term is separately inert because the architecture ships with
`shear_head=False` and `out_channels=5`, so `pred.shape[1] > PredChannels.SHEAR_RATE` is
`5 > 5` — false.  Worth knowing: `DEPLOY_FLOW_PLAN.md`'s "the Stage-A shear head is
anti-correlated at the wall (corr -0.11)" describes an architecture that is no longer built,
and `precache_rgp_deq` never writes `sr0_pred` for the same reason.  **No shear head is needed**:
the gate is built from `sr` and `dsrx` computed by MLS from `u, v`, which is what D2 supervises.

### 11.2 B19 — a 5.76 GiB allocation for 5x5 matrices

`rank_aware_pinv_sym` and `_deficient_rows` both call batched `eigh` / `eigvalsh` on `[N, 5, 5]`.
Arithmetically trivial; cuSOLVER's batched path is not — on a 15,754-node graph it requested
**5.76 GiB** and OOM'd a 4 GB card.  Both now decompose on the CPU, where the same operation
costs a few MB.  Neither is on the training hot path (one is a per-graph precompute, the other
is memoised per operator).

**This would have killed the retrain on the first elevated graph.**

### 11.3 Efficiency: what P2 elevation actually costs

```
                       P1 (as stored)     P2 (elevated)
nodes                        2,859            11,133
training step                289.7 ms         495.8 ms      1.71x
peak VRAM                     0.45 GB          1.10 GB
370 graphs x 100 epochs        3.0 h            5.1 h
```

**Step time scales 1.71x for 3.89x the nodes** — the DEQ's cost is strongly sublinear in `N`,
so the topology fix is affordable.  Dataset load costs 566 ms/graph (elevation 205, the
potential-flow CG 462, prior rewrite 356), i.e. ~3.5 min once per rheology swap; not worth
caching.

Memory *drops* 131 MB -> 5.8 MB per graph, because the P1 packs carry `G_x`/`G_y` as sparse
`[N, N]` operators (64.6 MB each) that `graph_gradient_operators` does not use in its default
MLS mode.  Elevated graphs simply do not carry them (see B18).

### 11.4 Aligning the objective with the metric that predicts the outcome

§10.3 established that **gate union Jaccard is the only Stage-A metric that predicts the clot
model's own oracle-F1** (+0.918 pooled, +0.765 within one flow arm; `dsrx` correlation reads
-0.073).  Nothing in the objective optimised it — `sr` and `dsrx` were supervised as continuous
fields, which is a *proxy* for the threshold crossing that actually matters.

`_soft_gate_bce` closes that gap.  The shipped gate is `(sr < lss) | (dsrx < sgt)`; its soft
form is the complement of "both branches stay off",

```
p_fire = 1 - (1 - sigma((lss - sr)/tau_s)) * (1 - sigma((sgt - dsrx)/tau_d))
```

with temperatures from the ground truth's own spread, against the **hard** GT gate as target.

*The target is well-posed under the training operator*, which was checked rather than assumed:
computed with the kernels' WLS the GT gate fires on 10.1 / 36.0 / 10.1 % of wall nodes on
patient020 / 001 / 041 — identical to the shipped 3-hop MLS convention to one decimal.

Behaviour, `patient020`:

```
prediction              l_band_gate    gate Jaccard
exact GT                     0.110           0.948
smooth 15% error             0.620           0.533
smooth 40% error             2.010           0.274
0.4x under-scaled            4.068           0.114
```

Monotone against the metric, and differentiable (grad norm 0.915).  Weight
`KINEMATICS_GATE_WEIGHT`, default 0.

**On operator agreement**, an earlier claim here was too strong.  `wall_band_shear_losses` uses
the kernels' WLS operator, not `build_mls_gradient(hops=6)` — the latter is numpy and not
differentiable.  They agree on *structure* (wall-node correlation 0.966-0.999 for `dsrx`,
0.986-0.998 for `sr`) and differ in amplitude by 1.8-2.6x, the known stencil attenuation.
Because both terms are normalised by the ground truth's own spread, the amplitude difference
cancels; what is optimised is the structure the consumer reads.

### 11.5 Launch configuration, final

```bash
SPECIES_PRIOR_SOURCE=analytic
KINEMATICS_ELEVATE_P2=1
KINEMATICS_COORD_MODE=centered
KINEMATICS_NORMALIZE_SHEAR_GRAD=1      # without this every other term is inert (§10.1)
KINEMATICS_GATE_WEIGHT=<w>             # the term that optimises the predictive metric (§11.4)
KINEMATICS_WALL_SHEAR_WEIGHT=<w>
KINEMATICS_PRIOR_FLOOR_WEIGHT=<w>
KINEMATICS_MIN_GATE_JACCARD=0.25
KINEMATICS_MIN_DSRX_CORR=0.65
KINEMATICS_LOSS_WEIGHTS=outputs/kine_loss_weights.json   # s12 -- replaces the weight sweep
```

Four unfitted weights.  Sweep them together, on gate Jaccard — and note that with B16 fixed the
*four existing* weights (`data 500 / cont 50 / wss 10 / shear_grad 50`) have never competed
either, so they belong in the sweep too.  `l_cont` currently lands at 73% of the objective,
which is not a considered choice, just what the old numbers give once they mean something.

### 11.6 B21 — the seal fix was in the wrong place

B8 set `KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS` in `finetune_kine_patient_anchors.py`.  That
protects exactly one launcher.  Running the end-to-end path directly and printing the split
showed the fallback still in force:

```
[kin] Clinical split: holdout val stems=['patient007']
```

— so `patient013`, `patient031` and `patient043` were still training vessels for anything not
started through that script.  The default is now derived from `wall_cohort_splits` inside
`split_clinical_anchor_train_val`, and verified on the default path:

```
SEALED in TRAIN : []            DEV in TRAIN : []
SEALED in VAL   : ['patient007', 'patient013', 'patient031', 'patient043']
```

**Found only by running the real integration path**, not by reading the code — the same
end-to-end check also confirmed the launch config produces 49/49 P2 graphs, rewritten priors,
`width_d2` max 62.75 (training p95 73.8) and per-node `is_anchor`.

---

## 12. The weights: calibrate by gradient share, do not sweep

**A sweep is the wrong instrument.**  Eight weights, each configuration costing a training run,
so a grid is unreachable and a random search would spend its budget rediscovering what one
forward pass can measure.  Worse, the weights are doing two unrelated jobs at once -- unit
conversion between terms on wildly different scales, and expression of priority -- and a sweep
searches that tangle instead of untangling it.

**The Kendall weighter is not the answer either**, and the tree already says so:
`DynamicLossWeighter` exists and is disabled ("avoids negative weighted PDE collapse").
Homoscedastic uncertainty weighting assumes every term is a log-likelihood of the same data; a
PDE residual is not, and the learned precisions run away.

**What actually matters is the gradient a term puts on the parameters, not its value.**  So
measure it.  For each term separately, on a fixed batch, from a realistic reference state:

```
g_i = || dL_i / d(theta) ||          w_i = share_i / g_i
```

`src/utils/loss_calibration.py` + `scripts/calibrate_kine_loss_weights.py`.  One
forward/backward per term per graph; no training runs.  Measured from the shipped checkpoint,
8 elevated anchor graphs, under the launch environment:

```
term              |grad| median   max/min   share      weight
l_band_gate               84.67     415.2    0.30    0.009272
l_data_kine              0.5757       4.7    0.22      1.0
l_cont                    13.57       5.4    0.10     0.01928
l_band_sr                  54.7       3.1    0.08    0.003827
l_band_dsrx                1120      39.7    0.08   0.0001869
l_prior_floor             1.247      11.0    0.07      0.1469
l_wss                     17.71       2.0    0.05     0.00739
l_shear_grad               1410      23.5    0.04   7.426e-05
l_io                      2.174       2.4    0.03     0.03611
l_bc                  3.556e-09       2.1    0.02    DROPPED (inert)
l_mom                    0.2486       1.7    0.01      0.1053
```

Weights are relative to `l_data_kine = 1.0`; the absolute scale is the learning rate's job.
Against the shipped recipe on the same basis (`cont 0.1, wss 0.02, shear_grad 0.1, io 0.01,
mom 0.002`), calibration wants `l_cont` **5x lower**, `l_shear_grad` **1350x lower**, `l_mom`
**50x higher** and `l_io` **3.6x higher**.

**What this leaves as a judgement call is `share` alone** -- a short, readable statement of
priority that someone can disagree with on the merits, rather than eleven numbers spanning six
orders of magnitude that nobody can reason about.  The ordering follows the one measurement
that predicts the downstream outcome (§10.3): the gate term leads.

### Three things the calibration had to be taught

* **`l_bc` is structurally inert** -- gradient 3.6e-09.  The hard BC `u = uv_prior + sdf * uvp`
  satisfies the boundary condition by construction, so the term has nothing left to say.
  Solving `w = share / g` for it asks for a weight of ~1e+07, which would make numerical noise
  the dominant gradient.  It is **dropped**, not amplified: a term with no gradient cannot be
  given influence, only noise.
* **Aggregate with a median, not a mean.**  Derivative-of-derivative terms are heavy-tailed
  across graphs -- one 6-graph run put `l_shear_grad` at 3.0e+09 against 7.0e+03 on a 4-graph
  run because a single vessel dominated.  The `max/min` column above is reported for exactly
  this reason; `l_band_gate` spans 415x.
* **Anchor "inert" to the reference term, not to the maximum.**  The first version used
  `max * 1e-6`, and `l_shear_grad`'s outlier then dropped *every other term* -- it emitted a
  recipe containing one loss.  Caught by running it, not by reading it.

### Wiring

`KINEMATICS_LOSS_WEIGHTS=outputs/kine_loss_weights.json` sets the whole recipe;
`_resolve_loss_weights` puts every weight in one place, expressed relative to the data term.
**Unset reproduces the shipped numbers exactly** (pinned by a test).  The individual env vars
still win where set, so a sweep of one or two terms on top of a calibrated base is easy.

Re-run the calibration from a mid-training checkpoint if the balance drifts; it is cheap, and
the balance measured at a poorly-trained reference state is a starting point, not a law.
