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

The pattern in all three: a mechanism that was plausible from reading the code, asserted before
it was measured.  The measurements that did survive — the leak (43/43), the prior-degradation
ratio (45/52), the collinearity (100% of degree-2 nodes), the rank split (0.72 vs 4.2e-16), and
the shear-structure damage (§1j) — are the ones this plan rests on.
