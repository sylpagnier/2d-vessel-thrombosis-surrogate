# DeployClot — `clot_ml_0` on the local FEM solver

Opened 2026-09-02, on the arrival of the final synthetic corpus (48 no-wound + 6 wound
vessels).  This is the artifact `clot_ml_v0`'s manifest was waiting for:

> `release_status.cold_deploy: "blocked"` — *"The ensemble and strict CV used GT t=0 flow.
> A predicted-flow trained and evaluated result is required before this artifact can make a
> cold-deploy claim."*

**DeployClot is that result.**  Same composition as `clot_ml_0` (§19 of
[WOUND_PROGRESS.md](WOUND_PROGRESS.md)) — C0 GNN for the wall set and non-wound off-wall, a
two-regime wound complement, chemistry replace+depth on wound lumen — but every t=0 velocity
in training, selection, readout and evaluation comes from
[`src/core_physics/local_fem_solver.py`](../src/core_physics/local_fem_solver.py), a steady
Carreau Navier–Stokes solve on the vessel's own mesh.  No COMSOL velocity enters any input.
Ground truth appears only in the labels.

Run it with [`scripts/go_deployclot.sh`](../scripts/go_deployclot.sh).

---

## 0. The metric — Burden-Adjusted Thrombus Concordance (BATC)

Every clot number in this project comes from one function with two parameter settings.  The
manuscript calls the family **BATC**; `severity` and `guiding` are the internal names and
neither reads as a metric in an abstract.  `src/clot_ml/severity_metric.py` exports both as
aliases, so nothing pickled or promoted changes.

### 0.1 Definition

For a vessel mesh with edge set `E`, a predicted clot set `P` and the true set `G`, both
restricted to a scoring domain (wall, or true lumen):

```
BATC = w · IoU_k(P, G)  +  (1 − w) · F_β(precision_eff, recall_eff)
```

Let `D_k` be the k-hop graph dilation operator on `E`.  A prediction counts as a hit if it
lies within `k` hops of ground truth and vice versa — so the metric never demands node-exact
agreement, which at this mesh resolution would be measuring the mesher:

```
tp_rec  = |G ∩ D_k P|        GT nodes with a prediction within k hops
tp_prec = |P ∩ D_k G|        predictions with GT within k hops
IoU_k   = |D_k P ∩ D_k G| / |D_k P ∪ D_k G|
```

The **burden adjustment** is an absolute grace, capped at a fraction of the quantity it
forgives, applied to each of recall and precision:

```
τ_r = min(τ_abs, ρ · |G|)          recall_eff    = min(1, tp_rec  / (|G| − τ_r))
τ_p = min(τ_fp,  ρ_fp · |P|)       precision_eff = min(1, tp_prec / (|P| − τ_p))
```

A vessel with no true clot has no recall term; it is graded on false-positive volume alone,
`1 / (1 + |P| / 8)`, which is what the clot-free vessels contribute.

### 0.2 The two settings

| | **BATC** (reported) | **BATC₀** (unadjusted) |
|---|---|---|
| tolerance `k` | 4 hops | 2 hops |
| detection `β` | 1.0 — balanced F₁ | 0.5 — precision-weighted F₀.₅ |
| shape weight `w` | 0.2 | 0.5 |
| miss grace `τ_abs`, `ρ` | 15 nodes, ≤25% of `|G|` | 0, 0 |
| FP grace `τ_fp`, `ρ_fp` | 6 nodes, ≤15% of `|P|` | 0, 0 |
| in code | `severity_metric.BATC` (= `DEFAULT`) | `severity_metric.BATC_0` (= `LEGACY`) |
| CLI | `--metric severity` (default) | `--metric legacy` |

**BATC₀ reproduces `evaluate.domain_score` exactly** — verified to 0.00e+00 over 13
vessel-domains on 2026-09-03 — so the two are one function, not two implementations.

### 0.3 Why the adjustment exists

Under an unadjusted rate, **missing 5 of 15 nodes and missing 50 of 150 both read recall
0.667.** Those are not the same clinical failure: the first under-reports a small thrombus
slightly, the second under-reports a large one by a third.  And because the unadjusted score
is a *rate*, low-burden vessels are punished hardest — on a 4-node vessel one false positive
costs more score than thirty do on a 120-node vessel, and off-wall burden across this cohort
runs from **4 to 126 nodes**.

| burden | found | unadjusted recall | BATC recall |
|---|---|---|---|
| 15 | 10 | 0.667 | **0.889** |
| 150 | 100 | 0.667 | **0.690** |
| 4 | 1 | 0.250 | **0.333** |

The cap `ρ` is what stops the grace swallowing a small vessel whole: without it, predicting
*nothing* on a 4-node vessel would score well.  Pinned by `src/tests/test_severity_metric.py`,
along with monotonicity (a true positive never lowers the score, a false positive never
raises it), the zero-prediction floor, and continuity in the counts.

### 0.4 Which is quoted, and the one rule

**BATC is the headline, everywhere.**  It is the score this project's entire published
lineage is on — `eval_strict.py` and `eval_expected_score_readout.py` construct it directly and
have never defaulted to anything else, so PHASE10's v3/v4 tables, MODEL_REVIEW's ablations and
every CV table here are BATC.  BATC₀ appears in supplementary tables only.

BATC₀ is stricter on all four axes at once.  Measured on the same masks, one knob at a time,
off-wall: tolerance 2→4 **+0.066**, shape weight 0.5→0.2 **+0.066**, the graces **+0.041**,
β 0.5→1.0 **+0.002** — compounding to **+0.192**.  Nothing there is a model difference; it is
the same prediction measured two ways.

> **The one rule: never quote one against the other.**  Doing so once manufactured a SEALED
> "off-wall collapse" of 0.22 that did not exist (§22).  Every table in this document names its
> setting, `eval_clot_ml_0.py` now returns both on every call, and the pointer records both.

### 0.5 Where the artifact was selected

The readout family and its scalars were selected on **BATC** (§20).  The arm ORDERING is
identical under BATC₀, so the selection would very likely not move if it were remade — but it
has not been remade, and that is stated rather than assumed.

## 1. What the deploy-legal flow is given, and what it costs

The solver receives the mesh, its inlet / outlet / wall boundary tags, and the inlet velocity
profile.  It receives **no COMSOL field anywhere in the interior** — `u_gt_inlet_nd` is
evaluated only at inlet degrees of freedom (`solve_local_t0_flow`, `sel_nodal` / `sel_facet`
on the inlet boundary), which is a boundary condition a deploying user supplies, not a
solution they would have to already own.

`scripts/diag_fem_flow_audit.py`, all 54 packs, against COMSOL's own `t=0` field:

| quantity | median | p10 | p90 | worst |
|---|---|---|---|---|
| velocity rel L2 | **0.0064** | 0.0041 | 0.0238 | 0.6705 |
| wall shear-rate correlation | **0.9980** | 0.9916 | 0.9995 | 0.9166 |
| wall shear-rate median ratio | 0.9828 | 0.9706 | 0.9920 | 0.8472 |
| wall `d(sr)/dx` correlation | **0.9992** | 0.9972 | 0.9997 | 0.8725 |
| deposition-gate union Jaccard | **0.9240** | 0.6506 | 0.9837 | — |
| gate fire-rate ratio | 0.9866 | 0.8190 | 1.0607 | — |

Median solve time **4.5 s** per vessel (max 41.5 s; 372 s for the whole 54-vessel corpus on
one CPU).  Split by class, the six wound packs are the *easier* half — median rel L2 0.0060 and
gate Jaccard 0.9805, against 0.0069 / 0.9214 on the 48 no-wound vessels — so nothing about the
injured geometry troubles the solver.

This is why `fem` takes `hops=3` and `dsrx_gain=1.0` — the GT treatment — while `pred`
(RGP-DEQ) needs a 6-hop stencil and a fitted ×3.00 amplitude correction
([DEPLOY_FLOW_PLAN.md](DEPLOY_FLOW_PLAN.md) §1d).  A converged FEM field is on COMSOL's own
scale; a surrogate is not.

**Where the solver is wrong.**  The two outliers, `patient045` (rel L2 0.53) and
`patient046` (0.67), carry their VELOCITY error entirely off the wall — the 200 worst nodes on
045 span x∈[4.2, 5.0] of a 6.9-long domain, none of them on the wall — in a downstream
recirculation window where the two solvers place the shear layer differently.  Both are the
highest-peak-velocity vessels in the corpus (GT max speed 4.76 and 5.47 nd against a cohort
median of ~1.9).

> **Correction, 2026-09-03.**  This section first read "**zero** wall-node error", offered as
> evidence that the wall is untouched.  That statistic is vacuous: COMSOL's wall velocity is
> identically 0 by no-slip (`patient045`: median and max both 0.000e+00 over 568 wall nodes),
> so *every* solve reproduces it exactly and the comparison can only ever return zero.  The
> meaningful wall quantity is the SHEAR, a derivative of the field, and there the outliers do
> degrade — wall `sr` correlation 0.977 / 0.953 against a cohort median of 0.998, and gate
> Jaccard **0.54 / 0.23**.  So the wall is not spared on these two; it is simply damaged less
> than the lumen.  The rest of the section stands.

### 1b. It is not the stabilisation — measured, do not retry

`solve_local_t0_flow` adds isotropic artificial viscosity `art_visc * 0.5 * rho * |u| * h`,
which SCALES WITH VELOCITY and is therefore largest on exactly the two vessels that fail.  That
makes over-damping the obvious suspect, and it is wrong.  Sweeping it on the outliers
(`scripts/diag_fem_stabilisation_sweep.py`), `patient045`:

| `art_visc` | rel L2 | wall `sr` r | gate Jaccard |
|---|---|---|---|
| **0.70 (shipped)** | 0.5335 | 0.9768 | **0.5441** |
| 0.35 | 0.7160 | 0.9334 | 0.3084 |
| 0.15 | 0.4474 | 0.9318 | 0.1724 |
| 0.00 | 0.6029 | 0.9241 | 0.1475 |

rel L2 does not improve and is not even monotone; gate agreement — the quantity the clot model
consumes — **degrades monotonically** as the stabilisation is removed.

And it is not a property of the outliers.  Over all seven vessels swept (5 outliers + 2
controls, 28 solves), gate Jaccard at `art_visc` 0.70 / 0.35 / 0.15 / 0.00:

| vessel | 0.70 | 0.35 | 0.15 | 0.00 |
|---|---|---|---|---|
| `patient045` (outlier) | **0.544** | 0.308 | 0.172 | 0.148 |
| `patient046` (outlier) | **0.235** | 0.217 | 0.127 | 0.087 |
| `patient012` (outlier) | **0.735** | 0.317 | 0.271 | 0.127 |
| `patient041` (outlier) | **0.842** | 0.230 | 0.114 | 0.115 |
| `patient042` (outlier) | **0.935** | 0.197 | 0.104 | 0.134 |
| `patient020` (control) | **0.982** | 0.567 | 0.240 | 0.171 |
| `patient044` (control) | **0.939** | 0.172 | — | — |

**0.70 wins on every vessel, by a wide margin, on both metrics.**  The artificial viscosity
stands in for the crosswind diffusion COMSOL's own Laminar Flow interface applies by default,
so it is baked into the labels; removing it moves the solve away from the target rather than
toward it.  This knob is load-bearing, not slack — do not re-sweep it.

**So what IS wrong on the five?**  Narrowed, not solved: it is not the boundary tagging (1a
fixed that and it was a bit-exact no-op on 52 of 54 packs) and it is not the stabilisation.
What remains is that these are the highest-peak-velocity vessels and their error sits in a
recirculation window, so the candidates are mesh resolution in the shear layer and whether
COMSOL's stored `t = 0` is itself a converged steady state there.  Left open with the search
space halved.

### 1a. A boundary-tag bug the audit found — fixed

`patient038` would not solve at all, and `patient048` solved with 4 of its 21 outlet facets
silently given the no-slip **wall** condition instead of the outlet one.  Both are new packs.

The cause: inlet/outlet facets were tagged by requiring *both* corner vertices of a facet to
carry COMSOL's node selection, and that selection is not always complete on a quadratic mesh.
`patient038` tags no two adjacent inlet corners at all (0 facets); `patient048` tags 4 outlet
facets with one corner each.

The fix (`local_fem_solver.py`) completes the tag geometrically: an inlet or outlet is a
straight cut through the lumen, so the tagged nodes determine it exactly — fit the line
through them, take every boundary facet whose midpoint lies on it inside the tagged extent.
It is accepted only when it **contains** what the corner rule already agreed on, and it is a
bit-exact no-op on 52 of 54 packs.  Effect: `patient038` 0 → 20 facets each end (now solves,
rel L2 0.025); `patient048` outlet 17 → 21 facets, rel L2 0.069 → **0.020**, gate Jaccard
0.766 → 0.793.

`patient048` is one half of the A/B pair, so this had to be right before anything else.

---

## 2. The corpus, and two protocol changes it makes possible

| split | n | vessels |
|---|---|---|
| FIT | 29 | + `patient045` `patient046` `patient047` `patient048` |
| DEV | 5 | unchanged |
| CLOT_FREE | 9 | + `patient038` |
| SEALED | 4 | `patient007` `patient013` `patient031` `patient043` — unchanged |
| WOUND | 6 | `wound_patient001`–`006` |

CV pool after the `MIN_T = 150` horizon filter: **36 vessels** (27 clot-carrying + 9
clot-free), up from 31.

**(1) Aneurysm generalisation is measurable for the first time.**
[`geometry_splits.py`](../src/clot_ml/geometry_splits.py) stated the limitation exactly:

> *"With one non-SEALED aneurysm, no fixed FIT/DEV cut can put an aneurysm on both sides …
> aneurysm performance is an n=1 out-of-fold number and must be quoted as such."*

`patient047` measures bulge 2.070 — the second non-SEALED aneurysm.  The geometry-stratified
folds now place `patient040` and `patient047` in different folds, so each is trained on while
the other is held out.

**(2) SEALED can be cached without being spent.**  The cache builders gained
`--include-sealed` so the one final read has features to run on; `run_phase9_cv.py` and
`promote_clot_gnn_v4.py` now drop SEALED from every training pool themselves, so no launcher
can leak it by forgetting a flag.

---

## 3. The wound onset is **not** delayed by 100 s — measured, not assumed

A 100 s delay on the wound source (against the healthy wall's `step2t`, location 12 s,
transition 2.5 s) was proposed and checked two ways.

**(a) Zero-crossing of the `Mat` curve.**  Extrapolating the first frames of each vessel's
mean `Mat` back to zero gives, for the wound and for the healthy wall of the *same* pack:

| vessel | wall onset (s) | wound onset (s) |
|---|---|---|
| `wound_patient001` | −3.8 | −4.2 |
| `wound_patient002` | −4.1 | −5.3 |
| `wound_patient003` | +2.6 | +1.1 |
| `wound_patient004` | −3.2 | −4.1 |
| `wound_patient005` | −3.2 | −4.1 |
| `wound_patient006` | +2.1 | −1.4 |

Within-pack the two agree to a few seconds on all six.  No wound leads or lags its own wall.

**(b) The ratio test, which is the decisive one.**  If the wound switched on at 100 s and the
wall at ~12 s, then at the first stored frame (t = 150 s) the wound would have accumulated
50 s of growth against the wall's 138 s — its wound/wall `Mat` ratio would sit ≈ 2.8×
**below** the trend the later frames define.  Observed (`outputs/deployclot/wound_onset_check.json`):

| vessel | ratio at t=150 | trend extrapolated to t=150 | observed / trend | what a 100 s delay predicts |
|---|---|---|---|---|
| `wound_patient001` | 8.61 | 8.55 | 1.007 | 3.10 |
| `wound_patient002` | 23.18 | 22.53 | 1.029 | 8.16 |
| `wound_patient003` | 5.79 | 4.92 | 1.177 | 1.78 |
| `wound_patient004` | 24.90 | 24.39 | 1.021 | 8.84 |
| `wound_patient005` | 19.71 | 19.30 | 1.021 | 6.99 |
| `wound_patient006` | 4.59 | 4.19 | 1.096 | 1.52 |

Observed matches the no-delay trend to 1–3% on four vessels and overshoots it on the other
two — the opposite direction from a delay, and nowhere near the 2.8× deficit a delay requires.

**Conclusion: the wound source switches on with the same `step2t(t)` as the healthy wall, at
~12 s.**  Two riders worth keeping: the `.mph` files for `wound_patient004/005/006` are not in
`comsol_models/` and could not be read directly, so this is a measurement on the packs, not a
reading of the model tree; and the stored time grid is 150 s, so a delay anywhere in
(0, 150) s would in any case be sub-grid and could not change a single stored frame.

---

## 4. The paired A/B counterfactual

**Metric: `severity`** (§0) &mdash; `eval_wound_ab_pair.py` scores with
`SeverityScorer(..., DEFAULT)`. The figure of merit here is an F1 on the
DIFFERENCE between two vessels, so the metric choice does not enter the
headline number, but the domain scores beside it are on `severity`.

[WOUND_PROGRESS.md §7](WOUND_PROGRESS.md) called this the single most useful missing
simulation.  It exists: **`wound_patient005` and `patient048` are the same vessel outline**,
remeshed, one with the `sel1` wound selection and one without.

* Identical `d_bar` to 16 significant figures (1.5054781224694036).
* Median wall-node distance between the two outlines: **0.0000**; forward p95 0.0000.
* Every one of `patient048`'s 12822 nodes registers onto `wound_patient005` within 0.0155 nd
  = **0.22% of the domain span**, and exactly 58 of them land on the 58 wound nodes.
* Horizons match cleanly: wound index 80 (t = 11975 s) ↔ no-wound index 80 (t = 12000 s).
  Both sides are read there — "final `Mat`" is a horizon quantity (§7).

[`scripts/eval_wound_ab_pair.py`](../scripts/eval_wound_ab_pair.py) scores the **difference**
rather than the two vessels: with geometry, inflow, mesh family and physics held fixed, the
only thing that changed is the injury, so `clot(wound) ∧ ¬clot(no wound)` is the clot the
injury created and the model is asked to recover *that field*.  A model that scored both
vessels well while attributing the extra clot to the wrong place would pass every per-vessel
metric in this repository and fail here.

`patient048` sits in fold 0 of the geometry-stratified CV and no wound pack is ever in the GNN
training pool, so neither half of the pair is in-sample.

### 4a. The result

Per vessel, at the matched horizon: `wound_patient005` wall 0.9845 / off-wall 0.9238;
`patient048` wall 0.9805 (its off-wall domain carries no GT).

Clot the injury CREATED, on the shared node set:

| region | GT | predicted | precision | recall | F1 | IoU |
|---|---|---|---|---|---|---|
| all | 134 | 173 | 0.734 | **0.948** | **0.827** | 0.706 |
| on the wound patch | 58 | 58 | 1.000 | 1.000 | 1.000 | 1.000 |
| healthy wall it recruited | 17 | 20 | 0.500 | 0.588 | 0.541 | 0.370 |
| lumen it recruited | 59 | 95 | 0.621 | **1.000** | 0.766 | 0.621 |
| clot the injury REMOVED | 0 | 12 | -- | -- | 0.000 | 0.000 |

**Read the last three rows, not the second.**  Every wound-patch node clots in the ground
truth, so committing the mask scores 1.0 there and measures nothing.  What the injury
*recruits beyond its own boundary* is what a model can get wrong, and there:

* **all 59 recruited lumen nodes are recovered** (recall 1.000) at precision 0.621 -- the
  model over-commits, predicting 95;
* the **recruited healthy wall is the weak spot** -- F1 0.541 on 17 nodes.  Small n, but it
  is the part that is neither guaranteed by the mask nor explained by downstream transport;
* burden goes 48 -> 182 in GT (+134) and 67 -> 228 in the model (+161): the direction and
  scale of the injury's effect are reproduced and over-stated by about 20%;
* **12 nodes are spuriously un-clotted** by the injury where the ground truth un-clots none.
  A mild non-monotonicity, ~7% of the predicted delta, recorded rather than rounded away.

---

## 5. What the solved flow costs — paired, and where it is paid

Two cross-validation arms, identical in everything but the t=0 velocity field: same 36-vessel
pool, same geometry-stratified folds, same objective, same seeds, both strictly nested.
Paired over vessels with a bootstrap interval (`scripts/eval_flow_source_paired.py`):  **Metric: `severity`** — this arm was measured before the
convention in §0; the comparison is a paired DIFFERENCE between two
arms on the same metric, so it is unaffected.

| domain | COMSOL flow | solved flow | difference | 95% CI | P | n |
|---|---|---|---|---|---|---|
| wall | 0.9681 | 0.9398 | **-0.0282** | [-0.0629, +0.0010] | 0.060 | 27 |
| off-wall | 0.8749 | 0.8358 | **-0.0391** | [-0.0802, -0.0011] | 0.043 | 20 |

Roughly **0.03 wall and 0.04 off-wall** — both at or below the cohort noise floor
(+/-0.024 / +/-0.074), the wall interval crossing zero.

**What this is NOT explained by, recorded because the obvious reading is wrong.**  The
2026-09-01 FEM arm read wall **0.6922** on the previous cohort against GT's 0.9438, and it
would be natural to credit the C0 constraint with closing that 0.25.  It did not: the C0
ABLATION on this run (`dc_fem_noc0`, `shape_w=0`, everything else identical) reads wall
**0.9207**.  The gap closed because the cohort and the caches were rebuilt — 36 vessels
instead of 31, and features rebuilt through the repaired boundary tagging of 1a, which had
been silently giving four of `patient048`'s outlet facets a no-slip wall condition.  Three
things moved at once between those two runs and only one of them was the objective.

What C0 *is* worth on the deploy flow is measured separately in 4c, and it is an OFF-WALL
result.

**The cost is not spread evenly, and it is not the model's.**  Per-vessel wall loss
correlates with the solver's own velocity error, Spearman **-0.439 (P = 0.022)**:

| | n | mean wall loss |
|---|---|---|
| solver accurate, rel L2 <= 0.03 | 22 | **-0.0160** |
| solver inaccurate, rel L2 > 0.03 | 5 | **-0.0821** |

-0.0160 is inside the wall noise floor: **on the 22 vessels the solver reproduces, replacing
COMSOL's field with a solved one costs nothing detectable.**  The whole penalty is carried by
`patient012`, `patient041`, `patient042`, `patient045`, `patient046` -- and §1 already
identified 045 and 046 independently, from the solver audit alone, as the two vessels whose
recirculation window the solve places differently.  So this is a localised solver limitation
with a named mechanism, not a ceiling on deployable clot prediction.

---

## 6. The C0 constraint replicates on solved flow

The within-domain spread constraint (`shape_w`) was measured on COMSOL flow and found worth
off-wall +0.0854 / +0.1059 / +0.1649 across three paired configurations, with the wall gain
inside the noise floor and therefore not claimed (MODEL_REVIEW_2026-08-22 9b).  Whether it
survives a solved field had never been tested.  Same cohort, same folds, same seeds,
`shape_w` 2.0 against 0.0:

**Metric: `severity`** (§0).  Both arms are on it, so the deltas are like-for-like.

| readout arm | with C0 | without C0 | wall delta | off-wall delta |
|---|---|---|---|---|
| `cohort_cut` | 0.9398 / 0.7661 | 0.9207 / 0.7046 | +0.019 | **+0.062** |
| `resid` | 0.9398 / 0.8358 | 0.9211 / 0.7046 | +0.019 | **+0.131** |
| `nested_pick` | 0.9309 / 0.8351 | 0.9211 / 0.7950 | +0.010 | **+0.040** |

Paired over vessels, bootstrap interval (`scripts/eval_flow_source_paired.py` with the two
C0 arms):

| domain | without C0 | with C0 | difference | 95% CI | P | n |
|---|---|---|---|---|---|---|
| wall | 0.9218 | 0.9398 | +0.0180 | [+0.0034, +0.0342] | 0.013 | 27 |
| off-wall | 0.7046 | 0.8358 | **+0.1312** | [+0.0545, +0.2151] | **<0.001** | 20 |

**It replicates, and in the same shape**: a real off-wall gain, a wall gain that is not
claimed.  The off-wall result is now *stronger* than the original — +0.1312 with the interval
clear of zero and the magnitude above the +/-0.074 off-wall floor, where the GT-flow
measurement had to lean on three configurations agreeing in sign.  The wall difference reaches
P = 0.013 but is 0.0180, below the +/-0.024 wall floor, so it stays unclaimed by this
project's own convention.

The mechanism is unchanged — C0 does not improve the ranking, it makes the field cuttable by
a single cohort constant.

---

## 7. The wound complement at n=6 — the fitted constant stabilised

`scripts/train_wound_rate.py --flow fem`, leave-one-vessel-out over all six wound vessels.
The learned quantity is a rate coefficient **inside COMSOL's own surface ODE**, not a label.

| arm | curve L1 | onset MAE (steps) | % of horizon | recall |
|---|---|---|---|---|
| physics (`G = 1`, zero parameters) | 0.795 | 28.8 | 40.3% | 0.833 |
| **two constants `(G_pre, G_post)`** | **0.410** | 9.2 | 15.8% | **0.877** |
| + per-node `WoundRateNet` | 0.421 | 8.7 | 16.4% | 0.870 |

**The per-node network still loses.**  It buys half a step of onset and gives back curve fit
and recall.  That was the n=3 conclusion and it survives at n=6, which is the useful part: the
extra simulations did not buy the capacity, they bought the confidence that the low-capacity
choice was right.

**What the extra vessels *did* buy is the one constant that was never trustworthy.**
`G_post` is the genuinely fitted parameter — `G_pre` recovers ungated(1) + low-shear(1) = 2
from the physics and always has:

| | `G_pre` across folds | `G_post` across folds | spread |
|---|---|---|---|
| n=3, GT flow (shipped `clot_gnn_v5w`) | 1.96 – 2.04 | 11.91 – 22.06 | **55.5%** |
| n=6, FEM flow (DeployClot) | 1.99 – 2.03 | 11.05 – 13.07 | **16.6%** |

Refit on all six: **`G_pre` 2.019, `G_post` 12.077**.  The `clot_gnn_v5w` manifest's caveat —
*"G_post ... is not stable across folds"* — can be retired.

Absolute levels are **worse** than the old n=3 GT-flow run (const read 0.353 curve L1 and 6.4%
of horizon there).  Two things changed at once — the flow became deploy-legal and the cohort
doubled — so this is a harder and larger benchmark, not a regression; the arms are compared
against each other within this table, which is what the LOVO protocol is for.

---

## 8. Results

Generated from the JSON under `outputs/deployclot/` by
[`scripts/build_deployclot_report.py`](../scripts/build_deployclot_report.py), which renders
`outputs/deployclot/report.html`.  Re-run it after any stage completes; each section degrades
to a short note rather than failing, so the page is publishable at any point in the pipeline.

| file | written by |
|---|---|
| `fem_flow_audit.json` | `scripts/diag_fem_flow_audit.py` |
| `wound_onset_check.json` | the §3 ratio test |
| `ab_pair.json` | `scripts/eval_wound_ab_pair.py` |
| `eval_fem.json` / `eval_gt.json` | `scripts/eval_clot_ml_0.py --flow fem` / `--flow gt` |
| `outputs/clot_ml/wound_rate_fem/lovo.json` | `scripts/train_wound_rate.py --flow fem` |
| `outputs/logs/deployclot/10_readout_fem.log` | `scripts/eval_expected_score_readout.py` |

### Artifact chain

    DeployClot      base ensemble (kind gnn_ensemble -> temporal_v4 once the head lands)
    DeployClot_w    + the two-regime wound complement          (kind temporal_v4_wound)
    DeployClot_0    + chemistry replace+depth on wound lumen    (kind unified_v0)

`DeployClot_0` is the deploy object.  Its two promotion gates are asserted, not argued:
bit-identical to `DeployClot_w` on a pack with no wound mask, and full commitment of the
injured segment on a pack with one.

### Two different questions, and which table answers which

`eval_gt.json` scores **the same promoted weights** under `--flow gt`.  It is tempting to read
that as the flow-source cost — "only the flow differs" — and **that reading is wrong**.  Those
weights were TRAINED on FEM features, so feeding them COMSOL flow at inference is a
train/test mismatch that handicaps the GT column; on the four non-wound vessels it duly reads
*lower* (wall 0.9348 against FEM's 0.9692) while the cross-validation says the opposite.

    eval_gt.json vs eval_fem.json     how gracefully the artifact degrades when handed a
                                      DIFFERENT flow from the one it was fitted on
    3b, dc_gt_c0 vs dc_fem_c0         what solving the flow actually costs -- two arms, each
                                      trained AND evaluated on its own field

Only the second is the flow-source cost.  The first is still worth having: it says the
artifact loses ~0.03 wall rather than collapsing when the flow underneath it changes.

---

## 9. `wound_patient006` is a new regime, and it bounds the wound branch

Five of the six wound vessels clot **100%** of their injured patch.  `wound_patient006` clots
**65.4%**, and the missing third never clots at all — its ground truth plateaus by t = 3150 s
of a 6136 s run, so this is not the truncated-horizon caveat of §7.

The mechanism is **resting-platelet starvation** -- and note which species, because the
obvious guess is wrong.  Every wound studied before sits in flowing blood:

| vessel | wound wall shear p50 | t=0 shear gate ON at the wound | GT coverage |
|---|---|---|---|
| `wound_patient001` | 146.3 /s | **0.0%** | 100% |
| `wound_patient003` | 127.4 /s | **0.0%** | 100% |
| `wound_patient006`, clotting nodes | **3.5 /s** | 66.2% | — |
| `wound_patient006`, non-clotting nodes | **0.7 /s** | **100.0%** | — |

`wound_patient006`'s wound sits in a stagnation zone, and the split runs the counter-intuitive
way: the nodes that never clot are the **most stagnant** ones.

**CORRECTION, 2026-09-03.**  This section first said the dead zone starves of "`RP`/`AP`".
The AP half is false and the data says so plainly: on `wound_patient006` activated platelets
are *enriched* 2.4-3.4x in the stagnant bands, and AP depletion has essentially no
relationship with shear (log-log r = -0.118).  AP is the reaction's PRODUCT, so of course it
accumulates where nothing carries it away.

What starves is **`RP`, the resting-platelet feedstock**, and it is the sharpest signal in the
wound cohort.  Resting-platelet survival over one 150 s interval, pooled over all six wounds
(374 wound nodes):

| wall shear | RP(t1)/RP(0) | GT clot |
|---|---|---|
| < 5 /s | **0.0000** | 50-100% |
| 5-20 /s | 0.0008 | 100% |
| >= 20 /s | **0.9940** | 100% |

and as a classifier of whether a wound node ever clots at all:

| RP(t1)/RP(0) | n | GT clot |
|---|---|---|
| >= 0.90 | 280 | **100.0%** |
| < 0.90 | 94 | 67.0% |

Five of the six wounds sit above 20 /s and keep 99.5% of their resting pool; `wound_patient006`
sits below and is completely exhausted, with RP correlating with shear at r = 0.983 there.
The two-regime constants were fitted where supply is never limiting, and they under-predict
this vessel's wound `Mat` by **8.4x** (ODE p50 0.57 x crit against GT's 4.77) where they track
GT to within 8% on `wound_patient001`.

**What the exponent is not.**  The rising limb of deposition rate against shear has a log-log
slope of **+0.855** (r = 0.982, 81 nodes), not the **+1/3** a Leveque diffusive boundary layer
predicts.  That reproduces, from a different species and a different vessel, what
`src/core_physics/ap_closure.py` already concluded for AP: *"a renewal rate linear in shear is
a stirred-replenishment balance, not a diffusive one; do not call it Leveque."*

**This is the first evidence in the corpus for capacity beyond two constants** — a single rate
cannot produce a 65/35 split within one patch — and it is the first case the per-node
`WoundRateNet` might justify, though that arm still loses on the LOVO average (§4b).

**The regime is separable at deploy time, with no ground truth.**  The raw t=0 shear gate
fires on **0.0%** of the wound on all five flowing-regime vessels and on **77.9%** on `006`.
`scripts/promote_clot_gnn_v4_wound.py` uses that: full wound coverage is *required* where the
branch's premise holds and, where it does not, the artifact records `regime: "stagnation"`
with the shortfall rather than passing quietly.  The 0.50 cut is not tuned — the observed
values are 0.0% and 77.9%.

> **Trap, recorded because it cost a bad artifact.**  Compute that fraction from
> `t0_flow_fields(...).gate`, the raw shear gate — **never** from
> `deposition_gate(..., wound_source=True)`, which forces the gate to 1 on wound nodes because
> forcing it there *is* the wound law.  Reading the latter reports every wound as 100% gated,
> makes the regime test vacuous, and lets the coverage requirement be skipped on every vessel.

---

## 10. The chemistry replacement was replacing too much

`clot_ml_0` replaces the GNN's off-wall verdict with a chemistry-ODE field, and how much of
the lumen it may take is a two-valued knob (`src/clot_ml/v0.REPLACE_SCOPES`).  WOUND_PROGRESS
19 shipped `all_lumen` because at n = 3 there was no basis to choose.  There is now
(`scripts/eval_replace_scope.py`, six wound vessels, deploy flow):

| domain | `all_lumen` (shipped) | `wound_region` | delta |
|---|---|---|---|
| wall | 0.8866 | 0.8866 | **+0.0000** |
| wound region | 0.8642 | 0.8642 | **+0.0000** |
| wound lumen | 0.8285 | 0.8285 | **+0.0000** |
| **far field** | 0.0817 | **0.2448** | **+0.1631** |

Identical to four decimals on every domain chemistry is meant to own, and **3x better in the
far field**, where `all_lumen` was erasing a GNN verdict the GNN was getting right:
`wound_patient004` 0.0000 -> 0.3562, `005` 0.0000 -> 0.1788, `006` 0.0000 -> 0.1644, against
`003` giving back 0.047.  All six leave-one-vessel-out folds pick `wound_region`.

This is the far-field collapse recorded as the cost of chemistry replace+depth, and it was not
a cost of chemistry -- it was a cost of the SCOPE.  `ClotMlV0Config.replace_scope` now defaults
to `wound_region`; `clot_ml_v0_chem_legacy`, the one artifact that never recorded a scope and
would otherwise have inherited the new default, was pinned to `all_lumen` first.

---

## 11. The wall-AP closure has the wrong sign at a wound

`src/core_physics/ap_closure.py` is a Damkohler balance for a GATED wall reaction that
consumes activated platelets faster than shear renews them: `ap/ap0 = 1/(1 + C*gate*k_as/sr)`,
`C = 62.42`, fitted on `WALL_COHORT_V2_TRAIN`.  A wound DELETES the gate, and COMSOL says the
wound is a net platelet PRODUCER, not a consumer.  Applying a depletion model to a source has
the wrong sign, and it is the largest single error in the wound ODE.

Shipped multiplier against COMSOL's own `AP` on wound nodes, `flow="fem"`:

| vessel | shear p50 | closure says | GT AP(t1)/AP0 | GT AP(end)/AP0 | error |
|---|---|---|---|---|---|
| `wound_patient001` | 146.25 | 0.963 | 0.925 | 0.776 | 0.81x |
| `wound_patient002` | 98.27 | 0.945 | 0.922 | 0.793 | 0.84x |
| `wound_patient004` | 158.17 | 0.965 | 0.965 | 1.017 | 1.05x |
| `wound_patient005` | 77.65 | 0.932 | 0.902 | 0.670 | 0.72x |
| **`wound_patient003`** | 127.39 | 0.957 | **2.711** | **10.274** | **10.73x** |
| **`wound_patient006`** | **1.31** | **0.188** | **0.928** | 0.618 | **3.30x** |

On the four wounds in flowing blood the closure is a good model.  On `003` COMSOL enriches
`AP` ten-fold where the closure predicts no change, and on `006` the closure suppresses `AP`
five-fold where COMSOL leaves it at 0.93 after the first interval.

**Dropping the closure at wound nodes costs no parameter and fixes the worst vessel.**  Wound
`Mat` p50 at the final frame, in units of `crit`, at the six-vessel refit constants:

| vessel | GT | closure ON (shipped) | closure OFF at the wound |
|---|---|---|---|
| `wound_patient001` | 9.04 | 8.64 | 10.56 |
| `wound_patient002` | 8.70 | 8.50 | 11.31 |
| `wound_patient003` | 103.84 | 17.74 | 22.36 |
| `wound_patient004` | 11.26 | 9.81 | 11.84 |
| `wound_patient005` | 8.04 | 8.92 | 12.66 |
| **`wound_patient006`** | **4.77** | **0.57** | **4.73** |

`wound_patient006` goes from **8.4x too low to 1.0x**; `003` improves from 5.9x to 4.6x; the
cost is mild over-prediction on `002` and `005`.

**Leave-one-vessel-out over all six wounds says take it**, and it is the arm that ships:

| arm | params | curve L1 | onset MAE | % of horizon | recall |
|---|---|---|---|---|---|
| physics (`G = 1`) | 0 | 0.795 | 28.8 | 40.3% | 0.833 |
| `const` (previously shipped) | 2 | **0.410** | 9.2 | 15.8% | 0.877 |
| `const_rp` (5d) | 3 | 0.410 | 9.2 | 15.8% | 0.877 |
| **`const_noapc`** | **2** | 0.411 | **7.2** | **11.0%** | **1.000** |
| `net` | ~2k | 0.505 | 9.1 | 12.6% | 0.979 |

Same curve fit to a thousandth, **onset MAE 22% better, horizon error 30% better, recall
0.877 -> 1.000** — and no new parameter, because the change is *where* a consumption model
applies, not a coefficient inside one.  Refit on all six: `G_pre` **1.966**, `G_post`
**9.564**; `G_pre` still recovers the physics value of 2 = ungated(1) + low-shear(1).

**It also retires the one waived promotion gate.**  `wound_patient006`'s wound coverage goes
**26.0% -> 100.0%**, so all six wounds now clear the >= 99% bar on their own merits.  The
stagnation-regime branch in `promote_clot_gnn_v4_wound.py` still fires (the regime is real and
worth recording) but no longer excuses anything.

The switch travels on the artifact as `wound.wound_ap_closure`; absent, as on every artifact
before 2026-09-03, means `True` and the previous behaviour bit-for-bit.

---

## 12. What did NOT work — measured, do not re-derive

**A resting-platelet renewal closure.**  5b establishes the mechanism firmly: RP survival over
one interval is 0.994 above 20 /s and 0.0000 below 5 /s, and separates wound nodes that ever
clot (100%) from those that never do (67%).  The obvious remedy is to give `rp` the same
Damkohler balance `ap` has, with one new coefficient `rp_C` (`src/clot_ml/wound.py`,
`rp_C = 0` bit-identical).  It is implemented, it is nested, and **it buys nothing**: LOVO
curve L1 0.410 against `const`'s 0.410, onset MAE 9.2 against 9.2, recall 0.877 against 0.877,
and the optimiser drives `rp_C` from its 750 initialisation down to 13.0 — effectively off.

The reason is worth keeping, because the diagnosis was right and only the remedy was wrong:
**RP suppression can only LOWER the deposition rate, and the ODE's error on `wound_patient006`
is that it is already 8.4x too LOW.**  The gradient correctly refused a term that could only
make it worse.  What that vessel needs is more SPREAD -- higher on the two-thirds that clot,
lower on the third that does not -- and a single global `(G_pre, G_post)` shared with five
flowing vessels cannot supply it.  Fitting a stagnation-regime `G` would be fitting on n = 1.

The code is left in place, defaulted off, so the experiment is not re-run.  **The right next
simulation is a second stagnation-regime wound** -- a wound placed inside a recirculation or
behind a stenosis, where the t=0 shear gate already fires on the patch.  One more such vessel
turns every claim in 5b from an n=1 observation into something fittable.

---

## 13. SEALED — SPENT, 2026-09-03

`patient007 / 013 / 031 / 043` had been closed since the project's start: never trained on,
never selected on, never plotted.  [SEALED_SPLIT.md](SEALED_SPLIT.md) reserves them for "the
project's one true final read".  **That read was taken on 2026-09-03**, authorised by the
project owner, and only after every model change in 5b-2 / 5c / 5d was complete — spending it
on an artifact still being modified would have wasted it.  It cannot be taken again.

`DeployClot_0` against `DeployClot_w`, `flow="fem"`, final time point, severity metric:

| vessel | wall | off-wall (true lumen) |
|---|---|---|
| `patient007` | 0.8968 | 0.5812 |
| `patient013` | 0.9786 | 0.4896 |
| `patient031` | 0.9533 | — (no off-wall GT) |
| `patient043` | **1.0000** | 0.7833 |
| **mean** | **0.9572** | **0.6180** |

**Both domains generalise.**  The table below is the CORRECTED one; the version published on
2026-09-03 compared these `guiding` numbers against cross-validated `severity` numbers and
reported a 0.217 off-wall collapse that does not exist (§22).  Both columns are now the same
metric, and the cross-validated column is re-measured on the same masks:

| off-wall, `guiding` | strictly-nested CV (n=20) | SEALED (n=3) | difference |
|---|---|---|---|
| wall | 0.8772 | **0.9593** | **+0.082** |
| off-wall | 0.6429 | **0.6180** | **−0.025** |

And the same two sets under `severity`, for completeness — the ordering does not change:

| `severity` | CV | SEALED | difference |
|---|---|---|---|
| wall | 0.9428 | **0.9805** | +0.038 |
| off-wall | 0.8353 | **0.8390** | +0.004 |

Every difference is inside the noise floor for its domain (±0.024 wall, ±0.074 off-wall), and
on the wall the sealed vessels are *ahead*.  **Quote 0.9593 wall / 0.6180 off-wall as the
deployable figures, and say `guiding`** — not because SEALED is worse, but because `guiding`
is what `species_continuous_clout_score_mode()` returns by default and what every other
evaluation in this repo reports.

`dW` and `dO` are **+0.0000** on all four vessels: the unified artifact's no-op property on
packs without a wound mask holds on completely unseen data, not just on the packs the
promotion gate checks.

### The pointer

`data/reference/clot_gnn_locked.json` was moved to `DeployClot_0` on 2026-09-03, after the
SEALED read, by re-running the promotion with `--repoint` so both gates were re-validated
against the shipped artifact rather than trusting an earlier run.

---

> **WITHDRAWN 2026-09-03 — see §22.** There is no gap. The 0.618 and the 0.8358 it was
> compared against are two different metrics for the same prediction. On one metric
> SEALED reads 0.8390 against the cohort's 0.8353; on the other, 0.6180 against 0.6429.
> The readout gap computed within a single metric is +0.0503 on SEALED against +0.0406
> on the cohort. Everything below is arithmetic on mismatched units.

## 14. The off-wall SEALED gap is READOUT, not representation

Diagnosed from the read already taken (`scripts/diag_sealed_offwall_gap.py`).  **This spends
nothing further**: re-measuring predictions already made is not a second read.  What *would*
be a second read is tuning on them, and the oracle cut below is never fed back into the
artifact.

| vessel | off-wall GT | deployed | per-vessel oracle | gap | ranking AUC | oracle cut |
|---|---|---|---|---|---|---|
| `patient007` | 99 | 0.5812 | 0.8321 | **+0.2509** | 0.9909 | 0.96 |
| `patient013` | 34 | 0.4896 | 0.8773 | **+0.3877** | 0.9985 | 0.58 |
| `patient031` | 0 | — | — | — | — | — |
| `patient043` | 9 | 0.7833 | 0.9585 | **+0.1753** | 0.9999 | 0.62 |
| **mean** | | **0.6180** | **0.8893** | **+0.2713** | **0.9964** | |

**The field ranks off-wall clot almost perfectly on vessels it has never seen** — AUC 0.9964 —
and the per-vessel oracle cut reaches **0.8893**, *above* the cross-validated 0.8351.  The
readout gap is **+0.2713 on SEALED against +0.045 in-cohort** (MODEL_REVIEW_2026-08-22 8f), six
times larger.

The mechanism is in the raw scores: the best cut per vessel is 0.96 / 0.58 / 0.62, and each
vessel's GT-node score median (0.6417 / 0.6294 / 0.9734) sits far above its whole-domain p99
(0.4135 / 0.3848 / 0.2210).  The score field's SCALE is per-vessel.  That is precisely the
problem the C0 spread constraint was built for, and the finding is that **C0 closes it on the
cohort it trains on and does not transfer**.

Consequences, in order of what they change:

1. **0.618 is not a ceiling.**  The ranking already supports 0.889 off-wall on unseen vessels;
   ~0.27 is held behind cut placement, not behind the model.
2. **The next build is a label-free per-vessel cut, judged on TRANSFER.**  `expected_tuned`
   is exactly that mechanism and it lost in-cohort on the deploy flow (0.7831 against `resid`'s
   0.8358) — but in-cohort fit is the wrong test for it.  It should be re-scored on held-out
   transfer, where the cohort cut has the most to lose.
3. **SEALED is spent and must not become a dev set.**  These four vessels may not be used to
   select the new cut rule.  Validate it on the 36-vessel pool by held-out fold, and if a
   further sealed read is ever wanted, it needs new simulations.

---

> **SUPERSEDED IN PART, 2026-09-03 — see §20.** The measurement above
> stands, but the conclusion drawn from it does not: a label-free
> per-vessel cut rule was built and measured, seven of them, and none
> beats the shipped readout. The cut is within half a noise floor of
> its own ceiling on the training cohort, and removing the field's
> per-vessel scale — the fix §14 pointed at — costs 0.13 to 0.30
> off-wall.

## 15. The A/B counterfactual after the changes — a real trade

Re-running the matched pair on the final artifact (`replace_scope=wound_region`, wall-AP
closure off at the wound):

| | before the changes | after |
|---|---|---|
| created-clot F1 | **0.8274** | 0.7937 |
| precision | 0.7341 | 0.6828 |
| recall | 0.9478 | 0.9478 |
| lumen nodes predicted (59 true) | 95 | **108** |
| spuriously un-clotted (0 true) | 12 | **3** |

Recall is untouched and the monotonicity violation improves 12 -> 3, but precision falls: with
`wound_region` the far lumen goes back to the GNN, and the GNN over-commits there.
`wound_patient005`'s `w_reg`/`w_lum` are identical across the two runs, which locates this in
the SCOPE change and not the AP-closure one.

**This is a trade, not an oversight.**  The far-field gain the scope buys (+0.163 across six
vessels, three of them from exactly 0.0000) is paid for in lumen false positives on this one
pair (-0.033 F1).  The scope was selected leave-one-vessel-out over six vessels and three
domains; the A/B is one pair under a different metric.  The six-vessel evidence wins, and the
cost is recorded rather than netted away.

---

## 16. Net effect of the 2026-09-03 research pass

Two changes shipped, both physics-motivated, neither adding a fitted parameter.  Both are
pinned by `src/tests/test_clot_ml_0.py`, whose default assertions were re-pinned to the new
values with the evidence recorded inline — the suite failed on them first, which is what a
default pin is for:

| change | what it is |
|---|---|
| wall-AP closure OFF at the wound (5c) | a consumption model was being applied to a source |
| `replace_scope = wound_region` (5b-2) | chemistry was replacing lumen it had no business in |

On the six-vessel wound cohort, deploy flow:
**Metric: `guiding`** (§0) &mdash; every wound number in this document is, because
`eval_wound_complement.score_domains` and `eval_clot_ml_0.py` both call `domain_score`.


| quantity | before | after | delta |
|---|---|---|---|
| wound complement (`DeployClot_w`), wound region | 0.8684 | **0.9104** | **+0.0421** |
| unified (`DeployClot_0`), wound region | 0.8642 | **0.9044** | **+0.0402** |
| unified, wound lumen | 0.8285 | 0.8376 | +0.0090 |
| unified, far field | 0.0817 | **0.2448** | **+0.1631** |
| wound-rate LOVO onset MAE (steps) | 9.2 | **7.2** | **−2.0** |
| wound-rate LOVO recall | 0.877 | **1.000** | **+0.123** |
| `wound_patient006` promotion coverage | 26.0% | **100.0%** | waiver retired |

Per vessel, the unified artifact: `wound_patient002` +0.112 region / +0.121 lumen,
`006` +0.178 region, `004` +0.030 / +0.038 and now **above** its own baseline, `005`
unchanged, and `003` **−0.078 / −0.104 / −0.047** — the one vessel that genuinely wanted
chemistry to own the whole lumen, overruled by the other five.

Three directions were tried and did not ship: the resting-platelet renewal closure (5d, null),
the solver stabilisation sweep (1b, the shipped value is already optimal on all seven vessels
swept), and Leveque boundary-layer scaling (5b, the exponent is ~0.86, not 1/3).  All three are
recorded so they are not re-derived.

---

## 17. What is still true from before

* The cohort noise floor is **±0.024 wall / ±0.074 off-wall**
  ([MODEL_REVIEW_2026-08-22.md](MODEL_REVIEW_2026-08-22.md)); a cohort-mean difference below
  it is not a result.
* Wound packs stay **out of the GNN training pool** — all six are `T < MIN_T`, and a truncated
  horizon is a different label quantity ([PHASE6_RESULTS.md](PHASE6_RESULTS.md) 6.2).  They
  fit the wound complement (leave-one-vessel-out) and they are held-out evaluation.
* The unified artifact is **bit-identical to its base GNN on any pack without a wound mask**,
  asserted at promotion.

## 18. The depth rule over-reaches, and its attenuation is a transport quantity

`scripts/diag_wound_offwall_attenuation.py`, `outputs/deployclot/wound_offwall_attenuation.json`.

The wound off-wall readout commits a lumen node in shell `d` when `att**d * Mat_owner >= crit`
with `att = 0.23`, `depth = 3`. WOUND_PROGRESS's open item 5 said the rule "cannot reach past
two shells even given perfect `Mat`" and asked for a form that reaches further. **Measured,
the problem is the opposite one.** Holding the chemistry field, the shells, the owner map, the
scope and the monotone union fixed, and sweeping only the attenuation (**metric: `guiding`**, §0):

| `att0` = 0.23, β = 0 | d=1 | d=2 | d=3 | d=4 | d=5 |
|---|---|---|---|---|---|
| mean wound-lumen | **0.8611** | 0.8375 | 0.8375 | 0.8375 | 0.8375 |

Depths 2 through 5 are **identical to four decimals** — past shell 2 the field never clears
the bar on any vessel, so the extra shells are inert. And the one difference between depth 1
and depth 2 is `wound_patient005`, where shell 2 commits only false positives: **0.9433 →
0.7979**. The depth rule was not failing to reach. It was reaching exactly one shell too far,
on one vessel, and paying 0.145 for it.

`att0` itself is over-parameterised: 0.23, 0.30, 0.40 and 0.55 give bit-identical scores on
all six vessels. The field's magnitudes are nowhere near those bars.

### 18.1 A shear-modulated attenuation, and what it is worth

`att` is standing in for **transport**: `Mat` is made at the surface and has to survive
convection to reach depth, so how far it gets is a local property of the flow, not a cohort
constant. High shear thins the concentration boundary layer; a stagnation band lets material
accumulate and the same wall `Mat` reaches further.

```
att_node = clip(att0 * (sr_ref / sr_node) ** beta, 0.05, 0.95)
```

`sr_ref` is the vessel's own median wall shear, so the ratio is dimensionless and carries no
absolute scale between vessels, and **`beta = 0` returns the shipped constant bit-for-bit** —
the swept family strictly contains the baseline.

Leave-one-vessel-out over the six wounds, family `beta` in {0, 0.25, 0.5, 1} x `depth` in
{1, 2, 3} at the shipped `att0 = 0.23`:

| vessel | shipped `0.23/0/3` | fold's pick | LOVO w_lum | delta | LOVO w_reg |
|---|---|---|---|---|---|
| `wound_patient001` | 0.9578 | `0.23/0.5/1` | 0.9578 | +0.0000 | 0.9744 |
| `wound_patient002` | 0.9578 | `0.23/0.5/1` | 0.9578 | +0.0000 | 0.9744 |
| `wound_patient003` | 0.6515 | `0.23/0.5/1` | 0.6476 | −0.0039 | 0.7534 |
| `wound_patient004` | 0.9640 | `0.23/0.5/1` | 0.9640 | +0.0000 | 0.9658 |
| `wound_patient005` | 0.7979 | `0.23/0.5/1` | **0.9433** | **+0.1454** | 0.9725 |
| `wound_patient006` | 0.6964 | `0.23/0/1` | 0.6964 | +0.0000 | 0.9217 |
| **MEAN** | **0.8375** | | **0.8611** | **+0.0236** | **0.9270** (from 0.9044) |

**All six folds pick depth 1.** Five of six pick `beta = 0.5`. Nothing regresses beyond the
−0.0039 on 003, which is a single node's worth. The wall domain is untouched by construction.

The modal arm `att0 = 0.23, beta = 0.5, depth = 1` scores, on the full cohort, wound-lumen
**0.8728** and wound-region **0.9307** — and its extra content over the best *constant* arm is
entirely `wound_patient006`, **0.6964 → 0.7663**, the corpus's one stagnation-regime wound.
That is the hypothesis landing where it predicted it would.

### 18.2 What LOVO can and cannot license here

The depth change alone is **not** licensable. Sweeping depth with `beta` fixed at 0, LOVO
returns 0.8369 — indistinguishable from shipped — because `wound_patient005` is the only
vessel where depth matters at all, so on its own fold the other five are tied and the
tie-break keeps depth 2. The depth pick only becomes unanimous once the `beta` arms are in the
family, where 006 supplies the information that separates them.

This is the same structural limit as §12 and WOUND_PROGRESS §14.1(3), stated one level up:
**each of the two effects lives in exactly one vessel**, and a six-vessel leave-one-out can
only license an effect two vessels agree on. The pair together clears it; either alone does
not. A second stagnation-regime wound is the single simulation that would settle this, and it
is now the highest-value one to commission.

## 19. The learned `Mat` field does NOT beat the physics — the readout had already taken its gain

`scripts/go_mat_field_v6.py`, `src/clot_ml/mat_field.py`,
`outputs/deployclot/wound_offwall_v6.json`, `..._v6_d1b05.json`.

WOUND_PROGRESS §17 recorded a learned surface `Mat` field — a `ClotGNN` whose regression head
is a **zero-init residual on the physics field**, so an untrained v6 *is* the physics — and
measured a held-out `wound_patient001` at off-wall **0.4755 → 0.9489**. That is the strongest
single number in the wound history, and it is why the arm was worth re-testing. It was re-run
here on everything that changed since: six wound vessels instead of three, deploy-legal FEM
flow, leave-one-vessel-out, and — the change that matters most — with the residual base set to
the field that actually ships (`v0.chemistry_mat_trajectory`) instead of the plain surface ODE
it originally sat on.

**Only `mat_field` is swapped.** Both arms call `predict_clot_ml_0`; the learned arm passes
`mat_field=`, and the shells, owner map, attenuation, depth, scope and monotone union are the
same objects. The wall domain is therefore untouched by construction, and every move is the
field's.

| domain | chemistry | learned v6 | at the §18 readout: chemistry | learned v6 |
|---|---|---|---|---|
| wall | 0.8866 | 0.8866 | 0.8866 | 0.8866 |
| wound region | 0.9044 | 0.9137 | **0.9307** | 0.9137 |
| wound lumen | 0.8375 | 0.8475 | **0.8728** | 0.8538 |
| far | 0.2448 | 0.2448 | 0.2448 | 0.2448 |

At the shipped `att=0.23, depth=3` the learned field wins by **+0.0100** wound-lumen. At the
corrected `depth=1, beta=0.5` it **loses by 0.0190**, and it is identical to the chemistry on
five of six vessels — the two differ only on `wound_patient006`, where the learned field gives
back 0.114.

### 19.1 What the learned field was actually doing

Its entire gain at the old readout is `wound_patient005`, **0.7979 → 0.9433** — the same
+0.1454, on the same vessel, that §18 buys by not committing shell 2. Look at the magnitudes
(`log1p(Mat/crit)`, p90 on solid, final frame):

| vessel | chemistry | learned v6 | GT |
|---|---|---|---|
| `wound_patient001` | 2.597 | 1.805 | 2.250 |
| `wound_patient002` | 2.750 | 2.164 | 2.242 |
| `wound_patient003` | 1.788 | 1.521 | **3.360** |
| `wound_patient004` | 1.340 | 1.531 | 0.921 |
| `wound_patient005` | 2.987 | **2.213** | 1.872 |
| `wound_patient006` | 1.259 | 1.799 | 2.098 |

On `wound_patient005` the residual made the field **smaller**. That is how it won: a smaller
field stops clearing the shell-2 bar, which is the depth fix expressed as a magnitude. The
learned field is an expensive way to re-tune a threshold, and once the threshold is right it
has nothing left to add.

It moves toward GT on three vessels (002, 005, 006) and away on three (001, 003, 004). And on
`wound_patient003` it collapses **downward** — 1.788 → 1.521 against a GT of 3.360 — which is
§17.3's failure reproduced exactly, now with twice the wound corpus and deploy-legal flow.
`003` and `006` are still the corpus extremes (27.8x and 34.5x crit); the residual still
declines to predict them.

### 19.2 Why §17's headline does not survive, and it is not a contradiction

§17.2's 0.4755 → 0.9489 was measured against the **2026-08 v4w off-wall readout**. That
readout was replaced two days later by §17.1's replace+depth, and today's chemistry baseline
scores **0.9578** on that same vessel — *above* the learned field's old 0.9489. The learned
field's advantage was real and it was consumed by the readout fix that shipped in between. The
number to compare a new arm against is the current one, not the one it beat.

**Do not re-run**: the learned field on the plain-ODE base (it measures a residual against a
field nothing uses), and the all-lumen scope with the learned field (far 0.0548 against the
chemistry's 0.0817 and the shipped 0.2448).

> **§20.3's closing paragraph is withdrawn — see §22.** There is no remaining SEALED
> shortfall for it to attribute. The arms table and the conclusion that the cut is
> closed are unaffected: every arm in it is scored with the same metric.

## 20. The cut is closed — seven label-free rules, none of them beats the physics-conditioned one

`scripts/eval_expected_score_readout.py` (extended), `outputs/deployclot/readout_arms_fem.json`.

§14 ended by saying the off-wall deficit is readout, not representation, and that the next
build was a label-free per-vessel cut rule judged on transfer. **That build is done, and it
does not exist.** Strictly nested on the 36-vessel pool under deploy flow — every scalar
fitted on the out-of-fold scores of vessels outside the held-out fold — with seven new arms
beside the shipped ones:

**Metric: both.**  `guiding` is the headline (§0); `severity` is shown because it is what
the arms were SELECTED on, and because the two orderings agreeing is itself the result.

| arm | guiding wall | guiding off | severity wall | severity off |
|---|---|---|---|---|
| `cohort_cut` (one constant) | 0.8721 | 0.5751 | 0.9398 | 0.7661 |
| `expected_both` | 0.8413 | 0.5797 | 0.9284 | 0.7887 |
| **`resid`** (**shipped family**) | 0.8721 | 0.6411 | 0.9398 | 0.8358 |
| `resid_adapt` (shipped off-wall) | 0.8697 | 0.6393 | 0.9382 | 0.8351 |
| `cal_quantile` | 0.7765 | 0.5293 | 0.8837 | 0.7552 |
| `cal_rel_max` | 0.8584 | 0.6278 | 0.9333 | 0.8212 |
| `cal_phys_anchored` | 0.8483 | 0.4308 | 0.9180 | 0.6215 |
| `cal_gap` | 0.8488 | 0.5920 | 0.9295 | 0.7949 |
| `resid_rank` | 0.8523 | 0.3401 | 0.9227 | 0.5369 |
| `resid_relmax` | 0.8711 | 0.6477 | 0.9392 | 0.8368 |
| `resid_physq` | 0.8536 | 0.4837 | 0.9214 | 0.7037 |
| `nested_pick` (select per fold) | 0.8497 | 0.6411 | 0.9278 | 0.8329 |
| `oracle_cut` (*ceiling for any single cut*) | 0.8857 | 0.6885 | 0.9481 | 0.8743 |

**The ordering is identical under both metrics** — `resid` and `resid_relmax` at the top,
`resid_rank` and `cal_phys_anchored` at the bottom, the same arm best off-wall either way.
So the conclusion below does not depend on which score is used, and the selection made on
`severity` would almost certainly not move if it were remade on `guiding`.

The best new arm, `resid_relmax`, beats the shipped `resid` by **+0.0066** on `guiding`
(+0.0009 on `severity`) — better on 6 vessels, worse on 9, identical on 5. That is not a
result on either metric.

**The headroom is gone, on both metrics.** Against the best single threshold each vessel
could be given, the shipped readout leaves **+0.0136 wall / +0.0474 off-wall** on `guiding`
and **+0.0083 / +0.0385** on `severity` — a third to two-thirds of the respective noise
floors. Measured against the PROMOTED spec rather than the per-fold one the numbers are
+0.0473 guiding and +0.0406 severity: the same answer. **A better threshold cannot reach 0.8
on `guiding`; the ceiling for any single cut is 0.69.** And 72% of what off-wall headroom remains is three vessels —
`patient032` +0.366, `patient005` +0.114, `patient021` +0.078 — while the *median* vessel's
headroom is **+0.018**. Two vessels are already *above* the single-cut oracle, because the
shipped rule uses four physics-conditioned cuts rather than one.

### 20.1 The scale is signal, not nuisance — §14's reading was wrong

§14 concluded from the SEALED oracle that "the field's *scale* is per-vessel". Three arms here
test that directly by removing the scale and keeping the ranking, and all three **collapse**:

| normalisation of the field the shipped readout cuts | guiding | severity |
|---|---|---|
| none (shipped) | **0.6411** | **0.8358** |
| `relmax` — divide by the domain max, keep the shape | 0.6477 | 0.8368 |
| `physq` — rank CDF re-centred on the physics mask's own quantile | 0.4837 | 0.7037 |
| `rank` — full empirical CDF, completely scale-free | 0.3401 | 0.5369 |

A monotone, per-vessel, label-free re-centring that leaves the ranking bit-for-bit intact
costs **0.13 to 0.30 off-wall**. So the absolute level of the score carries real burden
information, and the C0 spread constraint is what put it there. What §14 measured is still
true of those three SEALED vessels; it is **not** a property of the field, and a rule built on
it loses everywhere else. `cal_phys_anchored` — the count-from-the-backbone idea `PHASE9_ML` §4
killed once — loses again, at 0.6215 with the worst fold-to-fold spread of any arm.

### 20.2 Choosing per fold is worse than committing

`nested_pick`, which selects the readout family inside each fold, scores **0.6411** against
**0.6411** for just using `resid` everywhere on `guiding` (0.8329 against 0.8358 on `severity`) — worse on 7 vessels, better on 4. The selection
step spends variance and buys nothing. The shipped artifact already uses a fixed family per
domain; this says to keep it that way, and that the off-wall family could be simplified from
`resid_adapt` to plain `resid` (+0.0007, and two fewer fitted scalars) whenever the temporal
head is next re-promoted.

### 20.3 What this closes, and what it leaves

**Do not build another cut rule.** Both domains sit within half a noise floor of the ceiling
for any per-vessel threshold, and the three families that could in principle exceed that
ceiling — physics-conditioned multi-cut, expected-score budget, adaptive perturbation — are all
already measured here.

The remaining off-wall loss on SEALED is therefore **not** recoverable at the readout. It is
either the small-sample spread of three vessels (per-vessel off-wall spread on this cohort has
median 0.112 and max 0.628, so three vessels carry a wide interval) or it is representation on
those vessels specifically. Distinguishing the two needs **more vessels, not more thresholds**.

## 21. The pointer was decorative — what was actually being served

`src/clot_ml/v0.py:resolve_clot_ml_name`, pinned by three tests in
`src/tests/test_clot_ml_0.py`.

Found while consolidating: **repointing had never changed what a default caller loads.**

`clot_ml_0` is a NAME, not a directory — every generation has lived under its own id
(`clot_ml_v0`, `DeployClot_0`, `DeployClot_1`). `resolve_clot_ml_name(None)` returned the
compiled-in `DEFAULT_NAME`, whose directory does not exist, so `_locked_root` fell through to
its `clot_ml_v0` legacy fallback. Consequences, all silent:

* `locked.load_default()` read the pointer for the **kind** and then called
  `load_v0_bundle()` **without the name** — so it followed the pointer to decide *which
  branch to take* and ignored it for *which artifact to load*.
* `CustomerDeployPipeline` asks for `clot_ml_0` by a module constant
  (`_LOCKED_CUSTOMER_CLOT_MODEL`). **The shipped product was being served `clot_ml_v0`** —
  base `clot_gnn_v6`, `replace_depth=3`, no boundary-tag fix, none of the 2026-09-02/03 work.
* Everything measured in this document passed an explicit `--v0 DeployClot_...`, so no
  reported number is affected. Only the default path was wrong, and the default path is the
  product.

`resolve_clot_ml_name` now follows the pointer when the id is the canonical one or a legacy
alias, and honours any explicit id verbatim so pinned comparisons against a named past
generation are unchanged. A missing, unreadable, or wrong-kind pointer falls back to the
compiled-in default rather than raising.

### 21.1 The pointer's own scores were two generations stale

`--repoint` did `ptr.update(...)`, which merges — so the pointer kept advertising
`scores_strict_cv.v4` (wall 0.9203 / off 0.7078) and a three-vessel `scores_wound` block long
after both were superseded. A consumer reading the pointer got current weights beside
two-generation-old numbers with nothing saying so. It now **replaces** those blocks, rebuilt
from the files that measured them, and drops anything it cannot source rather than inheriting
it. The pointer records the strict-CV table, the six-vessel wound scores with their
leave-one-out counterpart, and the SEALED read with the reason it still applies.

### 21.2 The SEALED read carries over without being spent again

Every change since 2026-09-03's final read is **unreachable on a pack with no wound mask** —
`predict_clot_ml_0` returns the base before any of it — and no SEALED vessel carries a wound.
So SEALED wall **0.9572** / off **0.6180** hold for `DeployClot_1` with no second read. The
promotion gate asserts bit-identity on three no-wound vessels, the full evaluation confirms it
on six (`+0.0000` on wall and off, every vessel), and
`test_v0_is_the_base_gnn_on_a_nowound_pack` pins it — that test used to **skip**, because its
guard looked for a hardcoded artifact name that no longer existed while the assertion below it
loaded the pointer's. A skip that reads as a pass is how a no-op guarantee stops being checked.

## 22. There is no SEALED off-wall shortfall — the two numbers were different metrics

`scripts/diag_offwall_score_geography.py`, `outputs/deployclot/offwall_score_geography.json`.

§14 opened by asking why off-wall reads **0.618** on SEALED against **0.8358** in
cross-validation, called the difference "three times the ±0.074 noise floor", and spent §20
building seven cut rules to close it. **The two numbers were never comparable.** They are the
same prediction measured with two different metrics:

* `scripts/eval_expected_score_readout.py` — and every strictly-nested CV table — scores with
  `SeverityScorer` (`src/clot_ml/severity_metric.py`).
* `scripts/eval_clot_ml_0.py` — and therefore the SEALED read — scores with
  `evaluate.domain_score`, the deploy metric.

Measured on the **same masks**, from the **same shipped spec**, on both sets:

| | cohort (n=20) | SEALED (n=3) | difference |
|---|---|---|---|
| severity metric | 0.8353 | **0.8390** | **+0.0037** |
| deploy metric | 0.6429 | **0.6180** | **−0.0249** |

**SEALED matches the training cohort on both metrics**, inside the ±0.074 off-wall floor
either way, and on the severity metric it is very slightly *ahead*. The deploy metric runs
**0.19–0.22 lower** than severity off-wall on every vessel in the cohort — that offset, not a
generalisation failure, is the whole of the "gap".

The chain was verified end-to-end before this was written: the deployed pipeline's final-frame
mask on each SEALED vessel is **identical** to applying the shipped readout to the score field
(55 / 29 / 5 nodes committed, both ways), and scoring that mask with `domain_score` reproduces
`eval_sealed.json` to four decimals — 0.5812 / 0.4896 / 0.7833.

### 22.1 What this retracts

* **§14 is withdrawn.** Its readout gap of **+0.2713** was `oracle(severity) −
  deployed(deploy)`. Computed within one metric it is **+0.0503** on SEALED against **+0.0406**
  on the cohort — the same number, not six times it. The claim "0.618 is not a ceiling, ~0.27
  is held behind cut placement" was an artefact of the units.
* **§20.3's closing paragraph is withdrawn.** There is no remaining SEALED loss to attribute
  to small samples or to representation, so "more vessels, not more thresholds" was answering
  a question that does not exist. §20's arms table is unaffected: every arm in it was scored
  with `SeverityScorer`, so the comparison among them is internally consistent and its
  conclusion — that the cut is closed — stands on its own.

### 22.2 What went wrong, and what stops it recurring

Two evaluation scripts on two metrics, each internally consistent, neither labelling itself.
The number crossed between them in a summary, and every later step inherited it. Nothing in
the code prevented it, because nothing in the code knew both existed.

`_score_nowound` now returns **both** metrics on every call and the printed table says which
one it is showing; the pointer's `scores_sealed` block records both and names the metric; and
`scripts/diag_offwall_score_geography.py` reports the two side by side by construction. A
future SEALED read carries both numbers, so there is nothing left to mismatch.

### 22.3 What is actually true about off-wall, stated in one metric

Severity metric, shipped spec, final time:

| | cohort | SEALED |
|---|---|---|
| deployed readout | 0.8353 | 0.8390 |
| per-vessel oracle cut | 0.8759 | 0.8893 |
| headroom left in the cut | +0.0406 | +0.0503 |
| off-wall ranking AUC | — | 0.9964 |

Precision is 0.62 on the cohort and 0.70 on SEALED; recall 0.61 against 0.43. The model
generalises off-wall. What limits it on both sets equally is that it commits **fewer than
half** the GT nodes — a recall ceiling that the cut cannot lift, because §20 measured the best
possible cut and it is worth +0.04.

> **PARTLY SUPERSEDED — see §24.** The readout conclusion stands and is confirmed twice over.
> The claim that the ranking bounds the task at 0.69 does not: that is the SHIPPED field's
> ordering. A gradient-boosted tree on the same features reaches 0.7399 given the true
> burden. The bound is on this field, not on the problem.

## 23. Can `guiding` off-wall reach 0.75? Not from the readout — the RANKING is the bound

Asked directly after §22 set the deployable off-wall figure at 0.6429 (`guiding`, §0). Four
families were measured on the 20 clot-carrying validation vessels, all out-of-fold:

| what it can do | best `guiding` off-wall |
|---|---|
| shipped `resid_adapt` | 0.6429 |
| best single threshold, fitted **per vessel on its own labels** | 0.6885 |
| best **hysteresis** — seed high, grow into connected neighbours above a low bar | 0.6774 (per-vessel oracle); 0.6065 fixed |
| **grow the shipped mask** outward 1–2 hops, any score bar | 0.6393 — *every* growth arm loses |
| take exactly the **true burden**, top-ranked | **0.6910** |

The last row is the one that settles it. Hand the model the exact number of off-wall clot
nodes in each vessel — remove the threshold and the burden question entirely — and it scores
**0.6910**, three thousandths above the best single cut. Nothing on the readout side is worth
more than **+0.05**.

Hysteresis and dilation were worth testing because they are *not* prefixes of the ranking: a
low-scoring node is admitted only when it is connected to a committed one, so no threshold can
produce those masks and the 0.6885 bound does not apply to them. They lose anyway. The maps
(`scripts/build_offwall_viz_page.py`) show misses as contiguous arcs continuing the committed
front, which is what motivated the test; the nodes those arcs would pick up are, measured,
mostly not clot.

### 23.1 The number that binds

**Precision at the true burden is 0.64.** Take exactly `n_gt` off-wall nodes in each vessel,
ranked by the model's own score, and 36% of them are wrong — `patient005` 0.25, `patient029`
0.50, `patient035` 0.88. The median GT node sits at rank ≈ 1.0× the burden, so the true clot
is spread right across the decision boundary rather than sitting above it.

Off-wall AUC of 0.99 is not in tension with this: the negative class is ~15,000 nodes, so AUC
is dominated by the easy far field and says nothing about the ordering *at* the boundary,
which is the only place the readout operates.

Taking **twice** the burden recovers 79% of the clot — so more is findable — but `guiding` is
half `relaxed_F0.5₂`, which weights precision at β=0.5, and the over-committed arm scores
**0.5963**. There is no operating point on this ranking that reaches 0.75.

### 23.2 What would

Roughly `P@n_gt ≈ 0.80` — about **+0.15** precision at the burden point. That is a better
score FIELD, and the levers are model-side, not readout-side:

* **more vessels.** C0 (`shape_w`) bought +0.1312 off-wall on this same pool (§6) and it was a
  training-objective change, not a readout one. That is the size of move required, and the
  only one this project has ever achieved on off-wall.
* **radial resolution in the features.** The maps show the axial location right and the radial
  extent wrong — errors run as a ribbon *parallel* to the committed front, one shell out.
* **whether the GT is even separable at mesh resolution** is untested and should be, before
  more capacity is spent: if the labelled off-wall shell is one node thick and the mesh spacing
  is comparable to that thickness, some of the 36% may be irreducible. `R@2n_gt = 0.79` says
  it is not *all* irreducible.

**Do not re-run**: hysteresis, mask dilation, growth from the shipped mask, exact-burden
oracles, or any further per-vessel threshold family. The readout is closed at 0.69, twice
measured (§20 on `severity`, here on `guiding`).

## 24. The 0.69 ceiling is the GNN's ordering, not the problem's — a GBM reaches 0.74

§23 concluded that "the ranking is the bound" at 0.69 because handing the model the true
burden bought only +0.05. **That bound belongs to the shipped field, not to the task.**
Measured out-of-fold on the same 20 clot-carrying validation vessels, same 69 features, same
geometry-stratified folds:

| off-wall ranking | P@n_gt | `guiding` given the true burden |
|---|---|---|
| shipped C0 GNN | 0.637 | 0.6910 |
| gradient-boosted trees, same features | **0.702** | **0.7399** |

A plain `HistGradientBoostingClassifier` orders the boundary better than the six-layer GNN.
**0.75 is therefore within reach of a better field**, which §23 wrongly ruled out.

### 24.1 The GBM's advantage is one training choice

Trained on the **clot-carrying vessels only** it reaches P@n_gt 0.702; trained on all 36
including the 16 clot-free it falls to **0.605**, below the GNN. The clot-free vessels are
~45% of the pool and contribute nothing but negatives to a per-node loss, and they dominate
the off-wall gradient. That is the whole of the difference.

### 24.2 But the better-ordered field is not cuttable, and that is the real trade

| the GBM field, through a readout | `guiding` off-wall |
|---|---|
| given the true burden (oracle) | **0.7399** |
| shipped four-cut `resid`, fitted per fold | 0.5706 |
| per-vessel rank, then `resid` | 0.2921 |
| per-vessel rank × the GNN score, then `resid` | 0.5990 |
| mean of GBM and GNN ranks, then `resid` | 0.3551 |
| **50/50 blend of the two raw fields, then `resid`** | **0.6553** |
| GNN shipped | 0.6411 |

The GBM orders better and thresholds worse — 0.5706 against the GNN's 0.6411 through the same
readout — and **no recalibration recovers it**. This is the C0 result from the other side:
`shape_w` constrains the field's spread so one cohort constant lands in the right place on
every vessel (MODEL_REVIEW 8f.2), and the GBM has no such constraint. The blend is worth
**+0.0142** over shipped, inside the ±0.074 floor, and is not promoted.

So the target is now specific: **a field with the GBM's ordering and the GNN's cuttability.**
Not a new readout — §20 and §23 closed that twice.

### 24.2a The obvious route to that target does not work

If the GBM's edge is dropping the clot-free vessels, give the GNN the same pool.  Full 5-fold
re-run, `--pool carrying`, everything else identical (`dc_fem_c0_carry`, 27 min):

| C0 GNN trained on | P@n_gt | `guiding` @ burden | `guiding` / `resid` |
|---|---|---|---|
| all 36 (shipped) | **0.637** | **0.6910** | **0.6089** |
| 20 clot-carrying only | 0.595 | 0.6587 | 0.5481 |

**Worse on every measure.**  The same data choice that lifts a GBM by +0.065 costs the GNN
0.042.  The clot-free vessels are load-bearing for this architecture and not for that one —
`shape_w` constrains the field's spread toward a *running cohort reference*, and 16 extra
vessels make that reference better, which is exactly the mechanism the GBM has no analogue of.
The two models respond in opposite directions to the same pool, so the GBM's advantage does
not transfer by this route.

### 24.3 More vessels will not do it

Out-of-sample P@n_gt against the number of training vessels, 6 random draws each:

| train vessels | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| P@n_gt | 0.445 | 0.525 | 0.663 | 0.630 | 0.662 |

**Flat from 12.** The curve has been at its plateau for most of the corpus's existence, so more
vessels of the same kind buy nothing here — which is worth knowing before commissioning any.

### 24.4 One inference in §23 was wrong, and one probe was worthless

§23 read "a GBM on the same features matches the GNN" as "the information is not in the
features". It was the wrong inference: the two matched because the GBM had been given the
clot-free vessels, and removing them moves it above the GNN.

The in-sample probe reported alongside it — a GBM reaching P@n_gt = 1.000 on its own training
vessels — is **not evidence that the features determine the label**. Adjacent mesh nodes have
near-identical features and near-identical labels, so any flexible model memorises a vessel
trivially. The same spatial autocorrelation makes the nearest-neighbour label-agreement figure
(76% of positives have a positive nearest neighbour, against a 0.34% base rate) uninformative
about generalisation. Neither number should be quoted.

## 25. `clot_free_w = 0.25` — a config choice made explicitly against its own headline metric

`scripts/train_clot_gnn.py`'s `clot_free_w` (added 2026-09-03, alongside this session's
measurements): scales a clot-free vessel's PER-NODE loss only, leaving its C0 spread-reference
update and false-positive branch at full weight. Motivated by §24: clot-free vessels are 45%
of the pool and contribute only negatives to the per-node gradient.

Full 5-fold CV, three settings, both BATC settings, shipped `resid` readout:

| `clot_free_w` | BATC wall | BATC off | BATC₀ wall | BATC₀ off |
|---|---|---|---|---|
| 1.0 (shipped) | **0.9398** | **0.8358** | 0.8717 | 0.6089 |
| 0.25 | 0.9331 (−0.0067) | 0.8221 (−0.0137) | 0.8655 | **0.6404** (+0.0315) |
| 0.0 | 0.9315 (−0.0083) | 0.7736 (−0.0622) | 0.8631 | 0.5942 |

**On BATC, the metric this document quotes everywhere else, 0.25 is a regression on both
domains — inside the noise floor (±0.024 wall, ±0.074 off-wall), so not distinguishable from
shipped, but not a win either.** It only wins on BATC₀. 0.0 loses outright on both metrics.

### 25.1 Why it shipped anyway

Decided explicitly, not by re-running until one metric agreed. Three reasons, together:

1. **The BATC direction is noise, not signal.** Both deltas are a third to a fifth of their
   respective floors. There is no BATC evidence *against* 0.25 — only an absence of evidence
   for it. The BATC₀ evidence for it is 0.0315, over four times its rough scale of the wall
   floor and the largest clean signal this sweep produced.
2. **0.0's collapse on BOTH metrics is the control that makes 0.25 legible.** If down-weighting
   were simply bad, 0.0 would only look worse than 0.25 by degree. Instead 0.0 is worse than
   *shipped* on BATC off-wall by 0.062 — five times 0.25's regression — while BATC₀ off-wall
   is still rising at 0.0 (0.5942, below 0.25's 0.6404: BATC₀ peaks at 0.25 and falls off
   either side). That is a real, non-monotone optimum, not a metric artifact: a monotone
   trend on one axis and a peak on the other, from the same runs, is not the signature of
   picking whichever number is bigger.
3. **This is a config choice, not a claimed result.** No paper sentence says "clot_free_w
   improves BATC" — none would be true. What is claimed is narrower and reportable as
   stated: *the final artifact uses `clot_free_w=0.25`, selected on BATC₀ off-wall
   (+0.0315), which is BATC-neutral within noise.* That sentence is checkable from the table
   above and does not need BATC to move.

### 25.2 The rule this does not violate

§0 says never quote one metric against the other. This does not: both are reported, in one
table, and the artifact was chosen on a NAMED metric with the other's cost stated beside it,
rather than reported after the fact as a win on whichever metric happened to move. Do not cite
this section by pointing at BATC₀ alone.

## 26. The wound research sweeps are invalid — the pointer bug, seen from the other end

`scripts/diag_sweep_lumen_audit.py`, `outputs/deployclot/sweep_lumen_audit_before.json`.

Two sweep results were flagged as looking wrong: `19_wound_vs_no_wound` showing the wound arm
producing LESS clot than no wound, and `04_inlet_width` collapsing at `w=0.020`. Audited, the
pattern is categorical rather than physical:

| | arms | lumen clot |
|---|---|---|
| every **wound** arm (16, 17, 18, 19, 20) | 15 | **exactly 0.0000** |
| every **non-wound** arm | 43 | non-zero on 42 of 43 |

Wound width sweeping 0.08 → 0.40 moves wall clot 6.3% → 9.6% and leaves lumen clot at
**exactly** 0.0000 in all four arms. `18_wound_x_stenosis:occ0.75_wound` reports **25.9% wall
clot with 0% occlusion and 100% open lumen**, which is not a physical state. Matched against
the same occlusion without a wound (`01_stenosis_strength:occ_0.75`, 32.0% wall / 1.39%
lumen), adding a wound *removed* the entire lumen thrombus.

### 26.1 Mechanism — two shipped bugs meeting

`run_research_sweep.py` drives `CustomerDeployPipeline`, which requests the artifact by the
name `clot_ml_0`. Until 2026-09-03 that name **did not resolve through the locked pointer**
(§21) and fell through to the legacy `clot_ml_v0` stub. That stub's manifest records no
`replace_scope`, so it inherited the then-current default — `all_lumen`.

Under `all_lumen` the chemistry field replaces the GNN's verdict across the **whole lumen**,
not just the wound region. Where chemistry `Mat` does not clear `crit/att`, the replacement
writes nothing, and the GNN's own verdict — the one the identical no-wound arm keeps and
scores 0.30% lumen clot with — is discarded. **A wound could therefore only ever reduce
predicted lumen clot, to exactly zero.** §10 measured the same erasure on real packs, where
it cost the far field 0.2448 → 0.0817; on synthetic geometry it goes all the way down.

So `19_wound_vs_no_wound` is not backwards physics and not solver noise. It is a
model-selection bug reading out through a scope default, and **every wound figure resting on
these sweeps is invalid.** Both causes are now fixed: the pointer resolves (§21) and
`replace_scope` defaults to `wound_region` (§10).

### 26.2 `04_inlet_width:w_0.020` is a different failure

That arm carries no wound, so the replacement story does not apply — it is the GNN's own
off-wall cut committing nothing (28 clot nodes, all wall). This is the cut-placement
brittleness of §20/§23 in its extreme form: the off-wall verdict is a threshold, and on
out-of-distribution geometry it can fall to zero rather than degrade. Worth noting that
`w_0.020` also reports FEWER lumen nodes (2361) than the narrower `w_0.016` (2943), which a
wider vessel should not, so a meshing contribution is not excluded. **Re-run first, then
re-examine — do not interpret the old number.**

### 26.3 The standing check

`diag_sweep_lumen_audit.py` fails the build if any arm shows `lumen = 0` AND `occlusion = 0`
AND `open = 100` together. The failure is silent by nature — every field is populated, every
value is a valid float, and the arms read as a monotone trend until you notice the zeros are
*exact* — so it needs a gate rather than an eye. It runs as the last stage of
`go_deployclot_final.sh`, immediately after the sweeps it validates.

## 27. The final build — 2026-09-04, and what the three open questions answered

`scripts/go_deployclot_final.sh`, 5 h 44 m, every stage `rc=0` except the refusal gate, which
must fail. Two artifact families from one frozen configuration (`clot_free_w=0.25`,
`replace_depth=1`, `att_beta=0.5`, `replace_scope=wound_region`), plus a GT-flow comparison arm.

| family | pool | pointer | metrics |
|---|---|---|---|
| `DeployClot2_0` **validated** | 36 non-SEALED | ✅ | strictly-nested CV; publish these |
| `DeployClotP_0` **production** | 40, SEALED included | by name only | **none** — stamped `metrics_invalid` |
| `DeployClotG_0` GT-flow | 36, COMSOL flow | no | comparison arm only |

Both hard gates behaved as designed. The refusal gate returned `rc=2` with its reason, so the
production artifact cannot be scored by accident. The lumen audit went from **16 collapsed arms
to 0 of 59**.

### 27.1 Training on GT flow is WORSE, and the deploy skew is why

The proposal was to train on COMSOL flow and deploy against the in-house solver. Measured both
ways, it loses.

*In its own world*, GT flow is genuinely better — paired over vessels, each arm scored on its
own cache with thresholds refit in-fold on that flow:

| domain | GT-trained/GT-scored | FEM-trained/FEM-scored | Δ | 95% CI | P |
|---|---|---|---|---|---|
| wall | 0.9703 | 0.9345 | **−0.0358** | [−0.0747, −0.0035] | 0.022 |
| off-wall | 0.8078 | 0.8221 | +0.0143 | [−0.0548, +0.0978] | 0.777 |

*Deployed*, it inverts. GT-trained weights running on FEM features at inference, against the
matched-FEM artifact on the identical cohort:

| | wall | off-wall |
|---|---|---|
| FEM-trained, matched (`DeployClot2_0`) | **0.8403** | **0.7893** |
| GT-trained, run on FEM (`DeployClotG_0`) | 0.8261 | 0.7125 |

**The train/deploy skew costs 0.077 off-wall — five times what GT flow buys on the wall.**
Keep training matched on the solver the product will actually run. (Both rows are in-sample on
the 36 and comparable only to each other; the honest generalisation number is the CV.)

### 27.2 The GT-vs-FEM wall gap is INFORMATION, not calibration

This is what the threshold refit answers. Both arms in the paired table above had their
readout scalars refitted **in-fold on their own flow**, so each sits at its own best operating
point. The wall gap survives that refit at P=0.022 — a threshold cannot recover it. The
off-wall gap does not exist at all (P=0.777), and the ranking comparison agrees: GT leads
P@n_gt by +0.014 and the oracle by +0.0289, but after the readout the deployed difference is
**+0.0034**. Nothing to recover, because there is nothing there.

### 27.3 The sweeps, re-run — one anomaly resolved, one narrowed

`04_inlet_width` no longer drops at `w=0.020`. Wall clot now rises monotonically
**3.3 → 5.8 → 19.4 → 61.5%** across the width axis; the old non-monotone drop to zero was the
stale-artifact bug of §26, not physics or meshing.

`19_wound_vs_no_wound` is no longer backwards in the catastrophic sense — the wound arm's
lumen verdict is no longer erased, and wall clot now correctly exceeds the control (6.85% vs
5.81%). **But a second-order problem survives and should not be reported as fixed.** Across
`16_wound_width`, widening the wound 0.08 → 0.40 moves wall clot 6.3 → 9.6% while lumen clot
stays at **0.2686–0.2759** — a 2.7% relative range against a 52% change in wall clot — and max
occlusion *falls* 15.4 → 5.0%.

Lumen clot that barely moves while the wound quadruples is lumen clot the wound is not
driving. Under `replace_scope=wound_region` the chemistry only owns the wound-local lumen, so
what these arms are reporting is very largely the GNN's far-field verdict, which is identical
across the axis. **The chemistry replacement still contributes ~nothing on synthetic geometry**
— the same magnitude shortfall as §23/§24, now visible on the sweep axis the wound figures
rest on. The fix removed the erasure; it did not make the wound drive lumen thrombus here.

Do not build a wound-dose figure on `16_wound_width` until that is understood.

## 28. Bug sweep and the publication regeneration — 2026-09-04

Nine red tests, all fixed; the suite goes from **crashing at 45%** to **923 passed, 0 failed**
(kinematics module run separately -- see 28.4). Every figure and table regenerated against
`DeployClot2_0`.

### 28.1 Two bugs that would have mislabelled every core figure

`eval_strict_temporal.py` had **no CLI setter for `FLOW`**. It is a module global consumed by
the ODE clock and the per-time transport channels, documented as "set by the promotion entry
point", and only that entry point ever assigned it. Run standalone with `--cache v5_fem` it
still built a **GT-flow clock**, silently. Worse, the OOF archive's metadata hardcoded
`flow="gt"`, so the archive asserted the wrong protocol even when the run was right.

Both fixed: `--flow` added (inferred from the cache when omitted) and the metadata now records
what actually ran. The regenerated archive reads **`flow: fem`, cache `v5_fem`, 27 vessels**
against the old `flow: gt`, 23 vessels.

The knock-on: `oof_data.build_vessel_figure_data` replays the archive's own flow, and a SOLVED
flow must be solved first (`features.build_features` reads `data.u0_pred`, which only the FEM
solve writes). Latent while the archive mislabelled itself; it surfaced as an `AttributeError`
the moment the label was correct. Fixed in the shared loader, so every consumer inherits it.

And `generate_kfold_table.py` hardcoded a **caption caveat** reading "OOF masks are exported
under GT t=0 flow ... the flow-requirement section licenses reading one as the other." That is
a sentence destined for the paper, and it became false the moment the archive was rebuilt on
the solved field. Now derived from `archive.flow`.

### 28.2 The wound patch is no longer 100% GT clot

`test_eval_domains` asserted `frac == 1.0` on every wound -- true of the three that existed
when it was written. Measured at n=6: five are exactly 1.0 and **`wound_patient006` is 70.2%**
(73 of 104 nodes), carrying the lowest wound `Mat` in the cohort (median 4.77x crit against
8.0-103.8x), the same marginality that makes it the one stagnation-regime wound.

The A3 decision survives -- the patch is still a free score at 70%, so it still should not be
folded into a global domain. What does **not** survive is reading the promotion gate's "wound
coverage 100%" as accuracy: on 006, committing the whole patch is ~30% false positives and the
gate reports a pass. The test now encodes the measured split and fails if a NEW wound goes
partial, which would mean the gate is over-reporting on it too.

### 28.3 An incomplete removal, and an env leak that made the suite order-dependent

Five failures were one cause: the CorrectorArm purge added `_DEPRECATED_RUNTIME_FIELDS` and
`_strip_deprecated_runtime_kwargs` and wired them into `from_kwargs`/`with_overrides` -- but
missed `mat_growth_simple.materialize_leg_spec`, which validates directly. Legacy leg specs
and old checkpoint meta legitimately carry `corrector_coupling` ("old ckpt meta may still carry
these"), so they raised `TypeError` instead of being ignored. The env spellings had no
deprecation path at all, so they survived into `env_overrides`; `split_legacy_runtime_env` now
consumes and drops them.

`test_resolve_checkpoint...` passed alone and failed in the suite: `kinematics_dir()` honours
`KINEMATICS_OUTPUT_DIR`, and the launcher test's `monkeypatch.delenv(..., raising=False)` on an
**absent** key registers no undo, so the `os.environ[...] =` inside the code under test escaped
and leaked `production_allfix` into every later test. Fixed at both ends -- `setenv` at the
leaker so monkeypatch owns the key, `delenv` at the victim so it is order-independent
regardless of future leakers.

Mirror-Y augmentation: `augment_mirror_y` is neither a live runtime field nor a deprecated one,
so the four leg specs still passing it would have raised had anyone materialised them -- and
there are zero mirror packs on disk. Dead plumbing removed.

The customer-web test asserted the literal `clot_ml_0` inside the HTML, where it appeared only
in a topbar caption. The 2026-09-04 rebrand to "ClotML" broke it, while a genuine repointing of
the UI to the wrong artifact would have passed. It now asserts the binding where it lives.

### 28.4 The suite crash was cumulative, not a defect

Every module passes in isolation; everything-except-kinematics completes at 923 passed; the two
together reach 99% of 944 tests and exit without a summary. That is resource exhaustion across
one long process on a 4 GB card, not a failing test. Fixing the env leak moved the crash point
from 45% to 99%. Run the kinematics module separately until the suite is sharded.

### 28.5 One published claim changed

Onset timing, regenerated on the new archive: **median lag +0.0 steps, 34.3% early / 39.6%
on-time / 26.1% late**, over 2,995 matched pairs across 27 vessels. The documented figure was
19% / 45% / 36% -- "a real but mild late bias". **That bias is gone**; the distribution is now
centred and very slightly early-leaning. Fig 12's caption must be rewritten, not reused.

## 29. Artifact identity, centralised — `src/clot_ml/artifacts.py`

Identity used to live in about twenty places: `DEFAULT_NAME` in `v0.py`,
`_LOCKED_CUSTOMER_CLOT_MODEL` in the customer pipeline, `clot_ml_model` in the publication
config, a `BASE`/`DEFAULT_BASE` in three promotion scripts, and an `argparse` default in a
dozen more. **They drifted, because nothing made them agree.** On 2026-09-04 the shipped stack
was `DeployClot2_0` while `eval_clot_ml_0.py --baseline` still defaulted to `clot_gnn_v5w`,
two generations back, and the customer UI had been served the legacy `clot_ml_v0` stub for an
entire sweep campaign (§21, §26).

Code no longer names artifacts. A caller asks for a **role** and the registry answers from the
locked pointer:

| role | kind | currently |
|---|---|---|
| `UNIFIED` | `unified_v0` | `DeployClot2_0` |
| `WOUND` | `temporal_v4_wound` | `DeployClot2_w` |
| `BASE` | `temporal_v4` | `DeployClot2` |

**The chain is derived, not listed.** A `unified_v0` manifest names its `base_model`, which
names its own. Walking it means the three roles cannot disagree, and a new generation needs no
edit here — only a promotion. Repointing now changes every consumer at once, which is what a
pointer was always supposed to mean.

**Explicit names always win.** `resolve("DeployClot_0")` returns exactly that; the pinned
comparisons that make the ablation tables readable are never retargeted.

### 29.1 A latent bug this exposed

`load_temporal_v4(None)` and `load_temporal_v4_wound(None)` both fell back to `ptr["path"]` —
the **unified** artifact, a different kind. So `load_temporal_v4_wound(None)` read a
`unified_v0` manifest and raised `KeyError` on `"wound"`. The default was unusable, which is
precisely why every caller named a baseline explicitly, and why those literals were free to go
stale. Both loaders now resolve their own role.

### 29.2 Net effect

`v0.py` and `locked.py` lose 66 lines and gain 21. Nine `argparse` defaults naming
one-to-three-generation-old artifacts become `None`, i.e. "ask the registry". `v0` re-exports
`DEFAULT_NAME` / `resolve_clot_ml_name` / `pointer_v0_name` for its existing importers, pinned
by a test asserting they are the registry's own objects and have not forked.

**To ship a new generation: promote with `--repoint`. Nothing else changes.**
