# Deployable flow: closing the GT → RGP-DEQ drop on `clot_gnn_v5`

Everything in Phases 6-10 was measured on COMSOL's own `t=0` velocity (`flow="gt"`).  The
deployable arm substitutes RGP-DEQ's `u0_pred`, and until 2026-08-23 that substitution cost
roughly half the clot score.  This document records what the cost actually was, what has been
fixed, and what a retrain has to do — with the measured budget for each, so nothing here is
argued rather than counted.

Measurement set unless stated otherwise: 10 vessels (001/005/012/018/020/021/024/028/032/036),
per-node full-mesh F1 at the 0.5 cut plus oracle-F1 (the best cut this ranking admits).
Oracle-F1 is the honest ranking measure here — AUC reads 0.98 while F1 halves, because there
are ~150 positives against ~15k nodes.  Raw numbers: `outputs/diag_flow_*.json`,
`outputs/diag_width_causal.json`, `outputs/diag_why_d2.json`.

---

## 1. What the drop was made of

```
arm                                    AUC     F1@0.5   oracle-F1
GT t=0 flow                         0.9998     0.819      0.954
RGP-DEQ, as it stood                0.9757     0.377      0.509
  + edge-direction deadband         0.9886     0.619      0.651
```

Three independent defects, each with its own mechanism.

### 1a. `edge_features` amplified the surrogate's wall noise  (worth 0.242 F1)

`src/clot_ml/gnn.py` normalised velocity by `|f| + 1e-9`.  COMSOL's no-slip wall velocity is
**exactly 0.0**, so under GT every edge into a wall node got `cos = 0` and
`w_up = w_dn = 0` — the locked ensembles were trained with wall nodes receiving no anisotropic
messages at all.  RGP-DEQ cannot reproduce that: its hard BC is `u = uv_prior + sdf * uvp`,
and on these packs `sdf_nd` at the wall is clamped to `1e-6` while `u_prior` is `~2.9e-5`, so
its wall speed is `~5e-6` — physically zero (4e-6 of the lumen median) but a thousand times
the old floor.  Mean `|cos_d|` on wall-destination edges: **0.0000 (GT) vs 0.70 (RGP-DEQ)**;
total up/down aggregation mass **0 vs ~1100**.

**Fixed** — a field-relative deadband (`FLOW_DIR_DEADBAND = 1e-3`).  Under GT no node falls
in the band at all, so the trained regime is bit-unchanged; 1e-3 and 1e-2 give identical
scores, which is the signature of a noise floor rather than a tuned threshold.

### 1b. RGP-DEQ was being poisoned by one unnormalised input  (rel L2 0.375 → 0.138)

Not OOD — a corrupted channel.  COMSOL exports `triangle6`, so **74.5% of biochem graph nodes
are P2 mid-side nodes of degree 2** (the kinematics training vessels are P1: 85% degree-6).
`precompute_wls_operators` fits a 5-term 2nd-order WLS per node and inverts `M + 1e-6*I` with
`pinv(rcond=1e-5)`; at a degree-2 node the two neighbours are collinear, `M` is rank-deficient,
and the `epsilon*I` lifts the null directions just above `rcond` so they are inverted instead
of truncated.

**34 of 52 packs carry the resulting operator** (3-column collinear rows, transverse aspect
165-262, coefficients ±270 that cancel, row norms to 3296 against the training set's max of
83).  The other 18 had `G` built from a corner-only edge list and their mid-side rows are a
single zero.  `width_d1 = G·width_nd` and `width_d2 = G·width_d1` apply that operator twice,
so `|width_d2|` reaches **1.8e5** where the training p95 is **73.8** — and it separates the
cohort perfectly (it is the only variable that predicts per-vessel flow error, spearman +0.567).

`RGP_DEQ._apply_fourier_encoding` appends the three width channels **raw** to
`Linear(178, 256)`.  Encoder latents reach `‖z‖ = 64,387` against a median of 27; those nodes
then take **76-90%** of the Perceiver global-token read mass (10% when clamped); the poisoned
tokens are broadcast back to every node and `x_enc` is re-injected at each of the 25 Anderson
iterations.  0.01-0.6% of the mesh destroys the whole field.

**Fixed** — `clamped_width_priors` in `src/utils/kinematics_inference.py`, applied at the one
chokepoint every inference path routes through.

```
                rel L2 vs COMSOL t=0   ‖div u‖rms   cross-section flux CV
GT                        -               0.022            0.147
as stored               0.375             2.55             0.205
clamped                 0.138             0.74             0.147     <- mass conserved
```

0.138 is the checkpoint's own recorded benchmark (`rel_l2 0.1007`).  Mesh resolution was never
the problem: after the clamp patient020, 2.3x finer than training, reads **0.118** — the best
vessel in the set.

**All 52 packs re-precached** (`precache_rgp_deq.py --force`, now an atomic write).  Cohort
rel L2 is now **median 0.150, p90 0.205, max 0.372**, with only two vessels above 0.30
(`wound_patient001` 0.372, `wound_patient003` 0.309).  The previous cache was also *stale*: it
predated `repair_pack_wall_normals.py`, and `mod_adv`/`mod_rheo`/`mod_curve` — the GAT's
attention biases — are built entirely from `wall_normal`, so every deployable-flow number in
the project had been produced from zero wall normals.

### 1c. The gate's dominant input was differentiated at the worst stencil width

`build_features` used `hops = 4` for predicted flow.  Wall `dsrx` correlation against COMSOL:

```
hops     3       4       6       8      10
corr  -0.22   +0.24   +0.96   +0.98   +0.98
```

hops=4 sits exactly in the sign-flip band and is **anti-correlated on 3 of 10 vessels**
(036 -0.90, 024 -0.43, 018 -0.11).  `sr` peaks at 6 as well, so nothing is traded away, and a
leave-one-vessel-out probe on the same 69 columns reads oracle-F1 **0.482 at hops=4 against
0.551 at hops=6**.

**Changed to 6 — and the amplitude has to come back with it.**  A 6-hop stencil attenuates
`dsrx` by **2.18x** relative to a 3-hop one, measured on the GT field alone; the surrogate is a
further **1.38x** low like-for-like at hops=6 (corr 0.95).  `sgt` is a physical constant fitted
against the 3-hop convention, so uncorrected the gate branch fires on 0.56x the nodes it should
and agrees with the GT gate on almost none of them.

### 1d. The gate is now on one scale — `PRED_DSRX_GAIN`

**The obvious free version does not exist.**  A stencil ratio ought to be a property of the
operator, estimable per vessel with no ground truth.  It is not: the same operator chain
applied to a smooth **synthetic** field reads **1.00 on every vessel** (median error against
the GT target 54%), because a wide stencil is *exact* on a resolved field — the attenuation is
a statement about the near-wall shear field's own spectral content.  The pred field's own
3-vs-6 ratio is worse (median -0.21, unstable, since pred@3 is noise).  A constant beats both
at median error 5.6%.

So it is a **fitted calibration** and is treated as one.  Least squares on **FIT only** (n=25)
gives **3.00**; DEV (n=5, held out) would have chosen 2.56 — that spread is the honest transfer
error and is why the constant carries two significant figures.

```
                gate union Jaccard      fire rate vs GT
FIT (fitted)      0.20 -> 0.52            x0.61 -> x0.89
DEV (held out)    0.47 -> 0.54            x0.67 -> x1.24
```

Applied in `physics_wall_model.t0_flow_fields` and `clot_ml.features.build_features` under
`flow="pred"` only.  Deliberately **not** applied in `src/differentiable_wall_model`: its gates
are soft with *learned* thresholds, so they absorb the scale themselves.

---

## 2. What is left, and where it lives

**Everything above is shipped.**  Refreshed on the full cohort — 23 clot-carrying vessels, the
whole FIT+DEV pool rather than the 10-vessel scratch set:

```
frozen clot_gnn_v5 (IN-SAMPLE: these vessels are its training pool)
arm    split       AUC   F1@0.5   F1@oracle    n
gt     all      0.9998    0.849      0.949     23
pred   all      0.9911    0.462      0.694     23
gt     FIT      0.9998    0.849      0.953     19
pred   FIT      0.9857    0.419      0.635     19
gt     DEV      0.9996    0.862      0.940      4
pred   DEV      0.9958    0.717      0.736      4
```

**In the project's own currency** — the domain deploy score at the shipped thresholds
(`THRESH_WALL 0.73` / `THRESH_OFF 0.92`), mean over vessels with GT in the domain
(`outputs/diag_deploy_score_gt_vs_pred.json`).  Same weights, same packs, only the flow
source differs; the GT column is in-sample, so read the DELTA, not the level:

```
arm    split      wall   off-wall     full
gt     all       0.952      0.828    0.916
pred   all       0.586      0.350    0.545      delta  -0.366 / -0.478 / -0.371
gt     FIT       0.949      0.789    0.907
pred   FIT       0.542      0.306    0.506
gt     DEV       0.965      0.942    0.957
pred   DEV       0.792      0.481    0.731
```

For scale: PHASE7 10.7 measured the physics backbone under predicted flow at wall 0.515 / 0.505
(FIT / DEV) against 0.858 / 0.890 on GT — a -0.34.  `clot_gnn_v5` collapses by a comparable
amount from a much higher GT baseline, and **off-wall is hit hardest (-0.478)**, which is where
the shipped cut is highest (0.92) and therefore most exposed to a shifted score distribution.

**And the thresholds re-fitted for the pred arm** — on FIT (n=19), read on DEV (n=4).  This is
the readout question asked in the deploy metric rather than in a per-node proxy, and it splits
the two domains cleanly (`outputs/diag_pred_threshold_refit.json`):

```
domain  shipped   refit | FIT ship / refit | DEV ship / refit | ALL ship / refit
wall      0.730   0.775 |  0.542 /  0.543  |  0.792 /  0.793  |  0.586 /  0.586
off       0.920   0.500 |  0.306 /  0.382  |  0.481 /  0.574  |  0.350 /  0.430
```

* **The wall threshold is not the problem.**  Re-fitting it buys **+0.001**.  The entire -0.366
  wall collapse is ranking, not placement.
* **The off-wall threshold is badly stale**, and the optimal cut moves from **0.92 to 0.50**.
  Worth **+0.076 on FIT and +0.093 on DEV** — the held-out gain is the *larger* of the two, so
  this transfers rather than overfitting 19 vessels.  It is free and should be taken.
* That still leaves **-0.40 off-wall and -0.365 wall** after the readout is corrected.  The
  readout is a real lever and a small one; the collapse is in the representation.

The per-node proxy in the next block reads the readout as worth more (+0.10) than the deploy
metric does (+0.08, off-wall only).  Where they disagree, the deploy metric is the one that
counts.

On the original 10 vessels, so the arms are comparable like for like:

```
arm                                    AUC     F1@0.5   oracle-F1
GT t=0 flow                         0.9998     0.819      0.954
RGP-DEQ, nothing fixed              0.9757     0.377      0.509
  + edge deadband only              0.9886     0.619      0.651
  + flow fix + hops=6, NO gain      0.9597     0.209      0.502
  + everything shipped              0.9928     0.463      0.683   <- best ranking
```

**The ranking is the best it has ever been (oracle 0.509 -> 0.683) and the cut is not.**  The
gap between what the ranking admits and where a fixed cut lands is now **0.220**, where the
edge-deadband-only arm had **0.032**.  The flow fix, the wider stencil and the gain all moved
the score distribution, and any cut chosen against GT-flow statistics is now in the wrong
place — which is the best news in this document, because a stale cut is the cheapest thing on
the list to fix.

A leave-one-vessel-out **linear probe** on the same 69 columns — a freshly fitted head, so it
measures the features rather than the mismatch (23 vessels):

```
features                oracle-F1
GT flow                   0.823
pred (all fixes)          0.583
pred, minus `div`         0.579
pred, minus `gate_low`    0.581
pred, minus both          0.579
```

**P3 is dead — dropping the flow-source detectors does not help, it marginally hurts.**  `div`
is 9.76 sd outside the normaliser and `gate_low` agrees with the GT gate on nothing, and the
head still prefers to keep both.  Do not spend the retrain on channel pruning.

**How much of that 0.22 is just the threshold?**  Re-fitting ONE global cut on FIT and reading
it on DEV (`outputs/diag_cut_refit.json`; note this is per-node full-mesh F1, not the shipped
domain readout, which uses `THRESH_WALL 0.73` / `THRESH_OFF 0.92`):

```
arm     cut   | FIT @0.5 / @cut | DEV @0.5 / @cut | ALL @0.5 / @cut | oracle
gt     0.975  |  0.849 / 0.925  |  0.862 / 0.862  |  0.849 / 0.912  | 0.949
pred   0.880  |  0.419 / 0.537  |  0.717 / 0.688  |  0.462 / 0.566  | 0.694
```

So on the pred arm: **0.462 -> 0.566 from one re-fitted cut, and 0.694 at the per-vessel
oracle**.  Roughly +0.10 is a single scalar and another +0.13 needs per-vessel calibration.
The FIT-fitted cut does not transfer to DEV (0.717 -> 0.688, n=4), which is the usual warning
about a 4-vessel split, not a reason to skip the recalibration.

Three readings that survive all of the above:

* **~0.24 of the gap is in the features, not the weights** (probe 0.823 vs 0.583).  No retrain
  recovers that.
* **The frozen GNN's representation beats a refitted linear head** (oracle 0.694 vs 0.583), and
  by more than it did before the fixes.  The retrain's job is not to re-learn the task.
* **The readout is back in play**, worth ~0.10 as a single scalar and ~0.23 in total.

### Where the information goes: the gate

The gate `(sr < lss) + (dsrx < sgt)` is a **hard threshold on physical constants, computed
before the network**.  An amplitude bias there is information loss no head can undo.  Wall
node-level agreement with the GT gate:

```
                            gate_low   gate_sep   union    fires x
raw (hops=6)                  0.00       0.00      0.04      0.56
sr x1.44                      0.00       0.05      0.10      0.62
dsrx x2.73                    0.00       0.38      0.49      1.02
sr x1.44 + dsrx x2.73         0.00       0.38      0.51      0.96
```

* **The union Jaccard goes 0.04 → 0.51 on one scalar**, and the scalar is the `dsrx` one —
  the `sr` correction adds 0.02 on top and is not needed.
* **`gate_low` is 0.00 in every arm.**  The `sr < lss` branch is not recoverable from
  predicted flow; predicted shear never puts the same nodes below 25 1/s.  PHASE7 §10.7 already
  measured that the `sgt` branch dominates, so this is survivable, but it is a hard floor.
* Of the 2.73, **2.18 is the stencil convention** and only **1.38 is a genuine surrogate
  deficit**.  The first is derivable from GT data alone — no fitting against the flow model, no
  held-out split needed.  The second is a real calibration and needs its own split.

### Secondary facts

* **`div` is a flow-source detector** — 9.76 sd outside the v5 normaliser at wall nodes,
  because GT is divergence-free and the SIREN decoder has no continuity constraint (the
  residual is still 33x GT after the width fix).  Removing it costs **0.001** in the probe.
  Drop it for hygiene, not for score.
* **The Stage-A shear head is anti-correlated at the wall**: `sr0_pred` corr **-0.11**, scale
  3.87.  Worse than PHASE7's 0.17.  Never use it; MLS-on-`u0` is right.
* Wall `sr` from MLS-on-`u0` needs **x1.437** (per-vessel range 1.18-1.88); after rescaling the
  residual is **0.187** at corr 0.88.
* Inlet BC is reproduced at rel L2 **0.021**; `mu_eff` correlates 0.90 but runs 12-31% low.
* **patient018 scores 0.000 in every predicted-flow arm** (11 GT positives, commits 437-505).
  It has the worst interior edge-direction sign-flip rate, 9%.  Treat it as its own problem.

---

## 3. The retrain

`outputs/clot_ml_cache_pred` / `outputs/clot_ml_cache_v5_pred` are built from the corrected
packs, at hops=6, with the gain applied — 31 vessels, 68 columns, layout identical to the GT
cache, labels identical, and every flow-derived channel moved.  That is the input for
everything below.

The measured budget on the pred arm, per-node full-mesh F1:

```
0.462   frozen v5, shipped cut          <- where we are
0.566   + one re-fitted global cut      <- a scalar
0.694   + per-vessel calibration        <- the ranking's own ceiling, frozen weights
0.583   refitted LINEAR head            <- what the features alone support
0.912   GT arm at a re-fitted cut       <- the target
```

**P0 — retrain the v5 ensemble on the pred cache, and re-fit the readout with it.**  Take the
off-wall threshold move first (0.92 -> 0.50, +0.08 off-wall, held-out validated) and bank it
separately, so the retrain is judged against **wall 0.586 / off 0.430 / full ~0.58**, not
against the shipped 0.586 / 0.350 / 0.545.  The wall threshold is already right (+0.001), so
everything the retrain earns on the wall is genuine representation gain.

**P1 — flow-source augmentation: one artifact trained on GT ∪ pred.**  Promoted: with the
readout worth only +0.08 and the wall threshold already optimal, essentially the whole -0.37
is representation, which is exactly what augmentation targets.  Each vessel enters
twice with the same labels.  Still the most promising structural change: it forces the
representation to be flow-source invariant, it doubles the data in the regime where 31 vessels
is the binding constraint, and it means one artifact serves both scopes rather than a separate
deploy model needing its own validation.  Judge on both scopes at once — a GT regression is a
failure even if the pred arm improves.

**P2 — DONE** (§1d).  What remains of it: the gain is a global constant with a 3.00/2.56
FIT/DEV spread, so a per-vessel or per-geometry-class version is a real follow-up if the
retrain shows the gate is still the binding channel.

**P3 — DROPPED.**  Removing `div` and `gate_low` was measured and does not help (probe 0.583
-> 0.579).  Do not spend the retrain on channel pruning.

**P4 — raise the ceiling: retrain Stage-A.**  The only item that moves the 0.583 probe.  The
current recipe is `weight_data=500, weight_mu=10, weight_wss=10` on P1 meshes.  Needed:
  * normalise the width priors inside the encoder rather than clamping at the call site;
  * supervise **wall shear magnitude and its along-wall gradient** directly — the downstream
    consumes derivatives, and the model is scored on velocity L2, which is why its own shear
    head drifted to -0.11 correlation at the wall;
  * a divergence penalty (continuity is still 33x GT);
  * train on P2 meshes, or make the WLS builder rank-aware so P1 and P2 packs agree.

**How to judge any of this.**  Not by velocity rel L2 — the width fix halved rel L2 and made
the frozen clot model *worse*.  Judge by **wall `dsrx` correlation**, **gate union Jaccard**,
and **oracle-F1**, in that order, and separate the readout move from the representation move
before crediting either.

**Known-bad vessels.**  `patient018` (0.000 in every pred arm, 11 GT positives), `patient029`
and `patient019` (both 0.000 at the shipped cut, oracle 0.116 / 0.036).  All three are FIT.
They are a different failure from the cohort trend and should be looked at on their own rather
than allowed to drive a cohort median.
