# Publication notes

Running notes for the paper. **Claims go in here only once they are measured**, with the script
that produced them, so a number in a draft can always be traced back to a command.

Three standing rules, all learned the hard way in this repo:

1. **Quote the statistic that was actually computed.** Per-node and per-case means of the same
   quantity are different numbers; the metric of record (`guiding`) and Deploy Score v2
   (`severity`) are different scores. Every entry below names its scope.
2. **A result measured on n<10 vessels at one operating point is a hint, not a finding.** This
   file has already had to retract several such numbers (see *Retractions*) -- most recently an
   n=1 tolerance threshold that did not survive being measured on three vessels.
3. **State the convention beside the statistic.** A correlation without its sign convention, or
   a domain mean without saying which vessels entered it, is not a reportable number. Both
   failures have happened here (§2's signs; the empty-GT off-wall means in §7).

---

## 1. Flow coupling: why the deployed model does not close the clot -> flow loop

**Status: ready to write. ~150 words in the main text, no figure needed.**

A reviewer will ask why a clot-growth model does not feed the clot back into the flow field.
This is the answer, and it is a measurement rather than an argument.

We built an *oracle* closed loop -- ground-truth clot occupancy at each timestep, the measured
post-gelation wall-shear collapse applied under it, the deposition gate re-evaluated with the
same operator the consumer uses. No model error, no localisation error: an upper bound on what
any flow corrector could contribute.

Scored on the domain-restricted deploy metric, GT t=0 flow, 8 vessels:

```
0.748   physics backbone, open loop
0.903   physics backbone + ORACLE closed loop      +0.155, 8/8 vessels
0.990   clot_ml_0, open loop                      +0.087 ABOVE the oracle backbone
0.983   clot_ml_0 + ORACLE                        -0.0065, inside the noise floor
```

**The learned model already sits above the oracle-corrected physics backbone.** Closing the
loop supplies information the model has already recovered from t=0 geometry and shear, so it
adds distribution shift and no signal. The loop is therefore left open by measurement, not by
omission.

*Scope to state honestly in the paper:* this bounds the non-wound cohort under GT t=0 flow,
where the model is saturated (0.984-0.994) and headroom is 0.010 against a +/-0.024 noise
floor. It does not bound the wound vessels (n=3, mixed) or the predicted-flow regime.

Reproduce: `CLOT_ML_ORACLE_BLOCKAGE=1 python scripts/eval_clot_ml_0.py`
Artifacts: `outputs/diag_oracle_blockage_{off,on}.json`

---

## 2. How accurate must a t=0 flow surrogate be?

**Status: the requirement is measured, and as of 2026-09-02 the wall-field table and the
correlation table are both n=33 -- findings, not hints. What remains at n=5 is the three-way
score comparison (GT / FEM / surrogate), which needs the `.nas` exports.**

A reviewer will ask what is lost by replacing COMSOL's t=0 velocity with a learned surrogate.
The answer is not a smooth degradation, and it is not governed by velocity error.

**A converged solve costs nothing.** Ground-truth flow, a local Carreau FEM solve on the same
mesh, and the learned surrogate, all pushed through one scoring path so no comparison crosses
conventions:

```
                 GT flow    FEM     surrogate
patient010         0.882   0.969       0.000
patient005         0.715   0.571       0.483
patient020         0.929     ---       0.366
patient003         0.326   0.574       0.224
patient011         0.697   0.703       0.737
mean               0.710   0.705       0.362      (wall, clot_ml_0)
```

FEM sits inside noise of ground truth. So the readout is **not** hypersensitive to reconstructed
flow, and the -0.35 collapse is attributable to surrogate field accuracy alone.

**The requirement is on the derived wall fields, not on velocity.** Wall nodes, the consumer's
own gate constants and MLS stencil. **Regenerated 2026-09-02 at n=33 -- the whole cohort, not
the 4 mesh-bearing vessels** (median [IQR]):

```
                    rel-L2 u          gate Jaccard        fire ratio       dsrx corr    empty gate
FEM            0.0075 [.006,.009]  0.935 [.842,.972]  0.982 [.920,1.01]  0.999 [.999,.999]   0/33
RGP-DEQ        0.1270 [.111,.193]  0.353 [.145,.609]  0.674 [.178,1.13]  0.826 [.728,.939]   5/33
```

**This supersedes the old n=4 table** (`FEM 0.016 / 0.847`, `surrogate 0.118 / 0.102`). The
conclusion is unchanged and now much better powered: a converged local FEM solve reproduces the
gate's firing set almost exactly (median Jaccard **0.935**, `dsrx` correlation **0.999**, and it
**never** leaves the wall gate empty), while the learned surrogate agrees on roughly a third of
it and empties the gate on 5 of 33 vessels.

*The n=4 selection-effect caveat below is therefore resolved for this table.* It still applies
to the 5-vessel three-way score comparison above, which needs the `.nas` exports.

Reproduce: `python scripts/publication/generate_flow_diagnostics.py --flow {pred,fem}`.
Artifacts: `outputs/runs/flow_diagnostics{,_fem}.json`.

**Velocity accuracy is close to uninformative about the outcome.** Over 33 vessels with paired
ground-truth and predicted deploy scores, correlation of each candidate diagnostic with the
measured wall-score **drop** (`wall_gt - wall_pred`; **larger = worse**):

```
empty-gate indicator                 +0.745
gate Jaccard                         -0.687
wall-gate firing ratio               -0.251
dsrx correlation                     -0.132
velocity rel-L2                      +0.029
```

Regenerated 2026-09-02, n=33. Every sign now reads the way a health check should: an empty gate
predicts a **larger** drop, better gate overlap predicts a **smaller** one.

> **SIGN CONVENTION -- corrected 2026-09-02.** The previous table read
> `+0.613 / -0.395 / -0.350 / +0.131 / -0.030`. Those **magnitudes are reproduced closely**
> (rel-L2 0.030 vs 0.029; gate Jaccard 0.613 vs 0.687) but the **signs were not on a stated
> convention and were internally inconsistent** -- a gate Jaccard of `+0.613` against a *drop*
> would mean better gate agreement predicts a worse outcome. Do not re-quote the old signs.
> Always state the convention next to the table.

A surrogate can improve its velocity error and move the deployed score not at all. Any paper
reporting a flow surrogate for this pipeline should quote wall gate agreement, not rel-L2.

**The error is in the spatial pattern, not the scale.** An *oracle* monotone remap of the
predicted wall fields onto the ground-truth distribution -- an upper bound on what any
calibration, quantile matching or gain correction can achieve -- raises wall gate Jaccard from
0.339 to only 0.382 (n=30). Rank order at the wall is wrong, so no post-hoc correction recovers
the gate.

**Why the failure is discontinuous.** The readout's physics mask seeds from
`(gate > 0) & wall`. When a surrogate's wall gate fires nowhere the seed is empty, and thirteen
downstream physics/advection/ownership channels become identically zero rather than degraded --
on `patient010`, mask 131 -> 0 nodes and wall F1 0.969 -> 0.000. Measured on the full cohort,
**5 of 33 vessels have a wall gate that fires nowhere** under the surrogate
(`patient010/018/021/028/037`), and the empty-gate indicator is the strongest single predictor
of the drop (+0.745).

**The tolerance is real but vessel-dependent -- do not quote a single threshold.** The blend
curve (`u = (1-a)*u_gt + a*u_pred`), n=3, wall score:

```
             a=0    0.05   0.10   0.20   0.35   0.50  |  0.75   1.00
patient010   0.882  0.901  0.901  0.912  0.907  0.963 |  0.000  0.000
patient020   0.929  0.928  0.932  0.912  0.909  0.897 |  0.695  0.366
patient005   0.715  0.556  0.538  0.510  0.501  0.477 |  0.474  0.486
```

Two vessels hold a plateau to `a~0.5` and then fall -- `patient010` to exactly 0.000, the
empty-gate discontinuity, confirmed by its own `empty_gate = 1`. **`patient005` has no plateau
at all**: it loses 0.16 by `a = 0.05` and is then flat and insensitive out to `a = 1`.

> **RETRACTED SENTENCE, 2026-09-02.** This section previously read: *"that vessel holds F1 ~0.90
> up to ~5% velocity error and falls to zero between 5% and 8%."* That was measured on
> `patient010` alone and does not generalise -- on n=3 the cliff's location and depth vary, and
> one vessel has no tolerance window whatsoever. State the range and show all three curves.

*Scope to state honestly.* The three-way comparison is n=5 and the accuracy table n=4, limited
to vessels carrying a `.nas` mesh (8 of 30), so a selection effect cannot be excluded. The
33-vessel correlation table is the best-powered result here. One discrepancy is unresolved:
median wall `dsrx` correlation is +0.703 across 30 packs but -0.537 on the four mesh-bearing
ones, so which population sets the bar is open.

**Reproduce (all of it, one script):** `bash scripts/publication/_run_flow_requirement_inputs.sh`
then `python scripts/publication/generate_flow_diagnostics.py` ->
`generate_flow_requirement_data.py` -> `plot_flow_requirement.py`.

Individually:
```bash
python scripts/eval_clot_ml_0.py --cohort --flow gt   --out outputs/runs/eval_gt.json
python scripts/eval_clot_ml_0.py --cohort --flow pred --out outputs/runs/pred_all.json
python scripts/diag_flow_sensitivity.py patient010 patient005 patient020 --source pred --out outputs/runs/flow_sensitivity.json
python scripts/publication/generate_flow_diagnostics.py
```
Artifacts: `outputs/runs/{eval_gt,pred_all,flow_sensitivity,flow_diagnostics}.json`,
`outputs/publication/data/flow_requirement.json`, `figures/flow_requirement.pdf`.

> **The reproduction path had been deleted, and was restored 2026-09-02.**
> `scripts/diag_flow_sensitivity.py` and `scripts/diag_wall_gate_health.py` were removed in
> commit `b2eebb9` ("Fix customer_pipeline.py couple unpack error"), which also dropped 182
> lines of diagnostics; `outputs/runs/` was empty, so none of the artifacts this section used
> to name existed either. Both scripts are restored with their stale `clot_ml_v0` imports
> patched. The correlation table and tolerance curve above are **regenerated from scratch**,
> not carried over. `diag_wall_gate_health.py` operates on kinematics selection packs and needs
> an RGP-DEQ checkpoint; the per-vessel diagnostics for the correlation table now come from
> `scripts/publication/generate_flow_diagnostics.py`, which computes them on the biochem packs
> using the consumer's own `lss`/`sgt`, `_flow_hops` stencil and `dsrx_gain`.

---

## 3. Not going in the paper, and why

Recording these so they are not re-proposed later.

**The local kinematic corrector.** Fails quantitatively (worse than doing nothing in the severe
regime: MAE 0.684 vs null 0.630) *and* qualitatively (diversion `cos = -0.142` against FEM
truth, magnitude ratio 0.000). It is a characterised negative result and belongs in
`docs/LOCAL_KINEMATIC_CORRECTOR.md`, not in a paper. Do not present it as a component.

**The physics-informed rebuild (Tier 2).** 12-vessel LOVO gives `corr_log ~ 0` for every arm
and loses to a one-line analytic prior on the case metric. Same disposition.

**"Shielding vs acceleration" as a physics finding.** It is real and the figure is good, but
elevated wall shear at a stenosis throat is textbook haemodynamics. What the sweep actually
demonstrates is that our solver reproduces known behaviour. Use it as a *validation* figure or
as motivation -- never as a claimed discovery.

**"Shear redistribution is elliptic."** Also not novel. Its value is internal: it explains why
two unrelated local architectures both failed, which is a design lesson for us, not a result
for readers.

---

## 4. Open, would strengthen the paper if measured

* **`--cohort` under GT flow.** The oracle-gate conclusion rests on 8 vessels whose baseline
  (0.9844) is well above the cohort-wide ~0.92, so the saturation argument may be an artifact
  of an easy subset. Cheap to settle and it either hardens or qualifies section 1.
* **The oracle under `--flow pred`.** The only regime where flow coupling is still live
  (deploy score collapses to wall 0.586 / off 0.350, i.e. ~0.37 of headroom). Never run.
* **Wound vessels.** The one population with headroom left (0.71-0.90) and the one where the
  oracle was mixed rather than null (w003 +0.032, w002 -0.013, w001 flat). n=3.
* **The 22 deploy vessels with no `.nas` mesh.** Section 2's FEM comparison runs on the 8 that
  have one. Exporting the rest would take the surrogate-requirement result from n=4-5 to n=30
  and settle the `dsrx`-correlation discrepancy noted there. This is the single cheapest thing
  that would harden section 2.

---

## 5. Methods details that must appear, because they change numbers

* **The deploy metric of record is `clot_guiding`.** Deploy Score v2 (`severity`) is materially
  more forgiving -- `patient012` wall reads 0.9679 vs 0.9895 -- and is opt-in via
  `SPECIES_CONTINUOUS_CLOUT_SCORE=severity`. Never mix them in one table.
* **Clot placement must be chosen by geometry, not node ordering.** A clot seeded on the inlet
  is pinned by the Dirichlet BC and cannot reroute anything; such a case looks real and is not.
* **Anchor meshes are quadratic (`MeshTri2`).** Vertices and mid-side nodes alternate along the
  boundary and carry different shear, so raw wall profiles are a sawtooth between two
  interleaved curves rather than noise.
* **The gelation constant `sr/sr0 = 0.1226` is anchored on wound vessels at gelation only.** On
  a synthetic severe-occlusion sweep the case-median ratio spans 0.004-19.7. An
  occlusion-dependent replacement was fitted and rejected (LOVO 0.775 vs const 0.611). Report
  it as a measured constant with a stated validity domain, never as a blockage law.

---

## 6. Retractions

Kept visible so they are not silently re-quoted.

* **"The flux term flips wall-shear correlation to +0.554."** Measured on 15 per-case medians at
  a single viscosity. On the full 12-vessel, per-node corpus it is **-0.124**, with 5 of 12
  vessels the wrong sign. Do not use the +0.554 figure.
* **"`GELATION_SR_RATIO` is a shipped blockage law."** It is not shipped. It is used only by the
  `oracle_blockage` diagnostic (off by default) and by tests; C3' was never built.
* **"Wall F1 holds to ~5% velocity error and falls to zero between 5% and 8%."** (§2) Measured
  on `patient010` alone. On n=3 the cliff's location and depth vary and `patient005` has no
  tolerance window at all. Quote the three curves, not a single threshold.
* **The §2 correlation signs `+0.613 / -0.395 / -0.350 / +0.131 / -0.030`.** Magnitudes
  reproduce; the signs were not on a stated convention and were internally inconsistent. Use
  the 2026-09-02 regeneration and state the convention.

---

## 7. The paper: sections, figures, and where every number comes from

The plan and the argument live in [PUBLICATION_PLAN.md](PUBLICATION_PLAN.md); the novelty triage
in [RELATED_WORK.md](RELATED_WORK.md). **This section is the writing desk**: for each part of the
paper, what claim it makes, which figure carries it, and the exact command that regenerates the
number. If a row has no command, it is not yet a claim.

**Message:** *Learn the chemistry, solve the flow.* The expensive multi-species chemistry is what
a surrogate should replace; the flow is not, and we show that by measurement rather than
assertion.

**Ordering constraint:** §4 (the tool works) comes **before** §5 (the three negative results).
Inverted, the same material reads as a failure catalogue.

All generators live in `scripts/publication/`; `make_all.ps1` runs the set. Figure files use
**semantic names**, and `config.py::paper_map` maps them to paper numbers -- review reorders
figures, so never bake a number into a filename.

---

### 7.0 BLOCKING: our ground truth is not the published model

**Read this before writing a single sentence of §2 of the paper.**

The anchor citation is **Cardillo, G. & Barakat, A. I. (2025)**, *A 2D computational model of
chemically- and mechanically-induced platelet plug formation*, Biomech Model Mechanobiol
**24**(5), DOI [10.1007/s10237-025-01966-3](https://doi.org/10.1007/s10237-025-01966-3);
preprint bioRxiv [2023.01.26.525741](https://www.biorxiv.org/content/10.1101/2023.01.26.525741v1)
(Hydrodynamics Laboratory / LadHyX, École Polytechnique).

Read against the **2023 preprint**, our `.mph` is an **extended variant**, not the published model:

| | Preprint (2023) | Our `phase2_*.mph` (verified 2026-09-02) |
|---|---|---|
| Species | **7**: RP, AP, apr, aps, PT, T, AT | **9**: those 7 **+ FG (fibrinogen) + FI (fibrin)** |
| Thrombus -> flow | **ALE domain deformation**, mesh displacement `= V_plt * M_at / h` | **Viscosity step.** `mu1` = "viscosity platelets", COMSOL Step, location **2.0e7**, 1 -> **80**; plus `mu2` = "viscosity fibrin", location **0.6**, 0 -> **80**. Momentum sees `mu1(Mat) + mu2(FI)` |
| Geometry | backward-facing step w/ 90% stenosis; straight; bifurcation | parametric 2D vessels (stenosis / aneurysm / bend / width) |
| Operating point | inlet 17 cm/s; bifurcation Re 100 / 190 / 250 | **Re = 450**, above the published range |
| Duration | 120 s (validation to ~180 s) | T up to 201 frames |

What is **confirmed identical** and worth saying so: `Omega = APS/APScrit + APR/APRcrit + T/Tcrit`
(chemical activation), `kpa_mech = if(spf.sr > shear_crit, spf.sr/shear_crit, 0)` (mechanical),
`k_pa = kpa_chem + kpa_mech`, and `Sat = 1 - M/M_inf`. This *is* the Cardillo-Barakat model
family -- the "chemically- and mechanically-induced" of the title is literally `k_pa`.

**RESOLVED 2026-09-02 -- see §7.7.** The fibrin species and the viscosity coupling are
Giulia's own ongoing, **unpublished** work on the CFD. Q1 = no (not in the 2025 paper), Q2 =
hers. Q3 (the wound law) follows the same answer unless she says otherwise. The questions are
kept below because they are what had to be asked.

**Three questions for Giulia:**

1. **Does the published 2025 version already carry fibrin (FG/FI) and the viscosity coupling?**
   Preprint-to-publication is exactly where such a change lands. If yes, most of this section
   dissolves and we cite the 2025 paper cleanly. **We only read the preprint** -- this is the
   single cheapest thing to check and it is on you, since you have the PDF.
2. **If not, whose extension is it?** Hers, or this project's? This decides authorship and
   decides whether §2 of the paper says "we surrogate the published model" or "we surrogate an
   extension of it, described here for the first time."
3. **The wound law** (`WoundFlux_9spec` / `SfcRxn_3spec`, absent from `phase2_nowound_*`) -- same
   question, unchanged from [PUBLICATION_PLAN](PUBLICATION_PLAN.md) §1.

**Do not write "we build a surrogate for the published Cardillo-Barakat model" until Q1 and Q2
are settled.** As it stands that sentence is not supported, and it is the kind of claim a
reviewer who knows the lab will check.

### 7.0b The validation lineage is an asset -- use it precisely

The published model was **validated against in vitro data from Nesbitt et al. (2009)**
(aggregate-size evolution in a stenosed micro-channel, no fitting procedure). That is real
experimental grounding, and it substantially answers the "your ground truth is only a
simulation" objection -- **but only for the model that was validated.**

Say exactly this and no more: *the source model is experimentally validated; our surrogate is
validated against the source model.* The validation does **not** transfer automatically to (a)
the viscosity-coupled variant if it differs from the published one, (b) Re = 450, which is above
the published range, or (c) our parametric geometries. Chain-of-evidence, stated honestly, is
worth more than an overreach.

Two scope statements that follow, both cheap to make and expensive to omit:

* **Re = 450 is outside the published operating range** (17 cm/s inlet; bifurcation Re 100-250).
  Our claim is geometry generalization *at our operating point*; we are not asserting the source
  model's validation holds there.
* **Disaggregation.** The source model gives high shear a dual role -- it activates platelets
  *and* disaggregates the plug, and the thrombus grows until occlusion. Our surface law
  **accumulates monotonically** (`Mat` is monotone by construction, PHASE7 §9). If the ground
  truth embolizes and the surrogate cannot, that is a scope limit, and §8 of the paper should
  say so rather than let a reviewer find it.

---

### 7.1 The table

| § | Claim | Figure / table | Status | Regenerate with |
|---|---|---|---|---|
| 1 | COMSOL is ~48 h/vessel; nobody learns the clot field | **Fig 1** pipeline schematic | DRAW | hand |
| 1 | 11 studies, none field-level, none a GNN | **Table 1** the 11 studies | HAVE | Al Bannoud 2026 Table 1, recast |
| 2 | The ground truth: species, `k_pa`, the gate, the viscosity step | **Fig 2** physics + the gate | DRAW | hand; **blocked on §7.0** |
| 3 | Geometry classes are *measured*; the stenosis cut fails | **Fig 3** (narrowing, bulge) plane | **DONE** | `plot_geometry_classes.py` |
| 4 | Architecture | **Fig 4** | DRAW | hand |
| 4 | C0: +0.13 off-wall from ~10 lines, replicated x3 | **Table 2** C0 ablation | TODO | MODEL_REVIEW §9b; no generator yet |
| 4 | **2,156x faster than COMSOL** | **Table 3** staged cost | **DONE** | `generate_timing_data.py` -> `plot_timing.py` |
| 5 | Geometry generalization, out-of-fold | **Table 4** nested-CV OOF | **DONE** | `generate_kfold_table.py` |
| 5 | Untouched confirmation | **Table 5** SEALED, once | TODO | run last, once |
| 5 | Field-level fidelity | **Fig 5** final-time clot maps | HAVE | `plot_fig3_biochem_final.py` |
| 5 | Temporal fidelity | **Fig 6** evolution | HAVE | `plot_fig4_biochem_temporal.py` |
| 5 | The model tracks shape, not vessel identity | **Fig 7** geometry sweeps, 4 axes | TODO run | `run_research_sweep.py` -> `plot_research_sweep_figures.py` |
| 6 | Three independent attempts to learn the flow all fail | **Table 6** the three attempts | TODO | assemble w/ provenance |
| 6 | RGP-DEQ vs FEM vs GT fields | **Fig 8** flow fields | HAVE | `plot_fig1_flow.py` |
| 7 | **rel-L2 is uninformative; the gate is not** | **Fig 9** correlation + tolerance | **DONE** | `_run_flow_requirement_inputs.sh` -> `generate_flow_diagnostics.py` -> `generate_flow_requirement_data.py` -> `plot_flow_requirement.py` |
| 7 | The mechanism | **Fig 10** gate seeding (optional) | DRAW | hand |
| 7 | **The detector: 5/5 caught, 0/33 false alarms** | **Table 7** pre-flight validation | **DONE** | `validate_preflight.py` |
| 8 | Failure modes, stated | **Fig 11** known failures | HAVE | `plot_fig6_failures.py` |
| 8 | Onset timing: 19% early / 45% on-time / 36% late (n=2,510 nodes, 23 vessels) | **Fig 12** onset-timing histogram (new, supplement-first) | **DONE** | `generate_onset_timing_data.py` -> `plot_onset_timing.py` |
| 8 | An error doesn't always compound: wall dips recover, off-wall dips sometimes don't | **Fig 13** error trajectories (new, supplement-first) | **DONE** | `plot_error_trajectories.py` (reuses Fig 5/6/11's own metrics CSVs, no new data) |
| 6 (frozen) | Matched-geometry wound/no-wound example | **Wound A/B** preview, `wound_patient005` vs `patient048` | PREVIEW | `generate_wound_ab_data.py` -> `plot_wound_ab.py`; **not scored, not budgeted -- blocked on §7.0** |

### 7.2 Measured numbers, with their scope

**Table 3 -- cost.** Median **80.2 s** end-to-end (1.34 min), n=30, full horizon, `flow=fem`,
RTX 500 Ada Laptop GPU. IQR [53.5, 110.1] s; range 37.9-226.5 s. Stages: rollout ~69%,
features ~18%, FEM t=0 ~7.5% -- *the classical solve is not the bottleneck*. Against COMSOL's
~48 h: **2,156x**.
*Boundary:* pack -> FEM -> features -> rollout. **Meshing and geometry construction are excluded**;
COMSOL's 48 h covers geometry -> mesh -> solve. Say so, or add the meshing cost.
*Also state:* training cost is one-time; break-even is a handful of vessels at 48 h each.

**Table 4 -- geometry generalization.** 23 vessels, 5 folds, out-of-fold, final time, SEALED
untouched.

| class | n | wall (mean +/- SEM) | off-wall |
|---|---|---|---|
| baseline | 19 | **0.9070 +/- 0.0248** | 0.6574 +/- 0.0575 (n=12) |
| stenosis | 3 | **0.8305 +/- 0.0540** | 0.7806 +/- 0.0611 |
| aneurysm | 1 | 0.9507 (n=1) | 0.9097 (n=1) |

> **READ THE ERROR BARS BEFORE WRITING THE SENTENCE.** SEM is across vessels within the class.
> Baseline and stenosis wall differ by 0.077 with a combined SEM of ~0.059 -- **the classes are
> statistically indistinguishable**, and the same holds off-wall. That is exactly the claim the
> paper wants (*performance holds across geometry class*), and it is **not** licence to say
> stenosis is harder, or that aneurysm is best. Quote the overlap, not the ordering.
>
> Two dispersions, two questions, and they are not interchangeable (standing rule 3): **SEM /
> CI95** are across vessels -- would another draw of vessels move this mean. The **config noise
> floor** (+/-0.0037 wall, +/-0.0432 off; MODEL_REVIEW §9f) is across refits -- would another
> seed move it. Both are emitted by `generate_kfold_table.py`; name which one a stated
> uncertainty is.

*Two caveats that must appear in the caption.* (i) The OOF archive is exported under **GT t=0
flow**; §7's result is what licenses reading it as the deployed FEM configuration, since FEM sits
inside noise of GT (0.705 vs 0.710). (ii) **Aneurysm is n=1** -- one vessel, not a class mean;
n=1 again in SEALED.

> **Off-wall means exclude empty-GT vessels, and this matters.** 7 of the 19 baseline vessels
> carry wall clot but **zero** off-wall GT, so their off-wall F1 scores false positives, not
> recall. Including them gave **0.5749**; excluding them gives **0.6574** (the 7 score 0.4334 as
> a separate FP row). This is the same rule the clot-free protocol enforces
> (MODEL_REVIEW §8b) -- it was being applied to the 8 clot-free vessels and violated *within*
> the clot-carrying ones. **Any off-wall number quoted before 2026-09-02 is wrong.**

**Fig 9 -- the flow requirement.** See §2 above for the full table and both corrections. Headline:
velocity rel-L2 `r = +0.029` against the drop; gate statistics `0.69-0.75`; 5 of 33 vessels have
an empty wall gate.

**Fig 3 -- geometry classes.** 52 packs. Independently reproduces the documented failure of the
measured stenosis cut: **5 undesignated vessels sit below the most-open designated stenosis
(0.58)**, where §5 of this file cites one. Report the failure; it costs a panel and buys trust.

### 7.3 Threats to the claim, from the literature

Checked 2026-09-02. Each is a real objection; each has an answer, and the answer must be *in the
paper*, not discovered at review.

1. **"Mesh GNNs already generalize across geometry."** True, and this is the biggest framing
   risk. Geometric generalization with GNN surrogates is an established method -- MeshGraphNets
   and successors, [edge-augmented and multi-GNN architectures](https://www.nature.com/articles/s41598-024-53185-y)
   that "generalize well to unseen domains, boundary conditions and materials", and
   [GNN surrogates for parametrized PDEs handling geometric variability](https://www.researchgate.net/publication/376454313).
   **So geometry generalization is NOT our methodological novelty.** Ours is (a) the
   application -- first field-level surrogate of a continuum multi-species thrombosis model --
   and (b) the flow-requirement result. Claim those; cite the mesh-GNN line as established
   method rather than competing with it.
2. **"Published PI-GNNs report better wall shear than yours."** R = 0.94, 7.6% directional WSS
   ([RELATED_WORK](RELATED_WORK.md) §2). Answer: our claim is about the *metric*, not surrogate
   capability -- those papers all report in the norms we show to be uninformative. Frame §7 as
   the gated-coupling claim, never as "surrogates cannot do this".
3. **"Your ground truth is only a simulation."** Answer: the source model is validated against
   in vitro data (Nesbitt 2009). Use the chain-of-evidence framing in §7.0b -- and only that.
4. **"Which model, exactly?"** The §7.0 discrepancy. Unanswered, this is the most damaging
   question on the list, because it goes to whether the paper describes what it says it does.
5. **"How credible is the model for its stated use?"** Consider framing the paper's evaluation
   in [ASME V&V 40](https://www.asme.org/codes-standards/find-codes-standards/assessing-credibility-of-computational-modeling-through-verification-and-validation-application-to-medical-devices)
   terms -- *question of interest*, *context of use*, *quantity of interest*, credibility
   proportional to model risk. It is FDA-recognized, it is the vocabulary a biomedical
   reviewer/editor knows, and adopting it costs a paragraph. We are not making a regulatory
   claim; we are signalling that we know what evidence a surrogate owes its user.

### 7.4 Still to run, in order

1. **Fig 7** -- the four main sweeps (`CONFIG.main_sweeps`). Cheap, no COMSOL, strongest visual
   for the geometry claim.
2. **Table 2** (C0) and **Table 6** (the three attempts) -- currently transcriptions from
   MODEL_REVIEW §9b and §3 of this file. Write small generators or assemble at draft time with
   provenance; do not hand-copy constants into the manuscript.
3. **Table 5 -- SEALED, once, last**, after every other choice is frozen.
4. **Figs 1, 2, 4, 10** -- schematics. Fig 2 is blocked on §7.0.
5. **Blocked on Giulia:** §7.0 Q1-Q3. The wound section and its five sweeps stay frozen.

### 7.5 Rules that survive into the manuscript

* `clot_guiding` is the metric of record; **never** mixed with Deploy Score v2 in one table.
* Clot placement is chosen by geometry, never by node ordering.
* Anchor meshes are quadratic (`MeshTri2`): raw wall profiles are an interleaved sawtooth, not
  noise.
* `sr/sr0 = 0.1226` is a measured constant with a stated validity domain, **not** a blockage law.
* Empty-GT domains score false positives, not recall, and never enter a recall-bearing mean.
* Every correlation carries its sign convention; every domain mean names the vessels in it.
* Every number in the draft resolves to a command in §7.1.

### 7.6 Citations to have ready

* **Cardillo & Barakat 2025**, BMMB 24(5), DOI 10.1007/s10237-025-01966-3 -- the ground truth.
* **Nesbitt et al. 2009** -- the in vitro data the source model was validated against.
* **Al Bannoud et al. 2026**, J Thromb Thrombolysis 59:727-745, DOI 10.1007/s11239-025-03222-y --
  the PRISMA review that defines the gap (11 studies, none field-level, none a GNN).
* **Duraisamy**, *Predictivity and Utility of Neural Surrogates of Multiscale PDEs*,
  arXiv 2604.20061 -- predictivity vs utility; the position our §7 measures.
* **Grossmann et al. 2024**, IMA J Appl Math 89(1):143 -- can PINNs beat FEM (no).
* **Mesh-GNN surrogate line** -- Sci Rep 2024 (edge-augmented / multi-GNN, unseen domains);
  MeshGraphNets; GNN surrogates for parametrized PDEs with geometric variability. Cite as
  established method, per §7.3(1).
* **PI-GNN hemodynamics** -- Sci Rep 2026 stenotic coronary WSS (R = 0.94); arXiv 2212.05023
  SE(3)-equivariant mesh NN (7.6% WSS). The §7.3(2) comparison.
* **ASME V&V 40-2018** -- credibility framing, if §7.3(5) is adopted.

### 7.7 Ground-truth provenance: RESOLVED, and what it costs

**Answered 2026-09-02: the fibrin species and the viscosity coupling are Giulia's own ongoing
work on the CFD, and are UNPUBLISHED.** So §7.0's Q1/Q2 are settled: our ground truth is
Giulia's *current, unpublished* extension of the published Cardillo-Barakat model, not the 2025
paper as printed.

Consequences, all of which are manageable but none of which are automatic:

* **Giulia is a co-author.** Not optional -- the paper's ground truth is her unpublished model.
* **The paper must fully describe the extended model in Methods**, because a reader cannot look
  it up: the 9 species, `mu1(Mat) + mu2(FI)` with both Step functions and their locations
  (2.0e7 and 0.6, each 1/0 -> 80), and the wound boundary condition if that section survives.
  Put the full specification in supplementary. A surrogate paper whose ground truth cannot be
  inspected is not reproducible, and this is the fix.
* **Sequencing is now a real decision, and it is Giulia's call as much as ours.** Three options:
  (a) she publishes the extended model first and we cite it; (b) we cite it as *in preparation*
  and describe it in full; (c) co-submit as companion papers. **(a) is cleanest and (c) is
  strongest**; (b) is workable but weakest, since a reviewer cannot check the ground truth.
  Raise it with her early -- it affects our submission timeline, not just our citations.
* **The validation chain gets one link longer.** Nesbitt (2009) validated the *published* model.
  The extension is not covered by that validation, and neither is Re = 450. State the chain
  precisely (§7.0b) and do not let "validated against in vitro data" drift onto our variant.

*What this does NOT change:* the surrogate contribution stands regardless. Surrogating a
lab-internal model is normal and publishable, provided the model is specified.

---

### 7.8 Cheap ways to make the paper stronger, ranked by value per unit effort

Assessed 2026-09-02 against the constraints: no new COMSOL data, geometry is the generalization
axis, 2D, small n.

**1. DONE -- the wall-field table is now n=33, not n=4.** ~8 minutes of compute closed the
single largest stated weakness in §2 ("the accuracy table is n=4 ... a selection effect cannot
be excluded"). See §2. **This was the highest-value item on the list and it is spent.**

**2. DONE -- the wall-gate pre-flight check is implemented and validated (§7.9).**
5/5 detection, 0/33 false alarms. Original rationale kept below.

*Low lift, highest novelty remaining.*
The paper diagnoses a failure mode (an empty wall gate zeroes the readout) and shows the
diagnosis is predictive (gate statistics correlate 0.69-0.75 with the drop; rel-L2 does not).
The obvious next move is to **operationalise it**: before trusting a prediction on a new vessel,
compute wall-gate health on the flow you are about to deploy with, and refuse or flag when the
gate is empty. Evidence is already in hand -- FEM passes **33/33**, RGP-DEQ fails **5/33**.
This converts a negative result into a shipped safety feature and lets the paper say *we do not
merely characterise the failure mode, we detect it before it costs a prediction.* Reviewers
reward that, and the computation already exists in `generate_flow_diagnostics.py`.

**3. DONE -- error bars are on Table 4 (§7.2).** SEM across vessels plus the config noise
floor, both emitted by `generate_kfold_table.py`. The result changed the reading: the classes
are statistically indistinguishable, which supports the paper's claim and forbids an ordering
claim. Original rationale below.

*Nearly free.* The floor is measured
(+/-0.0037 wall, +/-0.0432 off). Put it on Table 4 and on every score in the text. It preempts
"is 0.9070 vs 0.8305 a real difference?" -- which, at n=19 vs n=3, a reviewer will ask.

**4. Make the speedup concrete with a design study.** *Nearly free -- Fig 7 already runs it.*
Do not stop at "2,156x". Say: *this four-axis sweep is N arms; at ~48 h each it is X COMSOL-days
of solve, and it completed in Y minutes.* A ratio is abstract; "a study that was previously
infeasible" is the actual contribution, and it costs one sentence over work already planned.

**5. Ensemble uncertainty.** *Medium-low lift, real added value.* `predict_scores`
(`src/clot_ml/locked.py:75`) averages ensemble members and discards the spread. Returning the
per-member stack gives a per-node predictive uncertainty for free, and the paper can then show
uncertainty is elevated where the model is wrong. Surrogate users need to know when to trust an
output, and the neural-PDE literature explicitly calls for calibration-aware UQ. The cost is a
small change to shipped inference plus one figure -- the only item here that touches deploy code,
so weigh it accordingly.

**6. An operating-point curve instead of a bare 0.5 cut.** *Low lift.* Precision-recall across
thresholds from the OOF archive, with the chosen operating point marked. Preempts "why 0.5?",
and a threshold-free number (AUC-PR) is more robust to the class imbalance here (~150 positives
against ~15k nodes) than F1 at one cut.

**Deliberately NOT recommended.**

* **Bolting a new component onto the clot GNN to look novel.** The novelty is the application
  plus the flow-requirement finding. An unmotivated architectural gadget invites "why is this
  here?", dilutes the message, and costs a re-validation of every number in Table 4. If the
  architecture needs a story, C0 already is one -- a ~10-line training constraint with a
  measured +0.13 and a replication, honestly reported including that its registered mechanism
  was wrong.
* **3D.** Not low-lift, and the ground truth is 2D.
* **Re-opening SEALED.** Buys one vessel, spends the only untouched evidence in the project.
* **Chasing a competitive PI-GNN baseline.** Genuinely strengthens §7.3(2), but it is a
  reimplementation, not a low-lift item. Park it as the answer if a reviewer presses.

### 7.9 The pre-flight gate check -- IMPLEMENTED, and it is a paper contribution

`src/clot_ml/preflight.py`, wired into `predict_clot_ml_0(..., preflight=...)`, tested in
`src/tests/test_preflight_gate.py` (8 tests), validated by `scripts/validate_preflight.py`.

**The design insight worth a paragraph in the paper.** §2 shows the empty-gate indicator is the
*strongest* predictor of the drop (r = +0.745) while velocity rel-L2 is the weakest (+0.029).
The useful accident is that the predictive statistics are exactly the ones needing **no ground
truth**: gate Jaccard, `dsrx` correlation and rel-L2 all require a reference field and are
unavailable on a new vessel, but *the firing set of the gate is self-contained*. **So the
diagnosis is deployable.** We can refuse a vacuous prediction before paying for the rollout.

That is what turns §7 of the paper from a characterisation into a contribution: we do not merely
explain the failure mode, we ship the detector for it.

**Measured detector performance** (`outputs/runs/preflight_validation.json`):

```
                              FAIL   WARN   PASS     fire_frac min/med/max
learned surrogate (RGP-DEQ)      5      6     22     0.0000 / 0.0917 / 0.4544
local FEM (shipped path)         0      0     33     0.0428 / 0.1245 / 0.4416

empty-gate vessels under the surrogate : 5
caught                                 : 5/5
FALSE ALARMS on the shipped FEM path   : 0/33
```

**5/5 detection, 0/33 false alarms.** The 6 WARNs on the surrogate are genuinely marginal
vessels, not misfires -- WARN never blocks a rollout.

**Calibration**, from the 33-vessel cohort (wall-node firing fraction):

```
GT   min 0.0465   p5 0.0563   median 0.1322   max 0.4286   empty 0
FEM  min 0.0428   p5 0.0569   median 0.1245   max 0.4416   empty 0
pred min 0.0000   p5 0.0000   median 0.0917   max 0.4544   empty 5
```

Bounds are `FIRE_FRAC_MIN = 0.040` / `FIRE_FRAC_MAX = 0.460`, set just outside the GT/FEM
envelope so a vessel is flagged when its gate behaves unlike anything the model was fitted or
validated against -- not merely when it is unusual. `test_thresholds_bracket_the_measured_cohort`
pins that property, so a future retune cannot silently start false-alarming on good flow.

**Verdicts.** `FAIL` = the wall gate fires nowhere; the seed `(gate > 0) & wall` is empty, the
rollout returns identically-zero channels, and the prediction is vacuous rather than degraded.
`WARN` = fires, but on a fraction of the wall outside the reference envelope -- inspect, do not
discard. Default mode is `"warn"` (print and continue, so no existing caller changes behaviour);
`"raise"` is for deployment, where a vacuous prediction is worse than a refusal.

**One definition of "the gate", shared.** `wall_gate_firing()` is used by both the shipped check
and `scripts/publication/generate_flow_diagnostics.py`, so the diagnostic and the deployed
detector can never drift apart about what firing means.

*Honest scope for the paper:* the detector is validated on 33 vessels at one operating point,
against two flow sources. It detects the discontinuous failure (empty gate) with certainty
because that is a property of the seed, not a learned threshold; the WARN band is an
envelope check and is only as good as the cohort that calibrated it.
