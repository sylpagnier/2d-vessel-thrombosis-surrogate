# Publication notes

Running notes for the paper. **Claims go in here only once they are measured**, with the script
that produced them, so a number in a draft can always be traced back to a command.

Two standing rules, both learned the hard way in this repo:

1. **Quote the statistic that was actually computed.** Per-node and per-case means of the same
   quantity are different numbers; the metric of record (`guiding`) and Deploy Score v2
   (`severity`) are different scores. Every entry below names its scope.
2. **A result measured on n<10 vessels at one operating point is a hint, not a finding.** This
   file has already had to retract one such number (see *Retractions*).

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
0.990   clot_ml_v0, open loop                      +0.087 ABOVE the oracle backbone
0.983   clot_ml_v0 + ORACLE                        -0.0065, inside the noise floor
```

**The learned model already sits above the oracle-corrected physics backbone.** Closing the
loop supplies information the model has already recovered from t=0 geometry and shear, so it
adds distribution shift and no signal. The loop is therefore left open by measurement, not by
omission.

*Scope to state honestly in the paper:* this bounds the non-wound cohort under GT t=0 flow,
where the model is saturated (0.984-0.994) and headroom is 0.010 against a +/-0.024 noise
floor. It does not bound the wound vessels (n=3, mixed) or the predicted-flow regime.

Reproduce: `CLOT_ML_ORACLE_BLOCKAGE=1 python scripts/eval_clot_ml_v0.py`
Artifacts: `outputs/diag_oracle_blockage_{off,on}.json`

---

## 2. How accurate must a t=0 flow surrogate be?

**Status: the requirement is measured; the headline comparison is n=5 and needs widening
before it is a finding rather than a strong hint (standing rule 2).**

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
mean               0.710   0.705       0.362      (wall, clot_ml_v0)
```

FEM sits inside noise of ground truth. So the readout is **not** hypersensitive to reconstructed
flow, and the -0.35 collapse is attributable to surrogate field accuracy alone.

**The requirement is on the derived wall fields, not on velocity.** Same two flows, wall nodes,
the consumer's own MLS stencil (`hops=3`), n=4 vessels:

```
              relL2 u   sr rel   sr corr   dsrx rel   dsrx corr   gate Jaccard
FEM             0.016    0.082     0.993      0.142       0.995          0.847
surrogate       0.118    0.489     0.431     -1.230*     -0.537          0.102
```

`*` the surrogate's `dsrx` error exceeds the signal it is estimating.

**Velocity accuracy is close to uninformative about the outcome.** Over 33 vessels with paired
ground-truth and predicted deploy scores, correlation of each candidate diagnostic with the
measured wall-F1 drop:

```
gate Jaccard (fraction of ceiling)   +0.613
wall-gate firing ratio               -0.395
empty-gate indicator                 -0.350
dsrx correlation                     +0.131
velocity rel-L2                      -0.030
```

A surrogate can improve its velocity error and move the deployed score not at all. Any paper
reporting a flow surrogate for this pipeline should quote wall `sr`/`dsrx` agreement, not
rel-L2.

**The error is in the spatial pattern, not the scale.** An *oracle* monotone remap of the
predicted wall fields onto the ground-truth distribution -- an upper bound on what any
calibration, quantile matching or gain correction can achieve -- raises wall gate Jaccard from
0.339 to only 0.382 (n=30). Rank order at the wall is wrong, so no post-hoc correction recovers
the gate.

**Why the failure is discontinuous.** The readout's physics mask seeds from
`(gate > 0) & wall`. When a surrogate's wall gate fires nowhere the seed is empty, and thirteen
downstream physics/advection/ownership channels become identically zero rather than degraded --
on `patient010`, mask 131 -> 0 nodes and wall F1 0.969 -> 0.000. Blending ground-truth and
predicted flow shows the same shape: that vessel holds F1 ~0.90 up to ~5% velocity error and
falls to zero between 5% and 8%.

*Scope to state honestly.* The three-way comparison is n=5 and the accuracy table n=4, limited
to vessels carrying a `.nas` mesh (8 of 30), so a selection effect cannot be excluded. The
33-vessel correlation table is the best-powered result here. One discrepancy is unresolved:
median wall `dsrx` correlation is +0.703 across 30 packs but -0.537 on the four mesh-bearing
ones, so which population sets the bar is open.

Reproduce: `python scripts/eval_clot_ml_v0.py --cohort --flow {gt,pred,fem}`; per-checkpoint
wall-gate health via `scripts/diag_wall_gate_health.py`; the tolerance curve via
`scripts/diag_flow_sensitivity.py`.
Artifacts: `outputs/runs/eval_gt.json`, `outputs/runs/pred_all.json`,
`outputs/runs/fem{2,3}.json`, `outputs/runs/flow_sens_pred.json`.

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
