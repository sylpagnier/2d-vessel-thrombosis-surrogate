# MODEL REVIEW — the whole stack, 2026-08-22

Written after reading [WOUND_PROGRESS.md](WOUND_PROGRESS.md), [PHASE10_V4.md](PHASE10_V4.md),
[PHASE7_FINDINGS.md](PHASE7_FINDINGS.md), [SEALED_SPLIT.md](SEALED_SPLIT.md) and the shipped
code (`src/clot_ml/*`, `src/core_physics/physics_wall_model.py`). Scope: what `clot_gnn_v4w`
is, whether the physics under it is right, and where the remaining deploy score actually is.
Everything here is either a citation to an existing measurement or a check run during this
review; nothing is a new experiment.

> **CURRENT STATE — read §8 first.** Several changes landed *after* the body of this document
> was written, and they matter for how to read it:
>
> * **Every score in §0 and in the body below is PRE-REPAIR** and superseded by §8f. The
>   caches and all three artifacts have been rebuilt; the mechanisms and negative results
>   below still stand.
> * **The cohort is different.** 23 clot-carrying vessels (VIZ_HALF released), plus 8
>   clot-free vessels for false-positive scoring. SEALED is `007/013/031/043` only. So "19
>   vessels" anywhere below means the *old* pool, and the ±0.024 / ±0.091 noise floor was
>   measured on it — do not carry it forward by assumption.
> * **SHIPPED: `clot_gnn_v5w`** (§9f) — final wall **0.9203**, off **0.7078**, readout gap
>   **0.045**, and the config floor tightened to **±0.0037 wall / ±0.0432 off**. Mean-over-time
>   is unchanged; quote C0 as a final-time result.
> * **Phases A, B and C0 are complete.** Phase B re-baselined at wall 0.9008 / off 0.5812
>   (23 clot-carrying vessels, floor **±0.024 / ±0.074**) and localised the whole off-wall
>   deficit to the **readout**, not the model (§8f). **C0 then closed it** (§9b): a ~10-line
>   training-time constraint on the logit spread takes final-time off-wall to **0.7078** and
>   the readout gap from **0.193 → 0.045**, replicated on all three configurations. Shipped.
> * **Quote C0 as a FINAL-TIME result.** Mean-over-time is unchanged (0.5713 → 0.5792) —
>   §9b.7. And it works by fixing the implied-burden tail, **not** the spread it was designed
>   to fix (§9b.5); the registered mechanism was wrong and the effect replicated anyway.
> * **Priority-class labels are correct again** — A2 (§8d) made the human designation
>   authoritative. But the *measured* stenosis cut is dead: it cannot separate the
>   labelled stenoses from `patient012`, so an UNLABELLED stenosis would be missed.
> * **The Phase B pool is 23 clot-carrying + 8 clot-free = 31.** Seven FIT/DEV vessels
>   are truncated below `MIN_T` and stay excluded (003/004/008/009/011/015/**039**) —
>   039 is an aneurysm, so `patient040` is still the only non-SEALED one.
>
> The mechanisms, the diagnoses and the negative results below all stand. The **numbers**
> need re-running, and §8 says in what order so the expensive step is paid once.

> **HEADLINE.** The modelling space has been mined honestly and is close to exhausted *at the
> current scope*. But the scope excludes the largest error source in the project: **every
> number in every phase doc is computed on COMSOL's own t=0 velocity field**, and the one
> time the deployable alternative was measured — on the physics backbone, PHASE7 §10.7 — it
> cost **−0.34 wall** and **−0.19 off-wall**. `clot_gnn_v4` has *never* been run this way.
>
> That measurement is **runnable today, on the full 19-vessel pool, with no new data and no
> retraining**: all 26 FIT+DEV packs now carry `u0_pred` (PHASE7's "012/041/044 carry no
> `u0_pred`" is stale — verified this session). Three code defects must be fixed first,
> because `build_sample(..., flow="pred", variant="v4")` currently returns a **GT-contaminated
> sample** (§6.1).
>
> Ranked by expected size, the remaining levers are:
> **(1) the flow input, ~0.3;  (2) closing the ODE loop, unmeasured but the only physics
> route with a measured mechanism;  (3) the readout, ~0.03 wall / ~0.09 off and mostly
> unreachable on this cohort;  (4) more vessels, which gates (2) and (3).**
> The project has been spending its effort on (3).

---

## 0. WHAT IS SHIPPED, AND WHAT IT SCORES

| | |
|---|---|
| pointer | `data/reference/clot_gnn_locked.json` → `clot_gnn_v4w`, kind `temporal_v4_wound` |
| set | 9-member GNN ensemble (3 configs × 3 seeds) on the v5 advective-transport cache |
| readout | per-domain, in-fold: `resid_adapt` on the wall, `expected_tuned` off-wall, `commit_final` |
| timing | 4-seed time-conditioned head + 3-seed off-wall lag, ODE-anchored |
| wound | +2 scalars (`G_pre=1.98`, `G_post=14.28`) into an ungated surface ODE; bit-identical to v4 on wound-free packs |
| backbone | accumulate-only surface ODE, gate frozen at t=0, `da_scale=40`, no washout |

Strict 5-fold CV, 19 vessels, **GT t=0 flow**:

```
                        mean wall   mean off   FIN wall   FIN off
clot_gnn_v4                0.8750     0.7188     0.9176     0.7372
oracle timing              0.9662     0.8709     0.9176     0.7372
per-vessel oracle cut           —          —     0.9447     0.8275
```

Held-out reality check (VIZ_HALF, [SEALED_SPLIT.md](SEALED_SPLIT.md)): wall **0.7179** (p042)
/ **0.7406** (p001) against a pool mean of 0.9176 — ranks 2 and 4 of 21, exact rank p=0.029.
Off-wall generalised exactly as CV predicted (0.696 / 0.785, 31st and 54th percentile).

Wound cohort, n=3, GT t=0 flow, LOVO constants: `w_reg` FINAL **0.878**, MOT **0.837**;
`far` unchanged across all arms (the regression check).

**Read those three blocks together.** CV says 0.9176. The only genuinely held-out wall
evidence says 0.72–0.74. And neither number was produced with a deployable flow field.

---

## 1. THE FLOW INPUT — the largest unmeasured quantity in the project

### 1.1 The whole stack defaults to `flow="gt"`

`src/clot_ml/locked.py` threads `flow: str = "gt"` through `build_sample`,
`predict_clot_series`, `predict_temporal_v3/v4/v4_wound` and `predict_default_series`.
`src/clot_ml/features.py:build_features` reads `data.y[0, :, 0:2]` — COMSOL's solved
velocity — and MLS-differentiates it for `sr` and `d(sr,x)`, which **are the gate's own
arguments**. `src/clot_ml/transport.py` solves the upwind operator on the same field.

That is the stated scope and it was a legitimate scope for Phases 6–10. It is not the
deploy scope: a surrogate that replaces COMSOL does not get COMSOL's velocity field.

### 1.2 The one time it was measured, it cost 0.34

PHASE7 §10.7, physics backbone (`predict_wall_clot`, hops=20 + speed lumen), 15 vessels:

```
                GT t=0 flow      RGP-DEQ u0_pred
wall deploy    0.858 / 0.890      0.515 / 0.505      (FIT / DEV)
off deploy     0.365 / 0.505      0.175 / 0.265
full mesh      0.801 / 0.807      0.461 / 0.443
```

PHASE9 §... and [WALL_MODEL_PLAN.md](WALL_MODEL_PLAN.md) both cite the −0.34, and both then
proceed on GT flow. **No document in this repo reports `clot_gnn_v4` under predicted flow.**

The diagnosed cause is Stage-A quality on *this* cohort: Rel L2 vs GT t=0 is **0.18 on
p005 and 0.39–0.60 on the rest** (040/041/044: 0.54 / 0.48 / 0.45). That is badly at odds
with `AGENTS.md`'s production allfix figure of **Rel L2 ~0.087**. Either the precache used a
different checkpoint, or the biochem cohort is out-of-distribution for the kinematics model.
**Resolving that discrepancy is the cheapest possible first move and may be worth most of
the 0.34 on its own.**

### 1.3 Why v4 in particular is exposed

v3 → v4's gain came from two sources of roughly equal size (PHASE10 §5, §10). Both are
velocity-differential quantities:

* the **advection operator** keys off `ubar·dhat` per edge, and the residence time `tau` and
  the finite-horizon cap are pure velocity integrals. A 40–60% Rel L2 error does not perturb
  `tau`, it reorders it — and `tau` is what produces the 0.16 attenuation without anyone
  writing 0.16 down;
* the **gate** is `sr < lss` and `d(sr,x) < sgt` — a *first and second* derivative of the
  velocity field. PHASE7 already measured that consuming the kinematics shear head directly
  (`sr0_pred`) drops FIT wall to 0.240, and that only MLS-on-`u0` keeps wall corr at 0.82.

So the honest prior is that v4's advantage over v3 is *smaller* under predicted flow, not
larger, and possibly negative. That is worth knowing before another round of readout work.

### 1.4 What to run — three steps, no new data

```bash
# 0. fix the GT contamination in the v4 feature block first (6.1) -- MANDATORY
# 1. build the deploy-faithful cache
python scripts/build_clot_ml_cache_v4.py --flow pred --out outputs/clot_ml_cache_v5_pred
# 2. score the SHIPPED weights on it -- no retraining, pure input swap
python scripts/eval_strict.py          --tags v5a,v5b,v5c --cache v5_pred
python scripts/eval_strict_temporal.py --arms v5a,v5b,v5c --cache v5_pred
# 3. the paired delta, which is the number that matters
python scripts/eval_significance.py --a v5a,v5b,v5c --b v5a,v5b,v5c --cache v5 --cache-b v5_pred
```

Then, and only then, decide whether the GNN needs **retraining under predicted flow**. Note
this is a covariate shift as well as a quality problem: the ensemble has only ever seen
GT-flow features, so even a *good* `u0_pred` presents it with a different input
distribution. A pred-flow-trained (or GT/pred-mixed) ensemble is a different, cheap arm.

### 1.5 A robustness prior the feature set does not currently have

If `u0_pred`'s error is largely a smooth scale/bias error (wall corr 0.82 supports this),
then the *absolute* thresholds are the fragile part: `sr_over_lss`, `dsrx_over_sgt`,
`gate_low`, `gate_sep` all compare a predicted quantity against a fixed physical constant.
**Within-vessel rank/percentile forms of `sr` and `d(sr,x)` are invariant to exactly that
error class** and cost nothing to add. PHASE10 §16.2 already found the shear *rank* to be the
best available ordering for stitch onset (+0.809 FIT), so the rank form is known to carry
signal. This is the one feature-engineering idea in this review that is motivated by a
measured error mode rather than by a hunch.

---

## 2. THE EQUATION IS SHORT A TERM, AND THE LOOP IS OPEN

### 2.1 What is already established

* **The surface ODE cannot order GT `Mat` even with perfect inputs.** Handed GT `RP`, `AP`,
  `M`, `Mas`, `sr`, `d(sr,x)` at every timestep, accumulate-only ranks **0.310** and is
  *anti*-correlated on 5 of 19 vessels (PHASE7 §9.2). No `da_scale` and no input model
  crosses this; it is a ceiling on the equation.
* **The missing structure is removal.** `−λ·sr·Mat` takes the same oracle to **0.464**
  in-sample, **0.447** LOVO, with 16 of 19 vessels picking the same `λ`. The nulls (`1/sr`,
  `−sr`, `J0/sr`) reach 0.271–0.287, so it is not a shear correlate in disguise. Saturation
  buys exactly nothing.
* **And it only works with flow *and* chemistry evolving** (PHASE7 §9.4):

  ```
  inputs                accumulate-only   with washout      Δ
  frozen both                 0.219          0.097       −0.123
  evolving flow only          0.395          0.356       −0.039
  evolving chemistry only    −0.026         −0.078       −0.052
  evolving BOTH               0.310          0.464       +0.153
  ```

  Accumulation is precisely what let the frozen-input approximation survive: a constant
  source against a linear sink has one attractor, and its ordering is the `1/sr` null.

### 2.2 The cell that has never been run

`scripts/eval_flow_washout_2x2.py` ran this cross on the real model path and every cell read
**0.765 ± 0.001** — because on that path *only the gate evolves*; `ap`/`rp` stay at t=0. So
the +0.153 cell has been measured on oracle inputs and **never on the deploy path**. It is
the only physics route in this repo with a measured mechanism, a measured magnitude and an
unmeasured deploy number.

### 2.3 The wound proves the frozen gate breaks, and proves it breaks *self-referentially*

WOUND_PROGRESS §3.3 is the cleanest evidence in the project that the t=0 approximation has a
limit, and it is not a calibration limit:

```
t (s)              0     1500   3000   3750   4500   6000
wound gate ON      0%      0%     0%     53%    94%    95%
wound sr (/s)    148     148    148      25     18     18
```

Ungated deposition → `Mat` crosses gelation → `mu1` steps 80× → near-wall flow stalls →
`sr` collapses → **the ordinary low-shear gate opens on top of the wound law**. The input
moves because of the model's own output. On a healthy wall this loop cannot start, which is
exactly why the frozen approximation survived for ten phases.

### 2.4 The machinery to close it already exists

Nothing here needs to be built from scratch:

| piece | where | status |
|---|---|---|
| gate-updating hook inside the ODE | `integrate_mat_trajectory(..., blockage=)` | implemented, unused on the deploy path |
| clot→flow blockage | `corrector_blockage` in `physics_wall_model.py`; `LocalKinematicCorrector` | implemented |
| removal term | `integrate_mat_trajectory(..., washout=, washout_sr=)` | implemented, `0.0` by default |
| self-consistent regime switch | `mat_trajectory_torch` in `src/clot_ml/wound.py` | implemented and differentiable |
| cheap flow-dependent transport re-solve | `src/clot_ml/transport.py` | linear operator, ~20 s/vessel/time |

**Proposal — a staged closed-loop rollout, evaluated as one arm rather than one term at a
time** (§9.4 says the three-way interaction makes term-at-a-time evaluation invalid):

```
t=0 flow  →  ODE(gate_t, ap_t, rp_t, washout)  →  Mat(t)  →  mu_eff(t)
          →  corrector re-solves flow           →  sr(t), d(sr,x)(t) → gate_{t+1}
          →  transport re-solve on the new flow → off-wall channels
```

**Test it on the wound vessels first.** This is the counter-intuitive but correct call: n=3
is hopeless for *fitting* anything, but it is the only regime in the repo where the effect of
closing the loop is **0% → 95%**, i.e. an order of magnitude above the ±0.024 / ±0.091 noise
floor. A mechanism that cannot be detected on 19 healthy vessels can be falsified decisively
on 3 wounded ones. Establish it there, then port it.

### 2.5 What NOT to redo

PHASE10 §11's "advective recurrence" failure is **not** evidence against §2.4. That arm fed
the GNN upwind-weighted *feature* channels and lost the ensemble (0.8958/0.7016 against
0.9176/0.7359). It is a representation change, not a physical closure. The two should not be
conflated in future write-ups.

---

## 3. THE READOUT — the accounting, and the one clean untried variant

### 3.1 What is left

```
                       wall      off
shipped v4           0.9176   0.7372
per-vessel oracle    0.9447   0.8275     <- upper bound (best of 33 cuts per vessel)
best prefix, oracle k     —   0.8205
```

so ~**+0.027 wall / +0.090 off**, and both are *upper* bounds.

### 3.2 The three failure classes, correctly separated

`diag_score_field_shape.py` is the sharpest diagnostic in the repo, and it splits the wall
error cleanly:

* **Ranking failure** (p042): AUC 0.9610, **z = −12.95** against the pool. No readout helps;
  its per-vessel oracle only reaches 0.7677. **No pool vessel has this failure mode**, so it
  cannot be attacked with available data.
* **Cut placement** (p001): AUC 0.9943 — *normal* — band occupancy z = **+4.44**, cut-gap
  **+0.239** against the pool's +0.006. Pure readout.
* **Burden compression** (p005/p020 over-predicted, p032 under-predicted): the chosen budget
  collapses toward the cohort middle.

And the reason CV never saw it: on the pool the cut crosses a **2.7%-occupied band**, so its
position is simultaneously unconstrained and nearly harmless. The readout looked solved
because it was never stress-tested.

### 3.3 What has been tried and lost — do not re-derive

Five substitution rules (`absolute`/`rel_max`/`quantile`/`phys_anchored`/`gap`), inner-CV
family selection, per-domain arm selection, the zero-parameter regression anchor, head
fusion, learned per-vessel cut (corr +0.075), learned per-vessel burden (−0.037, P=1.000),
magnitude head + physical cut (0.839 vs 0.902), owner-coupling, shell restriction, budget
anti-compression, morphological open/close, per-owner NMS, regression-head ordering, bagged
cut selection (+0.0026, inside the floor), `expected_tuned` on the wall (worse in every
band-occupancy stratum), and the training-time **mean** burden-consistency term (−0.007 wall
/ −0.017 off).

The pattern across all of them is stated correctly in the memory notes: **a fixed cut on the
learned field is already the best burden estimator available** (corr with `n_gt` = 0.967,
against physics 0.904 and a learned head 0.906), so every explicit replacement substitutes a
worse estimator for a better one.

### 3.4 The one variant that is genuinely different

The invariant that broke on p001 is not burden and not the cut — it is the **cross-vessel
calibration of the logit field** (p001 mean score 0.448 against a pool range 0.11–0.39). The
burden-consistency experiment attacked that at one cut, with a mean-over-vessels loss, and
fixed the median (11.6% → 5.7%) while leaving the tail alone (p90 28.3% → 32.2%).

Two things follow, and they are different experiments:

1. **Constrain the distribution, not the count.** A per-vessel penalty on the *shape* of the
   logit field (match its within-vessel quantile profile to a cohort reference, or standardise
   the logits per vessel before the head) makes the cohort cut per-vessel-adaptive **by
   construction and with zero fitted parameters** — which is the property every post-hoc rule
   in §3.3 failed to have. It is neither a burden match nor a readout substitution, so no
   measured negative covers it.
2. **Weight the tail.** The retry the memory itself recommends: CVaR or worst-vessel over the
   per-vessel burden error rather than a mean. The outliers *are* the vessels that matter, and
   the mean form provably does not move them.

**Caveat that must travel with both — AND ITS 2026-08-22 CORRECTION.** As written, this
said the pool has no headroom (wall AUC 0.9973, cut-gap +0.006, clot-free vessels already at
1.000) and that neither experiment could be *validated* on it, only argued from mechanism
plus a label-free burden-variance statistic.

**That is true of the WALL and false of OFF-WALL.** Phase B (§8f.2) measured the off-wall
cohort cut at 0.5812 against a per-vessel oracle of **0.7746** — 0.193 of headroom, against a
floor of 0.074. So the off-wall domain gives these experiments a direct, held-out, 2.6×-floor
target, which is the strongest validation surface this project has had for a readout change.
Judge on off-wall score, with the burden-variance statistic and the mechanism as corroboration
rather than as substitutes. The "no headroom" restriction still binds on the wall.

---

## 4. OFF-WALL — correctly closed. Stop working on the final-time set.

PHASE10 §14 is right and should be treated as settled:

```
best prefix of current ranking, per-vessel ORACLE k    0.8205
shipped                                               0.7372
within-shell decision, GBM on ALL 70 channels     AUC 0.912
the GNN's own score, alone                        AUC 0.887   <- within 0.025
```

The discriminator is within 0.025 AUC of a model trained on nothing but the decision
population; the band thickness is **sub-mesh** (`log(delta) − log(y)` ranks at AUC 0.598
against plain distance's 0.590); owner-attenuation with *oracle* wall `Mat` scores 0.1214;
occlusion-aware transport is at chance; and all twelve t=0 species channels are spatially
uniform (CoV 0.0000), so there is nothing left on the packs to mine.

**The only real fix is a finer boundary-layer mesh in COMSOL**, which is a data-generation
task, not a modelling one (§7).

**But mean-over-time off-wall is a different question and is still open**: 0.7188 against an
oracle-timing 0.8709. PHASE10 §15.1 decomposes it — an oracle *lag* on our own predicted wall
onset gives 0.7569, so **two thirds of the gap is inherited wall-onset error, not the off-wall
model**. The §15.2 ODE anchor was the right correction and collected +0.011 of it. The rest is
§2: a better wall clock, which is a physics problem.

---

## 5. THE WOUND MODULE — right architecture, misattributed blocker

**What is right**, and should not be relitigated: the wound is the wall law with the shear
gates deleted; the set is free on the boundary; the learned quantity is a coefficient inside a
conservation law; `G_pre` recovers 2.0 = `ungated(1) + low-shear(1)` on all three LOVO folds;
composition rather than retraining is correct at n=3; the bit-identity property on wound-free
packs is the right ship gate; and §13's domain correction (`w_reg`/`w_lum` rather than the
degenerate `wnd`) was an honest catch that should have been the headline from the start.

**What is open:**

* **`wound_patient003` is a wall-timing failure, not a wound failure** (§11.3): the
  deploy-legal wall trigger gels at step 53 when truth is step 2. The oracle trigger takes
  wound onset MAE 18.0 → 6.6. Correctly diagnosed, and correctly *not* patched with the
  `gate_scale=20` fudge (which takes cohort wall-onset MAE from 18.1% to 43.7%).
* **PHASE10 §16 does not unblock it and the doc says so** — §16 moves stitch nodes *later*,
  003 needs its near-wound wall *earlier*, and 003 is locally anti-correlated (ρ −0.246).
  §16's +0.042 LOVO is also **explicitly not a deploy number** (§16.4: on the real mask,
  +0.0076, CI crossing zero, positive on 4/19). Do not quote §16.3 as a wound fix or as a
  deploy gain.
* **The wound complement cannot be evaluated deploy-faithfully at all right now**: verified
  this session, `wound_patient001/002/003` carry **no `u0_pred`**. WOUND_PROGRESS's Next item
  5 is blocked on `scripts/precache_rgp_deq.py` over the wound packs.
* **The eval domain split is still undecided** (§9 last bullet, §12.3): `mask_wall` is the
  healthy-wall label, so wound nodes still score in the off-wall domain where the floor is 4×
  worse. `solid_boundary_mask` exists and is the obvious answer. This is a one-line decision
  that changes reported numbers; make it deliberately and re-baseline once.
* **`G_post` is not stable across folds** (22.1 / 20.9 / 11.9) and the per-node `WoundRateNet`
  loses LOVO. Both are n=3 symptoms. Correctly not shipped.

---

## 5b. THE WOUND/NO-WOUND COMPOSITION — measured, and the defect is not where it looks

Raised 2026-08-22: *"the wound part of the model is given full control of the wound section,
so we don't use the potential of our no-wound model — especially where the wound overlaps a
section that would have clotted anyway."* Measured with
[`scripts/diag_wound_composition.py`](../scripts/diag_wound_composition.py). The concern is
right that something is wrong there, and the mechanism turns out to be a different and worse
one.

### 5b.1 The gate-level override is CORRECT — settled from the `.mph`, not by argument

WOUND_PROGRESS §1 parked this ("override vs additive is currently unobservable ... it becomes
a real question after gelation"). It is observable in the model tree. Both surface-reaction
nodes on `tds2` carry **the same COMSOL feature type**:

```
[3] wall_surface_reactions_3spec  srf1   apiType SurfaceReactionsFlux   type Surface_reactions
[4] SfcRxn_3spec                  srf2   apiType SurfaceReactionsFlux   type Surface_reactions
```

Same-type boundary features are **exclusive** in COMSOL: the later node in the tree overrides
the earlier one on the overlapping selection, so `srf2` overrides `srf1` on `sel1`. The
expressions confirm it independently — `srf2`'s `J0_Mat` restates the **complete** bracket
(`Da*(Sat(M)*k_rs*RP + Sat(M)*k_as*AP + (Mas/M_inf)*k_aa*AP)*step2t(t)`) rather than an
increment. An additive reading would double-count the same chemistry on the wound.

**So `gate_fields`' `torch.where(wnd, g_pre, base)` — discarding the healthy gate on wound
nodes — is right, and an additive variant would be wrong.** It is also currently inert:

```
vessel               n_wnd   healthy gate>0 at t=0
wound_patient001        80        0   (0.0%)
wound_patient002        80        0   (0.0%)
wound_patient003        26        0   (0.0%)
```

including on 003 — §11.1's "42% open at step 3" is the gate *after the flow evolves*, and the
module reads the frozen t=0 gate, which is empty. Override and addition are bit-identical
today on all three vessels.

### 5b.2 The composition deletes nothing — because v4 has nothing to delete

```
vessel             owned   v4 commits   wound commits   DELETED   of which GT+   ADDED
wound_patient001     160            0             160         0             0     160
wound_patient002     160            0             160         0             0     160
wound_patient003      55            0              55         0             0      55
```

`compose_with_v4`'s hard override (`mask[owned] = wound_out["mask"][owned]`) destroys **zero**
v4 predictions on all three vessels. The worry does not manifest as deletion. But the reason
is the real defect: **v4 commits zero nodes in the wound region because it cannot see the
wound at all.**

### 5b.3 The clot-ML feature path carves the wound out of "wall" — WOUND_PROGRESS §6, un-fixed

`src/clot_ml/features.py` re-derives its own geometry from `data.mask_wall` alone —
`dist_w, owner = cKDTree(pos_xy[wall]).query(...)`, `hop_w = hop_distance(wall, A)`,
`shell = resolve_offwall_shell(pos_xy, wall, ei)` — and `transport_fields` seeds its source
on the same mask. On `wound_patient001`, at the 80 wound nodes:

```
channel                  at wound        at healthy wall
is_wall                    0.000              1.000
hop_wall            mean 8.7, max 12           0
dist_wall_edges           10.92                0
log_mat_phys               0.000              0.154
log_mat_owner              0.000              0.154
transport source        ZERO at every wound node
```

**The wound is encoded as open lumen roughly 11 edge-lengths from the wall**, its `owner` is a
distant healthy wall node carrying no `Mat`, and it contributes nothing to the advection
operator's source.

This is exactly the bug WOUND_PROGRESS §6 diagnosed and fixed — and the fix did not reach
here. §6 wired `solid_boundary_mask` through `src/data_gen/lib/*` (the pack builders), so the
**pack** channels are repaired (`sdf_nd` at wound nodes is 0.0000, verified) while the
**clot-ML** channels re-introduce the split (`dist_wall_edges` 10.92). Same defect, different
file, outside `test_solid_boundary_mask.py`'s end-to-end assertion.

Note what must *not* change: `mask_wall` stays the healthy-wall label for the **deposition
law**, because §5b.1 says `srf1`/`srf2` really are two different laws on two selections. The
union belongs to **geometry and transport** only.

### 5b.4 The seam, and where 003's residual error actually lives

`owned` is the wound plus **one** corner shell, assigned by nearest-solid-node ownership.
Everything deeper falls to v4, which §5b.3 has just shown is blind there:

```
vessel              GT off-boundary clot BEYOND the owned shell
wound_patient001              0
wound_patient002              0
wound_patient003            214
```

001/002 have none, which is why they score `w_reg` 0.970 / `w_lum` 0.971. **003 has 214 such
nodes and scores 0.693 / 0.615.** So the outlier that WOUND_PROGRESS §11 attributed entirely
to wall-onset timing has a second, independent cause that is pure composition geometry — the
region the module owns is a shell, and 003's thrombus is a volume.

### 5b.5 The two fixes, both zero-parameter

1. **Union the geometry — LANDED 2026-08-22 (roadmap item A1).**
   `src/clot_ml/features.py` now resolves two masks and keeps them apart: `wall`
   (`mask_wall`, the gated `srf1` **law**, and for now the eval domain) and `solid`
   (`solid_boundary_nodes(data)`, the **geometry**). SDF/owner/hop/shell and `is_wall` take
   the union; the backbone rollout's `gate * wall` source does not. `features_v4` takes the
   union as the transport boundary and as `horizon_for`'s bulk-speed exclusion, falling back
   to `S["wall"]` when a pre-change cache carries no `solid` key.

   Measured on `wound_patient001`, at the 80 wound nodes:

   | channel | before | after |
   |---|---|---|
   | `is_wall` | 0.000 | **1.000** |
   | `hop_wall` | mean 8.7, max 12 | **0** |
   | `dist_wall_edges` | 10.92 | **0.000** |
   | `owner` | a distant healthy wall node | **itself** |

   And the correction is not confined to the wound itself — the lumen above it was dragged
   with it:

   | pack | nodes | owner changed | hop changed | shell changed | max SDF move |
   |---|---|---|---|---|---|
   | `wound_patient001` | 9 957 | 1 632 (16.4%) | 1 279 | 80 | 0.645 |
   | `wound_patient002` | 14 184 | 2 097 (14.8%) | 1 442 | 80 | 0.505 |
   | `wound_patient003` | 19 746 | 1 114 (5.6%) | 246 | 32 | 0.186 |
   | `patient020` (control) | 19 708 | **0** | **0** | **0** | **0.000** |

   Inert on every no-wound pack, as predicted — `solid == wall` bit-for-bit there, so the
   cohort numbers cannot move. Pinned by four new assertions in
   `src/tests/test_solid_boundary_mask.py`, including the `build_sample` end-to-end path and
   the no-`solid`-key fallback.

   **What this does NOT do:** `log_mat_phys` and `log_mat_owner` now read **0.000** at the
   wound rather than a spurious 0.154 borrowed from a distant healthy node. That is honest —
   the healthy-wall law deposits nothing there — but it means the wound still contributes no
   `Mat` to the transport operator. Item 2(a) below (C1) is what puts a source there.
2. **The lumen: one transport solve fed by both sources — a SOURCE change and a COMPOSITION
   change, and they are separate.**

   *Why override has no justification here.* §5b.1's COMSOL result is a rule about **boundary
   features**: same-type nodes on overlapping selections, later wins. Off-boundary nodes are
   domain nodes. There is no `srf`-style override in the lumen — there is only
   `u·grad(Mat) = 0` fed by whatever boundary lies upstream, and that boundary is in general
   *both* wounded and healthy. So on the wound boundary the override is right, and one node
   further out it is unmotivated.

   *(a) The source.* The wound module paints its lumen thrombus with `att = 0.16` and a lag
   of 4% of the horizon — two constants WOUND_PROGRESS §10.4 flags as *fixed, not fitted* —
   for exactly the nodes where v4 already ships a solved advection operator, and PHASE10 §5's
   whole argument is that nearest-owner rules transport along the mesh normal, the one
   direction the equation does not transport along. `transport_fields` already takes
   `(…, wall, wall_source)` and seeds `ws[wall] = wall_source[wall]`; pass the **union** as
   the boundary and a source array carrying healthy-wall `Mat` on healthy nodes and the wound
   ODE's `Mat_wound(t)` on wound nodes. Linear operator, one solve per stored time
   (`build_temporal_transport.py` already does this at ~20 s/vessel).

   *(b) The composition.* Drop `owned_off` — the nearest-solid-node shell — as the unit of
   ownership, and with it the `mask[owned] = wound_out["mask"][owned]` override in the lumen.
   The single transport field decides every off-boundary node, at any depth, with no
   ownership assignment and no seam. This is the half that reaches 003's 214 deep nodes;
   (a) alone still leaves them outside the shell the module owns.

   *The guard, because this can regress.* 001/002 currently score `w_lum` 0.971 with the
   shell rule, and their thrombus genuinely **is** one shell — so a depth-unlimited field can
   only add false positives there. Run it as an arm with a hard gate: **001/002 must not
   degrade.** The safe first variant is the *union* of the two rules (shell OR transport),
   which is monotone and gives a clean read — if it adds nothing on 001/002 and reaches 003's
   214, it is right; if it sprays on 001/002 the calibration is wrong and that is cheap to
   learn. Note PHASE10 §14.4 killed `att * Mat_owner >= crit` at 0.1214 with *oracle* wall
   `Mat`, so a threshold on a transported field is not automatically safe; what is different
   here is that the wound reaches 9–104× crit, so the question is extent, not a marginal call.

Both are independent of the n=3 problem: neither fits a parameter, so neither is limited by
the three-vessel cohort.

---

## 6. DEFECTS FOUND DURING THIS REVIEW

> **6.1–6.3 were FIXED on 2026-08-22**, pinned by
> [`src/tests/test_flow_source_threading.py`](../src/tests/test_flow_source_threading.py).
> `flow="gt"` reproduces `outputs/clot_ml_cache_v5` **bit for bit** (max|diff| 0.000e+00 over
> all 68 columns on `patient020`), so the locked artifact and its normaliser are untouched.
> Under `flow="pred"`, **55 of 69 channels now respond** — including the five that were
> GT-locked; the remaining 14 are pure geometry / topology / species-IC and correctly do not.
> The descriptions below are kept as the record of what was wrong.
>
> Decision 2026-08-22: **GT flow at t=0 stays in scope** for now; Stage-A quality (§1.2, §8
> row 2) is deferred. §1 remains the largest unmeasured quantity and the plumbing to measure
> it is now correct and waiting.

### 6.1 The v4 feature block is GT-locked — blocks every deploy-faithful measurement

`src/clot_ml/features_v4.py`:

```python
def indicator_physics(data, bio, wall, hops=3):
    f0 = t0_flow_fields(data, bio, hops=hops, flow_source="gt")   # <-- hardcoded
```

and `augment_sample(data, S, bio)` takes **no `flow` argument**. `locked.py:build_sample`
calls it without one (line 124). So `build_sample(..., flow="pred", variant="v4")` returns
**55 predicted-flow channels + 4 GT-flow channels** (`gate_ind`, `log_mat_phys_ind`,
`onset_phys_ind`, `log_mat_ind_owner`) plus `log_mat_adv_ind`, which transports a GT-derived
source. Any `--flow pred` number produced today is silently optimistic and uninterpretable.

**Fix:** thread `flow`/`hops` through `augment_sample` → `indicator_physics`, defaulting to
`"gt"` so existing caches reproduce bit-for-bit. Pin with a test asserting the two variants
differ on a pack with `u0_pred`.

### 6.2 `build_clot_ml_cache_v4.py` has no `--flow`

The v3 builder has it (`scripts/build_clot_ml_cache.py:28`); the v4 builder does not. Add it,
and default the out-dir to `outputs/clot_ml_cache_v5_{flow}` the same way v3 does.

### 6.3 `wound_features` reads GT velocity regardless of `flow`

`src/clot_ml/wound.py`, inside `wound_features`:

```python
u = data.y[0, :, 0].detach().cpu().numpy()   # GT, always
...
"speed": np.hypot(u, v),
```

The module docstring claims "`flow_source` is threaded through to `t0_flow_fields`", and it is
— for `f0`, but not for this channel. Inert today (the `WoundRateNet` that consumes it is not
shipped), but it will silently contaminate the first deploy-faithful wound run.

### 6.4 Wound packs have no `u0_pred`

Verified on all three. `precache_rgp_deq.py` over `wound_patient001/002/003` is a prerequisite
for WOUND_PROGRESS Next item 5.

### 6.5 Dead input channels — FIXED 2026-08-22

> **Both are now populated on all 45 packs**, by
> [`scripts/repair_pack_wall_normals.py`](../scripts/repair_pack_wall_normals.py), pinned by
> [`test_boundary_normals_and_node_type.py`](../src/tests/test_boundary_normals_and_node_type.py).
> `wall_normal` is unit-length at **100%** of solid nodes on every pack;
> `node_type_*` is a strict one-hot `[interior, solid, inlet, outlet]` on **100%** of nodes.
>
> **The normals are fitted from the graph, not the mesh** — `boundary_normals_from_graph`
> takes each solid node's solid-subgraph neighbours, fits the boundary tangent by total
> least squares and rotates it, orienting by the centerline. That is what makes it
> applicable to the three wound packs whose COMSOL exports are gone. 12 of 539 solid nodes
> on `patient008` turned out to be **degree-0 in the solid subgraph**, so they fall back to
> their nearest solid nodes by position — still mesh-free.
>
> **The write is a delta, and that mattered.** A wholesale `rebuild_x` does *not* reproduce
> the stored packs, for a reason unrelated to this fix: the cohort was extracted by two
> builder revisions that disagree about the prior channels. On `patient020` a fresh build
> puts `wss_prior_nd` at ~45 at the wall where the pack has **0**, and moves `u_prior` by
> 0.55 in the interior — the divergence WOUND_PROGRESS §8 recorded and never resolved. The
> repair therefore writes `x + (x_fixed − x_prefix)`, so any channel the builder disagrees
> about for unrelated reasons cancels and stays byte-identical. **That second divergence is
> still open and was deliberately not touched.**
>
> **What this invalidates, as accepted:** `clot_gnn_v4` / `v4w` (the v5 cache and the locked
> normaliser must be rebuilt and the artifact re-promoted), and the frozen RGP-DEQ, whose
> Fourier encoding consumes `wall_normal` and whose `NodeFeat.REST` consumes `node_type_*`.
> **`u0_pred` / `v0_pred` on every pack are now stale** and must be recomputed before any
> `--flow pred` number is quoted (§1). Backups are at `*.pt.prenormalfix`.
>
> **The blast radius is larger than "two dead channels", and this is the number to plan
> against.** `compute_hydraulic_width_nd` sphere-marches *along the wall normal*, so a zero
> normal made the width degenerate at the boundary. Measured old-vs-new:
>
> ```
> channel            max|d|   median|d| at wall   % of nodes moved
> sdf_nd                  0                   0        0.0%
> shear_potential         0                   0        0.0%
> wall_normal_y           1              0.9917    2.8 - 5.5%
> wss_prior_nd        33-68               18-25    2.8 - 5.5%
> width_nd            ~0.82              ~0.66    11.1 - 22.0%
> width_d1            70-254            0.16-8.4  12.5 - 24.8%
> width_d2         3e4 - 4e5             5.6-638  16.7 - 33.1%
> ```
>
> `width_nd`, `width_d1`, `width_d2` are v4 feature columns and they moved on **11–22% of
> nodes**. So this is a genuine **retrain**, not a re-normalisation — do not expect the
> shipped weights to transfer. `sdf_nd` and `shear_potential` are untouched, as they should
> be: neither depends on the normal.
>
> **The width channel got much better, and that is checkable independently of any model.**
> `src/clot_ml/geometry_class.py`'s docstring recorded that `width_nd` was *"unusable on 9 of
> 34 vessels: 001/010/011 read a constant 1.000 with a 10x spike, and 003/004/005/006/007/008
> read ~0.12, neither of which is anatomy"*, and predicted **"fix the channel and the abstain
> goes away."** It did:
>
> ```
>                                       old        new
> vessels with unusable width_nd     9 of 42     0 of 42
> aneurysm  bulge, designated       2.57-3.48   2.08-2.42   baseline max 1.66  -> still clean
> stenosis  narrowing, designated   0.281-0.323 0.448-0.476 next baseline 0.488
> ```
>
> **So the aneurysm cut (`BULGE_ANEURYSM = 2.0`) survives and the stenosis cut
> (`NARROWING_STENOSIS = 0.40`) does not.** The designated stenoses now sit at 0.448–0.476
> against a next-baseline of 0.488 — a **0.012** margin where the old channel had a real gap.
> `patient008` also reads a degenerate `narrowing = 0.000` that the usability guard no longer
> rejects.
>
> **The threshold was deliberately NOT retuned.** Refitting a cut with a 0.012 margin on
> three labelled vessels is precisely the selection this project's own noise discipline
> rejects (§3.3, and PHASE10 §13.3 on search-space size). The test is `xfail`ed with that
> reason attached, and recalibrating the classifier against the repaired channel — plus
> understanding `patient008` — is a follow-up task, not a threshold edit.
>
> One further bug this surfaced: `data.x_biochem` carries its **own** copy of `wall_normal`
> (`BIO_X_SCHEMA` channels 3:5) and `assert_anchor_dual_x_aligned` requires the two tensors
> to agree, so the repair has to write both. Caught by `test_anchor_dual_x_schema.py`.

The original finding, kept as the record:

`node_type_*` is **identically zero on 100% of nodes** on `patient005/012/020/032/041/044` and
`wound_patient001` (4 dead channels). `wall_normal_{x,y}` is nonzero in the interior but
**exactly zero at every wall node** — harmless for `u_n`/`u_t` (no-slip makes them zero
anyway), but it means the exact segment-normal extract path is dead on every pack, so *any*
future construction needing a true wall normal — a boundary-layer band thickness (§4), a
wound-normal growth direction — has no correct input to build on. WOUND_PROGRESS §8 reports
both and correctly declines to fix them mid-baseline. They should be fixed in the **same
change that rebuilds the cache for predicted flow** (§1.4), since that invalidates the
baseline anyway and the two re-baselines can be paid for once.

### 6.6 Minor

`src/clot_ml/features.py:FEATURE_ORDER` is a module-level global memoised on first call. Safe
today (the key set does not vary with `flow`), fragile if a variant ever changes it.

---

## 7. THE DATA ASKS, IN PRIORITY ORDER

The binding constraint on §2, §3 and §4 is evidence, not modelling. The generation targets are
already specified precisely across the docs and memory; consolidated:

1. **Stenoses spanning the throat-transport regime**, where `phys_mask` recall degrades —
   p042's ranking failure (AUC z = −12.95) has **no analogue in any pool vessel**, so it is
   currently unattackable. Templates exist (`comsol_models/phase2_nowound_*.mph`).
2. **Vessels with occupied score bands** — the pool never exceeds 7.5% band occupancy and p001
   is at 13.5%. Concretely: vessels where `|phys_mask ∩ wall| / n_gt` ≈ 1 *and* the score
   field is not bimodal. Without these the readout work in §3 cannot be validated even if it
   is right.
3. **A finer boundary-layer mesh** — the only route to final off-wall > 0.8 (§4). Two rows is
   not enough to resolve a band whose thickness decides the label.
4. **The paired wound A/B** — same `.nas`, with and without `sel1`. Nothing currently isolates
   the wound's effect on fixed geometry; `wound_patient001` is *not* `patient001`.
5. **3–4 more wound simulations**, so `G_post` can become a function of vessel state instead
   of an unstable constant.
6. **`patient039` re-run to full horizon** — the cheapest single addition to the pool.

---

## 8. RECOMMENDED ORDER OF WORK

**Rewritten 2026-08-23**, after Phases A, B and C0 shipped and C1/C2/C3 closed. Everything
below assumes the two standing constraints: **no new data**, and **the corrector is not being
changed yet**.

> **The shipped model is `clot_gnn_v5w`** (§9f): final wall 0.9203 / off 0.7078, readout gap
> 0.045, config floor ±0.0037 wall / ±0.0432 off. Mean-over-time 0.8694 / 0.5792, unchanged.

### Done

| | | |
|---|---|---|
| ✅ | **Phase A** — geometry union, clot-free data path, geometry classifier, eval domain | §5b.5(1), §8c, §8d, §8e |
| ✅ | **Phase B** — the re-baseline, and the off-wall deficit localised to the READOUT | §8f. AUC unchanged at 0.989; the oracle was within the floor |
| ✅ | **C0** — distributional logit constraint, shipped | §9b. off 0.5812 → 0.7078, gap 0.193 → 0.045, three configurations. Its two follow-ups add nothing (§9b.10) |
| ❌ | **C1** lumen source — closed, negative | §9c. Transport ratio 23× too small as a magnitude; `off_att = 0.16` already accurate to ±15% |
| ⏸ | **C2** recursive shells — implemented, gated, not shipped | §9d. +0.006 on n=3. Corrected §5b.4: 123 of 003's 206 deep nodes are >14 hops from the wound |
| ❌ | **C3** closed loop via the corrector — falsified | §9e. −3.5% shear against −87% required, insensitive to `delta_mu` over 100× |

### Open, in priority order

| # | action | cost | why |
|---|---|---|---|
| **1** | **§1 — the flow.** Re-run `precache_rgp_deq` (36 of 38 `u0_pred` are stale since the repair), then `--flow pred` end to end. **First reconcile Stage-A Rel L2**: `AGENTS.md` says 0.087, this cohort measures 0.39–0.60, and until that is explained a bad `--flow pred` number cannot be attributed | ~1 day | **the largest unmeasured quantity in the project.** The whole stack runs on COMSOL's t=0 velocity; the one deployable-flow measurement ever taken cost **−0.34 wall** and was never repeated on v4. The plumbing is correct and tested. It is also the only open item whose answer could invalidate a deploy claim |
| **2** | **C3′ — the closed loop as a blockage LAW.** `Mat >= crit -> sr <- sr/8 -> the ordinary low-shear gate` | ~1 day | §9e.4. `sr/sr0` is 1.000 before gelation and **0.1226** after (p25–p75 0.113–0.136, 584 observations, all three vessels) and it opens the gate unaided. Two stages instead of five, no corrector. The build is reconciling it against the fitted `G_pre`/`G_post`, which stand in for the same effect: **two fitted parameters out, one measured constant in** |
| **3** | **`stitch_onset` on the deploy path.** Implemented in `physics_wall_model.py`, tested, and called by **nothing in `src/clot_ml/`** | ~hours | `scripts/eval_stitch_onset.py` measures **+0.0462 wall LOVO [+0.0313, +0.0628], P = 0.0000, positive on 15/15 vessels** — larger than C0's wall gain and far above even the old floor. **But it is measured on the GT wall set**, so it isolates timing from the mask and the deploy number is unknown; its own docstring warns the apparent gain on a predicted set is "mostly a precision effect". Measuring it on the shipped set is cheap and it is the largest unclaimed result in the repo |
| **4** | **C4** — port C3′ to the cohort, if C3′ holds | ~days | size it only after C3′ reports. §2.3 argues the loop *cannot start* on healthy wall, so the cohort payoff is unproven |

### Blocked by the standing constraints

* **C2** shipping — needs more wound vessels (**D2**).
* **D1 / D2 / D3** — all data.
* **Reviving C3 proper** — needs a corrector that stalls rather than diverts. §9e.3 sizes it;
  it is Stage-A work and belongs with item 1, not with the clot stack.

### Do not

Spend FINAL_HALF (§8b) · make the wound gate additive (§5b.1) · retune `NARROWING_STENOSIS`,
whose classes now overlap outright (§8d) · put the metric's empty-GT branch into the training
loss (§8f.4) or into readout selection (§8f.3, measured harmful) · sweep `shape_w` (§9b.8) ·
build a mechanism story from a single CV configuration (§8f.4) · cite the `CorrectorArm`
`seed_ramp` sweep as evidence about flow coupling (§9e.3) · re-test the five substitution
rules, per-domain arm selection, seed-count increases off-wall, or owner-attenuation.


## 8b. SHOULD FINAL_HALF BE OPENED TO GROW THE COHORT?  No — it makes validation *worse*

Asked 2026-08-22. Recorded here because it will be asked again.

**The reframe that settles it.** The request was phrased as "increase our low-data regime and
bring back a solid training/validation split". But the train/validation split is *already*
sound: `eval_strict.py` / `eval_strict_temporal.py` select every readout scalar on the
out-of-fold scores of vessels outside the held-out fold, and PHASE10 §1 removed the three
leaks that were in it. What the project lacks is not validation data, it is **test** data —
and that is exactly what FINAL_HALF is. Opening it does not improve the split; it **converts
the test set into training data**, leaving no unbiased read at all.

**What it would buy, quantitatively.** Four vessels, pool 19 → 23. The noise floor goes from
±0.024 to roughly ±0.022 — the regime does not change, and nothing at the 0.01–0.03 scale
becomes measurable. More decisively: **the pool is saturated, not small.** Wall AUC 0.9973,
cut-gap +0.006, band occupancy 2.7%, and 8 clot-free vessels already scoring 1.000. Four more
vessels from the same generator are four more vessels at AUC ≈ 0.99. The binding constraint
is *zero coverage of the failure regimes* (ranking failure, occupied score band), not sample
count — §7, and `clot-pool-has-no-headroom-left`.

**And 043 is the argument against, not for.** VIZ_HALF's 042 (stenosis) turned out to be the
cohort's only ranking failure, AUC z = −12.95, a mode no pool vessel exhibits. FINAL_HALF's
043 is the aneurysm — the other priority geometry class. A vessel likely to reveal a failure
mode is worth far more as an unbiased test than as 1/23rd of a training set; train on it and
"does this generalise to aneurysms?" becomes permanently unanswerable.

**Timing is also wrong.** "Nearing the final model" is in tension with three structural
changes still queued (closed-loop physics §2.4, logit calibration §3.4, composition §5b).
FINAL_HALF should be spent *after* those land, which is what rule 5 of
[SEALED_SPLIT.md](SEALED_SPLIT.md) already says.

**Three cheaper sources of the same thing, in order:**

1. **The 8 clot-free vessels** — `patient017/022/023/026/027/030/033/034`, `maxMat=0`, empty
   GT, currently **neither pool nor SEALED and entirely unused**. The eligibility filter
   drops them because an empty-GT vessel is a different *quantity* to score (PHASE6_RESULTS
   §6.2), and that is right for scoring. It is not right for *training*: a vessel with
   `n_gt = 0` is the strongest possible constraint on the logit distribution, which is
   precisely what §3.4 needs and precisely the failure mode (over-commitment, burden
   compression) that §3.2 identifies. **Use them as training-only, scoring-excluded data.**
   Free, protocol-neutral, and available today.
2. **VIZ_HALF** (001/010/014/042) — already permanently open for inspection. Rule 2 still
   forbids using them to select between configurations; they remain legitimate as qualitative
   evidence and as the source of both diagnosed failure modes.
3. **Generation** (§7). The only route that can target the failure regime rather than
   resampling the saturated one.

**The condition under which to revisit.** If generation becomes genuinely unavailable *and*
the remaining work is confined to fitting rather than architecture, the graduated move is to
split FINAL_HALF again by the same deterministic, disclosed-in-advance rule SEALED_SPLIT
used — release two, keep two, keeping 043 closed. That is a fallback, not a recommendation.

---

## 9. THE ONE THING THIS REVIEW WOULD CHANGE ABOUT THE PROJECT'S FRAMING

Every phase since 7 has optimised a quantity of order 0.02 against a noise floor of 0.024,
while a measured 0.34 sat unexamined one argument away. The discipline in this repo is
genuinely unusual — the negative results are recorded with numbers, the leaks were found and
priced, SEALED was split rather than spent, and the degenerate wound metric was retracted by
its own author. That discipline is exactly what makes §1 stand out: it is not an oversight of
rigour, it is a scope boundary drawn early, honoured consistently, and never revisited once
the model matured enough for it to become the dominant term.

Revisit it before anything else.
