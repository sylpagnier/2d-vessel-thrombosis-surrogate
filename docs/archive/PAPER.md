> **SUPERSEDED 2026-09-11 — archived, do not write from this file.** The story was restructured
> around five legs led by the tool itself (Leg 0), and the project moved off manuscript-shaped
> prose until the story stops moving. The live system is **[../publication/](../publication/)**:
> the argument in [STORY.md](../publication/STORY.md), number-checking in
> [EVIDENCE.md](../publication/EVIDENCE.md), figures in
> [FIGURES.md](../publication/FIGURES.md), the run queue in
> [EXPERIMENTS.md](../publication/EXPERIMENTS.md).
>
> Kept because the claim-id discipline was invented here, §5's four-point ablation axis is the
> origin of Legs 1 and 3, and §7's gated-coupling argument is carried forward almost intact.
> Numbers here were artifact-verified **as of 2026-09-10** and several have since been restated
> or re-scoped — check against the ledger before reusing any of them.

# The paper — canonical narrative and verified numbers

Opened 2026-09-07. **This is the document the manuscript is written from.**

It supersedes nothing: PUBLICATION_PLAN.md remains the planning record
and PUBLICATION_NOTES.md the working desk with the full provenance and
the retraction list. What lives *here* is the argument in the order a reader meets it, and only
numbers that a machine has re-derived from an artifact.

**Every number below carries a claim id in ⟨angle brackets⟩.** `scripts/publication/verify_claims.py`
resolves each id against the artifact that produced it and fails if the two disagree. Run it
before quoting anything:

```bash
PYTHONPATH=. python scripts/publication/verify_claims.py --paper docs/PAPER.md
```

A number without an id has not been verified and **must not enter the manuscript**. There is a
list of those at the end (§10), kept short on purpose.

---

## 1. The one-sentence claim

> Which parts of a coupled biophysics pipeline should be **solved** and which should be
> **learned** is an empirical question, and the answer is not uniform: it changes with the
> component, and — the finding we did not expect — *within a single component, with the spatial
> domain.* In a thrombosis pipeline we measure it. At the wall the governing equations suffice:
> a tuned physics rule matches the full model (+0.0245 ⟨pvlf.wall.full_minus_phystuned⟩, n.s.)
> and beats a physics-free network by 0.1070 ⟨pvlf.wall.phystuned_minus_plain⟩. In the lumen the
> same equations collapse (0.5277 ⟨pvlf.off.phys_tuned⟩) while the network reaches
> 0.8162 ⟨pvlf.off.full⟩, with physics conditioning still paying significantly on top
> (+0.1345 ⟨pvlf.off.full_minus_plain⟩, P=0.009 ⟨pvlf.off.full_minus_plain_p⟩). The flow is
> better solved than learned, and that one is bounded before accuracy enters the argument: a
> perfect flow surrogate is capped at 7.9% ⟨timing.flow_ceiling⟩ of end-to-end runtime, and a
> competent learned one costs −0.1075 ⟨rgp.off_delta⟩ off-wall to capture 6.8% of it.
>
> **Solve the flow. Solve the wall. Learn the lumen.** We ship the tool that answer implies.

**The per-domain flip is the load-bearing half.** "Profile the pipeline before choosing what to
learn" is close to established practice, and the Amdahl form of it is already in print (§7.1).
That the answer *reverses inside one component* — same network, same features, same fold
partition — is not reported anywhere in this literature or in the wider hybrid-modelling one.
Against a from-scratch mesh GNN the shipped model gains +0.2522 ⟨pvlf.wall.full_minus_naive⟩ at
the wall and +0.3313 ⟨pvlf.off.full_minus_naive⟩ off-wall — and roughly half of the wall gain is
the physics-informed *architecture* rather than the physics conditioning
(+0.1207 ⟨pvlf.wall.plain_minus_naive⟩), which no earlier ablation in this project could separate.

The tool is a geometry-generalizing surrogate for the Cardillo–Barakat COMSOL thrombosis model
that predicts the spatiotemporal clot field on unseen 2D vessel geometries in **58.9 s**
⟨timing.median_s⟩ against COMSOL's ~48 h — **2,934×** ⟨timing.speedup⟩ — with no ground-truth
velocity field anywhere in its inputs, and no ground-truth state of any kind at inference (§8).

> **Three legs, and they are not equally strong — the abstract must not imply otherwise.** The
> wall-versus-lumen result is significant in both directions on one verified fold partition. The
> flow result is strong but *different in kind*: an Amdahl ceiling, which is a property of the
> pipeline, plus one measured accuracy cost. The timing result (§5.3) is a **bounded null** and is
> deliberately not a fourth pillar — see §4's table and the scope note under it.

---

## 2. The gap, and why it is not the usual gap

The Dec-2025 PRISMA review in *J Thromb Thrombolysis* (Al Bannoud et al. 2026;59:727–745)
screened the ML × computational-thrombosis intersection and found **11 eligible studies**. Three
facts from it, all quotable:

1. **Within the 11, none learns the clot field.** Every one predicts a scalar, a class label,
   or a non-clot field. The nearest, Bouchnita 2023b, binary-classifies coagulation initiation
   from scalar thrombin parameters; Coagulo-Net solves and infers coagulation ODEs without
   generalizing a rollout to unseen geometry.
2. **No graph or mesh architecture appears anywhere** across the 11 (ANN, NARX, SVM, KNN, DT, RF,
   NB, LR, XGBoost, CatBoost, EBM, DNN, PINN).
3. **The review names this direction as its own future work**, repeatedly — 3D vascular
   geometries, patient-specific reconstructions, architectures that "improve the detection of
   complex spatial and temporal patterns in thrombus growth".

> **State the boundary of the "11", twice over.** First, the review's inclusion criteria required
> explicit ML–mechanistic integration and excluded reduced-order and metamodelling work, so "11
> studies" bounds the ML-hybrid literature, **not** thrombosis surrogate modelling generally.
> Second, and this is the one with teeth: **all three facts describe the review's screening
> window, which closed before the field moved.** Write every one of them in that scope.

### The near neighbour, and it is published — build the introduction around this

**Pelissier, Meliga & Hachem (2026), *Comput Biol Med* 208:111649** learn a spatiotemporal
thrombus field with a Transformer-GNN on held-out 2D intracranial-aneurysm geometries. It
appeared after the review's window and it is prior work by our submission date.

**Therefore: no priority claim anywhere in this manuscript.** Not "first field-level surrogate",
not "first mesh architecture in thrombosis", not "nobody learns the clot field" in the present
tense. Those sentences are falsifiable by a paper in a journal on our own venue list, and a
reviewer who knows the field will make that the entire review. Cite it in the introduction, in
§3, and — most importantly — in §7, where it is *evidence for us* rather than competition.

What it does not do, and what therefore remains ours to claim:

* **It runs no physics-versus-learning ablation at all** — its one ablation is multitask versus
  per-field heads. The division of labour (§5) is untouched, and §6.4 of
  [RELATED_WORK](../publication/RELATED_WORK.md) confirms nothing in the wider SciML literature reports it either.
* **It is a solver accelerator, not a predictor.** Its node features are the four physical fields
  at `t` and `t−Δt` plus the next inlet velocity — in its own words, *"the same input information
  as the CFD solver at each time step."* It is seeded from a real CFD state and time-steps
  forward. We take a mesh and emit the full horizon with no ground-truth state at any time (§8).
* **Its ground truth is in-house** (7-species, CIMLIB-CFD); ours is published (§3).
* **Its coupling is smooth.** No low-shear stagnation gate, no shear-gradient deposition; its only
  threshold is on agonist *concentration*. That difference is the whole of §7.4.

**And it is ahead of us on three axes. State them; do not argue with them.** Pulsatile
physiological flow with an out-of-distribution inflow test (we are steady at one Re);
patient-derived geometry from InTrA (ours are parametric-synthetic); and benchmarks against
published architectures — MeshGraphNet, BSMS-GNN, Transolver++ at matched compute — where our
`A_naive` is a from-scratch control, not a published architecture. All three belong in §9.

### The framing risk, named early

Mesh-GNN geometric generalization is **established method** (MeshGraphNets and successors), and
"we used a GNN and it generalized" is not a contribution. Published PI-GNNs report wall-shear
agreement well above ours (R = 0.94; 7.6% directional WSS error). A paper whose claim is
*"a learned flow surrogate cannot do this"* dies at review, because the honest reading is
*"**ours** cannot, and published ones look better."*

**§5 is the answer to that risk, and it is why this paper is not a mesh-GNN paper.** §5 trains a
**from-scratch** mesh GNN — geometry only, isotropic messages, one binary head on plain BCE, no
`Mat` auxiliary target, no metric-shaped loss, no C0, one global cut — and measures it against
the shipped model on identical folds: **0.6833 ⟨pvlf.wall.naive⟩ vs 0.9356 ⟨pvlf.wall.full⟩** at
the wall. A generic mesh GNN is a *measured* baseline here, not a rhetorical one.

**And the honest half belongs in the abstract too:** at the wall, the *learned* part adds nothing
measurable on top of a tuned physics rule (+0.0245, P=0.069 ⟨pvlf.wall.full_minus_phystuned_p⟩).
The learning earns its place in the lumen, not on the wall — and a paper that claims otherwise is
quotable against its own data.

---

## 3. What is being surrogated

The ground truth is **peer-reviewed and from our own lab**: Cardillo & Barakat (2025), *Biomech
Model Mechanobiol* 24(5). Verified against the `.mph` trees: **9 bulk species**
(`rp, ap, apr, aps, at, pt, th, fg, fi`) with `Reactions_9spec` / `WallFlux_9spec` /
`InletFlux_9spec` / `ExitFlux_9spec`; **3 surface species** (`M, Mas, Mat`) with
`wall_surface_reactions_3spec`; Carreau rheology; clot entering momentum as the `mu1(Mat)`
viscosity step at `viscosity_mat_crit`.

Saying so is the strongest framing move available: **the object of study is a published model,
not an anatomy claim**, so the synthetic-geometry objection largely dissolves.

Three properties of that model drive every design decision downstream, and each is measured, not
assumed:

* **The clot is `Mat`-driven, not fibrin-driven.** `mu2(FI) ≡ 0` everywhere: FI peaks ~46× below
  its 0.6 µM gelation threshold. *The manuscript must not describe fibrin as an active
  contributor.*
* **`Mat` grows autocatalytically.** `J0_Mat` deposition is ~1/133 of `dMat/dt`; ~90% is
  `(Mas/Minf)·k_aa·AP`. A model that learns "platelets stick to the wall" learns the wrong term.
* **Deposition is gated by low-shear stagnation.** `sr < lss` carries ~80%; the shear-gradient
  branch is minor *and* is the one graph operators resolve worst.

That third property — the coupling from flow into chemistry is a **threshold gate**, not a smooth
map — is the mechanism the whole paper turns on. Plant it here; §4 builds on it and §7 pays it off.

> **And it is what separates this ground truth from the nearest published surrogate's.** The
> thrombosis model in Pelissier et al. 2026 has no low-shear stagnation gate and no shear-gradient
> deposition; its only threshold is on agonist *concentration*, so flow reaches chemistry through
> advection and a smooth shear-activation rate. Shear-gradient dependence is precisely the feature
> Cardillo & Barakat position as their contribution over prior thrombosis models. **State the
> difference here as a property of the ground truth, without comparative language** — §7.4 turns
> it into the mechanism result, and it lands harder there if §3 has already laid the fact down
> neutrally.

---

## 4. The tool, and where the labour is actually divided

A reviewer needs this before §5 means anything, and stating it plainly resolves a framing the
project has carried loosely for a long time. The slogan has been *"learn the chemistry, solve
the flow."* §5 measures the division directly, and the accurate statement is:

> **Solve the flow. Solve the wall. Learn the lumen.**

**Two axes, not a flat list of four.** The division runs *per component* (flow versus chemistry)
and, inside the chemistry, *per spatial domain* (wall versus lumen). Do not present these as
"four couplings": §5 already uses "four" for its four-point predictor axis, and flattening the
two axes into one list discards the structure that makes the result novel. The per-domain flip is
the half nobody else has reported.

Every clause is a measured result, not a design intention — and they are **graded by how strong
the evidence is**, which the manuscript must preserve:

| clause | evidence | strength |
|---|---|---|
| **Solve the flow** | three independent attempts to learn it fail or add nothing; the prize is capped at 7.9% ⟨timing.flow_ceiling⟩ of runtime before accuracy enters, and the failure is a gated coupling rather than a field-accuracy problem (§7) | strong, but *different in kind* — an Amdahl ceiling is a property of the pipeline, not a measurement of learnability |
| **Solve the wall** | a tuned physics rule reaches 0.9111 ⟨pvlf.wall.phys_tuned⟩ against the full model's 0.9356 ⟨pvlf.wall.full⟩ — a gap of +0.0245 ⟨pvlf.wall.full_minus_phystuned⟩ that is **not significant** (§5.1) | **strong** — paired, one verified fold partition |
| **Learn the lumen** | physics collapses to 0.5277 ⟨pvlf.off.phys_tuned⟩ there while the network reaches 0.8162 ⟨pvlf.off.full⟩; conditioning still adds +0.1345 ⟨pvlf.off.full_minus_plain⟩ on top | **strong** — significant, and the reverse of the wall on identical folds |
| *(not a fourth pillar)* **Learn *when*** | the temporal head is worth +0.054 over a frozen mask at the wall and +0.098 off-wall (§5.3) — but ablating the ODE's onset channels costs **nothing** (−0.0017 / −0.0198, both null) | **weakest — a bounded null.** See the scope note below |

> **The timing clause is deliberately demoted, and the reason is scope, not modesty.** §5.3's
> defensible statement is that *the ODE's explicit onset channels are redundant given everything
> else the head already sees* — the head reads a physics-conditioned base-GNN score, so physics
> reaches it by another route. Worse, the **shipped deploy path** (`readout.lag_anchor: "ode"`)
> uses a different mechanism and **has no switch to ablate**, so it was never tested at all.
> Presenting "learn *when*" as a fourth measured coupling alongside the other three overstates it,
> and a careful reader who checks §5.3 will find the null and its caveats. Report it as a
> secondary result with its bound attached.
>
> **The previous slogan said "solve *when*", and this project's own measurement does not support
> that either.** The ODE supplies a clock in the deploy path, but the temporal head reconstructs
> an equally good schedule without the ODE's onset information. Corrected 2026-09-09; the earlier
> wording predates the test that could check it.

### 4.1 The pipeline

```
vessel mesh ──► local Carreau Navier–Stokes solve (t=0 velocity, pressure, shear)
                          │
                          ├──► physics backbone: gated surface-deposition ODE + upwind
                          │      transport  ──►  Mat trajectory, occlusion mask, onset index
                          │
                          └──► feature block (69 channels: geometry, flow, the deposition
                                 law's dimensionless groups, the backbone's own outputs)
                                          │
                                          ▼
                          flow-aware message-passing GNN  ──►  WHERE the clot is (final time)
                                          │
                                          ▼
                          temporal head, anchored on the ODE crossing  ──►  WHEN each node commits
```

### 4.2 The learned part: one network, two heads, a physics base

Encoder → 4 anisotropic message-passing layers (dim 64) → two heads. Edges carry the geometric
offset *and* the t=0 velocity projected onto them; aggregation is split into upstream and
downstream halves weighted by that projection, because the non-locality here is advective, not
diffusive.

```python
logit = head_cls(h)                                  # the scored output
r     = head_reg(h)
reg   = mat_phys + r                                 # physics as an additive base
```

`head_reg`'s last layer is **zero-initialised in weight and bias**, so at initialisation
`reg == mat_phys` exactly: *an untrained network is the physics prediction*, and training learns
only the residual `mat_gt − mat_phys`. Both are log-scaled and share units (base spans 0–2.21,
target 0–3.47).

**The base sits on the regression head, not the classifier** — and the classifier is what is
scored. The physics therefore reaches the reported output *indirectly*, by shaping a trunk that
both heads share. This is why `B_nobase` moves the score at all, and why the base could never
have been removed by zeroing feature columns.

Three shared-weight refinement rounds follow, feeding each node its own current occupancy plus
its neighbours' and its owner's back in as extra channels; round 0 is seeded from the physics
mask. Truncated BPTT — only the last round carries gradient — because a 4 GB card cannot hold
three rounds of activations at this width.

### 4.3 The gradient

One full-graph step per vessel per epoch, 80 epochs, ~29 training vessels per fold:

| term | weight | scope |
|---|---|---|
| `BCE(logit, y)`, `pos_weight = 30` | 1.0 | every node — clot is ~2.7% of nodes |
| `smooth_l1(reg, mat_gt)` | 1.0 | every node — the auxiliary `Mat` regression |
| `1 − soft_score`, a differentiable copy of the deploy metric | 2.0 | **per domain**, wall and off-wall separately, then averaged; starts after 30% of epochs |
| `(log1p σ(logit) − log1p σ_ref)²` — the C0 spread constraint | 2.0 | **per domain**; reference is an EMA over vessels seen so far, i.e. data, not a fitted parameter |

A clot-free vessel's whole loss is scaled by 0.25: it carries 10–20k all-negative nodes and no
information about where a thrombus boundary sits, but it still trains, still updates the C0
reference, and is still held out once like every other vessel.

### 4.4 Wall and off-wall are not two models

The domain split is **not architectural** — it is one network. It enters in exactly three places,
and conflating them has caused real errors here:

* **Domains.** `wall = mask_wall` (healthy no-slip wall); `off = ~solid_boundary`, i.e. true
  lumen, excluding both wall and wound. Every node is in exactly one of wall / off / wound.
* **The loss.** BCE and the regression are global. Only the soft-metric and C0 terms are
  per-domain.
* **The readout.** Separate thresholds per domain; within a domain the `resid` family uses two —
  one for *keeping* a node the physics mask already predicts, one for *adding* a node it does
  not. Four scalars, all selected out-of-fold.

### 4.5 The GNN never sees time — which is why the ODE is the clock

**The base GNN trains on exactly one timestep.** Both targets are taken at `t_eval`, the final
index: `y` is the binary clot there and `mat_gt` is `Mat` there. There is no temporal
supervision anywhere in it. It emits a static final-time field and has no representation of time.

The time axis is supplied entirely downstream, and the physics supplies it:

* **Wall commit times** come from the ODE's own crossing — node *i* switches on at the grid index
  where its integrated `Mat` trajectory first exceeds `c · crit`. Nothing is learned.
  *(This describes the **shipped deploy path**, `readout.lag_anchor: "ode"`. The evaluator
  `eval_strict_temporal.py` differs: its `predict_masks` builds the wall series by thresholding
  the **learned** temporal head, and calls `ode_wall_series` only to date the owner for the
  off-wall lag. Do not describe evaluator numbers with the deploy path's mechanism — §5.3.)*
* **Off-wall commit times** come from a per-node *lag* behind that node's owner wall node, either
  a cohort constant or a regression fitted on the 584 labelled off-wall nodes in the cohort.
* **A temporal head** predicts `P(clot at t)` over a (node, time) table whose static inputs
  include the cached features, the t=0 deposition rate, **the ODE onset index**, and the base
  GNN's out-of-fold score.

**Two consequences, and they govern how §5 must be read.**

1. **A final-time score is provably blind to all of this.** At final time the shipped arm scores
   0.9479 ⟨tmp.final_wall⟩ at the wall, the frozen arm with no temporal head scores
   0.9479 ⟨tmp.frozen_wall_final⟩, and an oracle clock scores 0.9479 ⟨tmp.oracle_wall_final⟩ —
   **identical to four decimals.** Mean-over-time separates them cleanly:

   | | wall (mean-over-time) | off-wall (mean-over-time) |
   |---|---|---|
   | frozen, no temporal head | 0.8599 ⟨tmp.frozen_wall⟩ | 0.5964 ⟨tmp.frozen_off⟩ |
   | shipped, ODE-anchored | 0.9039 ⟨tmp.wall⟩ | 0.7198 ⟨tmp.off⟩ |
   | oracle lag (ceiling) | 0.9814 ⟨tmp.oracle_wall⟩ | 0.8969 ⟨tmp.oracle_off⟩ |

   The temporal head redistributes *when*, never *whether* — worth **+0.044 wall / +0.123
   off-wall** over frozen, with a further +0.078 / +0.177 still on the table to a perfect clock.
**Temporal conditioning has been measured once, and it is worth a great deal.** The
`clot_gnn_v3` era (`docs/PHASE9_ML.md` §13.9) dropped the "rank the nodes, then map ranks onto
a reference schedule" factorisation and predicted `P(node is clot at time t)` directly, with
time as an input feature and predictions forced monotone by a cumulative maximum — *clot does
not un-clot; the production law has no sink*.

| arm | wall | off-wall |
|---|---|---|
| frozen mask | 0.7953 ⟨v3.frozen.wall⟩ | 0.4209 ⟨v3.frozen.off⟩ |
| rank + ODE schedule | 0.8547 ⟨v3.rank.wall⟩ | 0.5369 ⟨v3.rank.off⟩ |
| **time-conditioned** | **0.8845** ⟨v3.timecond.wall⟩ | **0.6110** ⟨v3.timecond.off⟩ |
| oracle schedule (reference) | 0.8908 ⟨v3.oracle_sched.wall⟩ | 0.6619 ⟨v3.oracle_sched.off⟩ |

That is **+0.089 wall / +0.190 off-wall** over the frozen mask, closing ~92% of the wall gap
and ~93% of the off-wall gap to an oracle schedule, with no oracle.

> **Scope, and it is strict.** These are `clot_gnn_v3` numbers: GT flow, a 19-vessel pool, 11
> timesteps, the old 56-column feature block. **They may not be differenced against, or averaged
> with, any current number** — the shipped family is FEM flow, 27 vessels, 69 columns. Quote them
> only as evidence that *time-conditioning pays*, never as a level.

**And the gap this leaves open is the paper's clearest piece of future work.** That head "does
not retrain the GNN": it is a gradient-boosted model over `{the GNN's features, the ODE onset,
the t=0 rate, the GNN's score, the query time, whether the ODE has fired}`. **The network itself
has never received a gradient that knows about time.** Everything temporal arrives through a
small head reading a frozen final-time score, so any timing information the representation did
not need for final-time membership is discarded before the head sees it — and the two physical
constraints (monotonicity, and an off-wall node never committing before its owner) are enforced
*post hoc* on the head's output rather than being architectural.

The well-posed experiment, which the archive does not contain, is to train the GNN itself with
time as an input and a monotone per-node onset target. It is motivated by a measured
92%-of-oracle result rather than a hunch, and it is the only design that puts a learned clock
head-to-head against the ODE — the comparison the feature-zeroing ladder was structurally
incapable of making.

> **Cost corrected 2026-09-09: this is NOT "one training run".** That estimate assumed
> time-resolved targets were available to train against, and they are not. The feature cache
> stores **final-time targets only** — in `v5_fem`, `y` and `mat_gt` are both `(N,)`, not
> `(T, N)`; `features.py` reduces to a single index at `t_eval = resolve_deploy_eval_time_index(...)`
> when the cache is built. So the experiment needs, in order: (1) a cache rebuild carrying
> per-time GT, which means re-reading the raw packs and which **also has to fix the missing
> feature fingerprint** that `v5_fem` already carries (§5.6); (2) a training loop over
> (node, time) rather than one full-graph step per vessel; (3) a monotone onset objective and a
> matching eval path. Three pieces of work and a cache regeneration, not one run.
>
> **The architecture itself is the cheap part**, which is worth recording for whoever picks this
> up: `ClotGNN.__init__` already takes `extra_dim` and `forward` already concatenates an `extra`
> tensor onto the encoder input (`src/clot_ml/gnn.py:93`, `:110`), currently used by the
> recurrent feedback channels. A time scalar needs no architectural change at all. The cost is
> entirely in the data path and the objective.

2. **So the ODE cannot be ablated by zeroing feature columns.** In the deployed stack it is not
   conditioning, it is the mechanism that produces the time axis. Any experiment that removes
   `onset_phys` from a final-time model and reports "the ODE adds nothing" has measured the ODE
   in the one place it does not act. §5.3 states this as the scope limit it is.

---

## 5. Physics, learning, and which one each domain needs  ⟵ **the paper's spine**

**Four points on one axis**, ordered by how much they know about the problem. One protocol, the
same 27 clot-carrying vessels, every score held out, every comparison **paired over vessels**:

| | predictor | what it knows | status |
|---|---|---|---|
| 1 | **naive GNN** (`A_naive`) | nothing — geometry and mesh topology, isotropic messages, one binary head on plain BCE, no `Mat` regression, no metric-shaped loss, no C0, one global cut | **not yet run** |
| 2 | **physics-informed GNN** (`A1_pure`) | the *architecture* only — anisotropic messages, `Mat` auxiliary target, metric-shaped loss, C0 — but no physics conditioning | measured |
| 3 | **pure physics** (`phys_tuned`) | the governing equations, nothing learned | measured |
| 4 | **shipped model** (`A4`) | both | measured |

**The axis is the argument.** 1 → 2 is what the physics-informed *architecture* is worth before
any physics conditioning enters. 2 → 4 is what the conditioning adds on top of it. 3 is the
equations alone. "Is the physics needed?" is a question about the gaps between these four, and
no single ablation answers it.

> **Point 1 is the one the paper cannot skip.** Without it, the baseline is our own model with
> feature columns zeroed — and that architecture was designed by people who already knew the
> physics, so a reviewer can fairly say the control was built by the answer. `plain − naive` is
> the number separating *physics as conditioning* from *physics baked into the architecture*.
> It is expected to be positive, which means every physics contribution reported below is
> **conservative**: a from-scratch GNN should score below `plain`, so the true gap is larger.

Detail on each predictor:

* **phys** — the physics backbone's own occlusion mask. No learning, no threshold, nothing
  fitted. (It therefore carries no selection noise, which advantages it; and it cannot adapt,
  which disadvantages it. State both.)
* **plain** — a GNN with geometry and flow but no physics *features*, no physics residual base,
  no physics rollout seed, and read out under the `plain` family so the backbone's mask cannot
  enter the threshold either.

  > **Do not call this "a plain GNN" in the manuscript — it is not one.** It is *our*
  > architecture with the physics columns zeroed and two doors shut, and that architecture was
  > designed by people who already knew the physics. It still has: **anisotropic, flow-directed
  > message passing** (the single most physics-informed choice in the model — PHASE6_RESULTS
  > §3.4 measured the non-locality to be advective, not diffusive, so a fresh attempt would use
  > an isotropic GNN); an **auxiliary head regressing the simulator's own `Mat` field**
  > (`reg_w=1.0`), which is supervision no naive approach would know to request; a
  > **metric-shaped loss** (`metric_w=2.0`) encoding the evaluation's structure; the **C0**
  > distributional constraint (`shape_w=2.0`) discovered on this problem; the
  > **wall / first-shell / owner** decomposition, which is itself physics (a shell node inherits
  > ~0.16× its owner's `Mat`); and hyperparameters tuned with the full conditioning on.
  >
  > **The bias is conservative**, which is why the headline survives: a genuinely from-scratch
  > GNN would very likely score *below* 0.8533 at the wall, so the true physics contribution is
  > **larger** than the +0.0975 reported. But the label overstates the baseline, and a reviewer
  > reading the methods will see it. Write "the ablated model" or "our architecture without
  > physics conditioning", and state the list above.
  >
  > `A_naive` (`src/clot_ml/ablation.py`) is the from-scratch control that lets the paper say
  > "against a naive mesh GNN" — geometry only, isotropic, one binary head on plain BCE, no
  > `Mat` regression, no metric loss, no C0, one global cut. **Run 2026-09-09 on the shipped
  > generation**; it scores 0.6833 ⟨pvlf.wall.naive⟩ / 0.4849 ⟨pvlf.off.naive⟩, and the
  > architecture gap it exposes (2 − 1 above) is +0.1207 wall / +0.1968 off-wall. The
  > conservative-bias argument above was correct: the true contribution is larger than the
  > ablated baseline suggested.
* **full** — the shipped conditioning, same readout family, so the *only* difference from
  `plain` is the physics.

A fourth predictor exists because the first three are not a fair fight, and the audit that
found it is worth repeating in the methods:

* **phys_tuned** — the *same* physics, given the *same* out-of-fold fitted cut the learned arms
  get. `phys` is a parameter-free mask with no threshold; `plain` and `full` each have one tuned
  per fold. Off-wall that gap is not cosmetic: the bare mask commits **nothing at all on 7 of 27
  vessels** and its burden spans 0.03×–9.3× of GT (against 0.97× at the wall). So part of any
  off-wall "learning beats physics" gap is really "fitted beats unfitted" — a bias running
  *toward* our own conclusion. `phys_tuned` threads the backbone's own continuous field
  (`log_mat_phys` at the wall, `log_mat_off_est` off-wall) through the identical tuner.

#### The result, on the SHIPPED generation (`v5_fem`)

All four predictors share one cache, one protocol and **one verified fold partition**, so every
delta is attributable. These are the manuscript's numbers.

| predictor | wall | off-wall |
|---|---|---|
| 1. naive GNN, from scratch | 0.6833 ⟨pvlf.wall.naive⟩ | 0.4849 ⟨pvlf.off.naive⟩ |
| 1b. *the same, at MeshGraphNet depth* | 0.6383 ⟨mgn.wall⟩ | 0.5207 ⟨mgn.off⟩ |
| 2. physics-informed architecture, no conditioning | 0.8041 ⟨pvlf.wall.plain⟩ | 0.6817 ⟨pvlf.off.plain⟩ |
| 3. pure physics, same fitted cut | **0.9111** ⟨pvlf.wall.phys_tuned⟩ | 0.5277 ⟨pvlf.off.phys_tuned⟩ |
| 4. shipped model | **0.9356** ⟨pvlf.wall.full⟩ | **0.8162** ⟨pvlf.off.full⟩ |

> **Row 1b answers the objection row 1 cannot, and it is the reviewer's first question once they
> have read Pelissier et al.** *"Your baseline is shallow — of course you beat it."* Run
> 2026-09-10: the identical geometry-only, physics-free arm at **15 message-passing layers**
> lands at 0.6383 / 0.5207, i.e. **in the same band as the 4-layer control** — slightly worse at
> the wall, slightly better off-wall. **Depth does not rescue a physics-free model on this
> problem**, so the physics contribution reported below is not an artefact of an undersized
> baseline. The shipped model's margin over row 1b is +0.2972 ⟨mgn.wall.full_minus_mgn⟩ wall /
> +0.2955 ⟨mgn.off.full_minus_mgn⟩ off-wall, and the physics-informed *architecture* alone buys
> +0.1657 ⟨mgn.wall.plain_minus_mgn⟩ over it at the wall.
>
> **Row 1b is a control, not a benchmark, and §9 states the three limits** — it is
> MeshGraphNet-*style*, not a reimplementation; it runs at lr 5e-4 because the shipped 3e-3
> collapsed the 15-deep post-norm stack to a constant field; and it is one arm, not the
> matched-compute suite Pelissier et al. report. Do not write "we beat MeshGraphNet".

Paired over 27 held-out vessels (20 off-wall, where off-wall GT exists):

| comparison | wall | off-wall |
|---|---|---|
| **2 − 1**, what the *architecture* buys | +0.1207 ⟨pvlf.wall.plain_minus_naive⟩ | **+0.1968** ⟨pvlf.off.plain_minus_naive⟩ |
| **3 − 2**, physics vs a physics-free net | **+0.1070** ⟨pvlf.wall.phystuned_minus_plain⟩ | *(physics loses off-wall)* |
| **4 − 3**, what learning adds to physics | +0.0245 ⟨pvlf.wall.full_minus_phystuned⟩ **n.s.** (P=0.069 ⟨pvlf.wall.full_minus_phystuned_p⟩) | **+0.2885** ⟨pvlf.off.full_minus_phystuned⟩ |
| **4 − 2**, what conditioning adds to the architecture | — | **+0.1345** ⟨pvlf.off.full_minus_plain⟩ (P=0.009 ⟨pvlf.off.full_minus_plain_p⟩) |
| **4 − 1**, the whole stack vs from scratch | **+0.2522** ⟨pvlf.wall.full_minus_naive⟩ | **+0.3313** ⟨pvlf.off.full_minus_naive⟩ |

**Read it as three sentences.**

1. **The wall is solved, not learned.** A tuned physics rule reaches 0.9111 and the full model
   0.9356 — a gap of +0.0245 that is **not significant**. Physics beats the physics-free network
   by +0.1070. Whatever the learned part is doing at the wall, it is not measurably improving it.
2. **The lumen is learned, not solved.** Physics collapses to 0.5277 there while the network
   reaches 0.6817 without any physics conditioning and 0.8162 with it — and that conditioning
   term (+0.1345) **is** significant. The equations inform the lumen; they cannot predict it.
3. **Half the wall advantage is architecture, not conditioning.** 2 − 1 is +0.1207 wall and
   +0.1968 off-wall from identical geometry-only features — anisotropic messages, the `Mat`
   auxiliary target, the metric-shaped loss and C0. Every previous ablation in this project
   shared that architecture and therefore attributed none of it.

#### The earlier `v5_split` ladder, retained for provenance only

Superseded by the table above and **not differenceable against it** (§5.4 — the fold partition
changed between the two runs). Kept because the retraction it drove is instructive.

| | wall | off-wall |
|---|---|---|
| physics, bare mask | 0.9078 ⟨pvl.wall.phys⟩ | 0.5200 ⟨pvl.off.phys⟩ |
| **physics, same fitted cut** | **0.9111** ⟨pvl.wall.phys_tuned⟩ | **0.5872** ⟨pvl.off.phys_tuned⟩ |
| plain GNN | 0.8533 ⟨pvl.wall.plain⟩ | 0.7656 ⟨pvl.off.plain⟩ |
| **full model** | **0.9508** ⟨pvl.wall.full⟩ | **0.7949** ⟨pvl.off.full⟩ |

Paired, over n=27 ⟨pvl.n⟩ vessels (n=20 off-wall, where off-wall GT exists). **Quote the
`phys_tuned` rows** — they are the like-for-like comparison:

| comparison | wall | off-wall |
|---|---|---|
| phys_tuned − phys | +0.0032 ⟨pvl.wall.phystuned_minus_phys⟩ **n.s.** | +0.0672 ⟨pvl.off.phystuned_minus_phys⟩ **n.s.** |
| phys_tuned − plain | +0.0578 ⟨pvl.wall.phystuned_minus_plain⟩ **n.s.** (P=0.096) | **−0.1784** ⟨pvl.off.phystuned_minus_plain⟩ |
| **full − phys_tuned** | **+0.0397** ⟨pvl.wall.full_minus_phystuned⟩ | **+0.2077** ⟨pvl.off.full_minus_phystuned⟩ |
| full − plain | **+0.0975** ⟨pvl.wall.full_minus_plain⟩ | +0.0294 ⟨pvl.off.full_minus_plain⟩ **n.s.** (P=0.247 ⟨pvl.off.full_minus_plain_p⟩) |
| *(superseded)* physics − plain, bare mask | +0.0545 ⟨pvl.wall.phys_minus_plain⟩ **n.s.** (P=0.121 ⟨pvl.wall.phys_minus_plain_p⟩) | −0.2456 ⟨pvl.off.phys_minus_plain⟩ — **inflated 27% by the unfitted-rule confound; do not quote** |

**The audit changed one number and confirmed the rest.** Giving physics a fitted cut recovers
+0.067 off-wall, shrinking the learning-versus-physics gap from 0.2456 to **0.1784** — still
with a CI excluding zero, so the conclusion survives a fair test with a smaller effect. At the
wall, tuning buys **+0.0032, nothing at all**: the parameter-free rule is *already at its
optimal operating point there*, which is a physics result in its own right and matches the
burden ratio of 0.97.

#### Physics enters through THREE doors, and at the wall the readout is the widest

The single most important methodological result of the ablation programme, found 2026-09-09.
The `resid` readout family thresholds physics-positive and physics-negative nodes separately —
i.e. **it reads the backbone's occlusion mask**. Under the shipped `auto` protocol the tuner may
choose it for *any* arm, so a from-scratch GNN can borrow physics at the readout:

| arm | `auto` readout (physics available at the cut) | `plain` readout (physics barred everywhere) |
|---|---|---|
| naive GNN | 0.9232 ⟨rd.naive.auto_wall⟩ | 0.6833 ⟨rd.naive.plain_wall⟩ |
| `A0_fresh` | 0.9102 ⟨rd.a0fresh.auto_wall⟩ | 0.6246 ⟨rd.a0fresh.plain_wall⟩ |
| shipped model | 0.9337 ⟨rd.a4.auto_wall⟩ | 0.9356 ⟨pvlf.wall.full⟩ |

**Under `auto`, a naive mesh GNN lands within 0.010 of the shipped model at the wall.** Barring
the physics mask from the threshold drops it to 0.6833. So the physics occlusion mask, used
*purely as a readout device with no learning and no features*, is worth **+0.24 wall** to a
physics-free network — and ~0 to the full model, which already has the physics.

That gives the paper a three-way decomposition of where the physics actually acts:

| channel | what it is | wall value to a physics-free model |
|---|---|---|
| **readout** | `resid` thresholds against the occlusion mask | **+0.24 to +0.29** |
| **architecture** | anisotropic messages, `Mat` auxiliary target, metric loss, C0 | +0.1207 ⟨pvlf.wall.plain_minus_naive⟩ |
| **conditioning** | the physics feature columns | the remainder |

> **This is why both readouts must be reported.** Had the ladder been run under `auto` alone,
> the honest conclusion would have looked like "physics barely matters at the wall" — a naive
> network is 0.010 behind the shipped model there. The effect was hiding in the readout, which
> is the one place nobody thinks to ablate.

**Reproducibility control.** `A4` on this cache scores 0.9337 ⟨rd.a4.auto_wall⟩ against the
archived shipped run's 0.9339 ⟨rd.shipped.auto_wall⟩ — agreement to +0.000, with off-wall
0.7946 vs 0.8089 ⟨rd.shipped.auto_off⟩ inside the noise floor. The ladder's reference **is** the
shipped model.

### 5.1 The two domains give opposite answers

**Off-wall is a learning problem.** The plain GNN beats the tuned physics baseline by
**0.1784** — the largest effect in the paper, and still the largest after the fair-baseline
correction shrank it from 0.2456. Meanwhile adding the physics conditioning back on top of the
learned model buys **+0.0294, not significant** (P=0.247). Off-wall, physics is neither
sufficient nor necessary. This is the same conclusion the archive reached from the other
direction: the deposition ODE cannot drive the off-wall rule, and `mat_field.py` exists
precisely because it cannot.

**The wall needs both, and that is the interesting cell.** The full model beats the plain GNN
(**+0.0975**) *and* beats tuned physics (**+0.0397**), both with CIs excluding zero — while
tuned physics and the plain GNN are **statistically indistinguishable from each other**
(+0.0578, P=0.096). So neither ingredient is sufficient, neither dominates the other, and the
combination is significantly better than either. That is a **synergy** claim, not an additive
one, and it is the sentence the paper should lead with.

**Independent confirmation that the off-wall null is real.** Two runs of the *identical*
configuration with the *same* seeds reproduce to 0.0002 at the wall but only **0.0321
off-wall**. The off-wall "physics adds nothing" effect (+0.0294) is *smaller than the
reproducibility of the configuration itself*, so that null does not rest on the bootstrap
alone. The wall effect (+0.0975) is ~500× the same-config reproducibility.

> **Report the interval, never the means alone.** The three wall means read 0.9078 / 0.8533 /
> 0.9508 and invite the sentence "the zero-parameter physics beats the learned model". Paired,
> that difference is not significant. This exact error was made and corrected on 2026-09-08;
> `scripts/publication/verify_claims.py` now pins the intervals, not just the means.

### 5.2 The feature-conditioning ladder is a null, and the null is informative

A 14-arm ladder (`src/clot_ml/ablation.py`) held each feature group at a constant in turn, at
6 seeds per arm. **No step above the first is resolvable at n=27**, and the arms must be read
against a spread measured on the ablated arms themselves — wall AP 0.0016–0.0555, off-wall AP
0.0081–0.0885 between two independent 3-seed replicates — not against the ±0.045 floor measured
on the shipped configuration, which does not transfer.

Read against that, only two things survive: geometry alone is decisively worst off-wall, and the
wall is carried by the deposition law's dimensionless *gate arguments* rather than by the raw
velocity field. **Neither integrating the deposition ODE nor the upwind transport solve produces
a step outside the noise.** The plan's intended headline — a measured decomposition of *which
solved equation pays* — is not available at this cohort size, and `docs/PHYSICS_ABLATION_PLAN.md`
§8 records why in full.

Two methodological findings from that null are worth a methods paragraph, because both are
general:

1. **A post-readout score is not a model-quality statistic.** The severity metric grants an
   absolute miss grace (`tau_abs = 15`, capped at `0.25·n_gt`) and the cut is fitted per fold,
   so an arm can buy score by committing more. Average precision on the raw field has neither
   property. The two disagree by design — `B_bce`, trained without the metric-shaped loss, has
   the best off-wall AP in the ladder and loses the severity table, which is the shipped
   objective's ranking-versus-cuttability trade working exactly as documented. **Report both and
   say which question each answers**; AP is the right diagnostic, severity the right deliverable.
2. **Ensemble size interacts with arm strength.** Six members help a weak arm far more than a
   strong one, so a 3-seed ladder flatters the strong arms. Compare only at equal seed count.

### 5.3 Is the ODE needed to predict clots in TIME? Measured — and no

The final-time ladder is structurally blind to this: the frozen, shipped and oracle-clock arms
score **identically** at final time, because the temporal head redistributes *when* a node
commits, never *whether*. Only mean-over-time separates them, so that is what this section
reports.

**Temporal prediction is worth a great deal.** On the shipped generation, against a frozen mask
replayed at every timestep:

| | wall | off-wall |
|---|---|---|
| frozen, no temporal head | 0.8459 ⟨odet.frozen_wall⟩ | 0.6262 ⟨odet.frozen_off⟩ |
| shipped temporal head | 0.8999 ⟨odet.with.wall⟩ | 0.7238 ⟨odet.with.off⟩ |
| oracle clock, same set (ceiling) | 0.9768 ⟨odet.oracle_wall⟩ | 0.9121 ⟨odet.oracle_off⟩ |

The head buys **+0.054 wall / +0.098 off-wall**, with a further +0.077 / +0.169 still available
to a perfect clock. Knowing *when* is real score, and there is headroom left in it.

**But the ODE is not the source of that ability.** Holding the ODE's timing information
constant in the head — the explicit `oon/T` input and the `onset_phys` / `onset_phys_ind`
cached columns — and refitting under the identical nested protocol:

| | with ODE timing | ODE timing ablated | paired delta |
|---|---|---|---|
| wall | 0.8999 ⟨odet.with.wall⟩ | 0.9016 ⟨odet.no.wall⟩ | −0.0017, CI [−0.0054, +0.0017] |
| off-wall | 0.7238 ⟨odet.with.off⟩ | 0.7436 ⟨odet.no.off⟩ | −0.0198, CI [−0.0590, +0.0109] |

Both null, and if anything the ablated arm is *marginally better*.

> **This is a live ablation, not a broken switch — checked explicitly.** The committed masks
> changed on **26 of 27** vessels at the wall and 17 of 20 off-wall while the score did not
> move: the head reorganised its schedule and arrived at the same quality. That check exists
> because an earlier attempt at this question *was* a no-op — `--lag-anchor` is consumed only
> inside the `--owner-lag` branch, so without it the flag did nothing and every field came back
> bit-identical. **If a temporal ablation returns exactly zero on every field, suspect the
> switch before believing the null.**

**Scope, and it bounds the claim from below.** The base GNN's out-of-fold score is an input to
the temporal head, and that score was trained with the full physics conditioning; the head also
keeps every physics *magnitude* channel. So physics reaches the head no matter what this flag
does. The defensible statement is **"the ODE's explicit onset channels are redundant given
everything else the head already sees"** — not "the ODE is unnecessary".

> **A doc error this experiment corrected — and the correction was itself half wrong, fixed
> 2026-09-09.** §4.5 says wall commit times come from the ODE's own crossing with nothing
> learned. That describes the **shipped deploy path** (`readout.lag_anchor: "ode"`). The
> evaluator is different, but *not* in the single way this box used to claim.
>
> **What `eval_strict_temporal.predict_masks` actually does is one of three things, and which
> one is a per-fold tuner outcome:**
>
> | when | wall clock | is the ODE in it? |
> |---|---|---|
> | tuner picks a `resid` family | `wall_by_residual` = the ODE's own onset `oon` **plus a learned integer offset** | yes, as the anchor |
> | otherwise | `series_masks` — a pure threshold on the learned temporal head | no |
> | `--wall-clock ode` (**new**) | `ode_wall_series`, nothing learned | yes, and only it |
>
> This box previously asserted the middle row unconditionally; `verify_claims.temporal_ol`'s
> docstring asserted the top row unconditionally. **Both were half right, and neither was safe
> as written** — the wall columns of the owner-lag pair are bit-identical because
> `--lag-anchor` never varied the wall clock under *either* mechanism, not because one of them
> was in force. Do not describe the evaluator's wall numbers with a single mechanism.
>
> **The switch that was missing now exists.** `--wall-clock {head,ode}`
> (`scripts/eval_strict_temporal.py`, added 2026-09-09) forces the pure-physics deploy clock and
> deliberately overrides `wall_resid`, because a learned residual on top of the ODE is a
> *correction to* the deploy mechanism rather than the mechanism itself — leaving it in place
> made the flag a no-op on exactly the folds where the tuner chose that family. Driver:
> `scratch/go_wall_clock.sh`. `score_vessel` now also emits `wall_on_sum` / `off_on_sum`, a
> fingerprint of the schedule rather than the score, so a null can be told apart from a broken
> switch.

#### 5.3.1 The deploy path's ODE wall clock — measured 2026-09-10, and it is redundant too

§5.3 above holds the ODE's onset information constant *inside the temporal head*. It could not
touch the mechanism the shipped artifact actually uses: `readout.lag_anchor: "ode"` dates each
wall node by its own integrated `Mat` crossing, with nothing learned. That path had no switch,
and this section adds one.

**Read the guard before the result.** `ode_wall_series` changes the committed schedule on
**27 of 27** vessels, so this is a live ablation and not another dead flag. The distinction
matters here more than usual: the previous attempt at this question (`--lag-anchor ode` vs
`pred`) returned wall columns that are bit-identical on all 27 vessels, because that flag only
ever selected how the *owner* is dated for the off-wall lag. `score_vessel` now emits
`wall_on_sum` / `off_on_sum` — a fingerprint of the schedule rather than the score — and
`scripts/publication/diag_wall_clock.py` refuses to interpret a delta until it has reported them.

| wall clock | mean-over-time | final-time |
|---|---|---|
| learned temporal head | 0.8999 ⟨wc.head.wall⟩ | 0.9321 ⟨wc.head.wall_final⟩ |
| **ODE, as deployed** | 0.9010 ⟨wc.ode.wall⟩ | **0.9111** ⟨wc.ode.wall_final⟩ |
| **ODE, clock only** (`ode_commit`) | 0.9027 ⟨wc.odecommit.wall⟩ | 0.9321 ⟨wc.odecommit.wall_final⟩ |

**Two results, and they must be quoted together.**

1. **The clock itself is worth nothing measurable.** `ode_commit` against the head is
   **+0.0028, CI [−0.0181, +0.0250]** — null, with the schedule moved on 27/27 vessels and the
   final committed set held *identical by construction* (final-time delta exactly 0.0000, 0/27
   moved). That identity is what makes the mean-over-time delta attributable to the clock and
   nothing else.
2. **The ODE clock as actually deployed also changes *whether*, not only *when*.**
   `ode_wall_series` is the only wall path with **no forced final commit** — `series_masks` and
   `wall_by_residual` both end with `M[-1] = gm & wall`. So a node whose integrated `Mat` never
   crosses inside the horizon is never committed at all, and final-time wall falls **−0.0210**
   (0.9321 → 0.9111) on 24 of 27 vessels. This is a genuine property of the shipped path, not
   an artefact of the experiment.

> **Scope, and it is a hard boundary: this is a WALL-ONLY result.** The off-wall columns are
> bit-identical across every arm — `off_on_sum` moved on **0 of 27** vessels *even under
> `--owner-lag`* — because the tuner keeps selecting the single-stage off-wall rule, which never
> reads the wall series. There is no off-wall number here. **Calling that equality a null would
> repeat precisely the error the `--lag-anchor` pair made**, and the manuscript must not.

**What this does to §5.3's conclusion: it strengthens it and sharpens the recommendation.** The
ODE is redundant as a clock in two independent places now — as onset *features* to the head, and
as the deploy path's *own* dating mechanism. But the follow-up advice below is unchanged and for
the same reasons: the trajectory is integrated anyway, so deleting the clock saves nothing, and
the +0.077 wall headroom to an oracle clock (§5.3) is still where the effort belongs. The one
thing that *is* now actionable is narrower: **the missing forced final commit costs 0.0210 at
final time for nothing**, and adding it is a one-line change to the deploy readout.

#### Should the shipped model drop the ODE timing? No — and not because it pays

Asked and answered 2026-09-09. The ablation says the ODE's onset channels are redundant, which
invites the obvious follow-up. **Do not make the change**, for reasons that are about cost and
evidence rather than about the null:

* **It saves nothing.** The ODE trajectory must be integrated anyway — `log_mat_phys`,
  `onset_phys` and `log_mat_owner` are cached feature columns, `mat_phys` is the regression
  base, and `phys_mask` drives the `resid` readout that is worth **+0.24 wall** to a
  physics-free model. The onset channels are *derived from a trajectory the pipeline already
  computes*. Deleting them removes part of a 17% stage that still has to run.
* **It gains nothing.** −0.0017 wall / −0.0198 off-wall, both null.
* **The null is bounded from below.** The head reads a physics-conditioned base-GNN score, so it
  may be recovering the schedule from physics by another route. "Redundant given what else the
  head sees" is not "unnecessary".
* ~~**The deploy path was not tested.**~~ **MEASURED 2026-09-10, and it is also null.** The
  switch now exists (`--wall-clock {head,ode,ode_commit}`, driver `scratch/go_wall_clock.sh`)
  and the answer does not change the section's conclusion — it extends it from the head's
  onset *features* to the deploy path's *clock itself*. See §5.3.1.
* **n=27, one operating point.** A physics anchor is exactly what would matter off-regime, which
  is the paper's stated scope limit and is untested.

**Where the effort should go instead.** The oracle clock sits +0.077 wall / +0.169 off-wall
above the current head (§5.3). That headroom is large, and neither the ODE nor the present head
is capturing it — it is the real target for anyone working on temporal prediction here.

### 5.4 The two ladders may NOT be differenced — folds changed between them

**Found 2026-09-09, and it invalidates one comparison while leaving the headline intact.**

`src/clot_ml/geometry_class.py` and `geometry_splits.py` were both modified on 2026-09-07 at
14:06. The `v5_split` ladder ran *before* that edit; the `v5_fem` ladder ran *after*. The
geometry-stratified folds are therefore **different partitions**, not merely relabelled ones --
the baseline vessels cluster identically but the priority vessels are dealt differently
(`comsol045` sits with `042` in one and with `040`/`048` in the other).

| comparison | status |
|---|---|
| any two arms **within** the `v5_fem` ladder | **valid** -- verified identical folds for `A_naive`, `A1_pure`, `A0`, `A1p`, `A2`, `A4` |
| any two arms **within** the `v5_split` ladder | valid on its own terms |
| **any `v5_split` arm against any `v5_fem` arm** | **CONFOUNDED -- generation and fold assignment vary together. Do not difference them.** |

**How it surfaced, because the mechanism is worth keeping.** `phys_tuned` is built from
`log_mat_phys` and `log_mat_off_est`, and those columns -- along with GT `y` and the `solid`
mask -- are **bit-identical across both caches on all 40 vessels**. Its score therefore *could
not* differ between ladders for any reason except the folds, and it did (off-wall 0.5872 vs
0.5277). A predictor that shares its input and its labels across two runs is an accidental but
excellent tripwire for a protocol change; keep one in any future cross-run comparison.

**Consequence for the manuscript:** quote the `v5_fem` (shipped-generation) four-point axis, and
do not present a split-vs-fem delta as a generation effect. Re-running the `v5_split` ladder
under the current fold code would be needed to make that comparison, and it is not worth the
GPU-hours -- the shipped generation is the one the paper reports.

### 5.5 Scope — what none of this measures

**Everything above is FINAL-TIME.** The shipped artifact is `kind: unified_v0` with
`readout.lag_anchor: "ode"`; a temporal head and a chemistry-replacement layer sit above the
base GNN that was ablated. Measured on `A4`, the temporal head is worth **+0.044 wall / +0.123
off-wall** over a frozen mask, with an oracle-lag ceiling a further +0.078 / +0.177 above that —
and **final-time scores are identical for frozen, shipped and oracle** (0.9479 wall for all
three). A final-time score is therefore *provably blind* to the temporal head, and since
`onset_phys` is an onset-timing prior, the ODE's contribution is measured here in close to the
one place it should not appear. Do not write "the ODE adds nothing" on this evidence.

Two further leaks bound what the ladder can claim, both found while writing this section:

* **The residual base was active in every A arm.** With `phys_base=True` the auxiliary
  regression loss is `smooth_l1(mat_phys + r, mat_gt)`, so the backbone's prediction is handed
  to the network as a free offset on a task that shares the whole trunk with the classifier.
  Only `A0_pure`, `A0_fresh`, `A1_pure` and `B_nobase` disable it — which is why the central
  comparison above uses `A1_pure`, not `A0`, as "no physics".
* **The flow reaches every arm through the architecture.** `apply_arm` zeroes only the node
  feature matrix; `gnn.to_device` builds the edge features and both aggregation weights straight
  from `S["u"]`/`S["v"]`. No column zeroing produces a flow-free model — `iso=True` is the only
  switch that does, and `A0_fresh` is the only arm that sets it.

### 5.6 Generation — RESOLVED 2026-09-09

**The generation mismatch is closed.** The four-point axis above is now measured on `v5_fem`,
the shipped `clot_ml_final_0` generation, which is the same generation §6 and every other
published number comes from. `A_naive`, `A1_pure`, `A4` and `A0_fresh` were all retrained there
(6 seeds each, verified identical folds), so §5 and §6 no longer cross generations.

Two residual notes, both minor and both stated rather than buried:

* The earlier `v5_split` ladder is retained for provenance only and **must not be differenced
  against these numbers** — the fold partition changed between the runs (§5.4).
* `src/clot_ml/features.py` was edited after the `v5_split` cache was built, so *that* cache
  fails its own feature fingerprint. It does not affect the `v5_fem` numbers, and within the
  split ladder every arm read the same cache, so that comparison remains internally sound.

## 6. It works — geometry generalization

Geometry-stratified 5-fold over the 36-vessel non-SEALED pool; every vessel held out exactly
once; priority-class vessels dealt into different folds by construction. Out-of-fold, final time,
27 clot-carrying vessels. Off-wall means exclude vessels with empty off-wall GT, which are scored
as a separate false-positive row and never folded into a recall-bearing mean.

| class | n | wall (mean, SEM) | off-wall (mean, SEM) |
|---|---|---|---|
| baseline | 20 ⟨t4.baseline.n⟩ | 0.9127 ⟨t4.baseline.wall⟩ (0.0239 ⟨t4.baseline.wall_sem⟩) | 0.6437 ⟨t4.baseline.off⟩ (n=13 ⟨t4.baseline.off_n⟩) |
| stenosis | 5 ⟨t4.stenosis.n⟩ | 0.7721 ⟨t4.stenosis.wall⟩ (0.0548 ⟨t4.stenosis.wall_sem⟩) | 0.7426 ⟨t4.stenosis.off⟩ |
| aneurysm | 2 ⟨t4.aneurysm.n⟩ | 0.9723 ⟨t4.aneurysm.wall⟩ | 0.9204 ⟨t4.aneurysm.off⟩ |

**Do not write "generalization holds across geometry class."** Baseline 0.9127 against stenosis
0.7721 is a 0.141 gap on a combined SEM of 0.060 — about 2.3 SEMs. **Stenosis is measurably
harder at the wall**, and that reading only exists because the classification was corrected by a
manual mesh read: `comsol045`/`comsol046` are stenosis (previously bucketed baseline) and
`comsol047` is a second aneurysm. Off-wall runs the other way (stenosis 0.7426 vs baseline
0.6437, ~1.5 SEMs — real in direction, not significant).

Two dispersions and they are not interchangeable: **SEM** is across vessels ("would another draw
of vessels move this"); the **config floor** — wall 0.0037 ⟨t4.noise_floor.wall⟩, off-wall
0.0432 ⟨t4.noise_floor.off⟩ — is across refits ("would another seed move this"). Name which one
any stated uncertainty is.

**A threshold-free summary, because a reviewer will ask what the readout is worth.** Pooling
nodes across all 27 ⟨aucpr.n_vessels⟩ vessels: **AUC-PR 0.908** ⟨aucpr.wall⟩ at the wall against
a base rate of 0.181 ⟨aucpr.wall_base⟩ (5× the floor), and **AUC-PR 0.604** ⟨aucpr.off⟩ off-wall
against a base rate of 0.0034 ⟨aucpr.off_base⟩ (177× the floor). *This answers "where should the
cut go", not "how well does it generalize per vessel" — the table above answers that. Report
both; never substitute one for the other.*

**The sealed holdout, read exactly once** (2026-09-03, and it may never be read again):
wall **0.9572** ⟨sealed.wall_mean⟩, off-wall **0.6180** ⟨sealed.off_mean⟩ over
`comsol007`/`013`/`031`/`043`. Scored under deployed FEM flow on the deploy metric — **not
directly comparable to the table above**, which is BATC₀. Report sealed as sealed.

**The C0 constraint** — a ~10-line training-time penalty on the field's within-domain spread,
which is what makes the score cuttable by a single cohort constant: off-wall
0.7046 ⟨c0.off_before⟩ → 0.8358 ⟨c0.off_after⟩, paired delta **+0.1312** ⟨c0.off_delta⟩,
n=20 ⟨c0.off_n⟩; wall 0.9218 ⟨c0.wall_before⟩ → 0.9398 ⟨c0.wall_after⟩, p=0.0125 ⟨c0.wall_p⟩.

> **Methods hygiene that must appear.** Architecture and hyperparameters were selected on the
> FIT/DEV split *before* the folds were run, so out-of-fold scores measure geometry transfer of
> the fitted **weights**, not of the whole pipeline end to end. Stating this is what makes the
> sealed number worth quoting.

---

## 7. Why the flow is solved and not learned

Ordering constraint: **§7 comes after §6, never before.** The reader must believe the tool works
before the negative results read as design justification rather than as a list of things that
broke.

### 7.1 The prize is capped before accuracy enters the argument

**Lead with this, because it needs no appeal to how good anyone's flow surrogate is.**

The t=0 flow solve is **7.9%** ⟨timing.flow_ceiling⟩ of end-to-end runtime — 4.66 s
⟨timing.fem_flow_s⟩ of 58.9 s ⟨timing.median_s⟩ — against 73% ⟨timing.share_rollout⟩ for the
rollout. So **an instantaneous, perfect flow surrogate would save at most 7.9%.** That ceiling is
a property of the pipeline, not of any model.

RGP-DEQ gets close to it: it does the flow stage in 0.667 s ⟨timing.rgp_flow_s⟩, a genuine ~7×
speedup *at that stage*, worth ~6.8% end-to-end. **And it costs −0.1075** ⟨rgp.off_delta⟩
off-wall (p=0.0045 ⟨rgp.off_p⟩).

> **This is an Amdahl argument, and it is the cleanest form of the result.** The trade is bad by
> construction: you are paying a significant, measured accuracy regression for a saving that is
> bounded at 7.9% before the first epoch is trained.
>
> **Cite the precedent rather than claiming the pattern.** A configuration-dependent Amdahl bound
> on what replacing one stage can buy is already in print for scientific-ML surrogates
> (e.g. arXiv 2606.15622, a GPU-native DNN surrogate for kinetic Fokker–Planck flows). What is
> ours is not the bound — it is that the ceiling is **computed and then paid against a measured
> downstream accuracy cost**, which turns a scheduling observation into a design decision.
> Presenting the bound itself as novel is an avoidable overclaim. A better learned flow surrogate — including
> the published ones that beat ours on wall-shear — moves the numerator toward 7.9% and changes
> nothing about the ceiling. *We are not claiming learned flow surrogates are bad. We are showing
> that in this pipeline the component worth replacing is not the one people replace.*

The corollary is the actionable one for anyone building a similar stack: **profile the pipeline
before choosing what to learn.** The 73% is the rollout, and that is where a surrogate pays.

### 7.2 RGP-DEQ is a good model. That is what makes the result worth reporting

**Do not present this as a strawman.** The learned flow surrogate is a deliberate,
physics-informed design — a graph attention network modulated by the local physics, Perceiver
global mixing for the elliptic pressure coupling, and a deep-equilibrium fixed point rather than
a fixed depth, with the boundary condition imposed *hard* (`u = u_prior + sdf · u_residual`) so
no-slip holds by construction. It is seeded from the FEM prior rather than an analytic one, its
residual head is ReZero-gated, and its authority is band-localised to where it has signal.

It works. Over the 33-vessel diagnostic cohort, median velocity **rel-L2 0.0165**
⟨rgpdeq.rel_l2_med⟩ — under 2% field error — with `dsrx` correlation
0.9938 ⟨rgpdeq.dsrx_corr_med⟩ and **0 ⟨rgpdeq.empty_gate⟩ empty gates**. Against the classical
solve's 0.0061 ⟨fem.rel_l2_med⟩ that is the same order of magnitude, and it is produced in
0.667 s ⟨timing.rgp_flow_s⟩ against 4.66 s ⟨timing.fem_flow_s⟩.

**And we still ship the classical solve.** The reason is one number:

| | velocity rel-L2 (median) | wall-gate Jaccard (median) |
|---|---|---|
| local FEM solve (**shipped**) | 0.0061 ⟨fem.rel_l2_med⟩ | **0.9240** ⟨fem.gate_jac_med⟩ |
| RGP-DEQ | 0.0165 ⟨rgpdeq.rel_l2_med⟩ | **0.7582** ⟨rgpdeq.gate_jac_med⟩ |

The two fields are close in norm and far apart in the statistic the consumer actually reads —
which is §7.3's whole point — and the downstream cost is **−0.1075** ⟨rgp.off_delta⟩ off-wall
(p=0.0045 ⟨rgp.off_p⟩). Paid for 6.8% of runtime, against a 7.9% ⟨timing.flow_ceiling⟩ ceiling.

> **This is a deployability decision, and it is worth stating as one.** What makes the tool
> prospective rather than retrospective is that *nothing in the pipeline needs a ground-truth
> velocity field* — we solve the flow ourselves on the user's own mesh. **Both** candidates
> satisfy that; the choice between them is purely accuracy-versus-speed, and at a 7.9% ceiling
> the accuracy wins without argument. We report RGP-DEQ because a competent learned field that
> loses downstream is the evidence, not because it failed to be built well.

### 7.3 Three attempts, decreasing ambition, one answer

| attempt | what it tried | outcome |
|---|---|---|
| RGP-DEQ | learn the whole t=0 field | wall −0.0078 ⟨rgp.wall_delta⟩ (p=0.5325 ⟨rgp.wall_p⟩, n=27 ⟨rgp.wall_n⟩, n.s.); off-wall **−0.1075** ⟨rgp.off_delta⟩ (p=0.0045 ⟨rgp.off_p⟩, n=20 ⟨rgp.off_n⟩) |
| local kinematic corrector | learn a local correction on frozen flow | worse than doing nothing |
| closed-loop coupling | feed clot back into flow every step | oracle upper bound inside the noise floor (below) |

Before asking whether flow is worth *learning*, ask whether it is worth *solving more than once*. We
built an oracle closed loop: ground-truth clot occupancy fed back into the flow at every step, the
measured post-gelation wall-shear collapse applied under it, the deposition gate re-evaluated with
the consumer's own operator. No model error, no localisation error — this is the upper bound on
what *any* per-step flow correction, learned or solved, could contribute over solving once at t=0.

| domain | closed − open | noise floor |
|---|---|---|
| wall, non-wound (n=5 ⟨closedloop.nonwound_n⟩) | +0.0093 ⟨closedloop.nonwound_wall_delta⟩ | ±0.024 |
| off-wall, non-wound | +0.0143 ⟨closedloop.nonwound_off_delta⟩ | ±0.074 |
| wound region, `w_reg` (n=6 ⟨closedloop.wound_n⟩) | +0.0015 ⟨closedloop.wound_wreg_delta⟩ | ±0.024 |
| wound lumen, `w_lum` | +0.0000 ⟨closedloop.wound_wlum_delta⟩ (exactly flat) | ±0.074 |

Every delta sits inside this cohort's own noise floor (±0.024 wall / ±0.074 off-wall). **An oracle
with zero model error and zero localisation error cannot buy a measurable gain by re-solving the
flow after t=0** — on the ordinary cohort or on the wound vessels, where the gate moves the most
(§8). That is the license for solving once: the ceiling on evolving-flow coupling is not "we
could not build it well enough," it is that the perfectly-built version is inside noise. Read the
sign as flat, not as a small win or a small loss — an earlier run of this comparison had the
opposite sign inside the same noise band (PUBLICATION_NOTES.md §1).

### 7.4 Why it fails, when it fails

**The diagnosis that unifies them.** Over 33 vessels ⟨flowreq.n⟩, correlation with the downstream
drop: wall-gate firing ratio **+0.715** ⟨flowreq.fire_ratio_r⟩ and gate Jaccard **−0.707**
⟨flowreq.gate_jaccard_r⟩ dominate, while velocity **rel-L2 is +0.224** ⟨flowreq.rel_l2_r⟩ —
weaker, though no longer negligible.

**Frame this as the gated-coupling claim, never as surrogate incapability.** The defensible core
is:

> *Where a learned field feeds a thresholded consumer, the surrogate's accuracy requirement is
> set by the consumer's decision statistic, not by field norms — so report the decision statistic.*

That claim is untouched by how good anyone else's flow surrogate is. Better: the published
PI-GNNs become **support**, because every one of them reports in exactly the norms shown here to
be the weakest predictor of downstream fitness.

#### The cross-lab test of the mechanism — the strongest paragraph available to §7

The gated-coupling claim predicts something checkable in someone else's data: *where the coupling
is smooth, learning the flow should work.* It does.

**Pelissier, Meliga & Hachem (2026)** learn the velocity field autoregressively, inside a
thrombosis model whose flow→chemistry coupling has **no low-shear stagnation gate and no
shear-gradient deposition** — its only threshold is on agonist *concentration*, so flow enters
chemistry through advection and a smooth shear-activation rate. Their learned flow succeeds. Ours
sits in a model where `sr < lss` carries ~80% of deposition (§3), and learning it costs
−0.1075 ⟨rgp.off_delta⟩ off-wall (p=0.0045 ⟨rgp.off_p⟩).

**Two labs, two couplings, opposite answers on the same component.** That is what converts "we
measured our pipeline" into *"whether a component should be learned is not knowable a priori"* —
the claim §1 leads with. It is also the reason to cite them generously: their result is not a
threat to ours, it is the second data point that makes ours a finding rather than an anecdote.

> **Write this fairly or it backfires.** They are not a foil. Their model is the easier coupling
> *and* they engineer it well — pulsatile flow, patient-derived geometry, benchmarks against
> MeshGraphNet / BSMS-GNN / Transolver++. The sentence is "the coupling differs, and so does the
> answer", never "they had it easy."


**And the diagnosis is deployable, which is what turns it from a characterisation into a
contribution.** The predictive statistic — the firing set of the wall gate — needs no reference
field, so it can be computed on a new vessel. `src/clot_ml/preflight.py` refuses a vacuous
prediction before the rollout is paid for.

> **Narrow this claim against the OOD literature, in one clause.** Reject-option deployment and
> out-of-distribution detection for surrogates is a mature field (SmOOD for aircraft surrogates,
> arXiv 2209.03438; the OOD-detection surveys; risk-control framings in clinical ML). **"We add a
> refusal gate" is not a contribution and will have a citation thrown at it.** What is defensible
> is narrower and true: our refusal criterion is computed from the **consumer's own decision
> statistic** — the wall gate's firing set — rather than from input-space distance or predictive
> variance, and it needs no reference field, so it runs on a vessel that has never been solved.
> That is the sentence. It costs one clause and it survives the citation.

> **Quote the detector carefully — the arm matters.** On the arm that *had* empty gates (the
> pre-cross-fit RGP-DEQ), the detector caught 5 of 5 with 0 false alarms; that is the only arm on
> which detection can be demonstrated at all, and it is a historical measurement.
> **On the shipped path today it has nothing to catch**, which is itself the point: the current
> artifact reports 0 ⟨preflight.fem_fail⟩ failures and 0 ⟨preflight.fem_warn⟩ warnings over
> n=37 ⟨preflight.fem_n⟩ vessels on the FEM path, and 0 ⟨preflight.pred_fail⟩ failures with
> 1 ⟨preflight.pred_warn⟩ warning over n=37 ⟨preflight.pred_n⟩ on the current cross-fit arm.
> So the honest sentence is *"zero false alarms on the deployed path, and the failure mode it
> guards against no longer occurs on the shipped flow arm"* — **not** "5/5 detection", which
> describes an arm the paper does not ship.

---

## 8. What the tool can do — the deployability argument

This is the section that says why anyone should care, and it is stronger than the accuracy table
alone.

* **No ground-truth state anywhere, at any time.** Training, readout selection and evaluation
  all run on a local Carreau Navier–Stokes solve on the vessel's own mesh. Most thrombosis-ML
  work presumes CFD is already available; this is what makes the tool prospective rather than
  retrospective.

  > **This is the sharpest distinction from the nearest published work, and it is a difference
  > in *product*, not in accuracy.** Pelissier et al. 2026 is a **solver accelerator**: its node
  > features are the four physical fields at `t` and `t−Δt` plus the next inlet velocity — "the
  > same input information as the CFD solver at each time step" — so it is seeded from a real CFD
  > state after the ramp and first cardiac cycle and time-steps forward, needing injected training
  > noise to keep the rollout stable. Its speedup is measured against *continuing* a simulation
  > already running. Ours is a **predictor**: a mesh goes in, the full-horizon clot field comes
  > out, and its speedup is measured against not needing the simulation at all. Both are
  > legitimate; they answer different questions and should not be compared on a single accuracy
  > number. Say which one you are, early.
* **~10³ speedup, and it changes the mode of use, not just the wall-clock.** 58.9 s
  ⟨timing.median_s⟩ median end-to-end over n=34 ⟨timing.n⟩ vessels, IQR
  [40.7 ⟨timing.q1⟩, 67.3 ⟨timing.q3⟩] s, against COMSOL's ~48 h — **2,934×**
  ⟨timing.speedup⟩. *Boundary:* pack → FEM t=0 → features → rollout. **Meshing and geometry
  construction are excluded and COMSOL's 48 h includes them** — say so, or add the meshing cost.
  Training is one-time; break-even is a handful of vessels.
* **The classical solve is not the bottleneck** — FEM t=0 is 7.9% ⟨timing.flow_ceiling⟩ of
  end-to-end and the rollout is 73% ⟨timing.share_rollout⟩. See §7.1: this caps what learning
  the flow could ever be worth, *before* accuracy enters the argument.
* **It refuses when it should.** The pre-flight gate (§7) computes the wall gate's firing set,
  which needs no reference field and therefore works on a new vessel, and declines to produce a
  vacuous prediction: **0 ⟨preflight.fem_fail⟩ false alarms in 37 ⟨preflight.fem_n⟩ vessels** on
  the shipped path.
* **What a user actually gets.** A per-node clot field on their own mesh, over the full horizon,
  with a commit time per node — not a scalar risk score and not a single final frame. That is
  the capability the 11-study review found missing (§2).

### 8.1 Honest operating envelope — quote this, not just the headline

A tool section that only reports the mean is the one a practitioner distrusts first.

| | |
|---|---|
| where it is strong | baseline vessels, wall: 0.9127 ⟨t4.baseline.wall⟩; aneurysm 0.9723 ⟨t4.aneurysm.wall⟩ |
| where it is weakest | **stenosis, wall: 0.7721** ⟨t4.stenosis.wall⟩ — a real, ~2.3-SEM deficit, not noise |
| threshold-free | AUC-PR 0.908 ⟨aucpr.wall⟩ wall against a 0.181 ⟨aucpr.wall_base⟩ base rate; 0.604 ⟨aucpr.off⟩ off-wall against 0.0034 ⟨aucpr.off_base⟩ |
| the one sealed read | wall 0.9572 ⟨sealed.wall_mean⟩, off-wall 0.6180 ⟨sealed.off_mean⟩, spent once |
| scope | 2D, Re = 450, simulation ground truth, n=27 scored vessels |

**Say "stenosis is measurably harder" in the abstract's limitations clause.** It is the single
most useful sentence in the paper for someone deciding whether to run it on their geometry.

---

## 9. Limitations, stated not hidden

* **One operating point, and the nearest published work does better here.** Re = 450 for every
  anchor. Framed as **scope**: the generalization claim is over geometry and flow is an input.
  State it in the abstract, not only here. **And name the comparison** — Pelissier et al. 2026 run
  pulsatile physiological waveforms over six cardiac cycles *and* stress-test on three unseen
  inflow profiles. A reviewer who knows that paper will notice; stating it first costs nothing and
  buys credibility for everything else in this list.
* **Do not write "any flow field."** The conditioning is not fully dimensionless — `log_sr` is at
  physical scale, shear aggregates clip at `min(sr, 500)`, and `sr_over_lss` / `dsrx_over_sgt`
  clip to ±40. The ratio channels are the right dimensionless groups; the **hard clips are
  regime-dependent**. Honest phrasing: *"conditioned on a t=0 flow field at the operating point
  of the training corpus."*
* **2D.** A 2D surrogate of a 2D model. The review's most-repeated future direction is 3D.
* **Simulation ground truth.** No in-vitro or in-vivo data. The claim is fidelity to a published
  model, never clinical prediction. One sentence of overclaim in the abstract costs the paper.
* **Aneurysm is n=2 out-of-fold**, plus n=1 in SEALED. A property of the data, not the split.
* **Sealed holdout n=4**, spent once.
* **Cohort size — state the arithmetic, not just the number.** We report n=27 ⟨pvl.n⟩ scored
  vessels. Contemporaneous mesh-surrogate papers train on 10²–10³ geometries (101 in Pelissier
  et al.; 105 patient-derived aneurysms in one released benchmark; 1,500 synthetic coronary
  bifurcations; 4,200 single-vessel geometries with paired OpenFOAM solutions). **The bare number
  read against that norm looks like a weakness; the arithmetic makes it motivation.** Those
  cohorts are steady or single-physics CFD. Ours is ~48 h of coupled 12-species COMSOL per
  vessel, so 40 vessels is roughly 80 machine-days of ground truth. **Sample size here is bounded
  by the simulator, not by effort — which is the same fact that makes the surrogate worth
  building.** One sentence, with the numbers.
* **Geometry provenance.** `comsol0NN` are parametric-synthetic vessels — rename for publication
  or a reader will believe they are people. Note also that the near neighbour uses
  **patient-derived** geometry (InTrA); our answer is the published ground-truth *model* (§3), and
  span across three geometry classes rather than one family — not a claim to anatomical realism.
* **One depth-matched control, not a benchmark suite — RUN 2026-09-10, and quote it carefully.**
  `A_naive` is a from-scratch control at the shipped depth, so on its own it bounds only the
  from-scratch floor. `A_mgn` adds the missing comparison: the same geometry-only, physics-free
  arm at **MeshGraphNet depth** (15 message-passing layers, dim 64 — the configuration Pelissier
  et al. report for their own MGN replication). It scores **0.6383 ⟨mgn.wall⟩ wall /
  0.5207 ⟨mgn.off⟩ off-wall**, against `A_naive`'s 0.6833 ⟨pvlf.wall.naive⟩ / 0.4849
  ⟨pvlf.off.naive⟩ — the same band. The shipped model's margin over it is
  **+0.2972 ⟨mgn.wall.full_minus_mgn⟩ wall / +0.2955 ⟨mgn.off.full_minus_mgn⟩ off-wall**.
  **The useful sentence is that depth does not rescue a physics-free model here**, so the
  reported physics contribution is not an artefact of a shallow control.

  > **Three limits on that sentence, all of which belong in the methods.** (1) `A_mgn` is
  > **not a MeshGraphNet reimplementation** — no updated edge features, no separate edge
  > encoder/decoder, our training loop rather than Pfaff et al.'s noise-injected rollout. Write
  > "a MeshGraphNet-style depth-matched control", never "MeshGraphNet". (2) It runs at
  > **lr 5e-4, not the shipped 3e-3**, because at 3e-3 the 15-deep post-norm stack collapsed to
  > a constant field (std 1.2e-06) and scored 0.1639 — a *failure to train*, not a result, and
  > it is excluded from the ledger for that reason. So `A_mgn` and `A_naive` differ in learning
  > rate as well as depth. (3) One retry, no learning-rate sweep on either arm. This is one
  > control, not the matched-compute benchmark suite Pelissier et al. run, and claiming
  > otherwise would be the same overclaim in a new place.
* **The `v5_split` and `v5_fem` ladders cannot be differenced** (§5.4) — the fold partition
  changed between the runs. Each is internally valid; the manuscript quotes `v5_fem` only.

---

## 10. Numbers NOT yet verifiable — do not put these in the manuscript

Kept short and visible. Each needs an artifact and a ledger row before it can be quoted.

* **FEM rel-L2 0.0064 median over 54 packs** (PUBLICATION_PLAN §12.3). No ledger row yet.
* **The gate-churn numbers from Fig C** — 3.7% of wall nodes fire, 2.4× as many sit within one
  threshold width above, and at 10% injected shear error the firing set changes size by 1.6%
  while 9.5% of it swaps members (Jaccard 0.910). These now resolve to an artifact
  (`outputs/publication/data/coupling_gate.json`) and a command
  (`scripts/publication/plot_coupling_gate.py`, seeded, deterministic), **but they have no
  ledger id yet** — add rows to `verify_claims.py` before any of them enters the manuscript.
  Note the scope: stagnation branch only, because the shipped cache carries no `dsrx`.
* **"Replicated ×3" for C0.** The current artifact is one paired comparison; the ×3 replication
  claim comes from the retired `v5w`/GT lineage.
* **The three regenerating commands for `split_vs_shipped.json`, `crossfit5_vs_shipped.json` and
  `c0_ablation_paired.json`.** `scripts/eval_flow_source_paired.py` was removed from the tree on
  2026-09-07 and only `docs/` still references it. **The JSONs survive and every number above
  resolves against them, but they can no longer be regenerated** — which violates
  PUBLICATION_NOTES §7's own rule that a claim must resolve to a command. Restore the script (a
  copy exists under `dist/LocalFEMSolver-Predict-win64-1.0/scripts/`) or record the removal
  deliberately.
* **COMSOL flow solve at t=0 ≈ 45 s** — recollection/estimate, not a logged artifact. No ledger
  row. If confirmed, it's a *second*, separate timing comparison from §7.1's pipeline-level
  2,934×: at just the t=0 flow stage, COMSOL (~45 s) vs local FEM (4.66 s ⟨timing.fem_flow_s⟩,
  ~10×) vs RGP-DEQ (0.667 s ⟨timing.rgp_flow_s⟩, ~67×). Do not conflate with
  `comsol_reference_hours` in `outputs/publication/data/timing.json` (48 h), which is the full
  coupled 12-species pipeline, not the t=0 solve alone. Needs an actual COMSOL log or a timed
  rerun before it gets a claim id and enters the manuscript.
* ~~**Two arms that would close §5's open defects have not been run**~~ **— CLOSED 2026-09-10.**
  `A_naive`, `A0_fresh` and `A_mgn` are all measured on the shipped generation (§5, §9), and the
  deploy-path wall clock (§5.3.1) is measured too. Historical text: — `A0_fresh` (the
  physics-free floor: no physics in features, base, seed *or* edges) and the `v5_fem` replicate
  of `full`/`plain`. Queued in `scratch/go_gpu_window.sh` (~6 GPU-h, resumable), together with `A_naive` --
  the from-scratch control that makes point 1 of §5's four-point axis measurable.
  Until they land, §5.5's generation caveat stands and the floor row is missing.
* **The wound section (§6 of the old PUBLICATION_PLAN) is still blocked on provenance** — whether the ungated
  wound boundary condition is Giulia's and whether it is published. That decides the author list.

---

## 11. Section-to-evidence map

| § | claim | evidence |
|---|---|---|
| 1/5 | **the lead claim** — physics and learning swap places between domains | **Fig A**, `scripts/publication/plot_division_of_labour.py` over `outputs/ablation/physics_vs_learned_fem.json` |
| 2 | the gap, and the near neighbour | PRISMA review, 11 studies; Pelissier, Meliga & Hachem 2026, *Comput Biol Med* 208:111649 |
| 3/7 | the coupling is a gate; it loses identity before size | **Fig C**, `scripts/publication/plot_coupling_gate.py` -> `outputs/publication/data/coupling_gate.json` |
| 3 | ground truth | `comsol_models/` `.mph` trees |
| 4 | how the tool works; where the labour divides | `src/clot_ml/gnn.py`, `scripts/train_clot_gnn.py`, `src/clot_ml/temporal.py`, `outputs/ablation/temporal_A4_ode.json` |
| 5 | physics vs learning vs both | `outputs/ablation/physics_vs_learned_fem.json`, `ablation_report_ablfem.json`; **Fig A** `plot_division_of_labour.py` |
| 5 | depth does not rescue a physics-free model | `outputs/ablation/physics_vs_learned_mgn.json` (`A_mgn`, 15 layers) |
| 5.3.1 | the deploy path's ODE wall clock is redundant | `outputs/ablation/temporal_wallclock_{head,ode,ode_commit}.json`; `diag_wall_clock.py` |
| 6 | geometry generalization | `outputs/publication/data/table4_kfold.json`, `outputs/deployclot/eval_sealed.json` |
| 7 | flow solved not learned; the cross-lab coupling test | **Fig B**, `scripts/publication/plot_amdahl_ceiling.py`; `crossfit5_vs_shipped.json`, `flow_requirement.json`, `preflight_validation.json`; Pelissier et al. 2026 §2.2 (smooth coupling, learned flow succeeds) |
| 8 | deployability | `outputs/publication/data/timing.json` |
