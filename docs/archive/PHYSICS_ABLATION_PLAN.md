> **ARCHIVED 2026-09-11.** This experiment ran, and its results became **Legs 1 and 3** of
> [../publication/STORY.md](../publication/STORY.md) — the wall/lumen division of labour and the
> architecture's own contribution. Kept for the ladder design, the frozen-arm definitions, and
> §8's correction to the round-1 reading, which is the reason the published version uses AP
> rather than severity and measures noise per arm.

# Physics ablation — how much of the model is physics, and does it matter?

Opened 2026-09-02. Scopes the experiment that would let the paper claim *why* the surrogate
works, rather than only that it does. Companion to
PUBLICATION_NOTES.md §7 (the writing desk).

---

## 0. Why this is worth GPU-hours

The paper's biggest framing risk ([RELATED_WORK](../publication/RELATED_WORK.md) §2, threat 1) is that
**mesh-GNN geometric generalization is established method** — MeshGraphNets and successors
already generalize to unseen domains. "We used a GNN and it generalized" is not a contribution.

But the shipped stack is **not** a generic mesh GNN, and calling it one undersells it badly. It
is a hybrid in which COMSOL's own governing equations are discretised and the learned parts sit
exactly where those equations were *measured* to fail. If that is what makes it work, the
contribution changes from *"a GNN surrogate for thrombosis"* to *"a measured division of labour
between solved and learned physics"* — which is the question the hybrid-surrogate literature is
actively asking (the greedy PDE router; neural-operator/FE coupling), and a materially stronger
claim.

**The catch: that is currently a hypothesis.** No run strips the physics and retrains. This
repo's own history is unkind to unablated mechanisms — C0's registered mechanism was wrong and
the effect replicated anyway (MODEL_REVIEW §9b.5). We do not get to assert this one.

---

## 1. What physics is actually in the stack

More than features. Four distinct kinds, and the ablation has to respect the difference.

### 1a. Discretised governing equations (real solvers, not features)

* **`src/clot_ml/transport.py`** — COMSOL's *actual* off-wall operator, read off the `.mph`
  node tree: `tds2` has `D_M = D_Mas = D_Mat = 0` and convection ON, so the governing equation
  is `dMat/dt + u·∇Mat = 0` — pure hyperbolic transport, wall flux source, **zero diffusion**.
  Solved here as vertex-centred first-order upwind FV with a residence-time cap. The
  stagnation term (`V_i / T`) is why the boundary-layer attenuation appears without anyone
  writing the 0.16 constant down.
* **`integrate_mat_trajectory`** (`physics_wall_model.py`) — integrates the surface deposition
  ODE, COMSOL's own `J0_Mat` law, driven by `deposition_gate` (which keeps gated `srf1` and
  ungated `srf2` distinct).
* **`src/core_physics/ap_closure.py`** — a fitted quasi-steady Damköhler balance for wall AP,
  `ap/ap0 = 1/(1 + C·consumption/sr^q)`. The exponent is **fit, not assumed**: `q = 1` beats
  Lévêque's 1/3 decisively (R² 0.748 vs 0.509), so it is stirred replenishment, not diffusive.
  Constants fit on TRAIN only.

### 1b. Law-derived features — the gate's own arguments

`sr_over_lss`, `dsrx_over_sgt`, `gate_low`, `gate_sep`, `gate_sum`, `dist_to_gate`,
`gate_owner`, `up_gate`, `dn_gate`. These are the dimensionless groups of `G_wall` itself, not
generic engineered features.

### 1c. Physics-model outputs fed back as features

`log_mat_phys`, `onset_phys`, `log_mat_owner`, `sr_owner` — the ODE/transport solution entering
the network as input.

### 1d. Learned components, each placed where physics was measured to fail

Documented, not chosen by taste — this is the part worth writing up:

* **`mat_field.py`** exists because the ODE **cannot** drive the off-wall rule. Holding shell,
  owner map and attenuation fixed and changing only the `Mat` source: ODE scores
  **0.0000 / 0.0000 / 0.0000** on the three wound vessels, GT scores 0.9755 / 0.9755 / 0.7897;
  on far-field candidates GT `Mat_owner` separates clot from lumen at **AUC 0.9961** while the
  ODE sits at chance (**0.5048**). A total flow stall — the strongest form of the flow
  hypothesis — does not close it.
* **C0** — a ~10-line training-time distributional constraint, +0.13 off-wall, replicated ×3.

---

## 2. The prior failures the paper can cite

The archive documents a systematic ladder of attempts on this problem, each with a *measured*
failure reason. This is the evidence that the problem is hard, and it is already written down:

| Attempt | Where | Why it failed |
|---|---|---|
| **GNODE** (graph neural ODE) | `gnode_biochem`, retired ([MODEL_NOMENCLATURE](../MODEL_NOMENCLATURE.md)) | retired era |
| **R0 gray-box ADR** | archive/R0_GRAYBOX_ADR_CHECK.md | Kinetics verified correct to machine precision — but **fibrin is not the clot driver**, `Mat` is. FI never reaches its 0.6 µM gelation threshold on any vessel |
| **R1 wall-deposition oracle** | archive/R1_WALL_DEPOSITION_ORACLE.md | The `dsrx < sgt` separation gate is the one the graph WLS operators **resolve worst** |
| **Species pushforward GNN** | `species_graphsage` | superseded |
| **Gray-box S0–S3 ladders, clot-ML rules, T0** | BIOCHEM_LEGACY_LESSONS | archived |
| **RGP-DEQ / local corrector / PI Tier-2** | PUBLICATION_NOTES §2–3 | flow-side, all negative |

Three binding constraints that the validation established, and that shaped the working design
(archive/SPECIES_LEARNING_STRATEGY.md):

1. The clot is **`Mat`-driven, not fibrin-driven** — `mu2(FI) ≡ 0` everywhere, always.
2. `Mat` grows **autocatalytically**: `J0_Mat` deposition is only ~1/133 of `dMat/dt`; ~90% is
   `(Mas/Minf)·k_aa·AP`. A model that learns "platelets stick to the wall" learns the wrong term.
3. Deposition is gated by **low-shear stagnation** (`sr < lss` carries ~80%); the shear-gradient
   gate is minor *and* is the one graph operators resolve worst.

> **CORRECTION to PUBLICATION_NOTES §7.0.** That section says momentum sees `mu1(Mat) + mu2(FI)`.
> The *expression* does, but `mu2` **never fires** — FI peaks ~46× below its 0.6 µM threshold
> (confirmed on `comsol001`: FI max 9.9e-05 nd at final time). The effective coupling is
> `mu1(Mat)` alone. This does not change the provenance finding (the `.mph` still carries FG/FI
> and a viscosity-step coupling the preprint does not), but the paper must not describe fibrin
> as an active contributor.

---

## 3. The ablation ladder

Five arms, each a full retrain under the **same geometry-stratified nested CV** as Table 4, so
every arm yields an out-of-fold score directly comparable to the shipped model.

| Arm | Features | Question it answers |
|---|---|---|
| **A0** geometry only | `is_wall, is_shell, is_midside, dist_wall_*, hop_wall, degree, sdf_nd, width_nd, width_d1, width_d2` | Can shape alone predict the clot? (floor) |
| **A1** + raw flow | A0 + `speed_nd, u_n, u_t, p_nd, log_sr, log_absdsrx, log_absdsry, vort, div` | **The generic mesh-GNN baseline.** Geometry + velocity, no knowledge of the deposition law. *This is the arm that answers the reviewer.* |
| **A2** + law-derived gates | A1 + §1b | What do the law's own dimensionless groups buy? |
| **A3** + physics-model outputs | A2 + §1c | What does the integrated ODE/transport solution buy on top? |
| **A4** full (shipped) | all 39 + directional aggregates + `rp0, ap0` | reference |

**The headline number is A4 − A1.** If it is large, the physics conditioning is the story and
the paper leads with it. If it is small, a generic mesh GNN does most of the work, the current
framing stands, and we have learned that before a reviewer told us.

**A2 − A1 vs A3 − A2 is the interesting decomposition:** is the value in *knowing the law's
arguments*, or in *actually integrating the equations*? Those are different claims, and only
the second supports "solved and learned physics, divided by measurement".

### Implementation, and why it is low-risk

**Zero the ablated columns after normalization; do not remove them.** That keeps the matrix
width, the `feature_fingerprint`, the normalization layout and the ensemble shapes unchanged —
no architecture change, no cache rebuild, no promotion-path edits. A column held at its mean
carries no information, which is exactly the ablation semantics wanted.

Do **not** zero before standardization (that injects an out-of-distribution value rather than a
neutral one), and record which columns were zeroed in each arm's manifest.

### What this ablation does NOT cover, and must not be claimed to

The structural physics — the upwind transport solve, the ODE integrator, the AP closure, the
topological shell/owner rules — enters the network through §1c features, so zeroing those
features removes *the information* but not *the architecture*. A0–A4 therefore bound the value
of physics **as conditioning**. Ablating the solvers themselves (e.g. replacing `transport.py`
with the 0.16 constant attenuation) is a separate, larger experiment. Say "conditioning", not
"architecture", when reporting.

---

## 4. Cost, and the decision it forces

One CV leg is **2.5–4.5 GPU-h** (WALL_MODEL_PLAN §0 budget notes), so five
arms is **12–22 GPU-h** on the 4 GB laptop GPU — two to three overnight runs. That is the single
largest remaining compute item in the project.

**It is worth it**, on this reasoning: without it the paper claims *"we built a surrogate that
works"* and competes against an established mesh-GNN literature it cannot outrun on n or
dimensionality. With it, the paper claims *"here is the measured division of labour between
solved and learned physics in a coupled biophysics pipeline"* — which nobody in the 11-study
PRISMA set has done, and which the hybrid-surrogate literature is explicitly asking for.

**Cheap first step before committing the full ladder:** run **A1 alone** against A4. That is one
leg (~3 GPU-h) and it already answers the reviewer's question. Run the full ladder only if
A4 − A1 is large enough to build a section on.

---

## 5. Honest risks

* **The ablation may come back small.** Then the physics story is not available, and the paper
  reverts to the current framing. Report it either way — a measured null here is exactly the
  kind of result this project already documents well.
* **Retraining changes the reference numbers.** Table 4's shipped-model scores come from the
  existing OOF archive; ablation arms must be scored through the *same* pipeline or the
  comparison crosses conventions (standing rule 1). Re-score A4 within the ablation run rather
  than quoting the archive.
* **Five arms invite cherry-picking.** Freeze the ladder and the metric before running, and
  report all five regardless of which way they fall.

---

## 6. As implemented — the frozen ladder, 2026-09-06

Wired in on 2026-09-06 and run on the laptop GPU.  Everything below is code, not intention:
the arms live in **`src/clot_ml/ablation.py`**, the runs in
**`scripts/run_ablation_ladder.py`**, the scoring in
**`scripts/publication/generate_ablation_data.py`**, the figure in
**`scripts/publication/plot_ablation.py`**, and `src/tests/test_ablation_ladder.py` fails if
the partition ever develops a hole.

### 6.1 What changed against §3, and why

**The cost estimate in §4 was wrong by an order of magnitude.** One CV leg (5 folds × 3
seeds, 80 epochs, 28 training vessels) measures **~35 minutes**, not 2.5–4.5 GPU-h — the
budget note it was taken from predates `dim=64, layers=4`.  A leg being cheap changes the
decision §4 forces: the whole ladder fits in one unattended run, so the "cheap first step"
(A1 alone) was skipped and every arm ran, plus three arms §3 did not have.

**Three arms were added, each closing a hole a reviewer would find.**

* **`A1p`** — A1 plus the neighbourhood and upstream/downstream aggregates *of the raw flow*
  (`sr_mean_h*`, `spd_mean_h*`, `up_sr`, `dn_spd`, `sr_owner`, …).  §3 put those in A4, which
  would have let the headline A4 − A1 be dismissed as a handicapped control: a 4-layer
  message-passing network computes neighbourhood aggregates itself, so denying them to the
  baseline and claiming the difference is *physics* is not honest.  A1p is the strengthened
  generic baseline; **A4 − A1p is the defensible headline** and A4 − A1 the plan's original.
* **`A1_pure`** — A1p's features with the two architectural physics doors shut as well (see
  6.2).  This is the genuinely generic mesh GNN.
* **`A3` split from A4 on the SOLVER, not on the feature count.**  §1a names two *different*
  discretised equations — the surface-deposition ODE and the zero-diffusion upwind transport
  — and §3's A3 merged them.  Here A3 adds the ODE and its occlusion mask, A4 adds the
  transport solve.  "Which solved equation pays" is a question the plan as written could not
  answer.

### 6.2 The confound §3 did not state

Zeroing columns cannot reach the physics backbone, because it enters the network through two
doors that are not features:

1. `mat_phys` is the **additive base** of the regression head (`gnn.ClotGNN.forward`);
2. `phys_mask` is the **round-0 occlusion seed** of the recurrent rollout (`gnn.rollout`);

and a third at *readout* time: the `resid` family thresholds physics-positive and
physics-negative nodes separately, i.e. it reads the backbone's mask
(`eval_strict.readout_resid`).

So under §3 as written, "A0, geometry only" is not geometry only — it still gets the whole
backbone through the architecture and the readout.  That biases the ladder **toward the
null**, which is the safe direction, but it makes A1 a poor stand-in for the reviewer's
generic mesh GNN.  Both doors are now switches (`phys_base`, `phys_seed`), `A1_pure` shuts
them, `B_nobase` / `B_noseed` price them at full features, and every arm is additionally
reported under a `plain`-only readout with the third door shut.

### 6.3 The arms

Feature groups partition all 69 columns (68 cached + the `phys_mask` column
`data.attach_physics` appends); an unpartitioned column is a test failure, because it would
be silently *kept* by every arm.

| arm | keeps | asks |
|---|---|---|
| `A0` | geom | can shape alone predict the clot? (floor) |
| `A1` | + raw flow | §3's generic mesh-GNN baseline |
| `A1p` | + flow aggregates | the *strengthened* generic baseline |
| `A1_pure` | A1p, physics base and seed removed | the genuinely generic mesh GNN |
| `A2` | + the law's dimensionless gate groups (§1b) | what is knowing the law's *arguments* worth? |
| `A3` | + the integrated deposition ODE (§1c) | what is *integrating* it worth on top? |
| `A4` | + the upwind transport solve, species IC | the shipped conditioning (reference) |
| `B_iso` | full | is the anisotropy claim real? (velocity removed from edges, aggregation made direction-free) |
| `B_nomp` | full | does message passing buy anything over a node-wise MLP? |
| `B_nobase` | full | is "physics as a base, not a competitor" real? |
| `B_noseed` | full | is the physics occlusion seed real? |
| `B_r1` | full | are the 3 refinement rounds real? |
| `B_bce` | full | is "the loss is the metric" real? |

`--arm A4` is a strict no-op — no column held constant, no switch flipped — so it reproduces
the shipped run bit for bit, and the ladder's own A4 leg is checked against the archived
`dc_v5_split` run as a reproducibility control.  Every arm trains at the shipped objective
(`--shape-w 2.0 --clot-free-w 0.25`) on `clot_ml_cache_v5_split`, over the same 5
geometry-stratified folds, and is graded through `eval_significance.nested_rows`, which *is*
`eval_strict.py`'s readout selection — satisfying §5's risk 2 by construction rather than by
discipline.

### 6.4 Reading the table

Round 1 is seeds 0–2 on every arm; round 2 is an independent 3-seed replicate (disjoint
seeds, same folds), because the off-wall null at three seeds is ±0.045 — larger than several
of the effects here — and a single 3-seed off-wall delta once came back with the wrong sign
(`go_deployclot_split.sh` BLOCK 4).  **Wall deltas are readable at three seeds; off-wall
deltas are not, and must not be quoted from round 1 alone.**  Deltas are paired bootstraps
over vessels against A4.

---

## 7. Round-1 results (3 seeds, 2026-09-07) — and where the run stopped

Round 1 ran all 14 arms in 8.3 h, no failures.  Round 2 (the independent 3-seed replicate)
completed **A4, A1, A0, A1p** and was stopped part-way through `A2_seedB`.  A killed arm
writes no score file, so `python scripts/run_ablation_ladder.py --round 0` resumes by
skipping what exists and re-running `A2_seedB` from the top.  **~5.4 GPU-h remain.**

Numbers below are the strictly-nested out-of-fold means over the 27 clot-carrying vessels,
severity metric, `auto` readout; full table and per-vessel rows in
`outputs/ablation/ablation_report.json`, figure at
`outputs/publication_split/figures/fig_ablation_ladder.png`.

**Validity check first.** `--arm A4` against the archived `dc_v5_split`: wall +0.0045
[-0.001,+0.012], off +0.029 [-0.033,+0.102] — both inside the pipeline's own floor.  (Not
bit-identical because `index_add_` on CUDA sums with atomics; that non-determinism *is* the
floor.)  The reference is a valid zero.

### 7.1 Physics as conditioning

| arm | wall | off-wall | step Δ wall | step Δ off |
|---|---|---|---|---|
| `A1_pure` generic mesh GNN | 0.8933 | 0.6188 | — | — |
| `A0` geometry | 0.9015 | 0.6668 | | |
| `A1` + raw flow | 0.9196 | 0.6576 | | |
| `A1p` + flow aggregates | 0.9021 | 0.6990 | — | — |
| `A2` + law's gate groups | 0.9464 | 0.7306 | **+0.044** | +0.032 |
| `A3` + deposition ODE | 0.9493 | 0.8086 | +0.003 | **+0.078** |
| `A4` + transport solve (shipped) | 0.9511 | 0.8411 | +0.002 | +0.033 |

Three findings, and they are not the ones §3 expected:

1. **`A0 ≈ A1 ≈ A1p`.**  Handing the network the velocity field, and then its neighbourhood
   and up/downstream aggregates, does not measurably beat handing it the shape.  The flow
   becomes valuable only once the deposition law has been applied to it.  This kills the
   framing that the contribution is "a mesh GNN on flow".
2. **The wall is won by knowing the law's ARGUMENTS.**  `A2 - A1p` = +0.044 wall (~9× the
   ±0.005 floor); the ODE then adds +0.003 and the transport solve +0.002, both inside it.
   Integrating the equations buys nothing at the wall the dimensionless groups had not
   already bought.
3. **The off-wall domain is won by INTEGRATING.**  `A3 - A2` = +0.078 off-wall is the only
   off-wall step that clears the ±0.045 three-seed null.  This is the arm that supports
   "solved and learned physics, divided by measurement"; the gate-argument story does not.

Against the genuinely generic baseline: `A4 - A1_pure` = **+0.058 wall / +0.222 off-wall**,
rising to +0.231 / +0.215 when the physics-derived readout is also removed (`plain` column).

**Honest limit on the wall claim.** Geometry alone already scores 0.9015 against the shipped
0.9511.  Physics conditioning is worth +0.050 there — real (10× the wall floor, CI excludes
zero) but modest.  The paper should lead off-wall and say so.

### 7.2 Architecture, at full conditioning

| arm | wall | Δ wall | off-wall | Δ off |
|---|---|---|---|---|
| `B_iso` isotropic messages | 0.9380 | **-0.013** [-0.028,-0.002] | 0.7845 | -0.057 |
| `B_nomp` no MP layers | 0.9448 | -0.006 | 0.7801 | -0.061 |
| `B_nobase` no physics base | 0.9462 | -0.005 | 0.7889 | -0.052 |
| `B_noseed` no physics seed | 0.9464 | -0.005 | 0.8015 | -0.040 |
| `B_r1` one round | 0.9517 | +0.001 | 0.7510 | -0.090 |
| `B_bce` plain BCE loss | 0.9485 | -0.003 | 0.7065 | **-0.135** [-0.228,-0.053] |

* **On the wall only anisotropy survives** (`B_iso` -0.013, ~3× the floor, CI excludes zero).
  Everything else — message-passing layers, physics base, physics seed, the refinement
  rounds — costs ≤0.006 and is at or inside the floor.  The wall does not need the
  architecture; it needs the gate features.
* **Off-wall the LOSS is the dominant design choice** (`B_bce` -0.135, 3× the null).  "The
  loss is the metric" is now measured rather than asserted.
* **The physics base and seed are individually near-null**, yet `A1_pure` (features gone AND
  both doors shut AND plain readout) collapses to 0.7185 wall.  Individually redundant,
  jointly load-bearing — worth stating as such rather than claiming each pays alone.

### 7.3 What must NOT be quoted yet

Every off-wall architecture delta except `B_bce` sits within ~1.5× the ±0.045 three-seed
null, and several **flip sign** under the `plain` readout.  Same for `A2 - A1p` (+0.032) and
`A4 - A3` (+0.033): the transport solve's own contribution is currently **unresolved**, not
small.  Round 2 is what makes these quotable.  Report them as unresolved until then.

---

## 8. CORRECTION to §7 — the round-1 reading was wrong, 2026-09-08

§7 above was written from **3-seed post-readout severity scores** and its headline
decomposition does not survive. Kept visible rather than deleted, because the *way* it was
wrong is the reusable lesson. The corrected result is `docs/PAPER.md` §4.

### 8.1 Two traps, and §7 fell into both

**Trap 1 — post-readout severity is not a model-quality statistic.** Each arm's score comes
from a per-fold threshold search, then a metric carrying an absolute miss grace (`tau_abs = 15`,
capped at `0.25·n_gt`) and a precision grace. Off-wall burdens here run 4–126 nodes, so on most
vessels the cap binds and `recall_eff = TP/(0.75·n_gt)` — committing about three quarters of the
true nodes already reads recall 1.0. **An arm can buy score by committing more**, and the
threshold search will find that trade. `B_bce` is the clean demonstration: it has the *best*
off-wall AP in the ladder (0.7035 against A4's 0.4562) and *loses* the severity table. That is
the shipped objective's deliberate ranking-vs-cuttability trade, not a defect.

*The fix:* for an ablation, the statistic of record is **average precision of the raw
out-of-fold field** — `scripts/publication/diag_ablation_ranking.py`. No cut, no grace, no
readout-family choice.

**Trap 2 — ensemble size interacts with arm strength.** Pooling six seed members instead of
three helps a *weak* arm far more than a strong one: `A1_pure` reads off-wall AP 0.340 and 0.428
on its two 3-seed replicates but **0.497 pooled over all six**. So §7's 3-seed table
systematically flattered the arms that needed the ensemble least, and the 6-seed table's
apparent "physics-free arms beat the full model" inversion was that correction landing and
overshooting through trap 1.

*The fix:* compare only at equal seed count, and never read a 3-seed table against a 6-seed one.

### 8.2 The noise scale, measured rather than inherited

§7 read its deltas against the project's ±0.005 wall / ±0.045 off-wall floor. **That floor was
measured on the shipped configuration and does not transfer to ablated ones.** Measured directly
as the spread between two independent 3-seed replicates of the same arm:

| | min | median | max |
|---|---|---|---|
| wall AP | 0.0016 | ~0.014 | 0.0555 |
| off-wall AP | 0.0081 | ~0.030 | 0.0885 |

### 8.3 What actually survives

Read against the **max** column above:

| step | Δ wall AP | Δ off AP | verdict |
|---|---|---|---|
| A0 → A1, add the flow field | +0.025 | **+0.198** | off-wall robust (2.2×); wall inside noise |
| A1p → A2, add the law's gate arguments | **+0.084** | +0.106 | marginal both (~1.5× / 1.2×) |
| A2 → A3, integrate the deposition ODE | −0.015 | +0.024 | noise |
| A3 → A4, add the transport solve | −0.030 | −0.112 | noise to negative |

* **Geometry alone is decisively insufficient off-wall** (AP 0.2371 against ≥0.435).
* **The two domains are won by different things**: the wall by the deposition law's dimensionless
  *gate arguments*, the lumen by the *flow field itself*. This asymmetry survives both statistics
  and is the paper's usable finding.
* **Solving the governing equations adds nothing this cohort can resolve.** §0 asked whether the
  contribution is "a measured division of labour between solved and learned physics". At n=27 the
  honest answer is that the division between *knowing the law's arguments* and *integrating the
  law* is **not measurable here** — §3's stated headline (A4 − A1) and its A2−A1 vs A3−A2
  decomposition both dissolve.

### 8.4 What §5's "honest risks" got right, and what it missed

Risk 1 ("the ablation may come back small") is what happened, and reporting it is the plan's own
instruction. What §5 did **not** anticipate is that the ladder could come back *non-monotone*
because the reported statistic was readout-coupled. **Add to any future ablation: name the
statistic before running, prove it is not fitted downstream of the thing being ablated, and
measure the replicate spread of the ablated arms rather than inheriting a floor.**


---

## 9. Two arms added 2026-09-10, both answering objections rather than the original question

Neither belongs to the feature ladder above. Both were added because a *reviewer* question, not
a physics question, had no measured answer — and both are recorded here so the ladder's scope
stays clean.

### 9.1 `A_mgn` — the depth-matched published-architecture control

**Why.** `A_naive` is a from-scratch control at the shipped depth (4 layers), so it bounds the
from-scratch floor and nothing else. It cannot answer *"is your margin over a competent published
mesh surrogate, or only over a shallow one?"* — which is the first thing a reader asks after
Pelissier, Meliga & Hachem (2026), who benchmark against MeshGraphNet, BSMS-GNN and Transolver++
at matched compute.

**What it is.** `A_naive` at MeshGraphNet depth: **15 message-passing layers**, `dim` 64 — the
configuration Pelissier et al. report for their own MGN replication. Physics absent everywhere:
no base, no seed, no physics columns, no metric-shaped loss, no C0, one binary head on plain BCE,
one global cut, read under the `plain` readout so `phys_mask` cannot enter the threshold either.

| | wall | off-wall |
|---|---|---|
| `A_naive` (4 layers) | 0.6833 | 0.4849 |
| **`A_mgn` (15 layers)** | **0.6383** | **0.5207** |
| shipped `A4` | 0.9356 | 0.8162 |

**The finding: depth does not rescue a physics-free model on this problem.** The deep arm lands
in the same band as the shallow one — slightly worse at the wall, slightly better off-wall. So
the physics contribution reported in PAPER §5 is not an artefact of an undersized baseline.

> **ATTEMPT 1 COLLAPSED, and the failure is the instructive part.** At the shipped `lr=3e-3` the
> 15-layer stack produced a **constant field** — std 1.2e-06 over 9,490 nodes, 0.1691 everywhere,
> i.e. the base rate — and scored 0.1639 wall / 0.1116 off. **That is a failure to train, not a
> result about MeshGraphNet, and it is deliberately absent from the claims ledger.** Cause: the
> per-layer block is POST-norm residual (`self.norm(x + self.drop(h))`, `gnn.py:88`), and a
> 15-deep post-norm stack driven by OneCycleLR at a peak rate tuned for 4 layers diverges into
> the prior. Attempt 2 lowers the peak lr to 5e-4 and changes nothing else.
>
> **Three limits, all of which must travel with the number** (they are in PAPER §9):
> 1. It is **MeshGraphNet-STYLE**, not a reimplementation — no updated edge features, no separate
>    edge encoder/decoder, our training loop rather than Pfaff et al.'s noise-injected rollout.
>    Never write "we beat MeshGraphNet".
> 2. `A_mgn` runs at **lr 5e-4 and `A_naive` at 3e-3**, so the two controls differ in learning
>    rate as well as depth. Stated, not engineered away.
> 3. **One retry, no lr sweep on either arm.** One control, not a matched-compute benchmark suite.

### 9.2 The deploy-path wall clock — the switch that did not exist

**Why.** PAPER §5.3 holds the ODE's onset information constant *inside the temporal head* and
finds it redundant. It could not touch the mechanism the shipped artifact uses
(`readout.lag_anchor: "ode"`), and said so. That path had no switch.

**What was wrong before.** Every `ode_wall_series` call in `eval_strict_temporal.py` sat in
off-wall lag scaffolding, where the wall series only dates the *owner*. `--lag-anchor` therefore
never varied the wall clock, and the existing `temporal_A4_ownerlag_{ode,pred}` pair is
**bit-identical on the wall across all 27 vessels**. Two docs had asserted *opposite* mechanisms
for that bit-identity and both were half right; which one holds is a per-fold tuner outcome.

**Result** (PAPER §5.3.1, ledger rows `wc.*`): the clock is worth **+0.0028, CI
[−0.0181, +0.0250]** — null — with the schedule moved on **27/27** vessels and the final set held
identical by construction. Separately, the ODE clock *as deployed* also changes **whether**, not
only when: it is the only wall path with no forced final commit, costing **−0.0210** at final
time on 24/27. That last one is a one-line fix to the deploy readout.

**Scope, and it is hard: this is a WALL-ONLY result.** The off-wall columns are bit-identical
across every arm — `off_on_sum` moved on 0/27 vessels *even under `--owner-lag`* — because the
tuner keeps selecting the single-stage off-wall rule, which never reads the wall series. There is
no off-wall number here, and calling that equality a null would repeat the `--lag-anchor` error.

> **The guard is the reusable part.** `score_vessel` now emits `wall_on_sum` / `off_on_sum`, a
> fingerprint of the *schedule* rather than the score, and `diag_wall_clock.py` refuses to
> interpret a delta before reporting them. **Any future temporal ablation here should use it**:
> this project has now produced a dead-switch null twice, and a score-only comparison cannot
> tell "the clock changed and the quality did not" from "the flag did nothing".
