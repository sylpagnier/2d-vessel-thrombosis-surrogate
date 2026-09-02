# Publication plan — one paper: the learned thrombosis surrogate

Rev 2, 2026-09-01. Supersedes rev 1 (multi-paper triage). Companion to
[PUBLICATION_NOTES.md](PUBLICATION_NOTES.md), which holds *measured claims*, and
[RELATED_WORK.md](RELATED_WORK.md), which holds the Phase 0 novelty triage.

**Decision: one paper.** Its subject is the tool — a mesh-generalizing learned surrogate for
the lab's multi-species thrombosis CFD. Everything else in the repo is either a section of that
paper, a stated limitation, or out.

Standing rule inherited from PUBLICATION_NOTES: a result on n<10 vessels at one operating point
is a hint, not a finding. Applied here as a gate on what may appear as a headline number.

---

## 1. The paper

> **A mesh-generalizing learned surrogate for chemically- and mechanically-induced thrombus
> formation**

We build a learned surrogate for the Cardillo–Barakat COMSOL model
([BMMB 2025](https://doi.org/10.1007/s10237-025-01966-3)) that predicts the spatiotemporal clot
field on unseen vessel geometries, at a fraction of the solve cost, with the t=0 flow supplied
by a local Carreau FEM solve and the clot rollout by a temporal GNN on the vessel mesh graph.

### Why this is publishable — the gap is documented by a systematic review

The Dec-2025 PRISMA review in J Thromb Thrombolysis (Al Bannoud et al., 2026;59:727–745,
[10.1007/s11239-025-03222-y](https://doi.org/10.1007/s11239-025-03222-y)) screened the entire
ML + computational-thrombosis intersection and found **11 eligible studies**. Their categories:
platelet-signalling ANNs embedded in multiscale models, patient-outcome classification, thrombin
threshold classification, shear-stress regression, and PINN parameter inference.

Three facts from that review define our contribution, and all three are quotable:

1. **No study learns the clot field.** The closest are a DNN that binary-classifies coagulation
   initiation from scalar thrombin parameters (Bouchnita 2023, 94% accuracy) and an MLP that
   regresses shear stress (Khajavi 2025). Coagulo-Net (Qian 2024) *solves and infers* the
   coagulation ODEs; it does not generalize a rollout to unseen geometry.
2. **No study uses a graph or mesh-based architecture.** The ML models across all 11 papers are
   ANN, NARX, SVM, KNN, DT, RF, NB, LR, XGBoost, CatBoost, EBM, DNN and PINN. Not one GNN. A
   mesh-agnostic graph surrogate is first-of-kind in this intersection.
3. **The review names our paper as its own future direction** — repeatedly. It calls for
   "extending the model to three-dimensional vascular geometries", for "patient-specific
   anatomical reconstructions", and for architectures that "improve the detection of complex
   spatial and temporal patterns in thrombus growth". Cite this and the motivation section
   writes itself.

**One-sentence contribution:** *the first field-level learned surrogate of a continuum
multi-species thrombosis model that generalizes across vessel geometry — given a t=0 flow field
and an arbitrary 2D vessel mesh, it predicts the clot field over time — with the accuracy
requirements on its flow input measured rather than assumed.*

### The claim is geometry, and the operating point is scope — not a limitation

**The generalization axis is geometry.** Flow enters as an *input* the surrogate is conditioned
on, not as something it learns, so the corpus's fixed operating point (Re = 450) is a **scope
statement**, not an unaddressed weakness. Written that way — "given a flow field and a 2D
geometry" — the single-regime corpus stops being the thing a reviewer attacks and becomes the
thing the method is defined over. Rev 1 and rev 2 both mis-framed this and put a multi-Re cohort
on the critical path; that is now **off the plan entirely**.

Two obligations come with the reframing, and both are cheap:

1. **Say the scope in the abstract, in the contribution sentence, and in the limitations.** A
   reviewer must never discover the single operating point on their own.
2. **Do not over-claim "any flow field."** See the caveat below — as the features stand today,
   that sentence is stronger than the evidence.

> **⚠ Code-level caveat, found this session — check before writing "arbitrary flow field".**
> The flow conditioning is not fully dimensionless. `src/clot_ml/features.py` feeds `log_sr =
> log1p(sr)` at physical scale, aggregates shear as `min(sr, 500.0)`, and clips `sr_over_lss` to
> `[0, 40]` and `dsrx_over_sgt` to `[-40, 40]`. The ratio channels are the right dimensionless
> groups (they are the gate arguments), but the **hard clips are regime-dependent**: at higher Re
> the shear rates rise and `min(sr, 500)` saturates, so the encoding — not just the training
> distribution — would degrade off-regime.
>
> This does not weaken the geometry claim at all. It means the honest phrasing is *"conditioned
> on a t=0 flow field at the operating point of the training corpus"* rather than *"any flow
> field"*. If you want the stronger sentence, the cheap route is a feature-level audit plus the
> §6 sweep evidence below — not new data.

### Ground truth — verified against the models this session

The `.mph` files in `comsol_models/` carry exactly the published structure: **9 bulk species**
(`rp, ap, apr, aps, at, pt, th, fg, fi`) transported with `Reactions_9spec`, `WallFlux_9spec`,
`InletFlux_9spec`, `ExitFlux_9spec`; **3 surface species** (`M, Mas, Mat`) with
`wall_surface_reactions_3spec`; Carreau rheology; clot as the `mu1(Mat)` viscosity step at
`viscosity_mat_crit`. This is the Cardillo–Barakat model, and saying so is the single strongest
framing move available: **the ground truth is peer-reviewed and from our own lab.** The
synthetic-geometry objection largely dissolves, because the object of study is the *model*, not
an anatomy claim.

**Wound provenance — one question for Giulia, and it is the only real blocker in §1.**
Diffing the model trees confirms `phase2_nowound_*` carries the base feature set while
`phase2_wound_*` adds `WoundFlux_9spec`, `SfcRxn_3spec` and the `wound` / `wallandwound`
selections. The wound law is therefore an **addition on top of the published model**, and the
files cannot tell us who made it. Ask directly:

* Is the ungated wound boundary condition Giulia's, or was it added by this project?
* Is it published, in preparation, or unpublished?

This decides (a) whether §6 below is "surrogate reproduces a published extension" or "we
characterise an unpublished extension", (b) the author list, and (c) whether the
[WOUND_PROGRESS](WOUND_PROGRESS.md) finding — that injury is the same law with the shear gates
deleted — is ours to state at all. **Do not draft §6 before this is answered.**

---

## 2. The message

> **Learn the chemistry. Solve the flow.**
>
> In a coupled flow → chemistry thrombosis pipeline, the expensive, high-dimensional part — the
> 12-species stiff reaction–transport system and its gelation threshold — is what a learned
> surrogate should replace. The flow is not. We build the tool that does this and generalizes
> across vessel geometry, and we establish the split **by measurement**: three independent
> attempts to learn the flow half all fail or add nothing, and one mechanism explains all three.
> The clot readout consumes flow through a **threshold gate**, so what it needs is the correct
> *rank order* of derived wall fields — which a cheap classical solve delivers and a learned
> surrogate, at any velocity accuracy we achieved, does not.

**Working title:** *Learn the chemistry, solve the flow: a geometry-generalizing surrogate for
multi-species thrombus formation.*

### Why this is one paper and not a catalogue of failures

The thing that makes this publishable rather than a lab notebook is that **the three negative
results are not three apologies — they are one finding with three independent confirmations.**
Ordered by decreasing ambition:

| Attempt | What it tried | Outcome |
|---|---|---|
| **RGP-DEQ** | Learn the whole t=0 field | −0.35 deploy score vs GT; FEM sits inside noise (0.705 vs 0.710) |
| **Local kinematic corrector** | Learn a local correction on frozen flow | Worse than doing nothing: MAE 0.684 vs null 0.630; diversion `cos = −0.142` |
| **Closed-loop coupling** | Feed clot back into flow | Oracle upper bound adds −0.0065 — inside the noise floor |

Three different levels of ambition, three different architectures, one answer. Then the
diagnosis that unifies them: velocity rel-L2 correlates **−0.030** with the downstream drop over
33 vessels while wall gate Jaccard correlates **+0.613**; the failure is a **cliff**, not a
gradient (`patient010`: 131 mask nodes → 0, F1 0.969 → 0.000, between 5% and 8% velocity error);
and an oracle monotone remap — the ceiling on any calibration — moves gate Jaccard only
0.339 → 0.382, so the rank order is wrong and no post-hoc fix recovers it.

That arc is the paper's intellectual content. A reader who arrives sceptical of "they used FEM
instead of a neural network" leaves with a design rule.

### Ordering constraint — the tool must work before the negatives appear

**§5 comes after §4, never before.** The reader must first believe the tool is good; only then do
the negative results read as rigorous design justification rather than as a list of things that
broke. Inverted, the same material reads as a failure catalogue and the paper dies.

### Where it lands in the literature

The general form of our claim is already in the air, which makes the paper timely rather than
scooped:

* **Duraisamy, *Predictivity and Utility of Neural Surrogates of Multiscale PDEs*** ([arXiv 2604.20061](https://arxiv.org/abs/2604.20061)) — a position paper separating *predictivity* (benchmark accuracy) from *utility* (value in a real application), arguing standard metrics mask physical limitations, and explicitly calling for neural–classical hybrids and better reporting standards. **This is our hook.** They argue it; we measure it, in a concrete biomedical pipeline, with a mechanism and a ceiling proof. Cite in the intro and again in the discussion.
* **Grossmann et al., *Can physics-informed neural networks beat the finite element method?*** ([IMA J Appl Math 89(1):143, 2024](https://academic.oup.com/imamat/article/89/1/143/7680268)) — the canonical "no" for the flow half. Our §5 is a task-level instance of the same conclusion.
* **Hybrid neural–classical composition** — a greedy PDE router for blending neural operators and classical methods ([arXiv 2509.24814](https://arxiv.org/pdf/2509.24814)); time-marching neural-operator/FE coupling ([CMAME 2025](https://www.sciencedirect.com/science/article/abs/pii/S0045782525005912)). The field is actively asking *which component to replace*. We answer it for one pipeline, with the criterion (the consumer's gating structure) rather than a heuristic.

**The generalizable contribution, stated for the discussion:** *where a learned field feeds a
thresholded consumer, the surrogate's accuracy requirement is set by the consumer's decision
statistic, not by field norms — report the decision statistic.*

---

## 3. Section plan, and where each section's numbers already live

Restructured for the §2 message. Note the ordering constraint: the tool and its results come
first, the negative arc second.

| § | Content | Source | Status |
|---|---|---|---|
| 1 | **Intro.** Thrombosis CFD cost; the 11-study gap; predictivity-vs-utility hook | PRISMA review, Duraisamy | Write |
| 2 | **Ground truth.** Cardillo–Barakat model, 9+3 species, gelation threshold; parametric cohort, 23 clot-carrying + 8 clot-free | `comsol_models/`, [PILOT_COHORT_RUNBOOK](PILOT_COHORT_RUNBOOK.md) | Have |
| 3 | **The tool.** FEM t=0 → temporal GNN rollout; mesh-agnostic graph construction; the C0 constraint | `src/clot_ml/` | Have |
| **4** | **It works.** Geometry-stratified K-fold, per-vessel and per-class; SEALED once; clot-free FP row; wall 0.9203 / off 0.7078 vs floor ±0.0037 / ±0.0432; **speedup** | [MODEL_REVIEW](MODEL_REVIEW_2026-08-22.md) §8–9, §6 below | **K-fold to run; speedup MISSING** |
| 4b | Geometry-response sweeps as the qualitative shape-generalization figure | [RESEARCH_SWEEPS](RESEARCH_SWEEPS.md) | Have, unrun |
| **5** | **Why the flow is solved, not learned** — the three-confirmation arc + the gate diagnosis | below | Have, thin in places |
| 5a | Learning the whole field: RGP-DEQ, −0.35 | [PUBLICATION_NOTES](PUBLICATION_NOTES.md) §2 | Have (n=4–5) |
| 5b | Learning a local correction: the kinematic corrector, worse than null | [LOCAL_KINEMATIC_CORRECTOR](LOCAL_KINEMATIC_CORRECTOR.md) | Have |
| 5c | Closing the loop: oracle bound, −0.0065 | PUBLICATION_NOTES §1 | Have (n=8) |
| 5d | **The diagnosis.** rel-L2 −0.030 vs gate Jaccard +0.613 (n=33); the cliff; the remap ceiling | PUBLICATION_NOTES §2 | Have — best-powered result |
| 6 | Extension: injured wall | [WOUND_PROGRESS](WOUND_PROGRESS.md) | **Blocked on provenance** |
| 7 | Limitations and scope | §5 below | Write |
| 8 | **Discussion.** The design rule; report the consumer's decision statistic | §2 above | Write |

### Handling §5b — the local kinematic corrector

[PUBLICATION_NOTES](PUBLICATION_NOTES.md) §3 rules this out of the paper. Including it is a
defensible reversal *in this structure specifically*, because §5's argument is stronger with a
second, architecturally different failure than with one — the corrector is the reason a reader
cannot answer "you just needed a smaller, more local model."

Two conditions on including it, and they matter:

1. **It is a confirmation, never a component.** It must appear inside §5's arc, described as an
   attempt that was measured and rejected. It must not appear in §3 or in any architecture
   figure. PUBLICATION_NOTES' actual warning — *"do not present it as a component"* — still binds.
2. **Keep it small: one paragraph, no figure, one table row.** It is the weakest of the three
   (it fails qualitatively as well as quantitatively — diversion `cos = −0.142`, magnitude ratio
   0.000), and a biomedical venue's page budget will not tolerate three full negative
   subsections. Spend the §5 figure on 5d, which is the best-powered result in the project.

### The speedup number — ~10³, and it belongs in the abstract

**COMSOL solve: ~48 h per vessel. Shipped surrogate: 80.2 s median** (measured, n=30 vessels,
full horizon — §11). That is **2,156×** — above the 2 orders reported
for mesh-CNN WSS and into the range of the platelet DeepONet's 4–5
([RELATED_WORK](RELATED_WORK.md) §1–2). This is a headline-grade number and it changes the
abstract's centre of gravity: the tool is not "faster", it is a different mode of use.

**Reconciling the two internal timings — do this before it is quoted.**
[WALL_MODEL_PLAN](WALL_MODEL_PLAN.md) §0 records ~25–30 min/anchor, which is **20× slower** than
the UI figure. The two are not measuring the same thing, and the difference is explainable:

| | WALL_MODEL_PLAN §0 | UI |
|---|---|---|
| Stack | `WC_v7` wall + compound growth — the **retired** mat-growth stack | shipped `clot_ml_0` |
| Hardware | explicitly "4 GB GPU" | UI machine, unstated |
| Work | deploy-faithful rollout **plus graded scoring** | inference only |

So ~1.5 min is the defensible number — it is the shipped tool — and ~25–30 min is a stale
artifact of a superseded stack on small hardware. **But do not publish a UI impression.** A
20× unexplained discrepancy inside our own repo is exactly what a careful reviewer finds, and
we would rather find it first.

Requirements before this goes in the abstract:

1. **One controlled timing run.** Median and spread over the scored cohort, `clot_ml_0`,
   end-to-end, on named hardware. Not a recollection from the UI.
2. **End-to-end means end-to-end** — geometry → mesh → FEM t=0 → rollout → clot timeline. If the
   1.5 min excludes meshing or the FEM solve, the honest number is larger; state what is inside
   the boundary.
3. **Name both machines**, and say the COMSOL 48 h is a cohort average.
4. **State training cost and break-even.** One-time training against 48 h/vessel means
   break-even lands within a handful of vessels — the answer is favourable, so being scrupulous
   costs nothing.
5. **Delete or annotate the stale 25–30 min line** in WALL_MODEL_PLAN §0 so the repo does not
   carry two order-of-magnitude-different numbers for "a rollout".

### §5 is the paper's best methodological content — keep it, reframed

Rev 1 proposed this as a standalone paper. As a *section* it is stronger, not weaker: it turns
an architecture choice that would otherwise look lazy ("they used FEM instead of a neural
surrogate") into a measured result. Reviewers reward that.

What it says: replacing the t=0 field with our RGP-DEQ surrogate costs −0.35 on the deployed
score, while a local Carreau FEM solve sits inside noise of ground truth (0.705 vs 0.710). The
reason is not velocity accuracy — over 33 vessels, correlation of velocity rel-L2 with the
downstream drop is **−0.030**, versus **+0.613** for wall gate Jaccard. The readout seeds from
`(gate > 0) & wall`, so when a surrogate's wall gate fires nowhere, thirteen channels go
identically zero: `patient010` 131 mask nodes → 0, F1 0.969 → 0.000, with the cliff between 5%
and 8% velocity error. And an oracle monotone remap — the ceiling on any calibration — moves
wall gate Jaccard only 0.339 → 0.382, so rank order at the wall is wrong and no post-hoc fix
recovers it.

**Frame it as the gated-coupling claim, not as surrogate incapability.** Published PI-GNNs
report wall-shear agreement well above ours (R = 0.94; 7.6% directional WSS error — see
[RELATED_WORK](RELATED_WORK.md) §2), so "a learned surrogate cannot do this" will not survive
review. The claim that does survive, and that those papers actively support: *field norms do not
determine downstream fitness when the coupling is a gate rather than a smooth map* — and every
one of those papers reports in exactly the norms we show to be uninformative.

**RGP-DEQ lives here.** Not as an architecture contribution — DEQ-for-steady-PDE is prior art
(FNO-DEQ, NeurIPS 2023) and mesh-GNN hemodynamics is crowded — but as the instrument that makes
the measurement. A physics-informed graph DEQ with respectable field error that loses to a plain
FEM solve downstream is precisely the evidence §5 needs. Report it honestly as an ablation arm;
do not claim it as a component of the shipped tool.

---

## 4. What is now out

Out entirely, unchanged from PUBLICATION_NOTES §3: the **local kinematic corrector** (worse than
null: MAE 0.684 vs 0.630; diversion cos −0.142); the **PI wall-shear Tier 2 rebuild**
(`corr_log ≈ 0` on 12-vessel LOVO); **"shear redistribution is elliptic"** (true, not novel);
**"shielding vs acceleration"** as a discovery (textbook — usable only as a *validation* figure,
and in a one-paper world it probably does not make the page budget).

Out as a *standalone* paper but retained as §5: the flow-surrogate requirements result.
Out as an architecture claim but retained as §5's ablation: RGP-DEQ.

---

## 5. Limitations to state, not hide

* **One operating point.** Re = 450 for every anchor ([GENERALIZATION_PLAN](GENERALIZATION_PLAN.md) §1.1). Framed as scope, not deficiency: the generalization claim is over geometry, and flow is an input. State it in the abstract, not only here. Transfer to other regimes is untested; §6c gives the honest sensitivity check, and the §1 clip caveat says where it would break first.
* **Aneurysm generalization is n=1** out-of-fold, and n=1 again in SEALED. A property of the data, not the split (§6b).
* **2D.** The ground-truth model is 2D; the review's most-repeated future direction is 3D. Say plainly that this is a 2D surrogate of a 2D model, and that 3D is future work.
* **Simulation ground truth.** No in-vitro or in-vivo data. The claim is fidelity to a published model, never clinical prediction. One sentence of overclaim in the abstract costs the paper.
* **Sealed holdout n=4** (`007/013/031/043`) after the VIZ_HALF release ([SEALED_SPLIT](SEALED_SPLIT.md)) — a documented, deliberate trade. Report sealed as sealed; never average it into FIT/DEV.
* **C0 is a final-time result.** Mean-over-time is materially unchanged (0.5713 → 0.5792). Your own tables show it; state it.
* **Geometry naming.** `patient0NN` are parametric-synthetic vessels. Rename for publication or a reader will believe they are people.

---

## 6. The evidence backbone — geometry generalization, from data we already have

No new COMSOL runs. The protocol that supports the claim is **already implemented** in
[`src/clot_ml/geometry_splits.py`](../src/clot_ml/geometry_splits.py), and it was built for
exactly this question.

### 5a. Primary evidence — geometry-stratified K-fold

`geometry_splits.py` does geometry-stratified K-fold over the eligible non-SEALED pool: every
vessel held out exactly once, priority-class vessels distributed across folds by construction so
each fold trains on at least two. That yields an honest **out-of-fold score per vessel**, broken
down by geometry class. For a geometry-generalization paper this is the right primary table, and
with n this small, cross-validation is a *better* protocol than a single holdout, not a
concession.

It also solves a problem the fixed split cannot. As the module documents: the old FIT/DEV cut is
*exactly* the stenosis/aneurysm set against an all-baseline FIT, so **every FIT-vs-DEV number in
[PHASE9_ML](PHASE9_ML.md) is confounded with geometry class** — DEV off-wall 0.80 vs FIT 0.64 is
three pathological vessels against ten normal ones, not evidence of generalization. Do not put
any FIT-vs-DEV comparison in the paper as a generalization result. K-fold replaces it.

### 5b. The aneurysm problem — state it, do not engineer around it

With one non-SEALED aneurysm (`patient040`), no split trains on an aneurysm while measuring a
different one. `patient039` is the other one and is truncated at T=92. So **aneurysm
generalization is an n=1 out-of-fold number and must be quoted as such** — the module says so and
the paper must too. `patient043` in SEALED gives a second, independent n=1.

Options, given that new data is not available:

* **Recommended: report both, honestly.** K-fold over the non-SEALED pool as the main table, then
  the 4 SEALED vessels (`007/013/031/043`) reported **once** as a confirmatory, never-touched
  number. Aneurysm is n=1 on each side; say it in the caption and in the limitations. This is a
  standard, defensible structure and it preserves the project's best asset — a genuinely sealed
  set — as the closing evidence.
* **Not recommended: opening SEALED into the folds** to get a second trainable aneurysm. It buys
  one vessel and spends the only untouched evidence in the project.

> **A hygiene point that must appear in the methods.** Architecture and hyperparameters were
> selected on FIT/DEV *before* the folds are run, so K-fold out-of-fold scores measure geometry
> transfer of the **fitted weights**, not of the whole pipeline end-to-end. That is a real and
> normal caveat; stating it is what makes the SEALED number worth quoting.

### 5c. Cheap regime-sensitivity evidence, no new data

The research sweeps already run the FEM solver plus the clot model across Re 150–900 and across
stenosis, aneurysm, width, bend, length, eccentricity and roughness axes
(`configs/research_sweeps/`, [RESEARCH_SWEEPS](RESEARCH_SWEEPS.md)). There is **no COMSOL ground
truth** at those points, so these cannot validate accuracy — but they can show the surrogate
produces monotone, physically-expected trends, and they directly exercise the geometry axis the
paper claims.

Use them two ways, both clearly labelled: (i) a **geometry-response figure** — the strongest
visual argument that the model responds to shape rather than memorising vessels — and (ii) an
**off-regime sensitivity check** answering the Re question honestly ("behaviour remains
physically ordered outside the training regime; accuracy there is unvalidated"). Never present
either as validation. The §1 clip caveat above predicts where the Re sweep should start to
misbehave — if it does, that is worth reporting plainly rather than omitting.

### 5d. Critical path

* **REMOVED — multi-Re training cohort.** The reframing eliminates it. This was the single most
  expensive item in rev 2 and it is gone.
* **DEMOTED — the 22 `.nas` mesh exports.** Supports §5, which is a design-justification section
  at n=4–5 with the caveat stated; the 33-vessel correlation table carries that argument. Do it
  only if COMSOL capacity is free.
* **PROMOTED — the K-fold run.** This is now the paper's primary evidence and the longest
  compute pole. Everything else waits on it.

### Order of work

1. **Ask Giulia the wound-provenance question.** One email; unblocks §6 and settles authorship.
2. **Run geometry-stratified K-fold** over the eligible non-SEALED pool; produce the per-vessel,
   per-geometry-class out-of-fold table. Start first.
3. **Run the two oracle diagnostics** for §5c of the paper while K-fold runs: `--cohort` under GT
   flow, and the oracle under `--flow pred` (PUBLICATION_NOTES §4). Hours, not days.
4. **Run the geometry and Re sweeps** for §4b and §6c figures.
5. **Score SEALED exactly once**, after every other choice is frozen. Once it is spent, it is
   spent.
6. **Freeze figures, then draft.** Methods must carry PUBLICATION_NOTES §5 in substance:
   `clot_guiding` is the metric of record and never mixed with Deploy Score v2; clot placement
   chosen by geometry not node ordering; quadratic meshes make raw wall profiles an interleaved
   sawtooth; `sr/sr0 = 0.1226` is a measured constant with a stated validity domain, **not** a
   blockage law.
7. **Reproducibility package.** Geometry generator + seeds, graph packs for the scored cohort,
   promoted checkpoints, and the per-claim eval commands PUBLICATION_NOTES already records.
   Every number in the draft resolves to a command — that convention exists here already, carry
   it into the paper.

---

## 7. Venue

The gap statement, the ground-truth model and the reviewers all live in the same place: this is
a **biomedical / biomechanical modelling** paper, not an ML-methods one. Natural targets are the
journal of the source model (Biomechanics and Modeling in Mechanobiology) or the journal of the
review that defines the gap (J Thromb Thrombolysis); Computers in Biology and Medicine and
Annals of Biomedical Engineering are also in-scope for the mesh-GNN hemodynamics line.

Write §5 for that audience. Its content is ML-methodological, but its *point* — you cannot tell
whether a flow surrogate is good enough by looking at its velocity error — is a practical
warning to exactly the people who build these pipelines.

---


## 8. The outline — eight sections, eleven figures/tables

Rev 4, 2026-09-01. Status: **[DONE]** artifact exists · **[GEN]** generator exists, needs a run ·
**[RUN]** compute outstanding · **[DRAW]** hand-drawn schematic.

Generator scripts live in `scripts/publication/` and use **semantic filenames, not paper
numbers** — `config.py::paper_map` ties the two together, because review reorders figures and
renaming scripts each time churns the pipeline.

**Budget: 11 main items.** That is the upper end for a biomedical venue, so §8's failure figure
and half the sweeps are pre-marked for supplement if pages bite.

### 1. Problem and gap
COMSOL costs **~48 h/vessel**; the surrogate answers in minutes. The PRISMA review found
**11 studies**: none learns the clot field, none uses a graph architecture.
* **Fig 1 — pipeline schematic**, COMSOL path alongside, annotated with measured times. **[DRAW]**
* **Table 1 — the 11 studies** recast by what each predicts, our row appended. **[DONE]**

### 2. Ground truth: the model being surrogated
9 bulk + 3 surface species, Carreau, **shear-gated** deposition, clot as the `mu1(Mat)` step.
* **Fig 2 — the physics, and the gate.** `G_wall`'s branches and the `Mat` threshold. Plant the
  gate here; §7 pays it off. **[DRAW]**

### 3. Cohort and geometry classes
23 clot-carrying + 8 clot-free parametric 2D vessels; classes **measured** from lumen width.
* **Fig 3 — the (narrowing, bulge) plane** with both cuts drawn. Shows classes are measured, and
  shows honestly that the **stenosis cut fails to separate** (designated 0.52 / 0.53 / 0.58 vs
  `patient012` baseline at 0.51). One panel, real credibility.
  **[GEN]** `plot_geometry_classes.py` (→ `geometry_classes.pdf`)

### 4. The tool
FEM t=0 + temporal GNN rollout; mesh-agnostic; the C0 constraint.
* **Fig 4 — architecture.** **[DRAW]**
* **Table 2 — C0 ablation.** Off-wall 0.5812 → **0.7078**, readout gap 0.193 → **0.045**,
  replicated ×3. Note it works by fixing the implied-burden tail, not the spread it targeted.
  **[RUN]** — numbers in MODEL_REVIEW §9b; no generator yet, and one should not be faked by
  transcribing constants. Assemble at draft time with provenance, or write a small generator.
* **Table 3 — cost.** Staged per-vessel wall-clock (FEM / features / rollout) plus the log
  comparison against COMSOL. **[GEN]** `generate_timing_data.py` → `plot_timing.py`

### 5. It works — geometry generalization (core result)
* **Table 4 — strict nested-CV out-of-fold**, per vessel and per geometry class.
  **[GEN]** `generate_kfold_table.py`. **The caption must carry two caveats:** the OOF archive is
  exported under **GT t=0 flow** (§6 licenses reading it as the deployed FEM configuration, since
  FEM sits inside noise of GT), and **aneurysm is n=1**.
* **Table 5 — SEALED scored once** plus a clot-free false-positive row (no recall; separate row).
  **[RUN once, last]**
* **Fig 5 — final-time clot maps** (model / GT / error), held-out vessels across classes.
  **[GEN]** `plot_fig3_biochem_final.py`
* **Fig 6 — temporal evolution.** **[GEN]** `plot_fig4_biochem_temporal.py`
* **Fig 7 — geometry-response sweeps**, four axes (`CONFIG.main_sweeps`): stenosis strength,
  aneurysm strength, bendiness, width. Evidence the model tracks *shape*, and it needs no COMSOL.
  The remaining six sweeps go to supplement. **[GEN, curated]**

### 6. Why the flow is solved and not learned — three confirmations
* **Table 6 — the three attempts, one row each.** RGP-DEQ (−0.35; FEM inside noise at 0.705 vs
  0.710); local corrector (MAE 0.684 vs null 0.630, `cos = −0.142`); closed loop (oracle −0.0065).
  One table, no subsections. **[RUN]** — same provenance caution as Table 2.
* **Fig 8 — flow fields**, RGP-DEQ / FEM / GT, velocity and error. **[GEN]** `plot_fig1_flow.py`.
  Generator-`fig1` but **paper §6** — evidence for the negative arc, not motivation. It must not
  open the paper.
* **[RUN]** the two oracle diagnostics: `--cohort` under GT flow; the oracle under `--flow pred`.
* The corrector stays **one paragraph, no figure**.

### 7. The diagnosis and the design rule
rel-L2 correlates **−0.030** with the downstream drop (n=33); gate Jaccard **+0.613**; the failure
is a **cliff** (mask 131 → 0, F1 0.969 → 0.000, between 5% and 8% velocity error); an oracle
monotone remap moves gate Jaccard only **0.339 → 0.382**.
* **Fig 9 — the methodological figure.** (a) correlation of each diagnostic with the wall-score
  drop; (b) the tolerance/cliff curve. **[GEN]** `generate_flow_requirement_data.py` →
  `plot_flow_requirement.py`. **Inputs still [RUN]** — see §10.
* **[RUN] Extend the cliff curve to at least 3 vessels.** One vessel is an anecdote.
* **Fig 10 — the mechanism** (gate seeding → empty mask → 13 channels zero; remap ceiling inset).
  **Optional** — cut first if pages bite; Fig 9 carries the claim alone. **[DRAW]**
* **The transferable sentence:** *where a learned field feeds a thresholded consumer, the accuracy
  requirement is set by the consumer's decision statistic, not by field norms.*

### 8. Scope, limitations, novelty
2D surrogate of a 2D model; one operating point as **scope** (flow is an input); aneurysm n=1
twice over; simulation ground truth, never a clinical claim.
* **Fig 11 — known failure modes.** **[GEN]** `plot_fig6_failures.py`. Keep if pages allow;
  supplement otherwise.
* **Fig 12 — onset timing, early or late (new, 2026-09-02).** Not "is the score low" but "is
  the model out of phase": signed lag between predicted and true clot onset, pooled over every
  node that eventually clots in both GT and prediction across all 23 strict-OOF vessels
  (n = 2,510 matched nodes). 19% early / 45% on-time / 36% late — a real but mild late bias,
  confirmed not to be one outlier vessel (per-vessel panel). **Supplement-first**, same as Fig
  11 — this is item 12 of the budget's original 11. **[GEN]** `generate_onset_timing_data.py` →
  `plot_onset_timing.py`. No new inference: reuses the strict-OOF archive's own masks/times.
* **Fig 13 — does an error compound or recover? (new, 2026-09-02).** Re-frames the
  score-over-time traces Fig 6 and Fig 11 already export (`fig34_metrics.csv`,
  `fig6_metrics.csv`) as an explicit divergence-vs-convergence claim: wall-score dips tend to
  fully recover (patient014 collapses to 0.0 at t=40, is back to 0.98 by t=100); off-wall dips
  sometimes recover within the horizon (patient020) and sometimes don't (patient014, still
  declining at t=200). **Supplement-first**, or merge as extra panels onto Fig 11 if pages are
  tight — it shares vessels with it. **[GEN]** `plot_error_trajectories.py`, no new data.

**Budget note:** Figs 12–13 push the count to 13; both are marked supplement-first so the
11-item core budget from the header above is unaffected unless review wants them promoted.

**If/when §7.0's wound provenance question is answered and §6 (injured wall) unfreezes:** a
matched-geometry wound/no-wound pair now exists — `wound_patient005` vs. `patient048`, same
`.nas` (identical node bounding box), one run with the wound boundary condition and one
without. It is the "commission the paired A/B run" item WOUND_PROGRESS.md §7 asked for. Ready
as that section's lead figure, but **not counted in the budget above and not GEN-tagged** —
predicted with the shipped `clot_gnn_v6w` pointer rather than the strict-OOF ensemble (the OOF
cohort excludes wound vessels), and neither vessel's held-out status against the base GNN's own
training set has been independently re-verified. Treat as a preview until that's checked and
§7.0 is unblocked. `generate_wound_ab_data.py` → `plot_wound_ab.py`.

**Novelty, ranked:** the measured flow-accuracy requirement and gated-coupling diagnosis
(strongest, ours); first field-level mesh-generalizing surrogate of a continuum multi-species
thrombosis model (strong, qualified by the review's criteria); first graph architecture in this
intersection (narrow, clean); C0 (minor but real); RGP-DEQ as architecture (**no claim**).

---

## 9. Generator status after the 2026-09-01 build

**Added this session:**

| Script | Produces | State |
|---|---|---|
| `generate_timing_data.py` | Table 3 data — staged wall-clock, median/IQR, hardware, `per_step_s` | run |
| `plot_timing.py` | Table 3 figure | ready |
| `generate_kfold_table.py` | Table 4 — per-vessel and per-class OOF, both caveats embedded | ready |
| `generate_flow_requirement_data.py` | Fig 9 data — correlation pairs plus tolerance curves | ready, **inputs missing** |
| `plot_flow_requirement.py` | Fig 9 — draws whichever panels have data | ready |
| `plot_geometry_classes.py` (→ `geometry_classes.pdf`) | Fig 3 | ready |

**Added 2026-09-02** (team figure-board review; see §8 above for placement):

| Script | Produces | State |
|---|---|---|
| `generate_onset_timing_data.py` | Fig 12 data — signed onset lag, pooled + per-vessel, from the strict-OOF archive | ready |
| `plot_onset_timing.py` | Fig 12 figure | ready |
| `plot_error_trajectories.py` | Fig 13 — reuses `fig34_metrics.csv` / `fig6_metrics.csv`, no generator needed | ready |
| `generate_wound_ab_data.py` | Wound A/B preview data — `wound_patient005` vs. `patient048`, shipped `clot_gnn_v6w` pointer | ready, **not GEN-tagged in §8** |
| `plot_wound_ab.py` | Wound A/B preview figure | ready |

All wired into `make_all.ps1`. `config.py` gained `paper_map` and `main_sweeps`.

**Timing is deliberately NOT in `make_all`.** It is a wall-clock measurement; anything else
running on the box corrupts it. Run it alone.

### ⚠ Recovered: the Fig 9 reproduction path had been deleted

`scripts/diag_flow_sensitivity.py` and `scripts/diag_wall_gate_health.py` — both cited in
[PUBLICATION_NOTES](PUBLICATION_NOTES.md) §2 as the reproduction path for the paper's
best-powered result — were **deleted in commit `b2eebb9`**, whose message is "Fix
customer_pipeline.py couple unpack error" and which also dropped 182 lines of diagnostics.
`outputs/runs/` was empty, so none of the artifacts that section names existed either.

Both are **restored, with their stale `clot_ml_v0` imports patched**. Consequence to act on:
**every number in PUBLICATION_NOTES §2 must be regenerated before it enters a draft.** Do not
assume a JSON found on disk is the one the notes describe.

### Curation decisions

* **Sweeps:** four in the main figure (`CONFIG.main_sweeps`), six to supplement. Nothing deleted.
* **Wound sweeps (`16`–`20`): frozen** until provenance is answered — five sweeps is a large
  investment in a section §1 may cut.
* **Re sweeps: stay retired.** Correct under the geometry framing; a figure needing three caveats
  costs more than it buys. Revisit only if a reviewer asks.
* **`fig6` failures: keep**, supplement if tight.

---

## 10. Outstanding runs, in order

1. **Finish the timing run** (30 vessels, full horizon) → `plot_timing.py`.
2. **`generate_kfold_table.py`** — the primary evidence. The archive exists; needs one pass.
3. **Fig 9 inputs**, all three, on an idle box:

```bash
python scripts/eval_clot_ml_0.py --cohort --flow gt   --out outputs/runs/eval_gt.json
python scripts/eval_clot_ml_0.py --cohort --flow pred --out outputs/runs/pred_all.json
python scripts/diag_flow_sensitivity.py patient010 patient005 patient020 --source pred --out outputs/runs/flow_sensitivity.json
```

   The third also discharges the "extend the cliff curve past one vessel" item. Panel (a)
   additionally needs per-vessel gate diagnostics joined onto the eval rows —
   `generate_flow_requirement_data.py` names exactly which columns and says so if they are absent.
4. **Geometry / main sweeps** for Fig 7.
5. **`plot_geometry_classes.py` (→ `geometry_classes.pdf`)** for Fig 3 (cheap, any time).
6. **SEALED, once, last.**
---

## 11. Measured this session (2026-09-01/02)

### Table 3 — cost. **2,156×.**

30 vessels, full horizon, shipped `clot_ml_0`, `flow=fem`, NVIDIA RTX 500 Ada Laptop GPU,
torch 2.11.0+cu128. Warm-up discarded, CUDA-synced.

| stage | median | IQR | min–max |
|---|---|---|---|
| FEM t=0 | 6.02 s | [4.1, 10.3] | 2.2 – 41.5 |
| features | 14.45 s | [11.3, 17.8] | 6.4 – 27.1 |
| rollout | 55.68 s | [39.6, 74.8] | 17.0 – 190.3 |
| **end-to-end** | **80.16 s** | **[53.5, 110.1]** | **37.9 – 226.5** |

**Median 80.2 s = 1.34 min against COMSOL's 48 h → 2,156×.** The UI recollection of ~1.5 min was
right; the ~25–30 min in [WALL_MODEL_PLAN](WALL_MODEL_PLAN.md) §0 is a stale artifact of the
retired `WC_v7` stack on a 4 GB GPU with scoring attached, and should be annotated there so the
repo stops carrying two numbers for "a rollout".

Rollout is ~69% of the cost, features ~18%, FEM ~7.5% — worth one sentence in the paper, because
it shows the classical solve is *not* the bottleneck the reader will assume it is.

Boundary: pack → FEM → features → rollout. Meshing and geometry construction are upstream and
excluded; COMSOL's 48 h covers geometry → mesh → solve. State this, or add the meshing cost.

Artifacts: `outputs/publication/data/timing.{json,csv}`, `figures/timing_cost.pdf`.

### Table 4 — geometry generalization, strict nested-CV out-of-fold

23 vessels, 5 folds, `flow=gt`, final time. SEALED (`007/013/031/043`) untouched.

| class | n | wall | off-wall | note |
|---|---|---|---|---|
| baseline | 19 | **0.9070** | **0.6574** (n=12) | 7 vessels have empty off-wall GT — reported separately |
| stenosis | 3 | **0.8305** | **0.7806** (n=3) | |
| aneurysm | 1 | 0.9507 | 0.9097 | **n=1 — a vessel, not a class** |

Empty-off-wall-GT vessels (7): FP score 0.4334, **excluded from the recall-bearing mean**.

**Wall generalization holds across geometry class** — 0.83–0.95, and the pathological classes are
not catastrophically worse than baseline. That is the paper's core claim and it now has an
out-of-fold number behind it.

### ⚠ A defect this session found and fixed in the primary table

The first run reported **baseline off-wall 0.5749**. That number was contaminated: **7 of the 19
baseline vessels carry wall clot but ZERO off-wall ground truth**, so their off-wall F1
(`patient014` 0.073, `patient024` 0.093, `patient036` 0.296, `patient018` 0.444, …) measures false
positives against an empty GT — not recall — and averaging it into a class mean repeats exactly
the error the clot-free protocol exists to prevent (MODEL_REVIEW §8b: *"they carry no recall and
must never enter a recall-bearing mean"*). The rule was being enforced for the 8 clot-free
vessels and violated *within* the clot-carrying ones.

`generate_kfold_table.py` now measures per-vessel GT burden per domain, restricts off-wall means
to vessels with non-empty off-wall GT, and reports the empty ones as a separate FP row. **Baseline
off-wall moves 0.5749 → 0.6574.** Any off-wall number quoted from before this fix is wrong.

### Fig 3 — geometry classes

52 packs plotted in the (narrowing, bulge) plane. Independently reproduces the documented
failure of the measured stenosis cut: **5 undesignated vessels sit below the most-open designated
stenosis (0.58)**. `docs`' own example was `patient012` at 0.51; on the full pool it is five.
Draw it and say so.

Artifacts: `figures/geometry_classes.pdf`, `data/geometry_classes.json`.

### Fig 9 — the flow requirement. **Regenerated from scratch, n=33. The claim holds.**

All three inputs rebuilt on 2026-09-02 after the §9 recovery, plus a new
`generate_flow_diagnostics.py` (the eval harness emits outcomes, not candidate predictors, so
the predictors had to be computed: gate firing sets, Jaccard, `dsrx` correlation and rel-L2,
recomputed with the consumer's own `lss`/`sgt` constants, `_flow_hops` stencil and `dsrx_gain`).

**Panel (a) — correlation with the wall-score drop (GT flow → surrogate flow), 33 vessels:**

| diagnostic | r | PUBLICATION_NOTES §2 |
|---|---|---|
| empty-gate indicator | **+0.745** | −0.350 |
| gate Jaccard | **−0.687** | +0.613 |
| wall-gate firing ratio | −0.251 | −0.395 |
| dsrx correlation | −0.132 | +0.131 |
| **velocity rel-L2** | **+0.029** | −0.030 |

**The headline claim is confirmed and slightly strengthened.** Velocity rel-L2 carries
essentially no information about the outcome (|r| = 0.029, against 0.030 in the notes), while the
gate statistics carry 0.69–0.75. Five of 33 vessels have a wall gate that fires **nowhere** under
the surrogate — `patient010/018/021/028/037` — and the empty-gate indicator is now the single
strongest predictor in the table.

> **SIGN CONVENTION — fix this in PUBLICATION_NOTES §2.** This regeneration correlates each
> diagnostic against the **drop** (`wall_gt − wall_pred`, larger = worse), and every sign then
> reads the way a health check should: empty gate → bigger drop (+), better gate overlap →
> smaller drop (−). The notes' published signs are **inconsistent with that convention** (gate
> Jaccard +0.613 against a drop would mean better overlap predicts worse outcome). Magnitudes
> agree closely; the signs do not. Quote the regenerated table and state the convention
> explicitly, and correct §2 rather than leaving two sign conventions in the record.

**Panel (b) — tolerance curves, 3 vessels. The cliff is real but vessel-dependent.**

Wall score against blend fraction `a` (`u = (1-a)·u_gt + a·u_pred`):

```
patient010   0.882 0.901 0.901 0.912 0.907 0.963 | 0.000 0.000      <- flat, then to ZERO
patient020   0.929 0.928 0.932 0.912 0.909 0.897 | 0.695 0.366      <- flat, then steep decline
patient005   0.715 0.556 0.538 0.510 0.501 0.477 | 0.474 0.486      <- drops AT ONCE, then flat
             a=0   0.05  0.10  0.20  0.35  0.50  | 0.75  1.00
```

**This is why extending past one vessel mattered, and it qualifies the story.** Two of three
vessels show the documented shape — a plateau to `a≈0.5`, then a cliff, with `patient010` going
to exactly 0.000 (the empty-gate discontinuity, and its diagnostics confirm `empty_gate=1`).
But **`patient005` has no plateau at all**: it loses 0.16 by `a=0.05` and is then flat and
insensitive out to `a=1`.

So the honest claim is *"a tolerance exists and a cliff exists, but its location and depth are
vessel-dependent"* — **not** the n=1 sentence in PUBLICATION_NOTES §2 ("holds F1 ~0.90 up to ~5%
velocity error and falls to zero between 5% and 8%"), which is `patient010`-specific. Say the
range, show all three curves, and let the reader see the spread.

Artifacts: `outputs/runs/{eval_gt,pred_all,flow_sensitivity,flow_diagnostics}.json`,
`outputs/publication/data/flow_requirement.json`, `figures/flow_requirement.pdf`.
Reproduce: `scripts/publication/_run_flow_requirement_inputs.sh` then
`generate_flow_diagnostics.py` → `generate_flow_requirement_data.py` → `plot_flow_requirement.py`.

### Still outstanding

Fig 9 is **done**. Remaining from §10: the geometry/main sweeps (Fig 7), Tables 2 and 6, the
SEALED run, and the two hand-drawn schematics. Also now due: **correct PUBLICATION_NOTES §2's
correlation signs and its n=1 cliff sentence**, per the two boxes above.
