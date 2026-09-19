# Related work — Phase 0 novelty triage

Opened 2026-09-01. Output of Phase 0 in PUBLICATION_PLAN.md §4.

Scope of this pass: establish (a) the anchor citation for the COMSOL ground-truth model, and
(b) whether each candidate paper's claim is already in the literature. This is a *scoping*
pass — abstract-level for most entries, full-text for the few that decide a go/no-go. Entries
marked **[UNVERIFIED]** were read from abstract or search summary only and must be read in full
before anything is drafted against them.

---

## 0. The anchor citation — found

> **Cardillo, G. and Barakat, A. I. (2025).** *A 2D computational model of chemically- and
> mechanically-induced platelet plug formation.* Biomechanics and Modeling in Mechanobiology
> **24**(5). DOI [10.1007/s10237-025-01966-3](https://doi.org/10.1007/s10237-025-01966-3).
> Preprint: bioRxiv [2023.01.26.525741](https://www.biorxiv.org/content/10.1101/2023.01.26.525741v1).
> LadHyX, CNRS, École Polytechnique, Institut Polytechnique de Paris.

This is the ground truth. Confirmed matches to what this repo consumes: COMSOL Multiphysics
5.6 finite-element implementation; platelet transport, activation, adhesion and aggregation
driven by *both* biochemical and mechanical factors; and — the distinguishing feature of the
model, and the thing our `srf1` gate implements — **shear-gradient-dependent** deposition,
which the paper positions as its own contribution over prior thrombosis models.

**This citation reframes every claim in the project.** Everything we have is a surrogate *of a
published model from our own lab*, which is a good position: the ground truth is peer-reviewed,
the physics is documented externally, and we are not asking reviewers to accept an unpublished
simulator. Say "surrogate for the Cardillo–Barakat model" throughout and the synthetic-data
critique loses most of its force — the model, not the mesh, is the object of study.

**Species check — done, 2026-09-01.** Read from `smodel.json` in `phase2_wound_001.mph` and
`phase2_nowound_011.mph`: **9 bulk species** `rp, ap, apr, aps, at, pt, th, fg, fi` with
`Reactions_9spec` / `WallFlux_9spec` / `InletFlux_9spec` / `ExitFlux_9spec` /
`InitialValues_9spec`, and **3 surface species** `M, Mas, Mat` with
`wall_surface_reactions_3spec` / `InitialValues_3spec` / `NoFlux_InletAndOutlet_3spec`. Carreau
rheology node present; clot enters momentum as the `mu1(Mat)` step. Consistent with the
published model.

**Wound provenance — the one open question, and it is not answerable from the files.** Diffing
the two model trees: `phase2_nowound_*` carries the base feature set above; `phase2_wound_*`
adds **`WoundFlux_9spec`**, **`SfcRxn_3spec`**, and the **`wound`** / **`wallandwound`**
selections. So the wound law is an *addition on top of* the published model — but nothing in the
`.mph` records who added it.

**Ask Giulia:** is the ungated wound boundary condition hers, and is it published, in
preparation, or unpublished? This decides whether the wound section is "our surrogate reproduces
a published extension" or "we characterise an unpublished extension", and it decides the author
list. The WOUND_PROGRESS finding — injury is the same surface law with the
two shear gates deleted — is not ours to publish until this is answered.

---

## 1. Body A — ML in computational thrombosis (decides THE paper)

### The systematic review — read in full, 2026-09-01. It hands us the gap statement.

> **Al Bannoud, M., Dias Martins, T., de Lima Montalvão, S. A., Annichino-Bizzacchi, J. M.,
> Maciel Filho, R., Wolf Maciel, M. R. (2026).** *Artificial intelligence in computational
> modeling of thrombosis: bridging mechanistic insights and clinical translation.* Journal of
> Thrombosis and Thrombolysis **59**:727–745. DOI
> [10.1007/s11239-025-03222-y](https://doi.org/10.1007/s11239-025-03222-y). Published online
> 17 Dec 2025.

PRISMA systematic review of the entire ML + computational-thrombosis intersection. **11 eligible
studies**, in five categories: platelet signalling, outcome prediction, thrombin threshold
prediction, shear rate prediction, multiscale/physics-informed modelling.

The eligible set, from their Table 1: Flamm 2012 (ANN/NARX, platelet calcium in a multiscale
model); Lu 2017 (ANN + LKMC/LBM/FEM); Shankar 2022 and 2023 (ANN in a 3D stenotic aggregation
model); Bouchnita 2022, 2023 (ANN/SVM/KNN/DT outcome classification); Bouchnita 2023b (DNN
surrogate, **binary classification** of coagulation initiation from thrombin parameters, 94%);
Qian 2024 (Coagulo-Net PINN, parameter inference on coagulation ODEs); Khajavi 2025 (MLP-ANN +
LBM-IBM, **shear stress** regression); Al Bannoud 2025 ×2 (outcome classification for recurrent
VTE).

> ## ⚠ SCOPE CORRECTION 2026-09-09 — read §6.1 before quoting any of the three facts below
>
> They describe **the review's screening window**, which closed before the field moved. A
> Transformer-GNN clot-field surrogate on held-out geometries was published in *Comput Biol Med*
> in May 2026 (Pelissier, Meliga & Hachem, 208:111649). Fact 1 must be written as "within the
> 11"; fact 2 stands only as a statement about the reviewed set; fact 3 survives and is
> strengthened. **No priority language anywhere in the manuscript** — §6.1.

**Three facts, scoped to the review's eligible set:**

1. **Within the 11, nobody learns the clot field.** Every eligible study predicts a scalar, a
   class label, or a non-clot field (shear). The one called a "surrogate model" (Bouchnita 2023b) is a binary
   classifier over 7,675 simulations, taking scalar thrombin parameters and injury size as
   input. Coagulo-Net solves and infers ODE parameters; it does not generalize a rollout to
   unseen geometry.
2. **No graph or mesh architecture appears anywhere.** ML models across all 11: ANN, NARX, SVM,
   KNN, DT, RF, NB, LR, XGBoost, CatBoost, EBM, DNN, PINN. Zero GNNs. **Retracted phrasing:**
   "a mesh-agnostic graph surrogate is first-of-kind in this intersection" — see §6.1.
3. **The review asks for this direction by name**, and an independent group moved in it within
   nine months (§6.1), which makes the point rather than undermining it. It repeatedly names as
   future directions:
   extension "to three-dimensional vascular geometries", "COMSOL anchor-specific anatomical
   reconstructions", and architectures (CNN/RNN/transformer) that "improve the detection of
   complex spatial and temporal patterns in thrombus growth". Our motivation section is a
   quotation from the review that defines the field.

One caution worth noting in the draft: the review's inclusion criteria required *explicit
ML–mechanistic integration*, and it explicitly excludes reduced-order and metamodelling work
(it discusses Méndez Rojano's PFA-100 UQ surrogate as excluded-but-relevant). So "11 studies"
bounds the ML-hybrid literature, not the surrogate literature generally. Do not overstate it as
"only 11 papers exist in thrombosis surrogate modelling".

### The wider ML-thrombosis cluster (outside the review's criteria)

Four clusters, none a full-field spatiotemporal surrogate of a continuum multi-species
thrombosis CFD:

| Work | What it learns | Why it is not our contribution |
|---|---|---|
| **Neural-operator surrogate for platelet deformation** ([Bioengineering 12(9):958, 2025](https://doi.org/10.3390/bioengineering12090958)) | DeepONet: membrane deformation of a *single platelet* across capillary numbers; <1% median displacement error, 4–5 orders speedup | Sub-cellular scale. Complementary to us, not competing |
| **Latent neural DEs for clot growth** ([arXiv 2608.08165](https://arxiv.org/abs/2608.08165)) | SNODE/SNFDE: infers tissue-factor parameter and forecasts **scalar clot size** from sparse observations | Low-dimensional trajectory forecasting, not spatial fields on a mesh. **The nearest competitor in framing** — read in full |
| **Coagulo-Net** ([PMC11578045](https://pmc.ncbi.nlm.nih.gov/articles/PMC11578045/)) | PINNs for blood-coagulation equations; solution + inverse inference | Solves/infers the PDEs; does not generalize a rollout across unseen geometries |
| Hybrid ANN–ODE VTE risk; DNN thrombogenesis classifiers; LAA thrombus prediction | COMSOL anchor-level *risk classification* | Different problem entirely |

**Verdict — the paper: GO.** The gap is real and is now documented by a systematic review rather
than by our own search. Remaining condition: the multi-Re training cohort, which the one-paper
decision promotes to the critical path (PUBLICATION_PLAN §5).

---

## 2. Body B — GNN / operator surrogates for hemodynamics (decides RGP-DEQ, and threatens Paper 1)

This body is **crowded, mature, and reports strong numbers**. That matters more than it first
appears.

* **Mesh CNNs for WSS in 3D arteries** ([arXiv 2109.04797](https://arxiv.org/pdf/2109.04797))
* **SE(3)-equivariant mesh neural networks for wall hemodynamics** ([arXiv 2212.05023](https://arxiv.org/pdf/2212.05023); Comput Biol Med 2024) — directional WSS at **7.6%** approximation error, two orders of magnitude faster than CFD
* **Physics-informed GNN for real-time WSS in stenotic coronary arteries** ([Sci Rep 2026](https://www.nature.com/articles/s41598-026-47410-z)) — **R = 0.94** on WSS across stenosis types
* **PI-GNN for carotid flow fields from 4D flow MRI** ([arXiv 2408.07110](https://arxiv.org/pdf/2408.07110)) — PointNet++ with group-steerable layers, trained on in-vivo data
* **Graph transformers for pulsatile aneurysm flow** ([arXiv 2601.19876](https://arxiv.org/html/2601.19876)); **graph deep learning for aneurysm flow + risk** ([arXiv 2512.09013](https://arxiv.org/abs/2512.09013))
* **FNO-DEQ / deep-equilibrium neural operators for steady-state PDEs** (Marwah et al., [NeurIPS 2023](https://arxiv.org/pdf/2312.00234)) — the DEQ-for-steady-PDE idea is established prior art

**Verdict — RGP-DEQ as an architecture paper: NO-GO, confirmed.** DEQ-for-steady-PDE is taken
(FNO-DEQ). Graph/mesh hemodynamic surrogates are taken, several times, with numbers we do not
beat. The specific *combination* (physics-modulated GAT + Perceiver global mixing + DEQ fixed
point) is new, but defending a combination claim requires beating the baselines above, and we
have no such benchmark.

**Verdict — RGP-DEQ as a component of Paper 1: STRONG GO.** This is the right home for it, and
it is a real contribution rather than a consolation prize. A physics-informed graph DEQ trained
to a respectable field error that then *loses to a plain local FEM solve on the downstream task*
is precisely the evidence Paper 1 needs. It is the instrument that makes the measurement.

### ⚠ The one finding that changes the plan

**Published PI-GNNs report strong wall-shear agreement (R = 0.94; 7.6% WSS error). We report
our surrogate's wall `sr` correlation at 0.431 against FEM's 0.993.**

A reviewer will make this comparison immediately, and as currently framed Paper 1 has no answer.
The risk is that our headline claim —

> *"a flow surrogate cannot get the derived wall fields right well enough to drive the readout"*

collapses under review to

> *"**our** flow surrogate cannot, and published ones look better."*

The measurement is not wrong; the **scope of the conclusion** is. Two ways out, and they are not
mutually exclusive:

1. **Reframe to the metric claim, which is untouched by this.** The defensible core of Paper 1
   was never "surrogates are incapable" — it is *"velocity rel-L2 does not tell you whether a
   surrogate will work downstream, and the requirement is set by the consumer's gating structure,
   not by field norms."* That claim survives regardless of how good anyone's surrogate is. Under
   this framing the published PI-GNN numbers become **support**: they are all reported in exactly
   the norms we show to be uninformative, so none of them establishes downstream fitness.
   *This is the recommended framing.*
2. **Add one competitive baseline.** Reimplementing an SE(3)-equivariant mesh network or a PI-GNN
   on our packs and showing it *also* fails the gate would convert the claim from "our surrogate"
   to "this class of surrogate". Much stronger, materially more work. Worth scoping, not worth
   blocking on.

Recommendation: adopt framing 1 now, and treat framing 2 as the stretch that would make the
paper hard to argue with.

---

## 3. Body C — task-aware surrogate evaluation (decides Paper 1)

The *concern* is known. The *measurement* is not.

Known and published, in general form:

* **Decision-aware / task-based loss functions** — a body arguing that models should be trained and judged against end-task decision cost rather than MSE
* **Case for a unified surrogate modelling framework in the age of AI** ([arXiv 2502.06753](https://arxiv.org/html/2502.06753v1)) — explicitly distinguishes *diagnostic* from *task-based* evaluation and argues for functional accuracy on downstream tasks
* **EcoL2** ([arXiv 2505.12556](https://arxiv.org/html/2505.12556)) and rollout-quality critiques — "a single scalar error does not adequately describe PDE rollout quality"; rel-L2 can be less discriminative than alternatives where a model gets magnitude right and smears gradients
* **Coupled-multiphysics UQ via GP surrogates** ([arXiv 2601.18480](https://arxiv.org/abs/2601.18480)) — proves predictive variance stays **bounded** through iterative coupling *under mild regularity and stability assumptions*

Not found anywhere in this pass, and this is the opening:

1. A **quantified cohort-scale correlation** between a surrogate's field error and its downstream
   task outcome. Our −0.030 (rel-L2) versus +0.613 (gate Jaccard) over 33 vessels is, as far as
   this scan can tell, the first such number in a coupled biophysics pipeline.
2. A **discontinuous** coupling failure. The literature above reasons about graded error
   propagation, and the UQ result is explicitly a *boundedness* theorem under regularity. Our
   coupling is a **gate** — `(gate > 0) & wall` — so the regularity assumption fails by
   construction, and the response is a cliff: 131 mask nodes → 0, F1 0.969 → 0.000, between 5%
   and 8% velocity error. **We are a counterexample regime for the assumptions under which
   error propagation is provably benign.** That is a sharper framing than "L2 is a bad metric"
   and I would build the paper on it.
3. An **impossibility result for post-hoc correction** — the oracle monotone remap ceiling
   (0.339 → 0.382). Rank order at the wall is wrong, so no calibration, quantile matching or
   gain fix recovers the gate. This closes the obvious reviewer escape hatch.

> ## ⚠ REVISED 2026-09-09 — this verdict rested on numbers the shipped arm no longer produces
>
> The three claims above were measured on the **pre-cross-fit** RGP-DEQ arm. On the arm the
> paper actually reports, all three have weakened or gone:
>
> | claim above | current arm (see `docs/archive/PAPER.md` §7) |
> |---|---|
> | rel-L2 uninformative, −0.030 | **+0.224** — weaker than gate agreement, but informative |
> | gate Jaccard +0.613 | **−0.707** (sign convention now stated; magnitude larger) |
> | a **discontinuous** failure: 131 nodes → 0, F1 0.969 → 0.000 | **0 of 33 empty gates.** No vessel collapses; curves fall off and plateau |
> | "counterexample regime for benign error propagation" | the regularity failure it depended on **does not occur** on this arm |
>
> **So the sharpest version of this framing — a discontinuity theorem — is a historical mechanism
> illustration, not a description of what we ship.** What survives is a correlation result: the
> consumer's gate statistic predicts downstream loss far better than field norms do, on n=33.
> Real and worth reporting; not a counterexample to anything.
>
> **Revised positioning, and it is stronger for the tool paper.** Lead the flow section with the
> **flow-solve share of runtime**, which needs no appeal to anyone's field accuracy: the t=0 solve is 7.9% of
> end-to-end runtime, so a perfect instantaneous flow surrogate saves at most that. RGP-DEQ
> captures 6.8% of it and costs −0.1185 off-wall (p=0.0005). *The trade is bad by construction,
> not by accident* — and a better published surrogate moves the numerator, never the ceiling.
> That completely defuses the §2 threat, because we are no longer in a numbers fight about WSS.
>
> **And the paper's strongest claim is no longer this section at all.** It is the measured
> division of labour across all four couplings (`docs/archive/PAPER.md` §5), which did not exist when
> this triage was written: solve the flow, solve the wall, learn the lumen, learn *when*. This
> section is one row of that table.

*Superseded verdict, retained for provenance:* "Paper 1: GO, and it is the strongest thing in
the project", positioned as *"we measure how inadequate, in a pipeline where the coupling is a
gate rather than a smooth map, and we show the failure is discontinuous, predictable from the
consumer's structure, and not repairable post-hoc"*, under the title **"Gated couplings break
surrogate error budgets"**. Both the "discontinuous" clause and the title are withdrawn.

---

## 4. Reading queue, in priority order

1. **J Thromb Thrombolysis 2025 review** (DOI 10.1007/s11239-025-03222-y) — institutional access. Confirms or kills the Paper 2 gap statement.
2. **Cardillo & Barakat 2025**, full text, against `dmodel.xml` — species-level confirmation, and settle whether the **wound** law is published.
3. **arXiv 2608.08165** (latent neural DEs) — nearest competitor in framing; know exactly how our claim differs.
4. **arXiv 2502.06753** (unified surrogate framework) — the paper Paper 1 positions against most directly.
5. **Sci Rep 2026 PI-GNN, and arXiv 2212.05023** — the two whose WSS numbers create the §2 risk. Read carefully: *which* wall quantity, on what meshes, and is it comparable to our `sr` / `dsrx` at all? There is a real chance the metrics are not commensurable, which would defuse the risk outright.

---

## 5. Go / no-go summary

Updated for the one-paper decision (PUBLICATION_PLAN rev 2).

| Candidate | Verdict | Condition |
|---|---|---|
| **The paper** — mesh-generalizing clot-field surrogate | **GO, repositioned** | ~~Multi-Re cohort~~ **removed** — the generalization axis is geometry and flow is an input (PUBLICATION_PLAN rev 2 §1). **2026-09-09: no longer the lead claim and no priority language** — it is the tool the §5 measurement implies (§6.1) |
| **The measured division of labour** — per component, and per **spatial domain** within a component | **GO — the lead claim, and the one the near neighbour cannot touch** | Done: `docs/archive/PAPER.md` §5. Report as **two axes, not four couplings** (§6.1); grade the three legs — the per-domain flip is strong, the flow leg is a low flow-solve runtime share plus one measured cost, the timing leg is a bounded null |
| Flow-surrogate requirements | GO **as a section**, not standalone | Lead with the **flow solve's low share of runtime** (7.9%), not the discontinuity — see the revision box in §3 |
| RGP-DEQ | GO as the **instrument** that makes the measurement | Never as an architecture contribution. Present it as a *good* model (rel-L2 0.0165, 0 empty gates) that still loses downstream — a weak one would prove nothing |
| Open-loop oracle bound | GO as §5 | Two cheap runs (PUBLICATION_NOTES §4) |
| Wound result | GO as §6 — **blocked** | Provenance question to Giulia, §0 above |
| RGP-DEQ as architecture paper | **NO-GO** | Prior art: FNO-DEQ + a crowded mesh-GNN field |

---

# 6. Second literature pass — 2026-09-09

Run against the question *"what would make this both publishable and used?"* The pass found
**one paper that changes §1's gap statement and nothing else that changes a verdict.** Every
entry below was read to the level stated; the one that matters was read in full.

## 6.1 ⚠ THE SCOOP — a Transformer-GNN clot-field surrogate, published, in one of our venues

> **Pelissier, U., Meliga, P., Hachem, E. (2026).** *Multiphysics learning with graph neural
> networks for thrombosis prediction in intracranial aneurysms.* **Computers in Biology and
> Medicine 208:111649.** DOI [10.1016/j.compbiomed.2026.111649](https://doi.org/10.1016/j.compbiomed.2026.111649).
> PMID 41905249. Open access on HAL: [hal-05670914](https://hal.science/hal-05670914v1)
> (CC BY, submitted to CBM 2026-02-23, deposited 2026-06-26). CEMEF / Mines Paris, PSL;
> ERC CURE 101045042.

**Read in full.** This is a mesh GNN that learns a spatiotemporal thrombus field on held-out
vessel geometries, which is the sentence PAPER.md §2 currently says nobody has written.
Precisely what it does:

| | Pelissier et al. 2026 | ours |
|---|---|---|
| ground truth | 7-species continuum + surface platelet deposition (`kus/kas/kaa`), CIMLIB-CFD, in-house | Cardillo–Barakat 2025, 9 bulk + 3 surface species, COMSOL, **published** |
| flow→chemistry coupling | agonist **concentration** threshold (`f_act >= 1`) + smooth shear activation `k_spa`; **no low-shear stagnation gate, no shear-gradient deposition** | `sr < lss` **flow** threshold carries ~80% of deposition |
| geometry | 101 2D cross-sections, all sidewall ICA aneurysms (InTrA bulge ∩ 4 mm toroid), one family | 40 vessels, three classes (baseline / stenosis / aneurysm) |
| split | 80 train / 11 val / 10 test, held-out geometries | geometry-stratified 5-fold, every vessel held out once, + sealed n=4 |
| learned fields | `u`, `C_up`, `C_ap`, `phi_DBP` — **including the velocity**, autoregressive rollout | clot field only; **flow is solved** |
| architecture | Transformer-GNN (encoder–processor–decoder, training noise), no physics conditioning | physics-conditioned anisotropic MPNN + ODE backbone + temporal head |
| headline metric | IoU of the thrombus region **inside the aneurysm sac** = 0.964; ALL-RMSE, RRMSE | per-node severity + AUC-PR, wall and lumen separately |
| speed | CFD 36 min (2D) → inference 33–100 s: **~20–65x** | COMSOL ~48 h → 58.9 s: **2,934x** ⟨timing.speedup⟩ |
| physics/learning ablation | **none** | §5, four-point axis, per domain |
| injury | random reactive surface, 2 boundary edges of the bulge, **randomized per run**; no no-injury control | wound BC; **matched A/B counterfactual**, same geometry with and without |
| flow regime | **pulsatile**, physiological waveform, 6 cardiac cycles, + OOD test on 3 modified inflows | steady, single operating point Re = 450 |
| geometry provenance | **patient-derived** (InTrA) | parametric-synthetic (`comsol0NN`) |
| architecture baselines | **MGN, BSMS-GNN, Transolver++** at matched compute | from-scratch naive GNN only |
| inputs at inference | all four fields at `t` and `t-dt` plus inlet `u` at `t+dt` — "the same input information as the CFD solver at each time step" | mesh + own t=0 Carreau solve; **no GT state, ever** |
| stated limitation | "reliance on **absolute node coordinates** as input features, which may hinder generalization to different geometries" | — |

**The structural difference is the last row, not the geometry count.** They are a **solver
accelerator** — seeded from a real CFD state after the ramp and first cardiac cycle, time-stepping
forward, needing Sanchez-Gonzalez training noise to keep the rollout stable. We are a
**predictor** — mesh in, full-horizon clot field out, no ground-truth state at any time. Their
speedup is measured against *continuing* a simulation already running; ours against not needing
one. Different products. This is PAPER.md §8's deployability argument, and it is much sharper
stated this way than as an absence in the literature.

### What this kills

**PAPER.md §2's three "facts" can no longer be stated in the present tense.** They were true of
the review's screening window; they are not true of the literature as of 2026-09.

1. *"No study learns the clot field."* → **must be rewritten.** This one does.
2. *"No graph or mesh architecture appears anywhere."* → still true **as a statement about the
   review's 11 studies**, and only that. Never write it as a claim about the field.
3. *"The review names this paper as its own future direction."* → survives, and is now
   *stronger*: an independent group moved in exactly that direction within nine months. Frame
   the review as evidence the direction was open, not as evidence we are alone in it.

**Do not write "first field-level surrogate of a thrombosis model" anywhere.** That sentence is
now falsifiable by a paper in a journal on our own venue list, and a reviewer who knows the
field will make it the whole review. PUBLICATION_NOTES §7.3(1) already ranked "which model,
exactly?" the most damaging unanswered question; this replaces it at the top.

### What it does not touch — and why the paper gets *better* from here

Nine things survive, and the first is the important one:

1. **Their success is our mechanism's best evidence.** Their model has no flow threshold gating
   deposition — the coupling from flow into chemistry is smooth (advection plus a smooth
   shear-activation rate), and they learn the velocity field autoregressively and it works. Our
   model's coupling *is* a threshold on `sr`, and learning the flow costs −0.1075 ⟨rgp.off_delta⟩
   off-wall. **Two papers, two couplings, opposite answers on whether flow can be learned** —
   that is the gated-coupling claim (§3, PAPER.md §7.4) demonstrated across labs rather than
   asserted from one. Cite them as *support*, prominently, in §7. This is the single
   highest-value move available from this pass.
2. **Nobody has measured the division of labour.** They run one ablation — multitask vs
   per-field heads — and none on physics-versus-learning. PAPER.md §5's four-point axis, and
   especially the three-door decomposition (readout / architecture / conditioning), is
   untouched and remains the paper's spine. §6.4 below confirms nothing else in the wider
   SciML literature has it either.
3. **No ground-truth flow at inference.** Their rollout is seeded from CFD frames and predicts
   velocity as a learned field; ours solves the flow on the user's own mesh. The deployability
   argument (PAPER.md §8) stands, though it must now be stated as a *difference*, not as an
   absence in the literature.
4. **The published ground truth.** Theirs is an in-house CIMLIB-CFD model; ours is
   Cardillo–Barakat 2025, peer-reviewed and external. Keep leaning on this — it is the framing
   move §0 identified and it now also separates us from the nearest neighbour.
5. **Shear-gradient-dependent deposition.** The distinguishing feature of our ground-truth model
   is absent from theirs. Ours is the harder coupling, and we say why.
6. **Geometry span, not geometry count.** Their 101 shapes are one topology family
   (sidewall ICA bulges, one 4 mm parent diameter, one inflow waveform); ours are 40 across
   three classes with a measured, honest per-class breakdown including a 2.3-SEM stenosis
   deficit. **Do not fight them on n** (§6.5) — fight on span, and on their own stated
   absolute-coordinate limitation.
7. **The temporal question.** They roll out in time but never ask whether the ODE clock is
   needed; PAPER.md §5.3 does, and answers it.
8. **The refusal gate.** They have none. Ours is narrower than it looks (§6.6) but it is real.
   Likewise the **matched A/B wound counterfactual**: they randomise the injury site across runs
   (a generalization axis we lack) but never run the same geometry without it, so the
   difference-scored comparison stays ours.
9. **Speedup.** 2,934x against 48 h, versus ~20–65x against 36 min — because the ground truth
   is a fundamentally more expensive model. That is a fair comparison to draw and it favours us.

### Where they are ahead — name these, do not argue with them

Three, and none is answerable by framing. **State them in the limitations rather than letting a
reviewer find them:**

1. **Pulsatile physiological flow**, plus an out-of-distribution inflow stress test. We are steady
   at one Re. This is PAPER.md §9's first limitation and it is their strength.
2. **Patient-derived geometry** (InTrA). Ours are parametric-synthetic — the naming caveat in
   PAPER.md §9 is the same fact.
3. **Published-architecture baselines** at matched compute. §2 of this document called adding one
   "much stronger, materially more work" and deferred it; they did it. Our `A_naive` is a
   from-scratch control, not a published architecture, and the manuscript must not imply otherwise.

**Honest one-liner for internal use:** they built the better-engineered emulator of an easier
problem; we built a colder-start predictor of a harder one and measured why it has to be built
that way. Neither displaces the other.

### Venue consequence

**CBM is now a worse first choice, not a better one.** It has just published the near-neighbour,
so an editor sees overlap before contribution, and the obvious reviewer pool is the Hachem
group. Recommended order, revising §7 of PUBLICATION_PLAN:

1. **Biomech Model Mechanobiol** — home of the ground-truth model; the "surrogate of a published
   model from this lineage" framing lands hardest there.
2. **Annals of Biomedical Engineering** — receptive to measured-methodology papers, V&V-literate.
3. **Computers in Biology and Medicine** — still viable *if* Pelissier et al. is cited in the
   introduction, positioned as complementary, and the gap statement is the revised one. Do not
   submit there with the old §2 text.

## 6.2 Everything else checked, and nothing else moved

| checked | finding | verdict |
|---|---|---|
| [arXiv 2608.08165](https://arxiv.org/abs/2608.08165), latent neural DEs — flagged in §1 as "nearest competitor, read in full" | **Read. Not a competitor.** Predicts **scalar** clot-size trajectories; infers a tissue-factor parameter from four biochemical parameters plus sparse early clot-size observations. No spatial field, no geometry. | close the §4 reading-queue item |
| [arXiv 2512.09013](https://arxiv.org/abs/2512.09013), graph deep learning for IA (Garnier, Jeken-Rico, … Pelissier, Meliga, Hachem) | **Hemodynamics only** — velocity, WSS, OSI. No thrombosis. Generalizes across unseen geometries and inflows. | §2 established-method cite, unchanged |
| [npj Digit Med, physics-constrained GNN for IA hemodynamics](https://www.nature.com/articles/s41746-026-02404-z) | full 3D time-resolved hemodynamics, physics-constrained | reinforces §2's NO-GO on RGP-DEQ as an architecture paper |
| DeepONet platelet deformation ([Bioengineering 12(9):958](https://doi.org/10.3390/bioengineering12090958)) | sub-cellular, single platelet | complementary, as recorded |
| a newer review superseding Al Bannoud | none found | gap statement's anchor stands |

## 6.3 The stage-share argument has precedent — cite it rather than claim it

PAPER.md §7.1 leads with the 7.9% ⟨timing.flow_ceiling⟩ ceiling. The reasoning pattern
("replacing one pipeline stage can save at most that stage's share of the runtime") is **already in
print** for scientific-ML surrogates — e.g. the GPU-native DNN surrogate for kinetic
Fokker–Planck flows ([arXiv 2606.15622](https://arxiv.org/html/2606.15622)), which uses exactly
this bound to argue that further accelerating one component has diminishing return unless the
rest is optimised too.

**This is good news, and the action is one sentence, not a retreat.** Cite it, and claim the
thing that is actually ours: *the ceiling is computed and then paid against a measured
downstream accuracy cost* (−0.1185 off-wall, p=0.0005), which turns a scheduling argument into
a design decision. Presenting that bound as a novel observation is the kind of avoidable
overclaim that costs a reviewer.

## 6.4 Physics-vs-learning division of labour — still nothing comparable in print

Searched the hybrid / physics-informed / gray-box modelling literature for a paper that
**measures** which parts of a coupled pipeline are better solved and which better learned.
Found: surveys of physics-knowledge integration, PINN-vs-CNN-vs-hybrid ablations, hybrid
mechanistic–ML reviews in biosciences, modular physics-encoded layers. **Found no paper that
partitions its own domain and reports opposite physics/learning verdicts per partition.**

**PAPER.md §5 is the contribution.** Lead the abstract with it, not with the surrogate. That was
already the §5 go/no-go verdict; this pass confirms it against the wider literature rather than
just against the thrombosis literature, and it is now also the claim the scoop cannot touch.

## 6.5 Cohort size — the objection this project has not written down

Contemporaneous mesh-surrogate papers train on **10²–10³ geometries**: 101 here (Pelissier),
105 patient-derived aneurysms in a released benchmark, 1,500 synthetic coronary bifurcations,
4,200 single-vessel geometries with paired OpenFOAM solutions
([arXiv 2501.09046](https://arxiv.org/abs/2501.09046),
[arXiv 2605.27578](https://arxiv.org/abs/2605.27578)). We report **n=27 scored vessels**.

PAPER.md §9 states n as scope; it does not state *why*, and a reviewer will read the bare number
against the norm above. **The answer is the ground truth's cost and it should be arithmetic, in
the limitations:** those cohorts are steady or single-physics CFD; ours is ~48 h of coupled
12-species COMSOL per vessel, so 40 vessels is ~80 machine-days of ground truth. Sample size
here is bounded by the simulator, not by effort — and it is the same fact that makes the
surrogate worth building. Say it in one sentence with the numbers and the objection converts
into motivation.

## 6.6 The pre-flight gate is not novel as a concept — narrow the claim

OOD detection and reject-option deployment for surrogates is a mature field (SmOOD for aircraft
surrogates, [arXiv 2209.03438](https://arxiv.org/pdf/2209.03438); the OOD-detection surveys;
risk-control framings in clinical ML). PAPER.md §8 and PUBLICATION_NOTES §7.9 call the gate "a
paper contribution" without qualifying against that body.

**The defensible narrowing:** ours is a refusal criterion computed from the **consumer's own
decision statistic** — the wall gate's firing set — rather than from input-space distance or
predictive variance, and it needs no reference field, so it runs on a new vessel. That is the
novel part, it is one clause, and stating it costs nothing. Claiming "a refusal gate" flat is
the version that gets a citation thrown at it.

## 6.7 Actions, in priority order

1. **Rewrite PAPER.md §2** around the revised gap statement (§6.1). Remove every present-tense
   "nobody has", add Pelissier et al. to §3's ground-truth comparison and to §11's evidence map.
2. **Add Pelissier et al. to PAPER.md §7 as cross-lab support** for the gated-coupling claim
   (§6.1 item 1). Highest value per word in this list.
3. **Promote §5 to the abstract's first sentence.** The division of labour is the claim no
   contemporaneous paper contests.
4. **Add the cohort-size sentence** to PAPER.md §9 with the ground-truth cost arithmetic (§6.5).
5. **Narrow the pre-flight claim** (§6.6) and **cite the stage-share precedent** (§6.3). One
   sentence each.
6. **Revise the venue order** in PUBLICATION_PLAN §7 (§6.1).
7. **Release code and the vessel cohort.** The norm in this literature is now a released
   dataset, and it is also the whole answer to "will anyone use the tool". Nothing else on this
   list moves adoption as much.

## 6.8 Citations to add

* **Pelissier, Meliga & Hachem 2026**, Comput Biol Med 208:111649 — the near-neighbour. Cite in
  the introduction, in §3, and in §7 as cross-lab support.
* **Garnier, Jeken-Rico, Lannelongue, Pelissier, Meliga, Hachem**, arXiv 2512.09013 — IA
  hemodynamics GNN, established method.
* **npj Digital Medicine 2026**, physics-constrained GNN for IA hemodynamics — established method.
* **arXiv 2606.15622** — the stage-share bound on what a SciML surrogate can save (§6.3).
* **arXiv 2209.03438 (SmOOD)** and an OOD-detection survey — the body the pre-flight gate must
  be narrowed against (§6.6).
* **arXiv 2501.09046 / 2605.27578** — the released-benchmark norm, for the cohort-size
  paragraph (§6.5).
* **Cardillo & Barakat 2025** page range confirmed: BMMB **24**(5):1465–1484; open access at
  [PMC12454609](https://pmc.ncbi.nlm.nih.gov/articles/PMC12454609/).

---

# 7. Third pass — 2026-09-11, verification sweep

Run against the restructured story (STORY.md (local only)). **Purpose was verification, not
discovery**: confirm that every load-bearing citation resolves and that nothing new sits in the
intersection the Phase 0 triage missed. Searched PubMed, arXiv, Springer, Nature, and the
ML×thrombosis / mesh-GNN-hemodynamics intersection explicitly.

## 7.1 Everything load-bearing verified

| citation | status |
|---|---|
| **Cardillo & Barakat 2025**, *Biomech Model Mechanobiol* 24:1465–1484 | Confirmed. Open access. 2D COMSOL 5.6 platelet-plug model, shear-gradient activation — matches what Leg 0 says is being surrogated, exactly |
| **Pelissier, Meliga & Hachem 2026**, *Comput Biol Med* 208:111649 | Confirmed. Published May 2026, e-pub March 2026. Transformer-GNN, thrombosis, patient-derived IA geometries, 105-geometry benchmark released. The no-priority-claim rule stands |
| **Al Bannoud et al. 2026**, *J Thromb Thrombolysis* | Confirmed, but **the title we were paraphrasing is wrong**. Actual title: *"Artificial intelligence in computational modeling of thrombosis: Bridging mechanistic insights and clinical translation."* Fix any citation that paraphrases it — a reviewer will look it up |
| **arXiv 2606.15622** (stage-share precedent) | Confirmed. Roohi, GPU-native DNN surrogate replacing a cubic-FP closure inside the particle loop. Correctly characterised in Leg 2b |

## 7.2 Jeken-Rico et al. — promote it from "established method" to a Leg 2 citation

**Jeken-Rico et al., "Physics constrained graph neural network for real time prediction of
intracranial aneurysm hemodynamics," *npj Digital Medicine* 9:212 (2026)**
(DOI 10.1038/s41746-026-02404-z). Same lab cluster as Pelissier (CEMEF / Mines Paris).

§6.8 already had this listed as established method. **That undersells it, and the undersell is a
review risk.** It is the closest published analog to our own flow arm: a physics-constrained GNN,
near-real-time, generalizing to unseen patient-specific geometry **with no fine-tuning**, with a
released 105-geometry benchmark. A reviewer who knows this literature knows this paper.

It belongs in Leg 2 for two reasons, and both help us:

1. **It is evidence that a physics-constrained learned flow surrogate can be built well and
   generalize** — which is Leg 2's own position ("we are not claiming learned flow surrogates are
   bad"). Citing it generously strengthens the flow-share framing instead of exposing it.
2. **It predicts hemodynamics only, with no thresholded clot consumer downstream** — so it does
   not test the gated-coupling mechanism at all. Name that gap ourselves rather than letting a
   reviewer name it. The sentence is roughly: *a physics-constrained flow surrogate generalizes
   well to unseen vascular geometry when nothing downstream applies a hard threshold to its
   output; we measure what changes when something does.*

**This makes three papers from one lab cluster in our immediate neighbourhood** (Pelissier 2026;
Jeken-Rico 2026; Garnier et al., arXiv 2512.09013). Treat them as the reference group the work
will be read against, and cite all three.

## 7.3 Nothing new threatens the story

No paper found that runs a physics-versus-learning ablation of the kind Leg 1 reports, and
nothing reporting a *within-component* reversal by spatial domain. §6.4's conclusion holds:
that remains the contribution with no contemporaneous competitor.

## 7.4 §6.7's action list, remapped

§6.7 was written against `PAPER.md` sections that no longer exist. Current mapping:

| old action | now |
|---|---|
| "Rewrite PAPER.md §2 around the revised gap" | done — STORY.md (local only) "What we must never claim" |
| "Add Pelissier to §7 as cross-lab support" | done — STORY.md (local only) Leg 2, "the cross-lab test" |
| "Promote §5 to the abstract's first sentence" | done — Leg 1 is the spine, under Leg 0's framing |
| "Add the cohort-size sentence" | **open** — belongs in Leg 0's limitations |
| "Narrow the pre-flight claim; cite the stage-share precedent" | precedent done (Leg 2b); pre-flight narrowed to the false-alarm sentence, and its generator is a debt in [EXPERIMENTS.md](EXPERIMENTS.md) E6 |
| "Revise the venue order" | **open** — venue analysis has no successor doc; it is still in the local-only `PUBLICATION_PLAN.md` §7 |
| "Release code and the cohort" | **open, and still the highest-leverage item for adoption** |
