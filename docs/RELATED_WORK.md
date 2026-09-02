# Related work — Phase 0 novelty triage

Opened 2026-09-01. Output of Phase 0 in [PUBLICATION_PLAN.md](PUBLICATION_PLAN.md) §4.

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
list. The [WOUND_PROGRESS](WOUND_PROGRESS.md) finding — injury is the same surface law with the
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

**Three facts that define our contribution:**

1. **Nobody learns the clot field.** Every eligible study predicts a scalar, a class label, or a
   non-clot field (shear). The one called a "surrogate model" (Bouchnita 2023b) is a binary
   classifier over 7,675 simulations, taking scalar thrombin parameters and injury size as
   input. Coagulo-Net solves and infers ODE parameters; it does not generalize a rollout to
   unseen geometry.
2. **No graph or mesh architecture appears anywhere.** ML models across all 11: ANN, NARX, SVM,
   KNN, DT, RF, NB, LR, XGBoost, CatBoost, EBM, DNN, PINN. Zero GNNs. A mesh-agnostic graph
   surrogate is first-of-kind in this intersection.
3. **The review asks for our paper by name.** It repeatedly names as future directions:
   extension "to three-dimensional vascular geometries", "patient-specific anatomical
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
| Hybrid ANN–ODE VTE risk; DNN thrombogenesis classifiers; LAA thrombus prediction | Patient-level *risk classification* | Different problem entirely |

**Verdict — the paper: GO.** The gap is real and is now documented by a systematic review rather
than by our own search. Remaining condition: the multi-Re training cohort, which the one-paper
decision promotes to the critical path ([PUBLICATION_PLAN](PUBLICATION_PLAN.md) §5).

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

**Verdict — Paper 1: GO, and it is the strongest thing in the project.** Positioning: not "we
discovered L2 is inadequate" (argued before, and claiming it invites a priority fight you would
lose), but *"we measure how inadequate, in a pipeline where the coupling is a gate rather than a
smooth map, and we show the failure is discontinuous, predictable from the consumer's structure,
and not repairable post-hoc."*

Title direction, replacing the plan's working title:
**"Gated couplings break surrogate error budgets: task-relevant accuracy requirements for a
thrombosis pipeline."**

---

## 4. Reading queue, in priority order

1. **J Thromb Thrombolysis 2025 review** (DOI 10.1007/s11239-025-03222-y) — institutional access. Confirms or kills the Paper 2 gap statement.
2. **Cardillo & Barakat 2025**, full text, against `dmodel.xml` — species-level confirmation, and settle whether the **wound** law is published.
3. **arXiv 2608.08165** (latent neural DEs) — nearest competitor in framing; know exactly how our claim differs.
4. **arXiv 2502.06753** (unified surrogate framework) — the paper Paper 1 positions against most directly.
5. **Sci Rep 2026 PI-GNN, and arXiv 2212.05023** — the two whose WSS numbers create the §2 risk. Read carefully: *which* wall quantity, on what meshes, and is it comparable to our `sr` / `dsrx` at all? There is a real chance the metrics are not commensurable, which would defuse the risk outright.

---

## 5. Go / no-go summary

Updated for the one-paper decision ([PUBLICATION_PLAN](PUBLICATION_PLAN.md) rev 2).

| Candidate | Verdict | Condition |
|---|---|---|
| **The paper** — mesh-generalizing clot-field surrogate | **GO** | Multi-Re cohort (now critical path) |
| Flow-surrogate requirements | GO **as §4**, not standalone | Reframe as the gated-coupling claim, per §2 |
| RGP-DEQ | GO **as §4's ablation arm** | Never as an architecture contribution |
| Open-loop oracle bound | GO as §5 | Two cheap runs (PUBLICATION_NOTES §4) |
| Wound result | GO as §6 — **blocked** | Provenance question to Giulia, §0 above |
| RGP-DEQ as architecture paper | **NO-GO** | Prior art: FNO-DEQ + a crowded mesh-GNN field |
