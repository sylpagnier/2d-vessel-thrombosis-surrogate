# Figures — what to draw, and what each one has to argue

**Thirteen main-text items, organised by leg — and no supplementary figures.** A figure earns its place by carrying an argument a
sentence cannot. Generator scripts are named **semantically** and deliberately do not encode
figure numbers — review reorders figures, and renaming generators each time churns the pipeline.

Conventions: [../VIZ_STANDARD.md](../VIZ_STANDARD.md). Profile/output wiring:
`scripts/publication/config.py`, whose `paper_map` is the single place figure→section is tied
together.

**In-figure text rule (review, 2026-09-13).** Figures carry panel labels, axis labels and legends only — no titles, subtitles, callouts or explanatory notes. Everything a figure argues goes in its caption, below.

**Status legend.** `BUILT` — generator exists and runs. `EXTEND` — generator exists but does not
yet make this argument. `NEW` — nothing draws this yet. `BLOCKED` — waiting on a run.

---

## Leg 0 — the tool works, on intact and injured vessels

### 0.1 What we are replacing  ·  `BUILT`

**Argues:** the ground truth is a real, published, expensive coupled simulation — 9 bulk species,
3 surface species, Carreau rheology, a shear-gradient deposition gate, a viscosity step at
gelation, ~48 h per vessel. The object of study is a *peer-reviewed model*, not an anatomy claim,
which is what makes the synthetic-geometry objection largely dissolve.

**Draw:** the COMSOL reaction network and its coupling structure, with the wall-clock cost on it.
Best read as a schematic, not a screenshot — a reader needs the species graph and the two
couplings (flow→chemistry through the gate, chemistry→flow through `mu1(Mat)`), not a GUI tree.

**Generator:** `plot_comsol_ground_truth.py` → `figures/comsol_ground_truth.{pdf,png}`. Drawn
rather than reproduced from the published paper: their figure explains *their* model, this one has
to carry the two couplings, their discontinuity, the wound substitution and the wall-clock cost.
A redrawn schematic also cannot go stale against a permissions question at submission.

### 0.2 The shipped architecture, wound and no-wound  ·  `BUILT`

**Argues:** one artifact handles both. The wound path is a boundary-condition branch on the
physics backbone — `srf1` (gated) on healthy wall, `srf2` (ungated) on wound — and it is
**bit-identical to the base model when no wound mask is present**. The learned half is untouched
and still owns everything off-wall.

**Draw:** mesh → FEM t=0 → features → GNN ensemble → temporal head → readout, with the wound
branch shown as a *switch on one term*, not a parallel pipeline. The no-op property is the point;
make it visible.

**Box text, checked against the shipped artifact (2026-09-13):** (1) local FEM, steady Carreau
Navier–Stokes, solved once (~5 s); (2) shear gate (low-shear + separation) → surface deposition
ODE + advective transport into the lumen; (2b) ungated `srf2` + 2 fitted rates; (3) **4**
message-passing layers with separate upstream / downstream aggregation, edges carrying the t=0
flow, output = physics + residual, 9-member ensemble; (4) 3 or 5 occlusion-feedback rounds;
(5) gradient-boosted onset + lag heads anchored to the ODE crossing time. **The diagram used to say
6 layers** — that is `ClotGNN`'s class default; the locked checkpoints have 4 (hidden 64), per
`data/architecture_table.md`.

**Generator:** `plot_biochem_architecture.py` (on the shared `arch_diagram.py` primitives) →
`figures/biochem_architecture.png`. The wound branch hangs off box 2 with "no wound mask ⇒
bit-identical output" on it, so it reads as a switch on one term rather than a parallel pipeline.

### 0.3 Why BATC — three real vessels, scored three ways  ·  `REBUILT 2026-09-13`

**Argues, qualitatively:** look at the prediction, then at what each score says about it. One
column per out-of-fold vessel at final time: **model** (row 1), **ground truth** (row 2), and
**strict F1 vs BATC** (row 3). Above each column, a small whole-vessel strip with the zoom window
boxed, so the close-ups read as a window onto the right part of the vessel; (b) and (c) are also
zoomed out 2.5×. The model panel carries its BATC.

| vessel, domain | what it shows | strict F1 | BATC |
|---|---|---|---|
| comsol041, off-wall | every error is a near miss | 0.68 | 0.96 |
| comsol010, off-wall | a 12-node clot — the grace matters | 0.38 | 0.79 |
| comsol028, wall | a real miss — BATC does not rescue it | 0.57 | 0.66 |

`data/batc_examples.json` also keeps the no-grace score (0.96 / 0.67 / 0.56) for the text.

**Generator:** `plot_batc.py` → `figures/batc.{pdf,png}` + `data/batc_examples.json`, on the
shared scorers in `example_vessels.py`. Replaces the synthetic response-function figure on review.

**Still unbacked, unchanged:** the +0.192 BATC-vs-BATC₀ decomposition on the deploy cohort
([EXPERIMENTS.md](EXPERIMENTS.md) E2b).

### 0.4 An application — the study 48 h per geometry makes unaffordable  ·  `REBUILT 2026-09-15`

**Argues what the speedup is FOR.** A number is not a contribution; the study it enables is. Sweep
a pathology axis finely on one vessel and read off where the response changes character.

**Strength** is defined as the percentage REDUCTION (stenosis) or INCREASE (aneurysm) of the vessel
width at the pathology's centre, relative to the inlet, measured back from each point's geometry
(agrees with the requested value to 0.5 pp). The previous version plotted "aneurysm factor", a
wall-offset multiplier where 1.0 meant the width tripled. 80 points (38 stenosis: 2.5% steps with 1.25% midpoints at the
transitions; 42 aneurysm: 10% steps, 2.5% across 40–100%, 0.625% across the onset — refined
2026-09-16), straight 12 mm vessel, trained pathology taper,
all built with the 2026-09-14 aligned graph builder (docs/DEPLOY_ALIGNMENT.md).

**One row: total clot mass** (clotted nodes, % of the vessel). Wall clot and peak occlusion were
dropped on review (2026-09-16). Peak occlusion is the depth of the single deepest clot node in
whole wall hops, so 1–9 nodes one layer deeper at the aneurysm apex, late in the run, read as an
8.6–9.1% "occlusion" band at 82–90% widening — a metric artefact, and clot inside a dome never
narrows the parent lumen anyway.

* **(a) stenosis** — no clot up to a narrowing of 21.2 ⟨apps.sten_last_zero⟩ percent; clot appears at
  22.4%; at 50% narrowing it reaches 0.71 ⟨apps.sten_50_mass⟩ percent of the vessel, and it keeps
  accelerating to 2.51 ⟨apps.sten_77_mass⟩ percent at the trained maximum (77%).
* **(b) aneurysm** — no clot up to a widening of 46.2 ⟨apps.aneu_last_zero⟩ percent; at 46.8% it steps
  to 0.71 ⟨apps.aneu_step_mass⟩ percent (0.6 pp sampling brackets it), rising at the trained maximum
  (+100%) to 1.10 ⟨apps.aneu_100_mass⟩ percent. **The rise is not monotone:** mass reaches 0.86% at 67.4%, drops to
  0.76% at 69.9% and stays there to 75%, then climbs again — consecutive 2.5% points agree, and
  70/80/90% re-run bit-identically, so it is the model's response, not a bad run. Beyond the
  trained maximum mass keeps rising slowly (1.58% at +200%), so aneurysm clot is still a thin
  mural layer: the two pathologies are not one severity axis.
* **Cost, for the caption** — 80 geometries, 39.3 min ⟨apps.cost_min⟩ of rollout here against ~160
  days of COMSOL at ~48 h each. (Was 95.8 min before 2026-09-18, when the sweep queried every
  index of its own axis instead of the 11 the temporal head was fitted on.)

The shaded band marks strengths beyond the most severe pathology in the training cohort (77%
narrowing, +99% widening): extrapolation.

**Generator:** `plot_applications.py` → `figures/applications.{pdf,png}`. The joint
bendiness × stenosis surface is `plot_critical_point.py`, its companion.

> **Two caveats for the caption (removed from the figure itself on review).** There is **no COMSOL ground truth at any
> swept point** — these are the surrogate's own predictions, and the evidence they can be trusted
> is Leg 0's held-out cohort, not anything in this figure. And the connecting lines are reading
> guides, not fits: the sampling cannot localise the transition more finely than the runs
> themselves. Never write "the critical occlusion is 0.5–0.75" as a validated physical finding.

### 0.5 Tracked over the whole horizon — intact and injured, paired  ·  `REBUILT 2026-09-13`

**Argues:** the rollout is right throughout, not merely at the end, on both vessel classes.

**Split into three figures on review (2026-09-13)** — one panel set was too compressed to read.

* **0.5a — `wound_temporal`:** **total clot mass** (clotted nodes, % of the vessel — the app's
  `vessel_clot_pct`), model vs ground truth, over each vessel's horizon, every frame (replaced the
  BATC-over-time panels on review, 2026-09-15). (a) 27 out-of-fold intact vessels, median and IQR
  of each; (b)–(g) the 6 injured vessels, one panel each. Intact: the model median tracks GT
  throughout and ends slightly high (final median 0.82% vs 0.74%). Injured: w001/w002/w004/w005
  follow GT, ending 0.1–0.6 pp high; **w003 and w006 lag onset** — GT clot appears from the first
  frames while the model predicts almost none until about 40% of the horizon — and finish low
  (1.62% vs 2.66%, 0.64% vs 1.45%). Data: `data/clot_mass_series.json`
  (`example_vessels.clot_mass_series`).
* **0.5b — `wound_example_comsol003`** (long, gradual) and **0.5c — `wound_example_comsol006`**
  (the hardest of six): model beside ground truth at clot onset, two intermediate frames and the
  **final frame** (t = 128 and t = 41 — the wound runs are shorter than the intact horizon). The
  **wound is drawn exactly as the app draws it** (`customer_predict_web.py`): the injured boundary
  as wall, a faint band across the vessel and two dashed cut lines. Every model panel carries its
  BATC for the wound region, wound lumen and healthy wall at that frame.

**Snapshot rule:** any vessel snapshot in the paper carries its BATC in the model panel (0.3, 0.5b/c,
0.6b). The helpers are `example_vessels.score_tag` and `mark_wound`.

> **The pairing fix.** The old figure drew the injured half alone and scored it with **BATC₀**
> while every table reports BATC. Both fixed. **The protocols still differ** — intact is strict out-of-fold,
> injured is leave-one-vessel-out with a GNN that never saw a wound — and the caption must say which.

**Generator:** `plot_wound_temporal.py` → `figures/wound_temporal`, `wound_example_comsol003`, `wound_example_comsol006` (`{pdf,png}`), from
`data/oof_batc_series.json` and `data/wound_batc_series.json` (`example_vessels.py`).

### 0.6b A bad final frame is never judged  ·  `BUILT 2026-09-13`

**Argues:** mid-run errors are visible to the frames after them and mostly recover; an error at
the last frame has nothing after it, so we cannot tell whether it would have settled.

* **(a)/(b)** BATC over time, wall and off-wall, for comsol014 (orange) and the recovering
  counter-example comsol037 (blue, added 2026-09-16): its wall stays ≥ 0.89 throughout, its
  off-wall holds 1.00 to t = 140, drops to 0.00 at t = 160 and recovers to 0.86 / 0.93. The other
  out-of-fold vessels' faint lines were removed on review, 2026-09-15. The cohort counts stay caption text:
  wall, 8 vessels dip below 0.6 mid-run and all 8 end ≥ 0.9; off-wall, 13 dip, 3 recover, 5 end
  at their lowest frame.
* **comsol014** wall collapses to 0.20 at t=40 and is back at 0.94 one frame later; its off-wall
  domain never holds GT clot, yet predicted nodes grow 1 → 76 through the final frame.
* **(c)** comsol014 at t = 40, 60 and 200, **model** above **ground truth**.
* **(d)/(e) counter-examples (added 2026-09-16):** a late error that *has* recovered by the final
  frame, zoomed onto the off-wall clot. **comsol037** off-wall misses all 10 GT nodes at t = 160
  (BATC 0.00) and is at 0.93 by t = 200; **comsol040** is at 0.55 at t = 160 and 0.97 at t = 200.
  They are three of the out-of-fold vessels whose off-wall BATC drops below 0.6 at t ≥ 120 and ends
  ≥ 0.9 (comsol035 is the third; none on the wall) — `plot_error_trajectories.late_recoveries`.
  The panel is there so comsol014 is not read as the rule: a late error can settle, we just cannot
  see whether a *final-frame* one would.

**Generator:** `plot_error_trajectories.py` → `figures/error_trajectories.{pdf,png}` (the "does a
mid-run error recover or compound" figure, moved here from the supplement and re-scored with BATC).

### 0.7 The packaged app  ·  `BUILT (asset exists)`

**Argues:** a surrogate nobody can run is a claim, not a tool. Self-contained Windows bundle,
embeddable Python, CPU-only, ~11 MB of checkpoints, double-click launcher, no research
environment and no GPU.

**Asset:** `docs/assets/customer_predict_demo.png` already exists. Check it shows a *prediction*,
not an empty form, before using it.

---

## Leg 1 — why the tool is built this way

### 1.0 The shipped architecture, as a methods table  ·  `BUILT`

**Argues nothing, and is required anyway.** Every ML reviewer expects width, depth, parameter
count, ensemble composition, optimiser and loss weights, and none of it was in these docs.

**Generator:** `generate_architecture_table.py` → `data/architecture_table.{json,md}`. Read from
the locked artifact (the member checkpoints' `cfg` and `state_dict`, the manifest, `temporal.pkl`)
and **it fails if the nine members disagree** on anything but `rounds`/`off_mult`, so it cannot
drift from what ships. Headline: hidden 64, 4 layers, **151,874 params per member, 9 members
(1.37 M total)**, AdamW + OneCycle at 3e-3, 80 epochs, gradient-boosted temporal head.

> **One asymmetry the methods must state.** The shipped ensemble mixes three configs (`v5a`
> rounds 3; `v5b` rounds 5; `v5c` off_mult 2.5), but `run_phase9_cv.py` — and therefore every
> ablation arm — trains **one** config (rounds 3, off_mult 1) × 3 seeds. CV and ablation numbers
> describe the `v5a` recipe, not the 9-member mix.

### 1.1 Architectures, end to end  ·  `BUILT`

**One figure carrying the whole leg.** Argues both halves at once: what to solve vs. learn, *and*
which architecture, measured on identical folds.

**Draw:** grouped bars, arms on one axis, wall and off-wall panels side by side, **paired CIs on
every bar**:

| arm | what it isolates |
|---|---|
| `A_naive` — from-scratch mesh GNN | geometry alone |
| `A_mgn` — depth-matched MeshGraphNet-*style*, 15 layers | does depth rescue a physics-free model? |
| physics-tuned rule, no learning | the solve-only floor |
| `plain` — our architecture, no physics conditioning | architecture alone |
| `full` — shipped | |

**Two things the figure must not get wrong.** First, the wall null — full-minus-physics-tuned is
+0.0252 at P=0.066 (CI [−0.0084, +0.0533]) — must be stated in the caption with its interval, because the paired-delta
panel that drew it was removed on review (2026-09-13) and the figure is now panels (a) and (b)
only. Second, the caption must say `A_mgn` is a MeshGraphNet-style control, **not** a MeshGraphNet
reimplementation.

**Generator:** `plot_division_of_labour.py` → `figures/division_of_labour.{pdf,png}`. It already
carried all five arms; what changed 2026-09-11 was the framing, not the data — the title came off
the retired "Solve the flow. Solve the wall. Learn the lumen." message (this figure carries Leg 1
only, and a title asserting the flow result claims something the panels do not show), and the
`A_mgn` legend now names it a depth-matched MeshGraphNet-**style control**, not a MeshGraphNet.

---

## Leg 2 — the flow is solved, once, with FEM

### 2.1 How much coupling is needed  ·  `REBUILT 2026-09-14 (+ strict F1)`

**Argues:** a real, GT-free FEM re-solve against the model's own clot changes nothing, at any
coupling frequency — six schedules from one final solve to a re-solve on every new clot node, wall
within ±0.01 and off-wall within ±0.06 of uncoupled, no dose-response.

**Row 2 is strict F1** of the same final-time predictions (recorded by `eval_strict` beside BATC;
the 24 runs were re-scored and reproduce their published BATC exactly). One figure rather than two,
because the question is the same and the reader needs both metrics side by side: an effect that
only moved clot boundaries by a node would hide inside BATC's tolerance and show in strict F1. It
does not: strict F1 finds no gain either, and the two schedules outside their noise band are the
same on both metrics and negative (every 64 steps at the wall, every 5 new clot nodes off-wall).

**Generator:** `plot_coupling_frequency.py` → `figures/coupling_frequency.{pdf,png}` +
`data/coupling_frequency.json`. Healthy runs vs healthy runs, degenerate readouts screened on both
sides; **not seed-paired** (the paired version manufactured an off-wall gain from one bad baseline
run — STORY 2.1i retraction).

### 2.2 Where the time goes  ·  `BUILT 2026-09-13`

**Argues:** the local flow solve is a small share of the runtime — **7% median** per vessel against
**76%** for the rollout — so a learned flow surrogate could save at most that share, and it is not
worth its accuracy cost (FEM gate Jaccard 0.9240 ⟨fem.gate_jac_med⟩ vs RGP-DEQ 0.7582
⟨rgpdeq.gate_jac_med⟩) unless that share grows. Plus the 2,934× comparison against COMSOL.

**Generator:** `plot_timing.py` → `figures/timing_cost.{pdf,png}`: stacked per-vessel stage bars
(n=34), with the flow-share sentence on the panel.

> **Wording rule, from review.** Say "the local flow solve is a low share of total runtime". Do not
> call it an "Amdahl ceiling" or an "Amdahl bound" anywhere in the manuscript — the separate
> ceiling figure (`plot_amdahl_ceiling.py`) is dropped. The 7% on the figure is the median of
> per-vessel shares; STORY's 7.9% is median FEM seconds over median total seconds — say which.

---

## Not in the paper's figure set

**Dropped on review, 2026-09-13.** The figure set is the main-text figures above and nothing else:
no supplementary figures are carried.

| item | why |
|---|---|
| **Oracle closed loop** (`closed_loop_oracle`, old 2.1) | the ground-truth oracle leaks the label (STORY 2.1f); replaced by 2.1 |
| **Flow-share ceiling** (`amdahl_ceiling`, old 2.2a) | the argument is one sentence on 2.2; the separate figure and its name were dropped |
| **Gate identity vs size** (`coupling_gate`, 2.4) | dropped from the figure set; STORY 2.4 carries the argument in prose |
| **All supplementary figures** — feature-conditioning ladder, architecture detail, RGP-DEQ architecture, RGP-DEQ / FEM / GT flow fields, cohort and geometry classes, final-time clot maps, geometry-response sweeps, operating point, onset timing, wound A/B pair | dropped. The error-compounding figure is not dropped — it moved into 0.6b |
| **Where BATC and the eye disagree** (`fig6_failures`, old 0.6, comsol005) | dropped on review 2026-09-16; the case stays one sentence in STORY 0.3 (wall 0.9957, off-wall 0.5479 on a 4-node burden) |
| **Gate seeding mechanism** | the shipped arm has **0/33 empty gates** — the phenomenon does not occur on the path the paper reports |
| **Pre-flight validation table** | the live claim narrowed to one sentence (false-alarm rate 0 in 37), and the generator is gitignored |

**Kept as a table, not a figure:** the geometry generalization table (Table 4), main text.

---

## Build status — 2026-09-13

| fig | status | artifact |
|---|---|---|
| 0.1 what we replace | **BUILT** | `figures/comsol_ground_truth.{pdf,png}` — `plot_comsol_ground_truth.py` |
| 0.2 wound/no-wound architecture | **BUILT** | `figures/biochem_architecture.png` — `plot_biochem_architecture.py` |
| 0.3 why BATC | **REBUILT** | `figures/batc.{pdf,png}` + `data/batc_examples.json` — `plot_batc.py` |
| 0.4 application / sweeps | **REBUILT 2026-09-15** (60 points, strength = % width change at centre) | `figures/applications.{pdf,png}` — `plot_applications.py` |
| 0.5a–c whole-horizon tracking | **REBUILT, split in three** | `figures/wound_temporal`, `wound_example_comsol003`, `wound_example_comsol006` — `plot_wound_temporal.py` |
| 0.6b the unjudged final frame | **REBUILT 2026-09-16** (comsol014 only; + comsol037/040 recoveries) | `figures/error_trajectories.{pdf,png}` — `plot_error_trajectories.py` |
| 0.7 packaged app | **BUILT** | `docs/assets/customer_predict_demo.png` |
| 1.0 architecture table | **BUILT** | `data/architecture_table.{json,md}` — `generate_architecture_table.py` |
| 1.1 architectures end to end | **REBUILT** (panel c removed) | `figures/division_of_labour.{pdf,png}` — `plot_division_of_labour.py` |
| 2.1 coupling frequency | **REBUILT** (+ strict F1 row) | `figures/coupling_frequency.{pdf,png}` — `plot_coupling_frequency.py` |
| 2.2 where the time goes | **REBUILT** | `figures/timing_cost.{pdf,png}` — `plot_timing.py` |

All per-vessel scores on 0.3, 0.5 and 0.6b come from one module, `example_vessels.py`, with
the shipped BATC config.

`config.py`'s `paper_map` matches this list as of 2026-09-16 (13 main, dropped rows with reasons).
