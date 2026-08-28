# Cohort runbook — generate on the COMSOL box, inspect here, then commit

The full repair is in [`RGP_DEQ_REPAIR_PLAN.md`](RGP_DEQ_REPAIR_PLAN.md).  This is the operating
procedure for the step before a retrain: generate a cohort on the COMSOL machine, move it, prove
it is fit to train on, and workshop the recipe before committing.

**Why inspect at all.**  The previous corpus was stale in four independent ways — no
severe-stenosis tail, dead `node_type`, WLS operators that did not match their own graphs, and
a leaked prior block.  Every one was found by *measuring a pack*, none by reading the generator.
Two of them (`node_type`, the stenosis tail) would have survived a regeneration unnoticed.

---

## 1. On the COMSOL machine — generate and solve

Confirm the plan first.  This writes nothing and takes about a second:

```bash
python -m src.data_gen.pipeline_kinematics --batch --rheology carreau -n 250 --mixed-levels --pathology-mix "random:0.72,max_stenosis:0.18,max_aneurysm:0.10" --seed 20260828 --overwrite --anchor-max-new 250 --repair-rounds 2 --dry-run
```

```
--- DRY RUN: nothing will be written ---
  rheology      carreau
  vessels       250
  levels        L0=100, L1=100, L2=50
  pathology     random:0.72,max_stenosis:0.18,max_aneurysm:0.10
  seed          20260828
  mode          OVERWRITE
  mesh          lc=1.00mm x0.75, >=8 elements across the throat
  repair        2 round(s) -- unsolved vessels are re-meshed finer on their own geometry
  mix expands   {'random': 180, 'max_stenosis': 45, 'max_aneurysm': 25}
```

Then run it for real by dropping `--dry-run`:

```bash
python -m src.data_gen.pipeline_kinematics --batch --rheology carreau -n 250 --mixed-levels --pathology-mix "random:0.72,max_stenosis:0.18,max_aneurysm:0.10" --seed 20260828 --overwrite --anchor-max-new 250 --repair-rounds 2 --num-workers 8
```

**One command, not two.**  `--pathology-mix` assigns a mode per vessel — weights as fractions,
or exact counts summing to `-n` — and shuffles the assignment so pathology does not correlate
with the geometry-level schedule.  Random sampling alone under-represents the severe-stenosis
regime that deployment actually fails in, which is why this used to need a second run with a
second seed and a second index range.  `--pathology-mode` still works for a single mode.

**Generation now refuses to write into a populated cohort.**  Say `--overwrite` (replace,
indices restart at 0) or `--append` (continue from the highest index on disk).  This exists
because a 12-vessel smoke test replaced a 370-graph corpus and its meshes: `MeshToGraph.run()`
clears every `*.pt` in its output directory before converting, and `data/` is gitignored, so
there was nothing to restore from.

```
REFUSING TO GENERATE: the target cohort is not empty.
  graphs :     8  in ...processed/graphs_kinematics/carreau
  meshes :     8  in ...raw/kinematics/meshes

Say which you mean:
  --overwrite   replace the cohort (vessel indices restart at 0)
  --append      add to it (indices continue from the highest on disk)
```

Other notes:

* Drop `--seed` for a random cohort, but record whatever you use — a preflight report is only
  interpretable next to its seed.
* `--skip-anchor` builds meshes and graphs without the COMSOL solve: a fast way to confirm the
  flags behave before spending solve time.
* 250 vessels is roughly **500 MB** of graphs now that the dead sparse operators are no longer
  stored.  It would have been ~33 GB.

**Check it there before transferring** (CPU-only, seconds):

```bash
python scripts/preflight_kine_cohort.py --src data/processed/graphs_kinematics/carreau --expect-p1
```

If `severe-stenosis coverage` warns at 0%, the pathology mix did not take and there is nothing
worth transferring — that is precisely the defect the old corpus carried.

---

## 2. Transfer

Packs written by the current builder are already slim: `G_x`/`G_y` are no longer stored, so this
step is close to a no-op for a fresh cohort.  It still matters for anything generated earlier.

```bash
python scripts/slim_kine_packs.py --src data/processed/graphs_kinematics/carreau --out transfer_carreau --verify
```

On an older 4,019-node pack:

```
G_x           64.61 MB   (N, N) sparse
G_y           64.61 MB   (N, N) sparse
everything     2.07 MB
TOTAL        131.30 MB   ->  250 vessels = 33 GB
after slim     2.07 MB   ->  250 vessels = 500 MB
```

`graph_gradient_operators` defaults to MLS and rebuilds from positions + connectivity; the
stored operators are read only under `BIOCHEM_GRAD_OPERATOR=legacy`.

`--verify` also reports whether each pack's stored WLS operator matches its own graph.  Expect
notes on older packs — 3 of 3 sampled did not match (B13).  The tool deliberately does not "fix"
that: `--drop-wls` would force a correct rebuild on load, but that is a numerics decision, not
something a copy tool should make silently.

---

## 3. On this machine — prove it before spending GPU time

```bash
python scripts/preflight_kine_cohort.py --src transfer_carreau --expect-p1
```

Exit code 1 on any FAIL.  Every check is a bug that has already cost a run:

| check | why it exists |
|---|---|
| topology P1 / P2 | training was 0% degree-2 against a 74.5% deploy mesh (§8 A1) |
| prior block is not the CFD solution | the s17 Z2 leak, bit-identical on 43/43 packs (§1a) |
| `width_d2` within training range | 1e4+ means a stale WLS operator (B13) |
| `wall_normal` populated | identically zero for a year; drives the GAT's attention biases |
| `node_type` populated | was a hardcoded `torch.zeros((N, 4))` placeholder in the builder (B24) |
| severe-stenosis coverage | the old corpus had 0% at ratio ≥ 2.0 against deployment's 14% |
| `u_ref` overlaps deployment | BC range; currently fine, checked so it stays fine |
| COMSOL solve rate | 39/250 packs shipped with an all-zero `y` and every check passed (B27) |
| resolution matches deployment | the corpus sat 17% coarser than deploy in `h_nd` (B33) |
| geometry substitutions | how many vessels the repair re-drew (B31) |

**Failed solves now repair themselves**, in two stages: two rounds that re-mesh the *same*
geometry finer, then two that re-draw a **different vessel of the same class and severity**
(rejection-sampled to within 0.85x of the original's stenosis / aneurysm ratio, stamped
`reshaped_from`).  Refinement alone recovered only 2 of 39 on the 2026-08-29 run -- the extreme
tail is close to degenerate, not under-resolved.

**Old note, kept for context.**  `--repair-rounds N` (default 2) re-meshes any vessel
COMSOL could not solve at a finer element size and tries it again, on the *same* geometry — the
wall polylines are re-read from the vessel's own `.json`, so the cohort keeps its designed
pathology mix rather than drifting toward the shapes that solve easily.  The run ends with a
`COHORT HEALTH  <solved>/<total>` block naming anything still unsolved.  Meshes are also sized to
the local lumen now (`mesh_min_elems_across`, default 8), which is what the 39 failures were
asking for: a uniform 1 mm element puts ~5 elements across a 3.7 mm throat.  Cost is +6-8% nodes
on stenosed vessels, zero on open ones.

**A failed solve still writes a pack.**  `mesh_to_graph` emits `is_anchor=False` and a zero
placeholder `y` when COMSOL produced no `.npz`; training uses those as unsupervised
physics-only graphs.  That is fine, but it used to be invisible — the `labels present` check
tests `y is not None`, and a zero tensor is not None.  The 2026-08-28 cohort lost **15.6%** this
way, and the loss is not uniform: it rises monotonically with stenosis ratio (2.9% below 1.5,
40.6% above 3.0), i.e. it eats exactly the tail the cohort is generated to add.  Preflight now
counts the severe-stenosis tail over *solved* vessels and prints the failures worst-stenosis
first so they can be reopened in COMSOL.

---

## 4. Workshop the recipe before committing

```bash
python scripts/calibrate_kine_loss_weights.py --graphs 8
```

It attaches the PDE label floors first, because training does (`KINEMATICS_PDE_FLOOR`, on by
default).  Un-floored, `l_cont` and `l_mom` carry the labels' own near-wall stencil residual —
up to 22 at the training weight on the sharpest vessels — so calibrating without the floor
weights a different objective than the one that runs (§13.1).

Then a short run with the launch config (§12 of the repair plan; PowerShell users set these with
`$env:NAME = "value"` rather than inline):

```bash
SPECIES_PRIOR_SOURCE=analytic KINEMATICS_ELEVATE_P2=1 KINEMATICS_COORD_MODE=centered KINEMATICS_NORMALIZE_SHEAR_GRAD=1 KINEMATICS_LOSS_WEIGHTS=outputs/kine_loss_weights.json KINEMATICS_INCLUDE_PATIENT_ANCHORS=1 KINEMATICS_SELECT_MAX_GRAPHS=6 KINEMATICS_SELECT_PATIENCE=6 python -m src.training.train_kinematics_predictor --epochs 20 --adam-epochs 20 --stage1-end-epoch 0 --stage2-end-epoch 0 --no-prompt
```

Read the run by its one-line-per-validation summary:

```
[kin] ep12   SELECT gateJ=0.281 dsrxR=+0.402 | relL2=0.183 div=2.1e-03 comp=0.394
```

**`gateJ` is first, deliberately.**  It is the only Stage-A metric measured to predict the clot
model's own oracle-F1 (+0.918; `dsrxR` reads **-0.073** within a single flow arm, §10.3).
`relL2` is reported because it is cheap and familiar, not because it should drive a decision.

Knobs for this phase:

* `KINEMATICS_VAL_EVERY` — 2 by default (1 when total epochs ≤ 12).
* `KINEMATICS_SELECT_MAX_GRAPHS` — 6 by default.  Each selection graph is a full 25-iteration
  Anderson solve, so this is the main validation cost.  Raise it for a final selection pass.
* `KINEMATICS_SELECT_PATIENCE` — validations without selection-score improvement before the run
  aborts.  Off by default; set it while workshopping so a dead run stops itself.
* `KINEMATICS_MIN_GATE_JACCARD` / `KINEMATICS_MIN_DSRX_CORR` — promotion gates.  Leave them unset
  while workshopping.

---

## 5. Acceptance

```bash
python scripts/eval_deploy_flow_acceptance.py --checkpoint <run>/kinematics_best.pth
```

Baselines to beat, on the real clot task (§10.4, mean oracle-F1 over 12 vessels):

```
GT flow                    0.882
RGP-DEQ (leak-assisted)    0.675   <- the number to reach on LEGAL priors
analytic prior alone       0.370
```

---

## 6. Known cohort hazards

* **7 deploy packs are from an older extractor revision**: `patient002` and all six `*_mirror_y`.
  They have dead `node_type` where the other 45 are live, and anomalous prior blocks
  (`patient002` is the bit-identical leak; the mirrors read 0.06–0.45).  `patient002` is in the
  training pool.  Consider excluding or re-extracting all seven.
* **`patient018`** scores 0.000 in every predicted-flow arm and is its own problem
  (`DEPLOY_FLOW_PLAN.md` §2).
* **The clinical steady-kine anchors** (`graphs_kinematics_anchors/carreau`) are what
  `KINEMATICS_INCLUDE_PATIENT_ANCHORS=1` loads and what B14's geometry sync repairs.  If that
  directory is missing, they are regenerable from the biochem COMSOL exports —
  `biochem_extract_transfer.py` maps `kine.pt` to
  `data/processed/graphs_kinematics_anchors/carreau/{stem}.pt`.
