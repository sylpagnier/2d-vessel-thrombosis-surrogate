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

---

## 7. Before the NEXT generation run — read this first (2026-08-28)

Measured on the transferred 250-vessel cohort.  Full write-up: [`RGP_DEQ_REPAIR_PLAN.md`](RGP_DEQ_REPAIR_PLAN.md) §16.

**The cohort is fine.  The labels are not what deployment looks like**, in the one channel the
clot gate reads.  The deposition gate is `(sr < lss) | (dsrx < sgt)`, and at deployment the
median vessel has **50.8% of its firing wall nodes firing through the `dsrx` branch alone**.  In
this corpus that number is **0.0%** — the branch fires on no wall node at all in more than half
the vessels.  A model that learns everything this corpus can teach caps at **27% of the
achievable deploy gate agreement** (`scratch/tune/diag_branch_ceiling.py`); the shipped
checkpoint is already at 18%.

Two causes, both in the pipeline rather than the vessel designs, and **both are generation-side
fixes**:

### 7.1 Solve at the element order deployment uses

```
comsol_models/phase1_template.mph        order_fluid = P1+P1   <- every kinematics vessel
comsol_models/phase2_nowound_040.mph     order_fluid = P2+P1   <- every deployment vessel
```

With linear velocity elements the profile inside the first cell off the wall is linear by
construction, so the wall shear rate is an element average and its along-wall derivative is
largely whatever that piecewise-linear field leaves behind.

**Measure it before changing it** — one vessel, both ways, same mesh:

```bash
python scripts/exp_comsol_element_order.py --stems vessel_0,vessel_5,vessel_7
```

Read the `dsrx_sd` ratio column.  Then set `PhysicsConfig.comsol_order_fluid = 2`
(`AnchorGenerator._set_element_order` applies it through the COMSOL API — the template is not
edited) and regenerate.  Expect the solve to be slower and to need more memory per vessel; the
repair ladder already handles the failures that follow.

### 7.2 Give the graph TRUE mid-side labels, not interpolated ones

`KINEMATICS_ELEVATE_P2=1` inserts a mid-side node per edge and sets its label to the mean of its
two corners.  That is fine for velocity (0.2-2.2% error) and destructive for `dsrx`, which is a
second derivative: a mid-side value on the chord makes the field piecewise-linear along the
half-edge **by construction**, so it cancels curvature.  Controlled experiment on a deploy pack,
operator and node count held fixed:

```
native P2 (COMSOL)             wall dsrx_sd 398.5
corner P1                                   161.9
re-elevated by interpolation                 64.1     <- 6.2x destroyed
```

`AnchorGenerator._evaluate_at_coords` evaluates COMSOL's solution at **arbitrary coordinates**,
so the fix is cheap: elevate the mesh topology at graph-build time and ask COMSOL for `u/v/p/mu`
at the mid-side coordinates too.  Combined with §7.1 the corpus then matches deployment on both
topology and label fidelity, and `KINEMATICS_ELEVATE_P2` becomes an unnecessary no-op.

**Until then**, `KINEMATICS_BAND_ON_CORNERS=1` evaluates the wall-shear supervision on the P1
corner subgraph, where the labels are COMSOL's own.  It recovers roughly a factor of two of the
lost spread — not the whole thing.

---

### 7.3 MEASURED 2026-08-28 on the generation box — the two changes only work TOGETHER

`scripts/exp_comsol_element_order.py --stems vessel_0,vessel_5,vessel_7 --p2-nodes`, same mesh,
same node set, same operator; only the element order and the source of the mid-side labels vary.

```
                   corner nodes only            P2 node set: TRUE mid-side vs corner-mean
stem          sr ratio   dsrx_sd ratio      order=1 ratio        order=2 ratio    |du|max
vessel_0        1.00         1.01               1.00                 2.22       0.001 / 0.025
vessel_5        0.98         1.00               1.00                 4.14       0.001 / 0.027
vessel_7        0.99         1.00               1.01                 4.41       0.001 / 0.027
```

**Read it in this order.**

1. **`order_fluid` alone changes NOTHING** (0.98-1.01 on both `sr` and `dsrx`).  The graph is
   built from the P1 mesh points, and a P2 solution sampled at corner nodes matches the P1 one
   there.  §7.1's hypothesis, as stated, is **disproved**.
2. **Mid-side evaluation alone changes nothing either.**  At `order_fluid=1` the interpolant IS
   linear, so COMSOL returns the corner mean: ratio 1.00 and `|du|max` 0.001, which is float
   noise.  §7.2 alone is inert.
3. **Together they recover 2.2-4.4x of the wall `dsrx` spread.**  That is the same phenomenon,
   at the same order of magnitude, as the 6.2x measured on a deploy pack by decimating a native
   P2 pack to corners and re-elevating it (§16.5).

**So the generation change is a single change with two halves, and half of it is worthless.**
Set `PhysicsConfig.comsol_order_fluid = 2` **and** evaluate `u/v/p/mu` at the mid-side
coordinates (`AnchorGenerator._evaluate_at_coords` already takes arbitrary points; the P2 node
set and its wiring are in the script's `p2_node_set`).  The corpus is then native P2 with true
labels and `KINEMATICS_ELEVATE_P2` becomes an unnecessary no-op.

**Necessary, probably not sufficient.**  `sep%` is 0.00 on all three vessels in every arm, and
the corpus sits **10.7x** below deployment on wall `dsrx` spread.  A 2.2-4.4x recovery does not
close that on its own; the residual is the vessel DESIGNS — deployment's walls vary about twice
as fast (`width_d1` per-vessel max 0.51x of deploy, preflight's second WARN) — which is a
sampler question, not a solver one.  Cost: the P2 solve ran 5-7 s against 3-6 s at P1 on these
meshes, and the graphs get ~4x the nodes.

---

### 7.4 What changed 2026-08-28, and the loop to close

**Preflight now gates on the consumer's own statistics.**  Checks 1-8 are producer-side (mesh
order, operator sanity, element size, stenosis ratio, BC range, solve rate) and all eight passed
the 2026-08-28 cohort while its labels did not contain what `clot_ml` reads.  The new check
measures wall `sr`, `dsrx` spread and both gate branches through the consumer's own convention
(MLS hops=3 on the labels, **after** P2 elevation, because that interpolation is part of what it
exists to catch) and compares them to a band derived from **FIT deploy packs only**:

```bash
python scripts/derive_deploy_wall_shear_band.py      # once; writes data/reference/deploy_wall_shear_band.json
python scripts/preflight_kine_cohort.py --src <cohort> --expect-p1
```

The 2026-08-28 cohort now reads, correctly, **2 FAIL**:

```
[FAIL] wall-shear regime matches deployment (n=24)  sep-only median 0 vs deploy 0.9146
       -- the `dsrx` gate branch NEVER fires here
[FAIL]    wall_dsrx_sd   median 95.7 vs deploy 717.7 (0.13x); deploy p10-p90 [350.5, 2228]
[WARN]    wall_sr_med    median 48.0 vs deploy  91.8 (0.52x)
```

DEV and SEALED are excluded from the band so they stay independent evidence.  The deploy packs
are **never trained on** -- they define a target distribution, which is the same thing preflight
already did with `h_nd` and `u_ref`, applied to the statistic that matters.

**Generation now produces true P2 labels.**  `PhysicsConfig.comsol_order_fluid` is **2**, and
`AnchorGenerator._process_single_anchor` evaluates `u/v/p/mu` at the mid-side coordinates and
stores them in the `.npz`; `mesh_to_graph` carries them as a position-keyed probe set and
`elevate_to_p2` matches its own midpoints against it, falling back to the corner mean per node.
So a pack generated before this date, or solved at `order_fluid=1`, is **bit-identical to
before** -- pinned by a test.  Matching is by POSITION, not index, because an ordering contract
between the generator, the mesher and the elevation is exactly the kind of assumption that fails
silently.

**The wall-variation knobs are exposed and NOT tuned.**  `VesselConfig.wall_noise_*` were
hardcoded; they are the knob the gate keys on, and the corpus sits at 0.51x of deployment on
`width_d1` and 0.13x on wall `dsrx` spread.  Defaults are unchanged, so nothing moves until they
are deliberately raised.  **The value cannot be derived from geometry alone** -- the map from
wall roughness to wall `dsrx` runs through meshing and the CFD solve -- so close it empirically:

```
raise wall_noise_freq_* / wall_noise_amp_frac
  -> generate ~20 vessels (`-n 20 --overwrite`)
  -> preflight, read `wall_dsrx_sd` and `sep-only`
  -> repeat until both land inside the deploy band
  -> only then generate the full cohort
```

Twenty vessels is minutes, and it is the difference between a cohort that trains and 250 that do
not.

**One thing this does NOT fix, measured.**  An arm that put 17 real deploy vessels into the
training pool (right regime, COMSOL's own labels, held disjoint from selection) scored
`gateJ%` 33.6 against the analytic prior's 32.5 -- the same +-4 oscillation every other arm
shows, with `dsrxR` pinned at the prior's 0.57-0.62 throughout.  So the corpus regime is
necessary and not sufficient: something in the architecture or the objective is preventing the
residual from learning wall `dsrx` at all.  That arm was a **diagnostic only** -- its checkpoint
must never be promoted, because those vessels are the clot stack's own evaluation cohort.

---

## 8. Launch configuration (2026-08-28)

```bash
SPECIES_PRIOR_SOURCE=analytic
KINEMATICS_ELEVATE_P2=1
KINEMATICS_COORD_MODE=centered
KINEMATICS_NORMALIZE_SHEAR_GRAD=1
KINEMATICS_LOSS_WEIGHTS=outputs/kine_loss_weights_20260828.json
KINEMATICS_MAX_NODES=26000          # P2 nodes; peak GPU ~0.092 GB per 1000
KINEMATICS_SELECT_MAX_GRAPHS=8
KINEMATICS_VAL_EVERY=2
KINEMATICS_PREPARED_CACHE=outputs/cache/kine_prepared
BIOCHEM_GRAD_CACHE_CPU=300
KINEMATICS_TRAIN_SUBSAMPLE=100      # iteration only; drop it for a final run
```

`scratch/tune/launch.sh` sets all of it; an arm overrides one variable.  Read a run by its
one-line summary:

```
[kin] ep6  SELECT gateJ%= 41.2 gateJ=0.379 dsrxR=+0.612 | relL2=0.271 div=3.1e-03 comp=0.58
```

**`gateJ%` is the headline.**  It is gate union Jaccard as a fraction of the per-vessel ceiling
a *perfect* flow field reads under the same stencils — the raw Jaccard carries a ceiling of
0.53-1.00 that belongs to the metric, not the model.  Baselines on the same strided 8 deploy
packs:

```
GT flow          100.0
analytic prior    32.5     <- the number a retrain has to beat
shipped ckpt      24.4
```

**Selection now always runs**, on real deploy packs (`src/utils/kinematics_select_packs.py`),
and the best checkpoint is ranked on it.  Before 2026-08-28 neither was true: the block was
gated behind clinical sidecars that do not exist and promotion ranked on
`rel_l2 + 100*continuity`.

Iteration-speed settings that are not optional if you want to sweep anything:
`KINEMATICS_PREPARED_CACHE` (P2 elevation + priors + PDE floors, ~20 min, cached),
`BIOCHEM_GRAD_CACHE_CPU=300` (the MLS operator LRU held 12 against 225 graphs), and
`KINEMATICS_TRAIN_SUBSAMPLE` (stratified, applied after the cache so it needs no re-prep).
