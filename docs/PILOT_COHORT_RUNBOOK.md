# Pilot cohort runbook — generate on the COMSOL box, inspect here, then commit

The full repair is in [`RGP_DEQ_REPAIR_PLAN.md`](RGP_DEQ_REPAIR_PLAN.md).  This is the operating
procedure for the step before a full retrain: generate a **small** cohort on the COMSOL machine,
move it, prove it is fit to train on, and workshop the recipe on it.

**Why a pilot at all.**  The stored corpus is stale in at least three independent ways
(§9.1 stenosis tail, dead `node_type`, stale WLS operators), and every one of them was found by
measuring a pack rather than by reading the generator.  A 24-vessel cohort costs a fraction of
the solve time and exercises every one of those checks.

---

## 1. On the COMSOL machine — generate and solve

`pipeline_kinematics` runs Gmsh → COMSOL → PyG graphs in one pass.  Two runs, because
`--pathology-mode` is per-run and the random mode alone will not guarantee the severe-stenosis
tail the deployment cohort actually fails in.

```bash
# A. 18 mixed vessels, random pathology  (L0/L1/L2 = 4/6/8, weighted to the pathological tier)
python -m src.data_gen.pipeline_kinematics --batch --rheology carreau \
    -n 18 --level-mix 4,6,8 --pathology-mode random \
    --seed 20260828 --overwrite

# B. 6 more, forced severe stenosis -- guarantees the failing regime is represented
python -m src.data_gen.pipeline_kinematics --batch --rheology carreau \
    -n 6 --level-mix 0,2,4 --pathology-mode max_stenosis \
    --seed 20260829
```

Notes that matter:

* **`--overwrite` only on run A.**  It restarts vessel indices at 0; passing it to B would
  overwrite A.
* Drop `--seed` for a random cohort, but record whatever seed you use — the preflight report is
  only interpretable next to it.
* Add `--num-workers N` to parallelise Gmsh.  `--skip-anchor` builds meshes and graphs without
  the COMSOL solve, which is a useful 2-minute dry run to confirm the flags parse.
* Output lands in `data/processed/graphs_kinematics/carreau/`.

**Sanity-check on that machine before transferring** (CPU-only, seconds):

```bash
python scripts/preflight_kine_cohort.py --src data/processed/graphs_kinematics/carreau --expect-p1
```

If `severe-stenosis coverage` warns at 0%, the pathology run did not take and there is no point
transferring — that is exactly the defect the current stored corpus has.

---

## 2. Shrink for transfer

**98.4% of a pack is two tensors nothing reads.**  On a 4,019-node vessel:

```
G_x           64.61 MB   (N, N) sparse
G_y           64.61 MB   (N, N) sparse
everything     2.07 MB
TOTAL        131.30 MB   ->  24 vessels = 3.15 GB
after slim     2.07 MB   ->  24 vessels =   50 MB
```

`graph_gradient_operators` defaults to MLS and builds from positions + connectivity; `G_x`/`G_y`
are read only under `BIOCHEM_GRAD_OPERATOR=legacy`.

```bash
python scripts/slim_kine_packs.py \
    --src data/processed/graphs_kinematics/carreau \
    --out transfer_carreau --verify
```

`--verify` also reports whether each pack's stored WLS operator matches its own graph.  Expect
notes here: on the current corpus **3 of 3 sampled packs do not match** (B13).  The tool
deliberately does not "fix" that — dropping `V`/`W`/`M_inv` (`--drop-wls`) forces a correct
rebuild on load, but that is a numerics decision, not something a copy tool should make.

Copy `transfer_carreau/` across.

---

## 3. On this machine — prove it before spending GPU time

```bash
python scripts/preflight_kine_cohort.py --src transfer_carreau --expect-p1
```

Exit code 1 on any FAIL.  Each check is a bug that has already cost a run:

| check | why it exists |
|---|---|
| topology P1 / P2 | training was 0% degree-2 against a 74.5% deploy mesh (§8 A1) |
| prior block is not the CFD solution | the s17 Z2 leak, bit-identical on 43/43 packs (§1a) |
| `width_d2` within training range | 1e4+ means a stale WLS operator (B13) |
| `wall_normal` populated | identically zero for a year; drives the GAT's attention biases |
| `node_type` populated | **currently dead on both training corpora, live at deploy** |
| severe-stenosis coverage | the stored corpus has 0% at ratio ≥ 2.0 against deployment's 14% |
| `u_ref` overlaps deployment | BC range; currently fine, checked so it stays fine |

Run it on the existing corpus first to see what a failing report looks like — it returns
`1 FAIL, 1 WARN` today (dead `node_type`, no stenosis tail).

---

## 4. Workshop the recipe on the pilot

```bash
# 1. calibrate the loss weights on the new cohort (no training runs; see s12)
python scripts/calibrate_kine_loss_weights.py --graphs 8

# 2. a short run with the full launch config
SPECIES_PRIOR_SOURCE=analytic \
KINEMATICS_ELEVATE_P2=1 \
KINEMATICS_COORD_MODE=centered \
KINEMATICS_NORMALIZE_SHEAR_GRAD=1 \
KINEMATICS_LOSS_WEIGHTS=outputs/kine_loss_weights.json \
KINEMATICS_INCLUDE_PATIENT_ANCHORS=1 \
KINEMATICS_VAL_EVERY=2 \
KINEMATICS_SELECT_MAX_GRAPHS=6 \
KINEMATICS_SELECT_PATIENCE=6 \
python -m src.training.train_kinematics_predictor --epochs 20 --adam-epochs 20 \
    --stage1-end-epoch 0 --stage2-end-epoch 0 --no-prompt
```

Read the run by its one-line-per-validation summary:

```
[kin] ep12   SELECT gateJ=0.281 dsrxR=+0.402 | relL2=0.183 div=2.1e-03 comp=0.394 | ...
```

**`gateJ` first, deliberately.**  It is the only Stage-A metric measured to predict the clot
model's own oracle-F1 (+0.918; `dsrxR` reads -0.073 within a flow arm, §10.3).  `relL2` is
reported because it is cheap and familiar, not because it should drive a decision.

What "working" looks like on a 24-vessel pilot: `gateJ` moving off its starting value at all,
and the run not aborting.  **Do not read a pilot's absolute numbers as a result** — 24 vessels
is far below the cohort noise floor.  The pilot answers "does the pipeline run end-to-end and do
the metrics respond", not "is the model good".

Knobs for the workshop phase:

* `KINEMATICS_VAL_EVERY` — 2 by default (1 when total epochs ≤ 12).
* `KINEMATICS_SELECT_MAX_GRAPHS` — 6 by default.  Each selection graph is a full 25-iteration
  Anderson solve, so this is the main validation cost.  Raise it for a final selection pass.
* `KINEMATICS_SELECT_PATIENCE` — validations without selection-score improvement before the run
  aborts.  **0 (off) by default**; set it for pilots so a dead run stops itself.
* `KINEMATICS_MIN_GATE_JACCARD` / `KINEMATICS_MIN_DSRX_CORR` — promotion gates.  Leave unset on a
  pilot; a 24-vessel cohort should not be promoting anything.

---

## 5. Then, and only then

```bash
python scripts/eval_deploy_flow_acceptance.py --checkpoint <run>/kinematics_best.pth
```

Baselines it must beat, on the real clot task (§10.4):

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
  training pool.  Consider excluding or re-extracting all seven before they influence anything.
* **The stored synthetic corpus is stale** — no severe stenosis, dead `node_type`, WLS operators
  that do not match their graphs.  Regenerating fixes the first two by construction (§9.1).
* **`patient018`** scores 0.000 in every predicted-flow arm and is its own problem
  (`DEPLOY_FLOW_PLAN.md` §2).
