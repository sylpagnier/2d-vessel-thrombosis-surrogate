# WOUND PROGRESS — extending the clot model to injured vessel wall

Opened 2026-08-21. Scope: `phase2_wound_001/002/003.mph` and the packs extracted from them
(`wound_patient001/002/003`). Companion to [PHASE7_FINDINGS.md](PHASE7_FINDINGS.md), whose
§9.1 predicted this case and parked it; the prediction held.

> **HEADLINE.** The wound is not a new mechanism. It is the **same surface-deposition law with
> the two shear gates deleted** — the bracket is character-for-character identical to `srf1`
> and the gate multiplier is replaced by a hard `1`. Consequences, measured on all three
> vessels: **100% of wound nodes clot**, and the t=0 deposition gate — the thing the entire
> shipped model is built on — **fires on 0% of them**. The current physics scores a region of
> guaranteed clot as guaranteed clean. Forcing `gate := 1` on `mask_wound` and changing
> nothing else takes full-domain F1 from **0.482 / 0.487 / 0.826 → 0.851 / 0.930 / 0.887**,
> with precision *rising* — every wound node genuinely clots, so it adds only true positives.

---

## 0. WHERE TO READ THE PHYSICS FROM

**`comsol_models/phase2_template_*.mph` are gone and are not coming back.** Both were
mislabelled — the one named `nowound` carried the wound physics and the one named `wound` was
a stale 4-boundary tube. Read the model tree off the **latest `phase2_wound_*` /
`phase2_nowound_*` patient runs instead**; those are always the format actually in use.

```python
import zipfile, json
z  = zipfile.ZipFile("comsol_models/phase2_wound_001.mph")
sm = json.loads(z.read("smodel.json"))   # GUI tree: node labels, selections, structure
dm = z.read("dmodel.xml")                # the expressions — J0 strings live here, not in smodel
```

`smodel.json` gives the tree shape; the **`J0` expressions are only in `dmodel.xml`**. Trust
the physics node tree over the parameter list ([PHASE7_FINDINGS.md](PHASE7_FINDINGS.md) §0).

---

## 1. THE PHYSICS: FOUR BOUNDARY NODES, ONE SUBSTITUTION

| tag | label | selection | role |
|---|---|---|---|
| `fl1` | `WallFlux_9spec` | `uni1` (wall ∪ wound) | bulk 9-species, **gated** |
| `srf1` | `wall_surface_reactions_3spec` | `uni1` | surface `M`/`Mas`/`Mat`, **gated** |
| `fl2` | `WoundFlux_9spec` | `sel1` (wound) | **new** — ungated, overrides `fl1` there |
| `srf2` | `SfcRxn_3spec` | `sel1` | **new** — ungated, overrides `srf1` there |

`srf2` decides the clot, because the clot label is `mu1(Mat)` stepping at
`viscosity_mat_crit = 2e7`:

```
srf1 (healthy wall)
  J0_Mat = Da * G_wall * [ Sat(M)*k_rs*RP + Sat(M)*k_as*AP + (Mas/M_inf)*k_aa*AP ] * step2t(t)
  G_wall = [d(spf.sr,x) < sgt] * (L/gamma_m)*|d(spf.sr,x)|  +  [spf.sr < lss]

srf2 (wound)
  J0_Mat = Da *   1    * [ Sat(M)*k_rs*RP + Sat(M)*k_as*AP + (Mas/M_inf)*k_aa*AP ] * step2t(t)
```

The bulk flux differs the same way and **only on the three platelet channels** — `RP`, `AP`,
`APR` lose their gates. `J0_APS`, `J0_PT`, `J0_TH` are byte-identical between `fl1` and `fl2`,
because those terms key off `Mat` rather than shear: the thrombin autocatalysis is unchanged
in form, it just gets a much larger `Mat` to work on.

### What is NOT new — checked, not assumed

- **No flow BC changes.** The `spf` feature list is identical wound vs no-wound.
  `WallFluidBC` covers the wound; no injection, no porosity, no slip. The wound perturbs the
  flow only through `mu1(Mat)`.
- **No parameter changes**, no new species, `Reactions_9spec` untouched, fibrin still inert.
- **The wound is not a notch.** Flat, chemically-relabelled wall — same geometry class.

### Override vs additive is currently unobservable

`fl1`/`srf1` span the wound too (`uni1`), so the wound law could be an override or an
addition. It cannot be distinguished at t=0 on these vessels: no wound node trips a gate, so
`G_wall = 0` there and both readings agree. **It becomes a real question after gelation** —
see §3, where the gate opens on 95% of wound nodes.

---

## 2. GEOMETRY: A BILATERAL COLLAR, NOT A SPOT

`sel1` = boundary entities **4 and 5** — two opposed patches, one on each wall, over the same
axial window. The deposition front closes in from both sides.

| vessel | wall n | wound n | axial span | t=0 wound `sr` med | t=0 wall `sr` med | gate ON at wound |
|---|---|---|---|---|---|---|
| `wound_patient001` | 466 | 80 | 14.6% | 148.1 /s | 148.0 /s | **0.0%** |
| `wound_patient002` | 464 | 80 | 14.9% | 99.4 /s | 83.4 /s | **0.0%** |
| `wound_patient003` | 559 | 26 | 5.7% | 127.6 /s | 97.4 /s | **0.0%** |

Shear at the wound is **ordinary** — at or above the wall median on every vessel. That is
precisely why the gated law is blind to it, and why this is a structural miss rather than a
calibration error. The cohort noise floor (±0.024 wall / ±0.091 off-wall) is irrelevant at
this effect size.

---

## 3. DYNAMICS: TWO REGIMES, AND A FEEDBACK LOOP WITH NO COHORT ANALOGUE

### 3.1 `Mat` at the wound is a two-regime curve hinged on gelation

Sub-linear fill while `Sat(M)` decays and `Mas` builds, then — once `mu1` steps — **dead
linear** growth: 0.87 × crit per 750 s on `wound_patient001`, constant to three digits,
because `Mas` has saturated at `M_inf` and the autocatalytic term locks at `Da*k_aa*AP`.
It never saturates.

### 3.2 A wound in healthy flow is a geometry-independent clock

`wound_patient001` and `002` are different geometries with different wall shear, and their
wound trajectories nearly coincide: gelation at **3600 s vs 3750 s**, final `Mat` **9.04 vs
8.70 × crit**, within-wound spread **2.18× vs 2.12×**. With the shear dependence removed,
wound `Mat` is driven only by `RP`/`AP`, which sit near their inlet values.

`wound_patient003` is the exception that proves it: gelation at **600 s**, final **104 ×
crit**, 21× internal spread, *accelerating*. Its wall is 35% gated at t=0 — a
stagnation-dominated vessel where the wound sits inside a region that was already clotting.
**So: a wound in healthy flow is a clock; a wound in a stagnating vessel compounds with the
wall. Do not generalise the 3600 s constant.**

### 3.3 The gate opens — the wound bootstraps its own stagnation

`wound_patient001`, gate recomputed from GT velocity at each step:

| t (s) | 0 | 1500 | 3000 | 3750 | 4500 | 6000 | 7500 | 9750 | 10429 |
|---|---|---|---|---|---|---|---|---|---|
| wound `sr` med (/s) | 148 | 148 | 148 | 25 | 18 | 18 | 18 | 29 | 30 |
| **wound gate ON** | **0%** | **0%** | **0%** | 53% | **94%** | **95%** | 95% | 49% | 49% |
| wall gate ON (reference) | 16% | 16% | 16% | 17% | 12% | 11% | 13% | 14% | 14% |

Ungated deposition → `Mat` crosses gelation → `mu1` → 80× → near-wall flow stalls → `sr`
collapses 148 → 18 /s → **the ordinary low-shear gate opens on top of the wound law**. On a
healthy wall this loop cannot start: without the wound the gate never opens in the first place.

**This is the approximation that breaks.** [PHASE7_FINDINGS.md](PHASE7_FINDINGS.md) §9 found
that accumulation-only survived *because* inputs were frozen at t=0. At the wound the gate
genuinely moves, 0% → 95%, and it moves as a consequence of the model's own output. Evolving
flow coupling stops being optional here.

---

## 4. WHAT THE WOUND DOES NOT DO

- **It does not occlude.** Mean flow speed through the wound band is flat across the whole
  run — 0.9924 → 0.9932 on vessel 001 (0.1%), upstream unchanged — while `mu_eff` p95 in the
  same band goes 5.3 → 162. The clot is a thin high-viscosity **mural coating**, one node
  shell thick on 001/002. Caveat: `Mat` is still growing linearly at every cut-off, so
  occlusion may simply be beyond the simulated horizon.
- **It does not corrupt the rest of the wall.** Wall-only F1 of the plain `gate > 0`
  predictor on the wound vessels — **0.710 / 0.832 / 0.874** — sits inside the no-wound
  cohort range (`patient011/012/020/025/044`: 0.549–0.816). The wound is a local addition,
  which is what makes the fix cheap. (Caveat: this is an across-vessel argument, not a
  controlled one — see §7.)

---

## 5. THE SCORE ON THE TABLE

F1 on the full wall ∪ wound domain, t=0 GT flow, final-time GT clot:

| vessel | gate only (F1 / prec / rec) | + `gate := 1` on wound |
|---|---|---|
| `wound_patient001` | 0.482 / 0.800 / 0.345 | **0.851** / 0.903 / 0.805 |
| `wound_patient002` | 0.487 / 0.855 / 0.341 | **0.930** / 0.941 / 0.920 |
| `wound_patient003` | 0.826 / 1.000 / 0.704 | **0.887** / 1.000 / 0.796 |

Vessel 003 gains least because its wound is 26 of 585 boundary nodes. n = 3 — treat the
magnitude as indicative and the sign as solid.

---

## 6. PACK ENCODING — FIXED 2026-08-21

**The bug.** The extract carves `mask_wall` disjoint from `mask_wound` (`dif1` vs `sel1`) so
the two deposition laws stay separable. That split is right for physics and wrong for
geometry: a wound node is still a no-slip boundary node. Every builder measured wall-derived
features against `mask_wall` alone, so injured nodes were encoded as **open lumen** —
`sdf_nd` at the wound read **0.32 / 0.25 / 0.11** diameters instead of 0 (literally the
distance to the nearest *un-wounded* wall node), and `wall_normal`, `width_nd`, `wss_prior`
and the no-slip enforcement followed it off the wall. Interior nodes above the wound inherited
the same error: **16.4% / 14.8% / 5.6% of each mesh** had a wrong wall reference, matching the
wound's axial fraction almost exactly.

**The fix.** One canonical helper, `solid_boundary_mask(mask_wall, mask_wound)` in
[`src/data_gen/lib/mesh_wls.py`](../src/data_gen/lib/mesh_wls.py) — "every no-slip solid
boundary node", i.e. COMSOL `uni1`. Wired through every geometry consumer:

| file | what now uses the union |
|---|---|
| `data_gen/lib/kinematics_graph_builder.py` | SDF/normals KD-tree, centerline, `wall_tree`, x-tensor, y-labels, no-slip (one named `mask_solid`, replacing scattered `mask_wall if mask_wound is None else …`) |
| `data_gen/lib/mesh_to_graph_biochem.py` | same inline block + no-slip + `wss_mag` + `wss_prior` |
| `data_gen/lib/mesh_to_graph.py` | same |
| `data_gen/lib/node_feature_assembly.py` | `refresh_kinematics_node_x_on_graph` unions `data.mask_wound` before rebuilding `x` |

Gmsh line-tag collection for exact wall normals now also accepts `TAGS["Wound"]` (104). Inert
on the current packs — the COMSOL `.msh` files export **no line cells at all**, so the exact
segment-normal branch never runs and `wall_normal` is identically zero at wall nodes on
*every* pack, wound and no-wound alike. See §8.

**`mask_wall` semantics are unchanged** — it stays the *healthy* wall label, disjoint from
`mask_wound`. Downstream code that means "all solid boundary" should call
`solid_boundary_mask`. No-wound packs are provably unaffected: the helper returns `mask_wall`
untouched when the wound mask is absent or empty, and the repair skipped all 49 of them.

**Repairing packs already on disk.** The multi-GB COMSOL exports for the wound runs are gone,
so re-extraction is not possible. [`scripts/repair_wound_pack_geometry.py`](../scripts/repair_wound_pack_geometry.py)
rebuilds `data.x` through the same `build_kinematics_node_x_tensor` call the extractor uses,
with the same `resolve_anchor_kine_phys_cfg()` Carreau config — only the boundary mask differs.

```bash
python scripts/repair_wound_pack_geometry.py --verify patient012   # correctness gate
python scripts/repair_wound_pack_geometry.py --dry-run
python scripts/repair_wound_pack_geometry.py
```

Two safeguards, both load-bearing:

1. **`--verify` round-trips the rebuild on a no-wound pack**, where the union is a no-op. On
   `patient012` and `patient020` every geometry channel reproduces exactly (`sdf_nd`,
   `shear_potential`, `width_nd`, `width_d1`, `node_type_*`, `rheology_flag` all max|diff|
   0.0000). The *prior* channels do not reproduce on those two — see §8.
2. **The write is surgical.** Only rows whose wall reference actually moved (plus the wound
   itself) are rewritten; every other row stays bit-identical to the pack v4 was validated
   against. On the wound packs the rebuild reproduces the stored priors with median relative
   difference 0.0000 on unchanged rows, so the rewritten rows stay internally consistent.

Backups are written once to `*.pt.prewoundfix`. Post-repair `sdf_nd` at wound nodes is 0,
matching the healthy wall. Pinned by
[`src/tests/test_solid_boundary_mask.py`](../src/tests/test_solid_boundary_mask.py), which
includes an end-to-end assertion over the on-disk packs.

**The measured physics in §1–§5 is unchanged by the repair** — it is all derived from `y`,
node positions and edge topology, none of which the bug touched. The F1 table in §5 is
byte-identical before and after.

### 6b. The same bug, one layer up — FIXED 2026-08-22

The table above covers the pack **builders**. `src/clot_ml/features.py` does not read the
builders' geometry — it re-derives its own (`cKDTree(pos[wall])`, `hop_distance(wall, A)`,
`resolve_offwall_shell(pos, wall, ei)`) — and it was still doing that against `mask_wall`
alone, so the clot-ML feature layer re-introduced the identical encoding on top of repaired
packs. At the wound: `is_wall` 0, `hop_wall` mean 8.7, `dist_wall_edges` **10.92**, and the
v4 transport source contributing nothing from the injured segment.

Fixed by `solid_boundary_nodes(data)` (same module, same union, read off a pack), threaded
through `features.py`, `features_v4.py`, `locked.build_sample` and the v3 cache builder. The
node counts come out **16.4% / 14.8% / 5.6%** — the same three fractions as the builder bug,
which is the check that it was the same bug. `patient020` moves zero nodes. Full before/after
in [MODEL_REVIEW_2026-08-22.md](MODEL_REVIEW_2026-08-22.md) §5b.5(1).

The split this preserves: `wall` still selects the gated `srf1` **deposition law** (and, until
§9/§12.3's eval-domain question is settled, the scoring domain); only **geometry and
transport** take the union.

---

## 7. KNOWN LIMITS

- **Truncated runs are expected, not a defect.** 71 / 75 / 129 of 201 steps (10429 s /
  10980 s / 19118 s vs the cohort's 30000 s) — wound simulations are simply much slower to
  run. But `Mat` is still climbing linearly at every cut-off, so **"final `Mat`" and "final
  clot extent" are horizon quantities**: any metric compared against the no-wound cohort needs
  a matched horizon, or it compares a third of a run to a whole one.
- **There is no paired A/B.** `wound_patient001` is *not* the same vessel as `patient001` —
  median outline offset 1.28 nd, different `d_bar`, different node count. Nothing here
  isolates the wound's effect on a fixed geometry. **Re-running one existing cohort `.nas`
  with and without the `sel1` selection is the single most useful next simulation.**
- **n = 3.**

---

## 8. TWO PRE-EXISTING ISSUES FOUND ALONGSIDE — **BOTH FIXED 2026-08-22**

> Both were repaired on all 45 packs by
> [`scripts/repair_pack_wall_normals.py`](../scripts/repair_pack_wall_normals.py):
> `wall_normal` is now unit-length at 100% of solid boundary nodes (fitted from the graph,
> so it needs no mesh and works on the wound packs whose exports are gone), and
> `node_type_*` is a strict one-hot. **This invalidated `clot_gnn_v4`/`v4w` and the v5
> cache** — `width_nd`/`width_d1`/`width_d2` are computed by sphere-marching *along* the
> normal and moved on 11–22% of nodes. See
> [MODEL_REVIEW_2026-08-22.md](MODEL_REVIEW_2026-08-22.md) §6.5, which also records that
> fixing the normal took `width_nd` from **unusable on 9 of 42 vessels to 0 of 42** — exactly
> what `src/clot_ml/geometry_class.py` predicted — and broke the stenosis threshold in doing
> so.
>
> **The third item below — the `wss_prior_nd` disagreement — is still open**, and is now
> pinned down further: a fresh build puts `wss_prior_nd` at ~45 at the wall on `patient020`
> where the stored pack has 0. That is why both repair scripts write a delta rather than a
> rebuild.
>
> The original text follows.


1. **`wall_normal` is identically zero at wall nodes on every pack in the cohort.** The
   COMSOL `.msh` files contain only `triangle6` cells — no line cells — so the exact
   segment-normal branch never runs, and the KD-tree fallback gives a wall node its own
   position as nearest neighbour (zero vector). Confirmed on `patient012`, `patient020`,
   `patient044` as well as the wound packs. Fixing it would change every pack and invalidate
   the v4 baseline, so it is reported, not changed.
2. **`node_type_*` one-hots are all-zero on 100% of nodes**, in every pack checked. Four dead
   input channels.

A third, smaller one **was** fixed in passing because it is an active trap:
`refresh_kinematics_node_x_on_graph` built its `wall_tree` in **ND** while
`compute_hydraulic_width_nd` probes it in **SI** (`probe * d_bar`). The sphere-march therefore
never registered a hit and every width collapsed to the 1.0 sentinel — which is why that path
defaulted to `preserve_width=True`. The tree is now built in SI.

Also note the cohort disagrees with itself on `wss_prior_nd`: it is populated at the wall on
`wound_patient001` (median 35.25) and identically zero on `patient012` / `patient020`. Two
extractor revisions. Not investigated.

---

## 9. WHAT THIS IMPLIES FOR THE MODEL

Stated as claims to argue with:

- **The wound needs no new learned machinery.** It is one input channel — `mask_wound` —
  entering an existing term as a gate override. Bracket, constants, autocatalysis and
  gelation threshold all unchanged.
- **Wound `Mat` is nearly a known function of time, not a field to predict.** Within-wound
  spread 2.1× against 17 orders of magnitude across a healthy wall, and 001/002 share a
  trajectory. A one-parameter clock may beat a GNN here.
- **The frozen-t=0 gate is what breaks** (§3.3), and it breaks because of the model's own
  output — so this is where evolving-flow coupling has to earn its place.
- **The readout domain still needs a decision.** `mask_wall` remains the healthy-wall label,
  so wound nodes currently fall into the **off-wall** scoring domain, where the noise floor is
  4× worse. Either the eval domain split adopts `solid_boundary_mask`, or wound score gets
  quoted separately. Not yet done — it changes reported numbers and should be a deliberate
  choice.

---

## 10. THE COMPLEMENT — `src/clot_ml/wound.py`, first pass

Shipped as a **complement to `clot_gnn_v4`, not a retrain.** §4 is what licenses that: the
healthy wall is measurably unperturbed by the injury, so v4 stays valid everywhere except the
injured segment, and at n=3 vessels disturbing a 19-vessel artifact would be all risk.

**GT flow at t=0 only.** Everything below reads `t0_flow_fields(..., flow_source="gt",
time_index=0)` plus mesh geometry. No GT velocity after t=0, no GT species, no GT `Mat` at
inference. The §3.3 feedback closes on the rollout's **own** `Mat`, not on the answer.

### 10.1 The model

```
G_i(t) = G_pre  +  (G_post - G_pre) * sigma((Mat_i/crit - 1)/tau)      on the wound
G_i(t) = gate_i                                                        on the healthy wall
```

fed straight into COMSOL's surface ODE. The two-regime shape is not a fitting convenience —
it *is* §3.3: `G_pre` is the ungated wound law, and the node picks up the low-shear branch
once its own clot stalls the flow. `src/clot_ml/wound.py` carries a differentiable torch
mirror of `integrate_mat_trajectory` (matched to **6e-16 relative**, pinned by
`src/tests/test_wound_complement.py`), so the rate is fitted *through* the physics.

The ODE supplies the set, monotonicity and the gelation threshold. The learned quantity is a
**coefficient inside a conservation law**, never a label.

### 10.2 Three arms, leave-one-vessel-out

`python scripts/train_wound_rate.py`

| arm | what is fitted | curve L1 (log10) | onset MAE | % of horizon |
|---|---|---|---|---|
| `physics` | nothing (`G = 1`) | 0.817 | 30.7 | 33.5% |
| **`const`** | **two global scalars** | **0.353** | **7.3** | **6.4%** |
| `net` | + per-node `WoundRateNet` | 0.696 | 35.0 | 28.9% |

**`const` wins and the network loses.** `net` reaches a far lower *training* loss
(0.05–0.08 against 0.38–0.40) and then fails LOVO — on the fold that holds out
`wound_patient003` its onset MAE is 93.5 steps against the constant model's 18.3. Two
training vessels do not support a per-node head. It is kept in the module and reported here
so the comparison exists, exactly as PHASE10 §7 kept its losing selection layers; **do not
ship it without more vessels.**

**The fitted `G_pre` is the headline.** Across all three folds it lands at **2.04 / 1.96 /
2.01** — the fit independently recovers `ungated (1) + low-shear branch (1)`, the value §3.3
predicts from the mechanism. `G_post` is the genuinely fitted one and is not stable (22.1 /
20.9 / 11.9); the fold trained on `patient003` pulls it up.

### 10.3 On the metric of record

`python scripts/eval_wound_complement.py` — each vessel predicted with the constants fitted
**without** it. Deploy clot score, cohort mean over the 3 wound vessels:

> **SUPERSEDED BY §13 — these domains are the degenerate ones.**  `wound` is 100% GT clot
> so it cannot be scored, and `off` pools the wound's own lumen thrombus with clot from
> elsewhere in the vessel.  The corrected table is §12.2.  Kept here because the *relative*
> ordering of the arms is unchanged.

| arm | FIN wall | FIN wound | FIN off | FIN full | MOT wall | MOT wound | MOT off | MOT full |
|---|---|---|---|---|---|---|---|---|
| v4 alone | 0.7879 | 0.0000 | 0.0600 | 0.4989 | 0.6230 | 0.0000 | 0.0063 | 0.3662 |
| + wound physics (`G=1`) | 0.7879 | 1.0000 | 0.0600 | 0.7060 | 0.6230 | 0.5328 | 0.0063 | 0.4841 |
| **+ wound two-regime [LOVO]** | 0.7879 | **1.0000** | **0.5883** | **0.8058** | 0.6230 | **0.9445** | **0.5871** | **0.7230** |

Read it in this order:

1. **`FIN wall` is identical across all three arms.** That is the regression check, not a
   result: the complement never touches the healthy wall. It also holds end-to-end on a
   no-wound pack — every arm is bit-identical on `patient012` (0.9401 / 0.9237 / 0.9247),
   and `test_wound_complement.py` pins the no-op.
2. **`FIN wound` 0.0 → 1.0 is real but nearly free.** The wound domain is 100% GT clot, so a
   perfect final-time score there needs only the set, which the ungated law already gives at
   recall 1.000. **The content is in the timing**: `MOT wound` 0.533 → 0.944 is what the
   learned rate buys, and mean-over-time is where the two-regime model earns its two
   parameters.
3. **Full-mesh: +0.307 final, +0.357 mean-over-time.** Far outside the ±0.024 / ±0.091 noise
   floor — but that floor was measured on 19 vessels and this is 3, so treat the *magnitude*
   as indicative and the *sign* and *mechanism* as solid.

### 10.4 What this first pass does not do

- **`wound_patient003` carries all the remaining error** (onset MAE 18.3 against 2.0 / 1.6).
  It is the stagnation-dominated vessel, and one global `G_post` cannot serve both regimes.
  That heterogeneity is exactly what a learned rate *should* capture and cannot at n=3 —
  it is the argument for more wound simulations, not for a bigger model.
- **Off-wall uses fixed constants**, not fitted ones: attenuation 0.16 (the PHASE7 §12.5
  cohort median) and a lag of 4% of the horizon (the "off-wall lags its owner" measurement).
  Neither was tuned here.
- **v4's own wall score on these vessels is low** (`MOT wall` 0.623). That is v4 on
  three unseen geometries, untouched by this work, and a separate question.
- **The rate is global.** Nothing yet predicts `G_post` from vessel state, which is the
  first thing to revisit once there are more than three wound runs.

---

## 11. WHY `wound_patient003` IS DIFFERENT — it is not the wound model

> **THE MECHANISM IN THIS SECTION IS SUPERSEDED BY §14 (2026-08-23).**  The conclusion
> in §11.5 — that this is the wall model's problem and not the wound model's — stands and
> was reached again independently.  But the *mechanism* here is wrong: the near-wound wall
> gate is **already 1.0 at t=0 and stays 1.0 under the GT flow oracle at every step**, so
> there is no gate for a neighbour trigger to open.  §11.2–11.4's trigger machinery cannot
> help for that reason, and the `gate_scale` result in §11.4 was multiplying a rate, not
> opening a gate.  Read §14 first.

§10.4 left `wound_patient003` carrying all the residual error (onset MAE 18.3 against
2.0 / 1.6). The hypothesis worth testing was that its wound overlaps a region that was
already going to clot, so the wound and wall models fail to combine. **That is exactly what
happens, and the measurement traces the whole causal chain.**

### 11.1 The chain

| | `wound_patient001` | `wound_patient002` | `wound_patient003` |
|---|---|---|---|
| mesh hops, wound → nearest **gated** wall node | 25+ | 25+ | **9** (p10 11.5) |
| wall nodes gelled **before** the wound does | 0 | 0 | **21, by step 2 (300 s)** |
| hops from wound to those gelled nodes | 61 | — | **12–14** |
| wound gate ON at its own gelation step | opens *after* (step 26 vs 24) | after | **42% at step 3, gels at step 5** |
| wound `sr` before its own gelation | 148 → 148 → 148 | flat | **128 → 119 → 84 → 77** |

On 001/002 the wound gate opens because of the wound's **own** clot — the self-triggered
model in §10.1 is right. On 003 a clot-prone wall region 12–14 hops away gels at t=300 s,
stalls the flow through the wound band, and the wound's gate is already 42% open at step 3 —
**two steps before the wound itself gels**. The wound then runs at the doubled rate from the
very beginning, gels at step 5 instead of step 24-ish, and races to 104× crit.

So the wound module's assumption is externally violated on exactly this vessel.

### 11.2 The structural fix, and why it is safe

`mat_trajectory_torch` now takes an optional **neighbour trigger**: the two-regime switch
opens on `max(own committed weight, neighbourhood committed weight)` over healthy-wall nodes
within `TRIGGER_HOPS = 25` mesh hops (`predict_wound_series(..., trigger="wall")`). 25 is set
by the separation it must make — 12–14 hops against 61 — and the answer is flat over 25–40.

**It can only ever add, and it is provably inert where the neighbourhood is quiet.** On
`wound_patient001` every trigger source, including the GT oracle, leaves the trajectory
bit-identical (onset MAE 1.2 at every `k`). `test_wound_complement.py` pins that.

### 11.3 The blocker is the wall model, not the wound model

| trigger source on `wound_patient003` | wound onset MAE |
|---|---|
| `self` (ships today) | 18.0 |
| `wall` — the shipped gated wall ODE, **deploy-legal** | **18.0 (no change)** |
| `oracle` — GT wall `Mat`, k=25 | **6.6** |

The deploy-legal trigger does nothing because **the wall law gels those nodes at step 53 and
the truth is step 2** — a 25× timing error, at the one place and time it would have mattered.
On the deploy metric the same gap reads `MOT wound` 0.844 → 0.940 and `MOT full` 0.514 → 0.564
under the oracle; final-time is unchanged, because the *set* was already right.

### 11.4 The fudge that works, measured and rejected

Scaling the wall trigger's gate makes it gel early enough, and it fixes the vessel:

```
wall gate x1   -> wound onset MAE 18.0        x10 -> 9.4
wall gate x20  -> wound onset MAE  4.7        x40 -> 3.9
```

**It is not shippable.** The same scale on the 12 no-wound vessels takes wall-onset MAE from
**18.1% of the horizon to 43.7%**, and `x1` is the best value on 8 of those 12 (`x2` on the
rest). The shipped `da_scale = 40` is already correctly calibrated for the wall; `x20` is a
`wound_patient003`-shaped fudge that would wreck the wall model to rescue one vessel.
`wall_trigger_field(..., gate_scale=...)` keeps it reproducible and defaults to 1.

### 11.5 What this means

**`wound_patient003` is not a wound-model defect. It is the wall model's local onset-timing
error, surfacing through the wound**, and it is only visible here because the wound amplifies
it: everywhere else a late wall clot costs some timing score, but at a wound it changes the
regime the ungated law runs in.

That reframes the work. The remaining wound error is not a 3-vessel problem to be learned
around — it is the **19-vessel wall-timing problem**, where the cohort mean onset MAE is
already 18% of the horizon and PHASE7 §9 established the accumulate-only equation has a
ceiling. Fix wall onset timing and the wound complement picks up the gain automatically,
because the coupling is already wired and waiting.

---

## 12. SHIPPED — `clot_gnn_v4w` (2026-08-21)

`data/reference/clot_gnn_locked.json` now points at **`clot_gnn_v4w`**, kind
`temporal_v4_wound`.  Promote / re-promote with:

```bash
python scripts/promote_clot_gnn_v4_wound.py --repoint
```

**It supersedes `clot_gnn_v4` rather than sitting beside it, and the licence for that is one
property: on a pack with no wound mask it returns v4's output bit-for-bit.**  The GNN
ensemble, the temporal head, the readout and every threshold are byte-identical — this
artifact adds a boundary-condition branch, not a retrained model.  The property is asserted
per vessel *at promotion time* (`patient012` / `patient020` / `patient044`, mask, onset and
the whole series), and pinned by `src/tests/test_wound_complement.py`.  If it ever fails,
the artifact must not ship.

The 19-vessel strict-CV table in the pointer therefore carries over unchanged: that cohort
has no wounds, so the wound branch cannot touch it.

### 12.1 What is in the artifact

| | |
|---|---|
| base | `clot_gnn_v4`, loaded unmodified |
| fitted content | **two scalars**, `G_pre = 1.98`, `G_post = 14.28`, refit on all three wound vessels |
| fixed constants | off-wall attenuation 0.16 (PHASE7 §12.5), lag 4% of horizon, trigger `self`, 25 hops |
| NOT included | the per-node `WoundRateNet` — it loses leave-one-vessel-out at n=3 (§10.2) |

`G_pre` is barely "fitted": all three LOVO folds land on 2.04 / 1.96 / 2.01, independently
recovering `ungated (1) + low-shear (1)`, the value the mechanism predicts.  `G_post` is the
genuinely fitted one and is *not* stable across folds (22.1 / 20.9 / 11.9).

### 12.2 Deploy numbers (`scripts/eval_wound_complement.py`, LOVO constants, GT t=0 flow)

Cohort mean, n = 3.  Domains as defined in §13 — **read `w_reg` and `w_lum`, not `wnd`**:

| arm | wall | wnd | **w_reg** | **w_lum** | far | full |
|---|---|---|---|---|---|---|
| | | | *FINAL* | | | |
| `clot_gnn_v4` alone | 0.7879 | 0.0000 | 0.0604 | 0.0000 | 0.2614 | 0.4989 |
| + wound physics (`G=1`) | 0.7879 | 1.0000 | 0.6759 | 0.0000 | 0.2614 | 0.7060 |
| **`clot_gnn_v4w`** | 0.7879 | 1.0000 | **0.8777** | **0.8522** | 0.2614 | 0.8058 |
| | | | *MEAN OVER TIME* | | | |
| `clot_gnn_v4` alone | 0.6230 | 0.0000 | 0.0178 | 0.0000 | 0.0292 | 0.3662 |
| + wound physics (`G=1`) | 0.6230 | 0.5328 | 0.3612 | 0.0000 | 0.0292 | 0.4841 |
| **`clot_gnn_v4w`** | 0.6230 | 0.9445 | **0.8367** | **0.7288** | 0.0292 | 0.7230 |

Per vessel, FINAL `w_reg` / `w_lum`: **0.970 / 0.971** (001), **0.970 / 0.971** (002),
**0.693 / 0.615** (003 — the §11 outlier).

Wound coverage at promotion: **0.0% → 100.0%** on all three vessels, 160 / 160 / 55 nodes
owned, everything else bit-identical.

> **The baseline arm in that script is pinned to `clot_gnn_v4` and deliberately not read from
> the locked pointer.**  Now that the pointer resolves to v4w, a pointer-following baseline
> would already contain the arm under test and the table would read "the complement does
> nothing".  That bug was live for one run and is why `--base` exists.

**Two things the corrected domains reveal that §10.3's did not:**

1. **v4 is not merely blind to the wound boundary — it is blind to the whole wound
   thrombus.**  On 001/002 it predicts **zero** clot anywhere within 4 hops of the wound,
   162 GT nodes of which 82 are lumen.  `w_reg` 0.060 → 0.878 is the honest headline.
2. **The zero-parameter physics arm scores 1.0000 on `wnd` and 0.0000 on `w_lum`.**  The
   ungated law commits the boundary for free, but with `G = 1` `Mat` never reaches
   `crit / att`, so the lumen extension never fires at all.  **The lumen clot is bought
   entirely by the two fitted scalars, not by the physics** — which corrects §10.1's "the set
   is free": the set is free *on the boundary only*.

### 12.3 Caveats carried on the artifact

Recorded in the manifest so they travel with it: n = 3 wound vessels; GT flow at t=0 only;
and `wound_patient003` is externally triggered and remains an outlier (§11).

> **The fourth caveat — "wound nodes still score in the off-wall domain" — was RESOLVED on
> 2026-08-22** (roadmap item A3, MODEL_REVIEW §8e). `mask_wall` does stay the healthy-wall
> label, and that is now a decision rather than an omission: `mask_wound` is **100% GT clot on
> all three packs**, so folding it into the wall domain would award 80 free true positives —
> the degeneracy §13 identified. What changed is the other side: the off-wall domain is now
> `~solid_boundary_mask`, **true lumen**, so those 100%-GT nodes are no longer inside it
> either. The wound is scored by `wound_region_masks` (§13) and by nothing else.
> `src/clot_ml/data.eval_domains` is the single definition; no cohort pack carries a wound, so
> the change is inert on every published cohort figure.

---

## 12b. THE LUMEN RULE IS NOW SELECTABLE — and what each option measured (2026-08-23)

`predict_wound_series(lumen=...)`, and `scripts/eval_wound_complement.py --lumen`:

| rule | what it does | measured |
|---|---|---|
| `shell` | **shipped.** Commit when the owner reaches `crit/0.16`, then lag 4% of the horizon | `w_lum` 0.9708 / 0.9708 / 0.6151 |
| `transport` | C1: COMSOL's operator supplies a per-node attenuation, replacing BOTH constants | **fails the gate** — 0.8377 / 0.8377 / 0.4558 |
| `union` | shell OR transport | identical to `shell`; the transported set is a strict subset |
| `recursive` | C2: the same attenuation applied shell after shell, depth emergent from `Mat_wound` | 0.9708 / 0.9708 / ~~0.6329~~ **0.6018** — see §16.4, the gain was a bug |

**Why `transport` fails, and it is not a bug.** GT `Mat_off/Mat_owner` on the owned shell has
median **0.16–0.17** with a p10–p90 of 0.14–0.19 — the shipped constant is already accurate to
±15% here. The transported ratio is **0.007**, ~23× smaller, because a steady transport solve
is not the time-integrated `Mat` a growing wall source produces. `mat_adv` is a good *ordering*
feature and an uncalibrated *magnitude*, which is the same conclusion PHASE10 §14.4 reached
from the other direction.

**Why `recursive` gains so little — and, as of §16.4, nothing at all.** GT `Mat` decays
geometrically at 0.18–0.19 per shell, so the admissible depth follows from the wound's own `Mat`
with no new constant: 001/002 reach 9× crit and admit one shell (unchanged, bit-identical), 003
reaches 104× and admits two. **That 104× is GT `Mat`; the ODE reaches 17.19×, against the 39×
shell 2 requires, so the ODE admits no second shell on any of the three** — the 0.6329 above was
a subtractive bug, not a gain (§16.4). Separately, of 003's 206 GT lumen nodes beyond the
shipped shell, **123 sit more than 14 hops from the wound** — that clot does not radiate from
the injury. §11.3's "the blocker is the wall model, not the wound model" is now confirmed
geometrically as well as from timing.

Full write-up: [MODEL_REVIEW_2026-08-22.md](MODEL_REVIEW_2026-08-22.md) §9c and §9d.

**Amended 2026-08-24 (§16.4).** `lumen` is now an artifact field (`wound.lumen`) rather than a
call-site argument, still defaulting to `shell`; until then the deploy dispatcher never passed
the argument at all, so no artifact could have run anything else. §16.5 also corrects the last
paragraph above: 003's far-field clot is not out of reach of the shell rule — 116 of its 147
far-field GT nodes are in shell 1 — it is *unranked*, because the ODE's `Mat` orders those
candidates at chance.

---

## 13. HOW A WOUND MUST BE SCORED — the first metric was degenerate

`mask_wound` (COMSOL `sel1`) is **100% GT clot on every vessel**.  A deploy score restricted
to it is therefore 1.0 for any model that commits the patch, and the ungated law does that
with nothing fitted.  §10.3 and §12.2 originally led with that number.  **It measures
coverage, not skill, and it should never have been the headline.**

Worse, it hid the thing that actually matters.  The wound's thrombus grows *into the lumen* —
on `wound_patient001`, 82 of the 162 GT clot nodes within 4 hops of the wound are off the
boundary — and those were being scored in the global off-wall domain, pooled with clot grown
from healthy wall elsewhere in the vessel.  Neither domain answered "did we get the wound's
thrombus".

`wound_region_masks` (`src/clot_ml/wound.py`) defines the domains that do:

| domain | definition | GT+ rate (001 / 002 / 003) |
|---|---|---|
| `wnd` | the wound boundary, `sel1` | 1.00 / 1.00 / 1.00 — **degenerate, coverage only** |
| `w_reg` | every node within `WOUND_REGION_HOPS = 8` of the wound | 0.19 / 0.19 / 0.33 |
| `w_lum` | the **lumen** subset of `w_reg` | 0.10 / 0.10 / 0.25 |
| `far` | off-boundary and *beyond* 8 hops — clot the wound did not cause | — |

8 hops is four corner shells (the packs are quadratic, so one shell is *two* hops —
PHASE7_FINDINGS §8).  It is set to contain the whole GT thrombus with margin rather than clip
it: on 001/002 the GT clot count saturates by hop 4 at 162 nodes and hop 8 adds only true
negatives; on 003 it reaches 148 of 176.  The radius is chosen from the *geometry*, never from
where the prediction happens to be.

Two findings fall straight out of the split:

* **On 001 and 002, every off-boundary GT clot node is wound-caused.**  `far` has no GT
  positives at all on those two vessels, so it scores `nan`; the 0.2614 in §12.2 is
  `wound_patient003` alone.  The old pooled `off` column was measuring the wound on two of
  three vessels without saying so.
* **`far` is identical across every arm**, which is the second regression check: the
  complement cannot reach clot it did not cause.

---

## 14. `wound_patient003` RE-DIAGNOSED — §11 was wrong about the mechanism (2026-08-23)

Measured on `clot_gnn_v5` + the (unchanged) wound complement, which reproduces
**w_reg 0.6931 / w_lum 0.6151** on this vessel against 0.970 / 0.971 on 001/002.

**It is a RECALL failure, not the timing failure §11 diagnosed.** Precision 0.984 / 0.967,
recall 0.412 / 0.302: 67 of 96 region-lumen GT clot nodes are never committed *at final
time*, where timing cannot be the cause. The misses are structured by distance — 6/20 missed
at hops 0–2, 21/36 at 3–4, **27/27 at 5–6, 13/13 at 7–8**.

### 14.1 Four mechanisms measured and RULED OUT — do not retry these

1. **§11's neighbour trigger cannot help, and the reason invalidates §11.2–11.4.** At the
   near-wound wall the gate is **already 1.0 at t=0 and stays 1.0 under the GT flow oracle at
   every timestep**. There is no gate to open. That is why `trigger="wall"` is inert; and the
   `gate_scale=20` "fudge" of §11.4 never opened a gate either — it multiplied a *rate*. The
   deficit there is 34× in the deposition rate at a gate that is already saturated.
2. **The v5 GNN as an alternative trigger source.** Its own wall onset at those nodes is step
   **52** against GT step **2** — it inherits the ODE's clock through `oon` /
   `ode_wall_series`, so it is no earlier than the physics it is anchored on.
3. **Un-freezing RP/AP.** `wound_patient003` is the only vessel in the dataset with live
   near-wall platelet activation (AP 11–12× at the wound, RP down to 0.14); **all 31 cohort
   vessels and wound 001/002 sit at exactly 1.000**. But feeding GT species makes the *wall*
   strictly worse everywhere (003 171→102 ignitions, 001 55→47, patient044 61→50) because GT
   RP collapses and the wall loses its source. And the only vessel-level selector the code
   nominates — `wall_gate_frac_vessel` — is **falsified**: `patient001` (0.360), `patient014`
   (0.344) and `patient032` (0.322) are as gated as 003 (0.352) with AP amp exactly 1.000.
4. **`graded_gate` as the t=0 surrogate for the evolving gate.** 0/15 on the blind owners,
   bit-identical numbers in every mode — grading a margin cannot reach a node five times
   above the threshold.

### 14.2 The two causes that remain, and both route to the wall

- **29 misses are owned by 15 healthy-wall nodes with `gate = 0` at t=0** (`sr` 117 /s
  against `lss` 25). No source term exists there at all. The GT-evolving gate opens on them
  (0% at step 0, 47% by step 20 — their GT onset — 100% by the end) and ignites 15/15, taking
  the whole wall 171/254 → **227/254 at 1 false positive**. That is the size of the prize and
  it is a *flow* prize.
- **38 misses are "wound-owned" only by geometry.** Their GT onset is step **77**, against
  the wound's 5.5 and the near-wound wall's 20, and it climbs monotonically with distance
  (8.5 / 44 / 76 / 126 at hops 1–2 / 3–4 / 5–6 / 7–8). The thrombus is a **growth front**,
  not an attenuated shell — which the shipped `0.16^k · Mat >= crit` rule cannot express at
  all: it needs 39× crit for shell 2 and 244× for shell 3, so **even GT's 104× stops at two
  shells**. The normalised front speed is consistent across the three vessels (7.75 / 6.67 /
  9.59 % of horizon per hop). But GT clot at depth is **sparse, not a full shell** — 8%
  positive at hops 7–8, 24% at 5–6 — so committing shells at depth would trade 27 true
  positives for 87 false ones and destroy the 0.98 precision. Per-node commitment at depth is
  v5's own off-wall path, and that path is blocked by the same 15 owners.

**So both halves route to one root: the wall model cannot ignite nodes whose shear collapses
for non-local reasons. The wound complement is not the lever on this vessel** — which is
§11.5's conclusion, reached by a different and more direct route.

### 14.3 Why no local rule closes it — measured, not asserted

[`src/core_physics/gelation_wake.py`](../src/core_physics/gelation_wake.py) implements the
closed loop §3.3 and PHASE7 §9 both flagged: `Mat >= crit -> sr collapses -> low-shear gate
opens`, from a kernel measured on GT over eleven vessels. **It does not reach the blind
owners, and the measurement says no local kernel can**: the wake load at those nodes is
**0.00 at every step of the model rollout**, and in GT their shear has already fallen 32%
*while zero gelled wall sits within 12 hops*. The collapse there **leads** the local clot
load rather than following it — it is upstream flow reorganisation, which is the same wall
`MODEL_REVIEW` §9e hit when the corrector produced −3.5% against a required −87%.

### 14.4 What did come out of it — the gelation wake is a real wall-physics gain

Measured on the ODE trajectory directly, wall domain, 26 clot-carrying vessels:

| | final | mean-over-time |
|---|---|---|
| shipped frozen t=0 gate | 0.7966 | 0.7635 |
| **+ gelation wake** | **0.8234** | **0.7778** |
| | **+0.0268** | **+0.0143** |

Better on 18 vessels, worse on 7, unchanged on 1; **zero false positives on the clot-free
vessels**, and bit-identical on any vessel that never ignites. The final gain clears the
±0.024 wall noise floor; the mean-over-time gain does not.

**It is OFF by default.** Through the full deploy path with `clot_gnn_v5`'s head unchanged,
every final score is **bit-identical** — the head's committed set dominates and the ODE only
supplies timing features, so the gain is absorbed. `ode_trajectory(..., wake=True)` and
`promote_clot_gnn_v4_temporal.py --wake` exist for the generation that would bank it, and the
flag travels on the artifact (`temporal["wake_ode"]`) so train and deploy cannot diverge.
`wake=False` reproduces the shipped trajectory bit-for-bit, pinned by
[`src/tests/test_gelation_wake.py`](../src/tests/test_gelation_wake.py).

### 14.5 The retrain was done, and it does NOT bank the gain

`clot_gnn_v5wake` (same v5 GNN members, temporal head refitted against the wake-enabled
clock; `outputs/clot_ml/locked/clot_gnn_v5wake`, **not promoted**). The head does learn a
different readout — cuts 0.55/0.25 against 0.45/0.15 and `burden_gate` 25 against 0 — so the
clock genuinely changed what it fits. `scripts/eval_wound_complement.py`, cohort mean n=3:

| base | wall FIN | wall MOT | **w_reg FIN** | **w_lum FIN** | full FIN |
|---|---|---|---|---|---|
| `clot_gnn_v5` | 0.7330 | 0.6217 | **0.8777** | **0.8522** | 0.7963 |
| `clot_gnn_v5wake` | 0.7429 | 0.6343 | **0.8777** | **0.8522** | 0.7997 |

**The wound region does not move at all**, on any vessel — `wound_patient003` stays at
w_reg 0.6931 / w_lum 0.6151 — and the wall gain (+0.0099 final) is inside the ±0.024 noise
floor at n=3. In-sample on the 31-vessel fit pool the wake head is slightly *worse*
mean-over-time (wall 0.9693 → 0.9631, off 0.8761 → 0.8678).

So the ODE-level +0.0268 is real and the deploy-level gain is not demonstrated. Do not ship
`clot_gnn_v5wake` on this evidence; the honest next step is the strict 5-fold CV protocol on
the full cohort, which is where a wall gain of that size would either survive or not.


## 14.6 THE WOUND WAS NOT A `Mat` SOURCE — fixed 2026-08-23

The same defect as §6/§6b, one layer further up. Those sections gave the wound its
**geometry** and its **ownership**; nothing gave it its **deposition**. Every ODE call site
wrote `f0.gate * mask_wall` — a correct transcription of `srf1` and a wrong description of
the vessel:

| | `wound_patient001` | `002` | `003` |
|---|---|---|---|
| ODE `Mat` at the wound, max over all time | **0** | **0** | **0** |
| share of the vessel's total GT surface `Mat` the wound carries | **78.8%** | **87.6%** | **50.2%** |
| mesh whose nearest solid node is a wound node (`mat_owner_t` ≡ 0) | 16.4% | 14.8% | 5.6% |

So the largest `Mat` source in the vessel contributed **nothing** to `mat_phys`, to the
advective source, or to `mat_owner` — while being the one place clot is guaranteed.

**The fix** is one canonical helper,
[`physics_wall_model.deposition_gate`](../src/core_physics/physics_wall_model.py): gated
`srf1` on `mask_wall`, **ungated `srf2` on `mask_wound`** — prefactor a hard 1, not a fit
(§1). Threaded through `features.build_features` (`mat_phys`),
`features_v4.indicator_backbone` and `temporal.ode_trajectory`, and the transport source in
`locked.predict_temporal_v4` now spans the **solid** boundary rather than the healthy wall.
The law split §6b preserved still holds — this is not `gate * solid`; the wound gets `srf2`'s
value rather than `srf1`'s.

**Provably inert off a wound pack** — gate and trajectory are bit-identical on
`patient012/020/041/044`, pinned by `src/tests/test_gelation_wake.py`. Measured on the three
wound vessels: `clot_gnn_v5` **alone** goes w_reg 0.0604 → **0.1040** (003: 0.1811 → 0.3121),
and the composed model gains 003 w_reg 0.6931 → **0.7065**, wall 0.8243 → **0.8436**.

---

## 14.7 THE SCORE FIELD'S CALIBRATION COLLAPSES ON WOUND PACKS — the ranking does not

The readout's cuts are **absolute** (`resid`, 0.95/0.53/0.98/0.92) and they are only valid in
the regime they were fitted in. On a wound pack:

| | 001 | 002 | 003 |
|---|---|---|---|
| off-wall score p99 (cut is 0.92) | 0.531 | **0.001** | 0.374 |
| `w_lum` **AUC** | **0.9675** | **0.9536** | 0.5974 |
| `w_lum` shipped score | **0.0000** | **0.0000** | 0.0000 |
| `w_lum` oracle cut | 0.8935 | 0.8768 | 0.7068 |

**v5 ranks the wound thrombus almost perfectly on 001/002 and the cut throws all of it away.**
This is the [[clot cut passes through an empty band]] pattern in a new regime, and it is a
calibration failure, not a ranking one.

`expected_tuned` — rank the domain, commit the prefix that maximises the expected severity
score using the model's own `p`, **`kscale = 1.0` so the prefix is the raw argmax with no
fitted scale** and `gamma` on a plateau (2/3/5 give identical wall gains) — is applied
**only on packs carrying wound nodes**, from `clot_gnn_v5w`'s manifest. Through the shipped
dispatcher, wound mean n=3:

| readout | wall | w_reg | w_lum | far | full |
|---|---|---|---|---|---|
| absolute (as shipped) | 0.7394 | **0.8822** | **0.8522** | 0.2794 | 0.7976 |
| **wound-regime rank** | **0.8141** | 0.8723 | 0.8478 | **0.4929** | **0.8025** |
| | **+0.0747** | −0.0099 | −0.0044 | **+0.2135** | +0.0049 |

Wall and `far` clear their noise floors (±0.024 / ±0.091); the `w_reg`/`w_lum` losses do not.
Mean-over-time is slightly worse (full 0.7157 → 0.7021) — quoted because it disagrees.

**A global swap was measured and rejected**: it costs the 23-vessel cohort **−0.0222 wall /
−0.1616 off-wall** to buy this. The branch is therefore conditioned on the pack carrying wound
nodes, which makes it **structurally unreachable** on every cohort, clot-free and SEALED
vessel — `patient012` and `patient044` come out bit-identical, and a test pins that the spec
cannot fire without wound nodes.


## 14.8 TWO SWEEPS THAT DID NOT MOVE — recorded so they are not re-run

**The lumen schedule is already optimal on the deliverable.** The wound shell commits when
its owner reaches `crit / off_att` then waits `lag_frac`.  `frozen` beats the shipped
schedule on `w_lum` (mean-over-time 0.8184 against 0.5843), which looks like PHASE9 §12.4's
"every off-wall timing rule scores below frozen" recurring here — but committing earlier
trades `w_lum` against everything around it, monotonically:

| `off_att` / `lag_frac` | w_reg MOT | w_lum MOT | **full MOT** |
|---|---|---|---|
| **0.16 / 0.04 (shipped)** | **0.8133** | 0.7340 | **0.7204** |
| 0.16 / 0.00 | 0.8070 | 0.7433 | 0.7166 |
| 0.50 / 0.00 | 0.7458 | 0.7939 | 0.6797 |
| 1.00 / 0.00 | 0.7285 | **0.8042** | 0.6670 |

Final-time scores are identical across all of them — the set does not change. The shipped
constants are the best of the sweep on `full`, so the `w_lum` observation does not
generalise and nothing should be changed on it.

**The temporal time cut has no dominating value either.** Unlike the committed-set cut
(§14.7), sweeping `time_th_wall` on the wound packs trades final against mean-over-time with
no winner: 0.15–0.25 gives the best `full` FINAL (0.8063 against the shipped 0.8025) while
0.55 gives the best `full` mean-over-time (0.7127 against 0.7021). Both moves are inside the
±0.024 noise floor and picking either at n=3 would be a fit, not a finding. **Shipped 0.45
stands.**

The distinction worth keeping: §14.7's committed-set cut moved because the score field's
calibration genuinely collapses on wound packs (p99 0.001 against a 0.92 cut, AUC 0.95+).
The time cut does not show that signature, and it does not move.


## 16. THE OFF-WALL GAP IS THE ODE's `Mat`, AND THE MISSING INPUT IS CHEMISTRY (2026-08-24)

`wound_patient003` off-wall was the open deliverable: **wall 0.9120 / off 0.5293**, missing
169 of 243 GT lumen nodes. The hypothesis on the table was a **global flow stall** — the wound
is 100% GT clot from t=0, so a `blockage` callable modelling the flow reduction it causes would
drop `sr` at the distant clot station and let ordinary stagnation admission pick up the
far field. That hypothesis is now **falsified with a hard bound**, and the real cause is
located precisely. Reproduce: `scripts/diag_wound_gt_gate_ceiling.py`,
`diag_wound_offwall_depth.py`, `diag_wound_far_separator.py`, `diag_wound_ode_closure_cell.py`.

### 16.1 A total stall is a no-op, and the bound is exact

A stall is `sr → 0`, which pins `gate_from_shear` at exactly 1 (the stagnation branch) and
kills the separation branch. So **`gate ≡ 1` on the whole wall upper-bounds every
stall-shaped blockage that could ever be written** — reduced-order, corrector, or exact — and
it can be integrated directly instead of argued about. On `wound_patient003`, wall `Mat`/crit
at p90:

| arm | wall p90 | wall ignitions | off-wall TP |
|---|---|---|---|
| frozen t=0 gate (shipped) | 1.73 | 171 | 0 |
| GT-evolving gate (flow ORACLE) | 1.91 | 211 | 0 |
| **`gate ≡ 1` (TOTAL stall — the bound)** | **2.31** | 533 | **0** |
| `gate ≡ 10` (unphysical, 4× any real shear field) | 20.76 | 545 | 33 at 575 FP |
| GT `Mat` | **19.45** | 515 | — |

The off-wall rule needs `Mat_owner >= crit / off_att` = **6.25× crit**. A total stall reaches
2.31× and commits **zero** off-wall nodes. It is not close and it is not tunable — the only arm
that reaches GT magnitude is one no shear field can produce, and it destroys the ordering
(far-field AUC 0.84 → 0.33) because a uniform gate scales a near-constant.

The gate was already open where it mattered: the healthy-wall owners of 003's missed lumen sit
at **`sr` = 7.9 /s against `lss` = 25**, median gate **1.000**. There was nothing for a stall
to open.

### 16.2 The architecture is fine; one component is broken

Holding the shell, the topological owner map and the 0.16 constant **fixed** and changing only
which `Mat` feeds them:

| off-wall rule | 001 | 002 | 003 |
|---|---|---|---|
| shipped `clot_gnn_v5w` | 0.4755 | 0.6736 | 0.5293 |
| `shell & 0.16·Mat_ODE >= crit` | 0.0000 | 0.0000 | 0.0000 |
| **`shell & 0.16·Mat_GT >= crit`** | **0.9755** | **0.9755** | **0.7897** |

The shipped magnitude rule fires on **nothing, on all three vessels** — it has been
contributing zero the whole time. Fed GT `Mat` the same rule clears the 0.75 target on 003 and
takes 001/002 to 0.9755 (2 and 1 false positives, down from 87 and 39). **The off-wall
architecture is correct and the ODE's `Mat` is the single broken component.**

On 003's far-field candidates GT `Mat_owner` separates clot from lumen at **AUC 0.9961**
(median 16× crit for GT+ against 1.8× for GT−, straddling the 6.25× bar exactly as designed).
The ODE's `Mat_owner` is at **chance, AUC 0.5048** — its two medians are equal to three
significant figures. Every other candidate signal is also at chance: GNN score on the node
0.5654, on the owner 0.5265, local committed-wall density 0.3606.

It has lost **dynamic range**, not scale: across committed wall nodes the ODE spans 1.29×
p90/p50 where GT spans 7.96×.

### 16.3 The missing input is chemistry, not flow

PHASE7 §9.3–9.5 explains the mechanism — with the gate and `ap`/`rp` frozen at t=0 the source
is constant, `mas` saturates, and every gated node integrates the same trajectory — and
measured that the removal term alone makes the shipped model *worse* (`rho_corner` 0.482 →
0.084), because a constant source against a linear sink has one attractor whose ordering is the
`1/sr` null. MODEL_REVIEW §2.2 flagged the evolving-flow × evolving-chemistry × washout cell as
**the only physics route in the repo with a measured mechanism, a measured magnitude and no
deploy number**, and §2.4 said to run it on the wound vessels first. Run, on 003's far-field
candidates (GT flow and GT chemistry as ORACLES, not a deploy path):

| gate | chemistry | washout | wall p90 | p90/p50 | **far AUC** | off @ recal. |
|---|---|---|---|---|---|---|
| frozen | frozen | off | 1.73 | 1.29 | **0.5048** | 0.5037 |
| GT-evolving | frozen | off | 1.91 | 1.28 | 0.5541 | 0.5940 |
| frozen | **GT** | off | 2.22 | 4.64 | **0.8819** | 0.6934 |
| GT-evolving | **GT** | off | 3.78 | 6.34 | **0.9642** | **0.7930** |
| GT-evolving | **GT** | **on** | 3.75 | 6.60 | **0.9662** | 0.7845 |
| GT `Mat` | — | — | 19.45 | 7.96 | 0.9961 | 0.8364 |

**This inverts the hypothesis.** Evolving the flow into the gate is worth **+0.05** of the
far-field AUC; evolving the chemistry is worth **+0.38**; together they reach 0.9642 against
GT's 0.9961 ceiling. Washout adds a consistent but small **+0.02–0.03** and only in the
presence of both, exactly as §9.4 predicted — it is real, it is third in line, and it still
must not ship alone.

Two failures remain separable, and this is the actionable part:

* **Ordering** is nearly solved by evolving chemistry (0.5048 → 0.9662).
* **Calibration** is not: wall p90 reaches 3.78× crit against GT's 19.45×, still ~5× low, so
  at the shipped 6.25× bar the arm collects 64 of the 140 nodes GT collects. Recalibrating the
  bar recovers **off 0.7930** — above target. This is PHASE7 §7.2's warning in its sharpest
  form: the score is a threshold crossing, not a ranking, so ordering gains are gated on
  calibration. `att* = 0.784` is a fit on **one vessel** and is a diagnostic, not a constant.

### 16.4 What shipped: a wiring gap and a subtractive bug, and `recursive` buys nothing

**The depth rule now travels on the artifact** (`wound.lumen`, read in
`src/clot_ml/locked.py`), defaulting to `shell`. `predict_wound_series` had implemented four
modes since §12b, but the deploy dispatcher **never passed the argument**, so no artifact could
ever have run anything but one corner shell regardless of what was promoted. Artifacts predating
today lack the key and are bit-identical (`test_dispatcher_lumen_defaults_to_the_shipped_single_shell`).

**`recursive` was subtractive, and its recorded gain was that bug.** Widening `owned_off` to a
deeper ring also widens what `compose_with_v4` overrides — it applies
`mask[owned] = wound_out["mask"][owned]`, so a deep node the wound module declines is *removed*
from v4's set. Measured on 003: `recursive` removed **2 committed nodes and added none**. Both
happened to be false positives, so `off` went **up** 0.0034 and `w_lum` up 0.0311 and the defect
was invisible in every score anyone had looked at. Only a set comparison finds it. Deeper rings
now claim only the nodes they actually commit, which makes "strictly additive" true of the
committed **set** rather than of the ownership map; shell 1 is untouched, so `shell` stays
bit-for-bit. Pinned by `test_dispatcher_recursive_never_removes_a_committed_node`.

**With the bug fixed, `recursive` is exactly inert on all three vessels** — 0 nodes differ, not
2. So it has **no measured gain**, and the reason is §16.2 again: shell 2 needs
`Mat_wound >= crit / 0.16²` = **39× crit**, and the ODE's wound reaches **17.19×** even with the
two-regime rate of §15. §12b's "003 reaches 104× and admits two" was read off **GT** `Mat`
(103.84×), not the ODE's. The same magnitude deficit that starves the far field starves the
depth rule. The wiring and the additivity fix are worth keeping; the mode is not worth
promoting until `Mat` is fixed.

**The target is not met by anything deploy-legal today.** 003 stands at **off 0.5293**,
unchanged. It is met at **0.7930** by the oracle cell of §16.3 — which is the point of running
it: the build is now specified rather than guessed.

### 16.5 The recall ceiling nobody had measured

003's off-wall GT clot is **three layers deep** — 161 nodes in the first species row, 59 in the
second, 11 in the third, 12 in the empty bridge band — while 001/002 are **exactly one layer**.
Every off-wall arm in `physics_lumen_model` predicts inside `first_corner_shell` only, so:

| oracle | 001 | 002 | 003 |
|---|---|---|---|
| perfect inside shell 1 | 1.0000 | 1.0000 | **0.8667** (recall 0.663) |
| perfect inside shells 1–2 | 1.0000 | 1.0000 | 0.9599 |
| perfect inside shells 1–3 | 1.0000 | 1.0000 | 0.9763 |

So **0.75 is reachable without leaving the one-shell architecture**, but only just, and the
GT-`Mat` oracle's 0.7897 sits between the two. Iterating the topological shell outward does
recover the deeper layers, so the multi-layer option is available if 0.8667 ever binds. This
supersedes the framing in §12b's last paragraph: the far-field clot is not unreachable, it is
*unranked*.

---

## 17. v6 — THE `Mat` FIELD IS LEARNABLE, AND 003 IS OUT OF DISTRIBUTION (2026-08-24)

§16 ended by saying the off-wall architecture is sound and the ODE's `Mat` is the one broken
component. v6 tests that directly: hold the shell, the topological owner map and the
attenuation **fixed**, and replace only the field. Code: `src/clot_ml/mat_field.py`,
`scripts/go_mat_field_v6.py`, pinned by `src/tests/test_mat_field.py`. The shipped stack is
untouched — v6 is a new module and all 35 wound/near-stall tests still pass.

### 17.1 Two architectural facts found before any model was trained

**(a) The off-wall readout must be REPLACED, not unioned.** §16.2's headline "GT `Mat` scores
0.7897 on 003" was measured with `base & solid`, which *discards* v4w's own off-wall verdict —
so it was a replacement, and nobody had noticed. Union it back in and a **perfect** field
scores **0.6558**. The shipped readout carries off-wall false positives that a correct field
cannot undo, because the union can only add.

**(b) Depth is worth more than the constant.** Shells off the whole solid boundary — not off
the wound, since 003's misses sit ~14 hops away beside the *healthy* wall — with an oracle
field, `--combine replace`:

| arm | 001 | 002 | 003 |
|---|---|---|---|
| shipped v4w off-wall readout | 0.4755 | 0.6636 | 0.5343 |
| oracle, att 0.16, shell 1 (= §16.2) | 0.9755 | 0.9755 | 0.7897 |
| oracle, att 0.16, shells 1–3 | 0.9755 | 0.9755 | **0.8799** |
| oracle, att 0.23, shells 1–3 | 0.9578 | 0.9578 | **0.9240** |

So the reachable ceiling on 003 is **0.9240**, not 0.79. §16.5's 0.8667 was a shell-1 bound
and is superseded for the multi-shell rule.

**(c) Replacement is cohort-neutral.** Ten non-wound clot-carrying vessels, oracle field,
mean off-wall **0.8697 → 0.8789** — inside the ±0.091 off-wall noise floor, with large rescues
where the shipped readout is weak (`patient020` 0.4911 → 0.8122, `patient032` 0.8117 → 0.9667)
and modest give-back where it was already strong (`patient040` 0.9500 → 0.7333). The change is
not wound-specific and does not have to be scoped to wound packs.

### 17.2 The field IS learnable — held-out `wound_patient001` reaches 0.9489

`ClotGNN` unchanged, time-varying channels through its existing `extra` port, the ODE's own
`Mat(t)` as the regression head's zero-init residual base (so an untrained v6 *is* the
physics). Trained on 43 vessels with `patient032/012/041` and `wound_patient001` **held out**
alongside 003 and SEALED:

| vessel | held out | owner AUC | pred `Mat` p90 | GT p90 | off (v6) | off (oracle) | off (shipped) |
|---|---|---|---|---|---|---|---|
| `wound_patient001` | yes | 0.9944 | 9.84 | 8.49 | **0.9489** | 0.9755 | 0.4755 |
| `patient041` | yes | 0.9721 | 5.72 | 6.16 | **0.8383** | 0.8944 | 0.9603 |
| `patient012` | yes | 0.8852 | 1.95 | 6.62 | 0.7689 | 0.8910 | 0.9125 |
| `patient032` | yes | 0.9398 | 2.05 | 11.13 | 0.7316 | 0.9667 | 0.8117 |
| `wound_patient003` | yes | 0.7924 | **2.33** | **27.78** | 0.5255 | 0.9240 | 0.5343 |

**A held-out wound vessel doubles its off-wall score, 0.4755 → 0.9489.** The approach works.
This is the first evidence in this document that anything recovers the far field.

### 17.3 The failure mode is magnitude compression, and on 003 it is OOD

Two distinct failures, and they must not be conflated:

* **Compression (fixable).** `patient032` orders its candidates at AUC 0.9398 but predicts
  p90 2.05× crit where GT is 11.13×. Ordering is good, scale is shrunk toward the cohort
  prior — 30 of 47 training vessels have wall `Mat` p90 at or below 1.01× crit. Lowering the
  bar recovers most of it (032 0.5119 → 0.7316, 012 0.6222 → 0.7689). Re-weighting (pos_weight
  3 → 12, magnitude-weighted regression, clot-free vessels down-weighted to 0.25) moved 003's
  AUC 0.7624 → 0.7924, so the lever is real but partial.
* **Out of distribution (not fixable by tuning).** `wound_patient003`'s wall `Mat` p90 is
  **27.78× crit, the largest in the entire 52-pack dataset**; no non-wound vessel exceeds
  11.13× (`scripts/diag_mat_magnitude_cohort.py`). Its predicted p90 came back at 2.33×,
  essentially the ODE's own 1.96× — **the residual collapsed onto the physics base rather than
  mislearning**, which is exactly what a zero-init residual does off-distribution. Its
  ordering therefore inherits the ODE's, which §16.2 measured at chance on the far field.

**No threshold recovers 003.** Swept from 0.33× to 12× crit, at depths 1–3, the score plateaus
at 0.42–0.53 everywhere — flat, not peaked. That is the signature of bad ordering, not bad
scale, and it is why 003 is a different problem from 032.

### 17.4 Where this leaves the target

Target was >0.75 on both domains for `wound_patient003`.

| domain | shipped | v6 | target |
|---|---|---|---|
| wall | **0.9135** | 0.9135 (untouched by construction) | met |
| off | 0.5343 | 0.5255 | **not met** |

The wall was already met. Off-wall is **not** met on 003 and was not moved. What changed is
that the reason is now a measured quantity rather than a hypothesis: the ceiling is 0.9240,
the machinery reaches it on a held-out wound vessel, and the gap on 003 is that its `Mat`
field is 2.5× outside the dataset's range so the learned residual declines to predict it.
§16.3 already named the input that distinguishes it — **evolving chemistry, +0.38 far-field
AUC against the evolving gate's +0.05** — and §14.1(3) records that 003 is the only vessel in
the dataset exhibiting near-wall platelet activation, i.e. there is exactly one example of the
mechanism and it is the test vessel. That is not a training-recipe problem.

Replicated on the deployable configuration (seed 2, 80 epochs, only 003 and SEALED held out):
`wound_patient001`/`002` land **exactly** on their 0.9755 oracle — but they are in that
training set, so that is memorisation and not a result — while held-out 003 reads 0.4055 with
predicted `Mat` p90 2.44x against GT 27.78x. Three seeds and two training-set compositions all
put 003's predicted p90 at 2.0-2.4x, i.e. on the ODE's own 1.96x. The collapse is a property of
the vessel, not of a seed or a recipe.

Do not re-run: the low-bar sweep (flat), union-vs-replace (replace wins, measured above),
recursive depth beyond 3 (nothing clears the bar), `pos_weight`/magnitude re-weighting (worth
+0.03 AUC on 003, banked), and more seeds or epochs (replicated above).

---

## 18. CHEMISTRY IS SUFFICIENT, A SPECIES GNN IS NOT THE FIRST BUILD (2026-08-25)

v6 failed on 003 because its `Mat` is out of distribution. The next hypothesis was that
evolving `AP`/`RP` would restore the ODE's dynamic range, and that a GNN could supply those
fields at deploy. Both halves were measured (`scripts/diag_species_ood.py`,
`diag_chem_oracle_v6.py`, `diag_chem_oracle_cal.py`) before any new model was written.

### 18.1 `AP` is out of distribution in the same way `Mat` is

Final-frame AP on solid nodes, as a depletion ratio against t=0 (t=0 AP is spatially flat,
CV 0.0000 on every pack):

| | AP p10 (final/t0) | AP CV at T | Mat p90 / crit |
|---|---|---|---|
| `wound_patient003` | **0.189** | **1.230** | 27.78 |
| next most depleted non-wound | 0.256 (`patient043`, SEALED) | 0.309 (`patient043`) | 11.13 (`patient032`) |
| `wound_patient001` / `002` | 0.773 / 0.726 | 0.090 / 0.204 | 8.49 / 8.41 |

Zero non-wound vessels match 003's depletion or its spatial contrast. A species GNN trained
on the legal 47-pack set would be asked to extrapolate on the test vessel — the same trap
v6 fell into, one field upstream.

### 18.2 GT chemistry through v6's readout, and the rate scalar that actually closes 0.75

§16.3's 0.7930 was UNION + a per-vessel `att*` fit. The number that matters is the same
chemistry-driven ODE `Mat` through **replace + depth**, which is what v6 uses:

| arm on `wound_patient003` | solid p90 | far AUC | att 0.23 d3 (shipped family) |
|---|---|---|---|
| ODE frozen (t=0 AP × closure) | 1.96 | 0.80 | 0.4505 |
| ODE GT-chem | 3.57 | 0.877 | 0.5670 |
| ODE GT-chem + gate + wash | 6.79 | 0.965 | **0.6624** |
| same, `da_scale_auto=123` (COMSOL's 3.07× ratio) | **20.22** | 0.966 | **0.8512** |
| GT `Mat` ceiling | 27.78 | 0.998 | 0.9240 |

**Target is met at 0.8512**, at the shipped attenuation, with a rate scalar taken from 19
TRAIN vessels' own Damköhler split — not fitted on 003. Ordering was already solved by
chemistry (0.965); calibration was the remaining 4× magnitude hole, exactly as §16.3 said.

Frozen chemistry cannot be scaled into a solution: sweeping `da_scale_auto` to 400 lifts
p90 to 16.5× and the score plateaus at **0.6125**, because the *pattern* of t=0 AP is wrong
(far AUC stuck at 0.86). Amplitude is not the missing piece; the time-varying AP field is.

`AP_owner` as a raw ranker is only AUC 0.775 against GT `Mat_owner` at 0.998, and `RP_owner`
is anti-correlated (0.035). A species model must **feed the ODE**, not replace the readout.

On `wound_patient001`/`002`, replace+depth with the *frozen* wound-rate ODE already scores
**0.9578**. Those vessels do not need chemistry; they needed the §17.1 readout.

### 18.3 What to build, in order

The deploy stack is not another `Mat` GNN. It is:

1. **A time-varying wall AP**, residual on the existing Damköhler closure (so untrained =
   today's frozen arm). The missing physics is *upstream renewal*: `ap_closure.py` already
   measured that isotropic smoothing makes R² worse and that the residual is advective.
   First attempt is a wall AP ODE with the written sink plus upwind graph transport along
   flow — not a GNN. If that field's far-field AUC on 003 is still ~0.86, then a
   `ClotGNN` residual on *that* AP (same extra-port pattern as v6, target = COMSOL
   `AP_log1p_nd`) is the fallback. Do not train a species GNN against raw AP as the first
   move; 003's AP contrast is 4× the in-distribution max.
2. **ODE** `integrate_mat_trajectory(species=AP(t), washout=on, da_scale_auto=123,
   blockage=wound_rate + flow gate)`. `da_scale_auto=123` is COMSOL's ratio, not an 003
   fit. The evolving gate is +0.05 AUC and should use the kinematic corrector at deploy,
   GT flow only as the oracle bound already measured.
3. **Readout** replace+depth (`att=0.23`, depth 3, shells off the whole solid boundary).
   Do not union with v4w's off-wall verdict. Wall stays v4w (already 0.91).

Do not re-run: AP OOD scan, chemistry-oracle replace+depth, `da_scale_auto` sweep on frozen
vs GT chemistry (frozen cannot cross 0.61). Do not retrain v6 against the shipped ODE; its
residual has nothing to correct until the ODE sees evolving AP.

---

## 19. `clot_ml_v0` — one artifact for wounded and non-wounded vessels (2026-08-25)

The experiments in §16–18 were separate arms: a C0 GNN (`clot_gnn_v5` / `v5w`), a learned
`Mat` field (v6), upwind AP renewal, and a replace+depth readout.  Two classes of model
(`clot_gnn_v5` vs `v5w`) is the wrong packaging — a wound mask is an input, not a product
line.  `clot_ml_v0` (`kind: unified_v0`) is the composition §18.3 specified:

| piece | what ships | why |
|---|---|---|
| wall SET + non-wound off-wall | C0 GNN (`--base clot_gnn_v5w`) | already 0.9203 / 0.7078; ODE Mat through the same rule scores ~0.40 |
| wound boundary | two-regime `(G_pre, G_post)` on that base | structural no-op without a mask |
| wound off-wall | **REPLACE** (not union) with chemistry-ODE `Mat` through solid-anchored replace+depth, `att=0.23`, depth 3 | union of a perfect field with the GNN off-wall verdict scores 0.6558 where the field alone scores 0.7897 |
| chemistry ODE | upwind AP renewal + `da_scale_auto=123` + washout + wound-rate blockage | GT-chem oracle 0.8512 on 003; this is the deploy-legal stand-in |
| 003-like residual | optional v7 ClotGNN on AP; missing checkpoint = physics | untrained residual is the upwind field; train when more 003-like vessels exist |

The GNN's temporal ODE is **not** replaced — the chemistry integration is a second field
used only for wound off-wall.  On any pack without a wound mask the predictor is
bit-identical to the base GNN, asserted at promotion (`scripts/promote_clot_ml_v0.py`) and
pinned by `src/tests/test_clot_ml_v0.py`.

Compare against the pinned baseline, never the locked pointer:

```
python scripts/promote_clot_ml_v0.py --base clot_gnn_v5w
python scripts/eval_clot_ml_v0.py --baseline clot_gnn_v5w
```

Do **not** `--repoint`.  The comparison against `clot_gnn_v5w` (`scripts/eval_clot_ml_v0.py`,
GT t=0 flow, `--every 4`) is:

| | wall | w_reg | w_lum | far |
|---|---|---|---|---|
| **001** v5w → v0 | 0.8858 = | 0.9802 → **0.8517** | 0.9708 → **0.8251** | n/a |
| **002** v5w → v0 | 0.8408 = | 0.9802 → **0.8340** | 0.9708 → **0.8070** | n/a |
| **003** v5w → v0 | 0.9135 = | 0.7297 → **0.8420** | 0.6018 → **0.7819** | 0.4929 → **0.3048** |
| patient012 / 032 | identical (bit-identical by construction) | | | |

Wall is untouched.  Chemistry replace+depth **helps 003's wound region/lumen** and **loses
on 001/002**, where the v5w complement already scores 0.97, and it also loses 003's far
field.  Mean w_reg 0.897 → 0.843; mean w_lum 0.848 → 0.805.  Deploy-legal upwind AP is not
the GT-chem oracle (0.8512 on 003).  The AP residual hook stays in the artifact for when
more 003-like vessels exist.  v6 is not in this artifact.

---

## Next

1. ~~**Wall AP with upwind renewal, then the ODE, then replace+depth.**~~ **Assembled as
   `clot_ml_v0`** (§19) and **measured against v5w**.  Do not `--repoint`: replace+depth
   helps 003 w_reg/w_lum and loses 001/002 plus 003 far.  Deploy-legal renewal is not the
   0.8512 GT-chem oracle.  The AP residual GNN waits for more 003-like vessels.
2. ~~**Ship §17.1's replacement + depth independently of 003.**~~ In `clot_ml_v0`, scoped
   to wound packs so the cohort GNN off-wall (0.7078) is not replaced by an ODE field.
3. **Fix magnitude compression directly.** Ordering is already good on the compressed vessels
   (032 AUC 0.9398, 012 0.8852) and only the scale is wrong, so this is worth ~0.2 off-wall on
   two of five held-out vessels. A per-bar ordinal head is the obvious form — the classifier
   head is calibrated at the one bar it was trained on and beat the regression head's AUC on
   every vessel measured.
5. ~~Then calibration, not more ordering~~ — **done in §18.** `da_scale_auto=123` (COMSOL's
   3.07× split, not an 003 fit) takes GT-chem + gate + wash from 0.6624 to **0.8512** at
   att 0.23 d3. Frozen chemistry still cannot be scaled there (ceiling 0.6125). Do not
   fit the attenuation on n=3; the shipped-family number already clears 0.75.
3. ~~The blocker is a flow solve~~ — **superseded by §16.1/§16.3.** The flow route is bounded:
   a *total* stall moves wall `Mat` p90 from 1.73 to 2.31× crit against the 6.25× the off-wall
   rule needs, and the GT-evolving gate is worth +0.05 far-field AUC. Flow still pays on the
   *wall* set (§14, 171 → 227 ignitions) — but it will not open the lumen, and §14's "the
   blocker is a flow solve" should be read as scoped to the wall from here on.
4. **Retrain the temporal head against `ode_trajectory(..., wake=True)`** and re-promote. The
   wake is worth **+0.0268 final / +0.0143 mean-over-time** on the wall ODE over 26 vessels
   but is fully absorbed by v5's head, which was trained on the wake-free clock (§14.4).
   This is the cheapest unbanked gain in the stack.
5. **Replace the depth rule's form.** `0.16^k · Mat >= crit` is a magnitude threshold and the
   physics is a growth front at 7.75% of horizon per hop (§14.2) — the rule cannot reach past
   two shells even given perfect `Mat`. Needs per-node commitment, not shells, because GT
   clot at depth is sparse (8% positive at hops 7–8). §16.5 bounds what this is worth: shell 1
   alone caps 003 at 0.8667.
6. **More wound simulations.** Every remaining limit is an n=3 limit, and §14.1(3) makes it
   concrete: `wound_patient003` is the only vessel in the dataset that exhibits near-wall
   platelet activation, so there is nothing to fit a selector against and the one candidate
   feature is falsified.

7. Commission the paired A/B run (§7) — same `.nas`, wound and no-wound.
8. Re-measure §10.3 deploy-faithfully (predicted t=0 flow rather than GT t=0 flow).
