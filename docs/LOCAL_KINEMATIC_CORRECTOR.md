# Local Kinematic Corrector

> **STATUS: DELETED (2026-09-01)**
>
> The `CorrectorArm` and `predict_corrector` components were fully removed from `physics_wall_model.py`.
> The models described below failed to capture the non-local (elliptic) nature of flow diversion around a clot.
> See [The conclusion worth keeping](#the-conclusion-worth-keeping) below for details.
> An `oracle_blockage` remains for benchmarking the upper bound of any future flow-coupling research.
> This document is kept as the failure record only.

Optional Stage-A companion: a local k-hop GNN predicts velocity diversion `[dU, dV]` as a
**residual on frozen RGP-DEQ flow**. Instead of re-solving the global field when a clot
appears, the patch reroutes flow around clot nodes. Trained on synthetic COMSOL
"Patch Factory" residuals. Not required for the locked WC_v7 species baseline.

## Why
The deploy RGP-DEQ kine model is accurate on healthy hemodynamics but OOD on the extreme
`mu` spikes a clot imposes. Rather than retrain the global model, we learn a *local*
correction so flow reroutes over/around the clot, supplying the shear/stagnation structure
the biochem clot model needs.

## Data: Patch Factory (COMSOL `mph`)
- Generator: `src/data_gen/lib/patch_factory_comsol.py` (no Gmsh; mapped structured quad grid).
- One master template `local_kine_template.mph`: a flat 2000um x ~350um box, parametric
  continuous-viscosity clot (`Clot_Mask` Heaviside, high-viscosity porous zone -- never a
  hole), inlet linear shear `u = shear_rate*y`, no-slip bottom, prescribed freestream top
  (exact Couette so the analytical baseline is clean).
- Baseline subtracted analytically -> residual `dU = U - shear_rate*y`, `dV = V`.
- QC: `src/data_gen/lib/patch_factory_qc.py` (baseline purity, BCs, mass, SNR, clot slowdown)
  + default mesh-convergence check (re-solve at refined mapped mesh; `convergence_report.json`).
- Current cohort: 1000 patches, all passing QC; convergence rel L2 ~ 1e-14 (mesh-independent).

## Architecture: `LocalKinematicCorrector`
- `src/core_physics/coupled_shear_gnn.py`. 3x `GATv2Conv` (heads=4, hidden=64, concat=False)
  + 2-layer MLP readout. Attention is used so the model can learn the anisotropic diversion
  (flow reroutes over/around a clot far more than it reverses behind it).
- Readout init near-identity (gain 0.01) so an untrained model leaves the base flow intact.
- Input features (`in_channels=6`): `[dx, dy, dist_to_wall, u0, v0, delta_mu]`, where `dx,dy`
  are clot-COM-centered (translation invariant). Assembled by the single source of truth
  `assemble_local_corrector_features` (shared by train / live verify / deploy).
- ND convention: positions by length scale (`d_bar` on patient graphs, channel height `H` on
  patches), velocity by `PhysicsConfig.get_u_ref(H)`, viscosity by `mu_viscosity_nd_scale`.

## Stage-A lessons ported (2026-06-20)
From the Stage-A kinematics curriculum ([docs/KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md)):
- **Difficulty-weighted oversampling** (clinical-anchor-boost analog): train sampler weight
  `1 + hard_boost*difficulty`, where `difficulty = patch_difficulty(clot_mu, clot_w, clot_h)`
  in `[0,1]` (single source of truth in `patch_factory_comsol.py`). `--hard-boost` (default 3).
- **Curriculum ramp** (L0L1 -> L2-heavy analog): `--curriculum-frac F` ramps the boost 0->full
  over the first `F` of epochs (easy-first, then hard). Default 0 (constant boost).
- **Cosine LR** (Stage-A: scheduler helped, LBFGS hurt): on by default; `--no-cosine` to disable.
- **Stratified (per-bucket) eval**: `eval_local_corrector` prints energy-weighted relL2 by
  terciles of difficulty / clot_mu / clot_w / clot_h / occlusion / shear -> shows *where* the
  failure tail lives.
- **Harder data generation**: `patch_factory_comsol --hard-bias B` skews clot mu/width/height
  toward the hard end for a fresh cohort (over-sample the difficult corner). `B=0` = original.

## Train / eval / viz
- Train: `python -m src.training.train_local_kinematic_corrector --patch-dir data/processed/cfd_results_patch_factory --epochs 800 --batch-size 4 --stride 2 --device cuda --hard-boost 3 --curriculum-frac 0.3`
  - 5 GiB GPU: `batch-size 4 / stride 2` fits. Reports per-epoch val MSE_nd + val relL2 + lr.
  - Difficulty-weighted sampler + cosine LR on by default; `--hard-boost 0 --no-cosine` for the
    legacy uniform/constant-LR recipe.
  - Saves `outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth` (+ `_last`);
    checkpoint meta records `sampling` (hard_boost / curriculum_frac / cosine).
- Eval (held-out vs COMSOL truth): `python -m src.tools.eval_local_corrector --patch-dir data/processed/cfd_results_patch_factory --corrector outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth`
  - Global + per-sample relL2; truth/pred/error maps (best/median/worst) ->
    `outputs/reports/figures/kinematics/local_corrector_eval.png`.
- Live (overlay on RGP-DEQ, dummy clot on a patient graph): `python -m src.tools.verify_local_corrector_live --graph data/processed/graphs_biochem_anchors/patient007.pt --corrector .../local_kinematic_corrector_best.pth --num-hops 5 --clot-mu 3.0`
  - Panels: base | corrected | overlay (shared arrow scale) ->
    `outputs/reports/figures/kinematics/local_corrector_diversion.png`.

## Metric note
relL2 = `sqrt( sum||pred - truth||^2 / sum||truth||^2 )`, global over the split (energy
weighted). A per-sample mean would divide by the ~0 far-field norms; the global ratio is the
robust accuracy number. `max per-sample relL2 > 100%` means a patch where predicting the
correction is worse than predicting zero (a failure-tail flag).

## Run log
| date | config | data | val MSE_nd (best) | held-out relL2 (global / med / p90 / max) | note |
|------|--------|------|-------------------|-------------------------------------------|------|
| 2026-06-19 | 300 ep, bs4, stride2, hidden64, heads4, MSE | 1000 patches (900/100) | 3.00e-6 | 26.7% / 30.0% / 54.5% / 106.3% | First end-to-end. Healthy curve, still descending at 300 ep (undertrained). Heavy failure tail (max>100%) -> likely extreme clots (largest w/h, highest mu/shear). Live diversion smooth, no artifacts, max\|dUV_nd\|~0.31. |
| 2026-06-20 | 800 ep, bs4, stride2, hidden64, heads4, MSE | 1000 patches (900/100) | 1.30e-6 | 17.6% / 19.1% / 42.4% / 97.5% | Just more epochs (same arch/loss). Every metric improved: global 26.7->17.6%, median 30->19%, p90 54->42%, max 106->98% (tail no longer worse-than-zero). val relL2 still trending down at ep799 (noisy, val n=100) -> not fully plateaued; no overfit (val~train). Tail (p90/p95) now the bottleneck. |
| 2026-06-21 | 800 ep, bs4, stride2, hidden64, **relative loss** (floor 0.5*med, grad_clip 5), cosine, hard_boost 0 | 1000 patches (900/100) | 1.06e-6 | 15.9% / 16.5% / 28.9% / 65.5% | First stable relative-loss run (after fixing the 1e-12 -> median-energy floor; the naive version had collapsed, val relL2~100%). **Beats the MSE 800-ep baseline everywhere**: global 17.6->15.9%, median 19.1->16.5%, p90 42.4->28.9%, p95 55.9->39.5%, max 97.5->65.5%, val MSE 1.30e-6->1.06e-6. Smooth monotonic descent, cosine annealed LR->0, plateaued ~16% from ep600. Designed to help only the low-signal tail but improved the global metric too -> low-signal patches were dragging the whole model, not just their bucket. |

### Bucket verdict (relative loss vs MSE, low tercile -> the targets)
The stratified table **flattened**: `clot_h/H` now flat (15.7/15.6/16.5), `clot_mu` nearly flat.
Low-tercile drops: shear_rate 28.4->24.4, clot_h 25.1->17.3, difficulty 23.2->17.7,
occlusion 20.5->15.7, clot_mu 20.2->17.9, clot_w 20.5->18.1. The single remaining hot cell is
the **lowest-shear tercile (24.4%, [84-1613 1/s])** -- the intrinsically smallest-signal
patches (tiny `du_nd`; `u_ref` is geometry-only). Relative loss is now the recipe to beat.

## Tail diagnosis (2026-06-20, stratified eval on the 800-ep ckpt)
The failure tail is **the low-signal regime, not the hard clots.** Energy-weighted relL2 is
worst at the *low* end of every axis; `shear_rate` has the steepest gradient (28.4% low ->
15.1% high), then `clot_h` (25.1% -> 17.5%). The deployment-relevant big clots (high
mu/size/shear) are already the best-fit (~15-16%).

Mechanism: `get_u_ref(d_bar)` is **geometry-only** (no shear), so `du_nd = du_si/u_ref` scales
with shear / clot size. Plain MSE on `du_nd` is therefore magnitude-weighted -> it fits big
signals and neglects small ones -> high *relative* error on low-shear/small/thin clots.

Consequence: `--hard-boost` (oversample high mu/size) is the **wrong direction** here (those
are already best). The principled fix is the **relative loss** (`--loss relative`), which
normalizes each patch by its own target energy so all signal scales count equally.

## Where we are / next levers (priority)
At global relL2 **15.9%** (800 ep, relative loss), tail much tighter (p90 28.9%, max 65.5%).
Relative loss is the recipe to beat. The table is flat except one hot cell: **lowest-shear
tercile 24.4%**. Remaining levers (diminishing returns -- decide if <12% global is worth it):
1. **Lowest-shear cell** -- the only real outlier. Cheap A/B: `--rel-floor-frac 0.25` (push
   small-signal patches harder). Or oversample low-shear directly. Risk: the absolute error
   there is already tiny (deploy-irrelevant); may not be worth chasing.
2. **Capacity** -- hidden 64 is small; try `--hidden-dim 96/128` now that the loss is settled.
   Most likely lever for a uniform global drop.
3. **More data** -- 900 train patches is modest; a second 1000-patch cohort would help the
   tail percentiles (val n=100 is noisy).
4. **Hard data / `--hard-boost`** -- still not indicated (error is at the *low* end).
2. **Decide what matters**: if only the big-diversion clots matter for biochem coupling, the
   current model is already ~15-16% there and the tail is a low-impact metric artifact -> a
   signal-weighted acceptance metric may be more honest than raw relL2.
3. **Capacity** -- hidden 64 is small; try 96/128 once the loss is settled.
4. **Hard data / oversampling** -- only if a *future* diagnosis shows the hard corner
   regressing (`--hard-bias` data gen, `--hard-boost` sampler). Not indicated now.

Target: global relL2 <~12% with p95 well under 100% before wiring into the biochem deploy
rollout (`BiochemGNN.set_local_corrector` / `local_corrector_ckpt`).

## Deploy coupling (intercept the flow in the rollout loop)
`src/inference/corrector_coupling.py` is the single source of truth for dynamically bending
the frozen base flow around a growing clot before it is fed to the biochem model (Steps A-F):
- **A** base flow `[u0, v0]` from the frozen RGP-DEQ kine pass (cached per graph).
- **B** clot nodes = `delta_mu_si > BIOCHEM_CORRECTOR_MU_THRESH` (default 1e-3 Pa.s), where
  `delta_mu = mu_eff - mu_bulk_carreau` (clot elevation over the *clot-free Carreau bulk*, not
  over `mu_inf` -- see the 2026-06-20 confound fix; the bulk ref keeps Δμ~0 away from the clot,
  matching the corrector's training distribution).
- **C** `k_hop_subgraph` (`BIOCHEM_CORRECTOR_NUM_HOPS`, default 4) around the clot nodes.
- **D** `assemble_local_corrector_features` (same clot-COM-centered convention as train/verify).
- **E** `corrector(x_sub, sub_edge_index)` diversion patched onto the base flow on the subset.
- **F** coupled `[u, v]` published to a per-graph registry + optionally written into
  `data.y[:, :, 0:2]` so species/shear/nucleation consumers see the diverted field.

Entry points: `CorrectorCoupledFlow.couple(data, mu_eff_si)` (provider) and
`couple_flow_with_corrector(...)` (stateless). Enable with `BIOCHEM_CORRECTOR_COUPLING=1`;
the per-step coupled clot rollout (`clot_coupled_rollout.rollout_temporal_phi_coupled`) then
uses the cheap corrector diversion instead of a full DEQ re-solve. The species GraphSAGE
stagnation features (`SPECIES_STAGNATION_FEATS`) and vel-decay flow also read the coupled
registry when coupling is on.

### Fixing the GraphSAGE *primary* input: two-tier clot-aware flow (`ClotAwareFlow`)
The local corrector only patches `u, v`; it never regenerates the DEQ latent `z_kin =
predict_kinematics_latent(kine, data)` that is the GraphSAGE teacher's primary flow input.
`ClotAwareFlow.update(data, mu_eff_si)` escalates by clot burden:
- **frozen**  -- no clot -> frozen base flow + frozen latent.
- **corrector** -- small clot (`n_clot < BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES`, default 40) ->
  cheap local diversion on `u, v`; latent stays frozen.
- **resolved** -- significant clot (node count or `BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC`) -> the
  kine model **updates itself**: the clot `mu` is injected into `data.x[:, MU_PRIOR]` and the
  RGP-DEQ is re-solved, regenerating **both** the velocity field **and** `z_kin`. Hysteresis
  (`BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR`, default 1.5x growth since last solve) avoids a global
  solve every step.

The re-solved `z_kin` is threaded into the GraphSAGE band features via
`build_band_base_features(..., z_kin_override=...)` /
`prepare_species_gnn_rollout_static(..., z_kin_override=...)`, so once the clot is big enough to
reroute flow the teacher's primary input tracks the rerouted field instead of the clot-free
latent. Knobs: `BIOCHEM_KINE_RESOLVE_ON_CLOT` (default = coupling state),
`BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES`, `BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC`,
`BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR`.

**Phase 2 (system-level) test** -- does coupling fix `Mat` nucleation localization?
`python -m src.tools.compare_coupled_mat_rollout --graph data/processed/graphs_biochem_anchors/patient007.pt --species-ckpt <species best.pth>`
runs the species rollout uncoupled vs corrector-coupled and reports the spatial overlap
(Dice/F1) of the active `Mat` species vs the COMSOL ground truth -> a higher coupled F1 means
the diverted stagnation zone moved `Mat` to the correct downstream location.

### Coupling experiment log
**2026-06-20 -- Run #1 (INVALID, two confounds found & fixed).** First `compare_coupled_mat_rollout`
sweep vs the GraphSAGE `arch_ab/sage` teacher, t_last Mat Dice/F1 (baseline -> coupled):
p007 0.637->0.694 (+0.058), p007+stagnation 0.642->0.692 (+0.050), p006 0.366->0.458 (+0.091),
**p004 0.628->0.403 (-0.225)**, **p008 0.365->0.142 (-0.223)** -> 3 up / 2 (large) down, net inconclusive.
Two bugs made this **not** a test of the intended architecture:
1. **Clot mask flagged the whole mesh.** `delta_mu = mu_eff - mu_inf > 1e-3` tagged ~17,378/17,413
   nodes (the non-Newtonian bulk sits well above `mu_inf=0.0035`; Carreau `mu_0~0.056`). The
   corrector -- trained on *localized* clot patches where outside-clot Δμ≈0 -- was applied
   mesh-wide (OOD, `max|div|_nd~0.18`), which is the most likely cause of the p004/p008 collapses.
   Every run logged `clot_nodes<= <N_total>`.
   *Fix:* detect clot as `mu_eff - mu_bulk_carreau` (clot-free Carreau ref from the base flow).
   On p007 GT @ t_last this drops the mask from 17,378 -> **500 nodes** (~2.9%, the real gelation).
2. **The kine re-solve tier never fired.** The tool set `BIOCHEM_CORRECTOR_COUPLING=0` for the
   baseline pass, so during the coupled refresh `kine_resolve_enabled()` (default = coupling
   state) was False -> every run logged `final_mode=corrector kine_resolved=False (z_kin frozen)`.
   So the GraphSAGE *primary* input (`z_kin`) was never updated -- only the velocity-derived
   shear/vel-decay changed. *Fix:* enable coupling/kine-resolve **before** the refresh loop.
With both fixes the 500-node clot clears the 40-node burden gate, so the `resolved` tier engages
and `z_kin` is regenerated. **Re-run Run #2** with the same command to get a valid baseline-vs-
coupled comparison (watch for `final_mode=resolved kine_resolved=True`).

**2026-06-20 -- Run #2 (clot mask fixed, kine re-solve firing): still mixed/regressing.**
t_last Mat Dice/F1 (baseline -> coupled), `kine_resolved=True` on all: p004 0.628->0.495 (-0.133),
p006 0.366->0.174 (-0.192), p008 0.365->0.488 (+0.123), **p007 crashed (CUDA OOM in the DEQ
re-solve, 4 GiB)**. Clot masks were now sane (`clot_nodes<= 686/612/730`), but `max|div|_nd`
was **0.42-0.62** -- i.e. the diversion is as large as the *entire* freestream (~0.5). Diagnosis:
1. **Corrector is OOD on patient clots.** It was trained on *micro*-clot patches (small, μ 1.5-3
   Pa.s); late-rollout patient clots are large (600+ nodes) with μ up to ~4 Pa.s. It extrapolates
   unphysically large diversions that wreck the flow (p006 t100 -0.485). *Fixes:* clamp the Δμ
   feature (`BIOCHEM_CORRECTOR_MAX_DELTA_MU`, default 3.0); the spatial-extent OOD (huge subgraph)
   remains a fundamental scope limit -- the corrector is a *micro*-clot operator.
2. **OOM** on the big graph -> CPU fallback added to `ClotAwareFlow.resolve_full`.
3. **Fundamental: there is no clean channel for the corrector to help the GraphSAGE.** The species
   teacher's primary flow input is the *opaque DEQ latent* `z_kin`, and it was **trained on the
   frozen (clot-free) latent**. The corrector improves *raw velocity*, which the GraphSAGE does
   not consume. The only way to make `z_kin` clot-aware is to re-solve the DEQ with the clot μ --
   but the DEQ is *inaccurate on clots* (the very reason the corrector exists), so that latent
   carries the DEQ's clot error AND is a distribution the GraphSAGE never trained on -> mixed-sign,
   high-variance results (NOT a controlled improvement). Phase 1 already proves the corrector
   improves *flow fidelity* (17.6% relL2 vs COMSOL); the gap is purely *consumption*.

   **Real fix (requires training, not an inference swap):** fine-tune / retrain the species
   GraphSAGE to consume the corrector-improved flow in a representation it controls -- e.g. add
   coupled-`u,v`-derived stagnation/shear features (`SPECIES_STAGNATION_FEATS`) or feed coupled
   `u,v` directly -- so flow accuracy actually translates to Mat localization. Until then,
   inference-time flow swaps on a frozen model are distribution shift.

   **Decisive isolation experiment** (added): `compare_coupled_mat_rollout --oracle-mu` drives the
   diversion from the TRUE COMSOL clot μ (removes the predicted-μ feedback confound). If Mat still
   regresses under oracle μ, the bottleneck is the `z_kin` consumption/distribution-shift, not clot
   localization -> retraining is required.

**2026-06-20 -- Run #3 (oracle-μ + corrector-only ablation): regresses regardless -> consumption is
the blocker.** Two ablations, t_last Mat Dice/F1 (baseline -> coupled):
- **A) `--oracle-mu`, z_kin re-solved** (divert around the *true* clot): p004 0.628->0.420 (-0.208),
  p006 0.366->0.344 (-0.023), p007 0.637->0.588 (-0.049), p008 0.365->0.274 (-0.090). **4/4 regress.**
- **B) corrector velocity only, `BIOCHEM_KINE_RESOLVE_ON_CLOT=0`** (z_kin frozen): p004 -0.195,
  p006 +0.013, p007 -0.057, p008 -0.189. **3/4 regress** (p006 +0.013 is noise).

Interpretation -- the two ablations isolate the two suspects and **both** still regress:
- Oracle-μ rules out predicted-μ localization error: diverting around the *correct* clot still hurts.
- Frozen-z_kin rules out the DEQ-latent swap as the sole cause: changing *only* the velocity-derived
  features (vel-decay/shear) also hurts. So **any** inference-time flow change degrades this teacher.
- `max|div|_nd` was still **0.32-0.52** in every config -> the corrector, driven as ONE subgraph over
  the whole macro-clot (single COM, dx,dy spanning hundreds of nodes), is extrapolating far OOD.

**Fixes implemented (this run):**
1. **Apply the micro-clot corrector *locally*, not as one giant subgraph** (`BIOCHEM_CORRECTOR_LOCAL_CLUSTERS=1`,
   default on). `tile_clot_nodes` greedily partitions the clot into ND-radius balls
   (`BIOCHEM_CORRECTOR_CLUSTER_RADIUS_ND`, default 0.12) capped at `BIOCHEM_CORRECTOR_CLUSTER_MAX_NODES`
   (default 64); each patch gets its OWN local COM + small k-hop subgraph, and `couple_flow_with_corrector`
   accumulates the per-node diversions (averaged on overlap). This restores the training scale -- the
   direct answer to "apply at every clot node in a subgraph?": **yes, but as many small in-distribution
   patches, not one macro subgraph.**
2. **Clean GT-flow diagnostic** (`compare_coupled_mat_rollout --gt-flow`): feeds the TRUE COMSOL velocity
   (already in `data.y[:, :, 0:2]`), NO corrector, frozen z_kin. This is the gate the corrector cannot
   provide: does the GraphSAGE benefit from *accurate* flow at all? If `--gt-flow` ALSO regresses, the
   corrector is the wrong lever and the only path is retraining the teacher to consume coupled flow.

**Strategic read (micro -> macro adaptation).** There are two separable problems and they need
different fixes:
- *Corrector OOD (operator scale):* fixed by local tiling above (no retraining needed). Optionally
  retrain the corrector on patient-scale connected clots + the deploy μ range if tiling is insufficient.
-   *Consumption / distribution shift (the real blocker):* Run #3 (oracle-μ) shows even perfect flow
  hurts the frozen teacher. This is **not** fixable at inference -- the GraphSAGE must be **retrained
  with the coupled flow in the loop** (coupled-`u,v`-derived stagnation/shear features, and ideally the
  clot-aware latent), so flow accuracy maps to Mat localization. Run `--gt-flow` first to confirm the
  upside is real before paying for that retrain.

**2026-06-20 -- Run #4 (GT-flow gate + local tiling): the consumption blocker is now PROVEN structural.**
- **A) `--gt-flow`** (TRUE COMSOL velocity, NO corrector, frozen z_kin): p004 0.628->0.628 (+0.000),
  p006 0.366->0.373 (+0.006), p007 0.637->0.648 (+0.011), p008 0.365->0.365 (+0.000). **Feeding the
  *perfect* flow is a no-op (±0.01).**
- **B) `--oracle-mu` + local tiling** (z_kin re-solved): `max|div|_nd` dropped **0.32-0.52 -> 0.26-0.30**
  (tiling works -- diversions are now physical-scale), but Mat still regresses 3/4 (p004 -0.180,
  p007 -0.034, p008 -0.038; p006 +0.028) because the only channel that moves Mat here is the OOD
  clot-aware `z_kin`.

**Root cause (confirmed in code).** The species GraphSAGE node input is `build_snapshot_features` =
`[z_kin, sdf]` -- there is **no velocity feature in the model**. Flow enters the rollout only as a
learned `vel_decay` multiplier on the growth state (`band_speed_for_rollout`); swapping that to GT
COMSOL flow barely moves Mat. So Mat is ~entirely a function of the frozen, clot-blind `z_kin` + wall
distance + state carry. The corrector edits `u,v` -> a channel the model ignores; the only channel
that matters (`z_kin`) is reachable only via an OOD/inaccurate DEQ re-solve. **Inference-time coupling
on this teacher cannot work; it is missing wiring, not mistuned.**

**Retrain plan (the actual fix).** Give the teacher a clot-aware flow channel it can learn from:
1. Add explicit flow features to `build_snapshot_features` -- speed `|u|`, shear-rate proxy, and a
   stagnation/divergence indicator -- computed from the **GT COMSOL velocity** (which is already
   clot-aware) during training. New input dim = `latent_dim + 1 + k` (breaks old ckpts -> new run id).
2. Train the species GraphSAGE with those features so flow->Mat is actually learned (currently the
   velocity signal had no training variance, so the model learned to ignore it).
3. At deploy, the corrector-coupled flow (now in-distribution thanks to local tiling) supplies the
   approximation of that clot-aware flow; keep `z_kin` frozen (do NOT feed the OOD re-solved latent).
4. Re-run `--gt-flow` on the retrained teacher: it should now show a real, positive delta. Only then
   does the corrector have ROI.

**2026-06-20 -- Retrain plumbing implemented.** Flow-aware teacher wiring is in place:
- `species_pushforward_gnn.py`: `flow_feats_enabled()` (`SPECIES_FLOW_FEATS=1`) appends a 5-ch
  clot-aware flow block `[log1p(speed), log1p(shear), tanh(div), x_n, y_n]` (`_flow_band_features`)
  to the band inputs. Velocity source via `SPECIES_FLOW_FEATS_SOURCE`: `gt` (COMSOL `data.y`,
  training), `kine`, or `auto` (kine + corrector-coupled override, deploy default); representative
  GT time `SPECIES_FLOW_FEATS_TIME` (-1=last). Model in_dim auto-derives from `base_feats.shape[1]`
  (257 -> 262 confirmed on p007), so no manual dim wiring.
- Persisted: trainer writes `flow_feats` into meta; `load_species_gnn_rollout_bundle` re-enables
  `SPECIES_FLOW_FEATS` at deploy (source left `auto`, NOT the training `gt`).
- Launcher: `scripts/go_species_flow_aware.ps1` (canonical arch_ab sage recipe + flow feats, FRESH
  since input dim changed) -> `outputs/biochem/biochem_gnn/flow_aware/sage/species/best.pth`.
- Tests: `src/tests/test_species_flow_feats.py` (shape, gt/kine source, bounded divergence).
**Next:** run the launcher, then re-gate with `compare_coupled_mat_rollout --gt-flow` on the new
ckpt; a positive delta unlocks the corrector path (keep z_kin frozen, source `auto`).

**2026-06-21 -- Run #5 (flow-aware teacher trained; gt-flow gate fixed): teacher +0.08, corrector
upside only +0.008.** 75-ep flow-aware sage (`outputs/biochem/biochem_gnn/flow_aware/sage/species/
best.pth`, `best_score=0.752`; input 257->262 confirms the 5 flow channels). p007 t200 Mat F1:
- Old teacher `[z_kin,sdf]`: **0.637**.
- New teacher, **kine** flow features (deploy baseline): **0.717 (+0.080)**.
- New teacher, **GT** flow features (corrector upper bound): **0.725 (+0.008 over its own baseline)**.

**gt-flow diagnostic bug (fixed).** The first re-gate read +0.002 (no-op) because `--gt-flow` set
only `SPECIES_ROLLOUT_VEL_SOURCE=gt` (vel-decay) and NOT `SPECIES_FLOW_FEATS_SOURCE=gt`; with
coupling off the flow features fell back to `auto`->kine in BOTH passes. Fixed
`compare_coupled_mat_rollout` to export `SPECIES_FLOW_FEATS_SOURCE=gt` in the gt-flow branch (and
clear it after). Re-gate then showed the true +0.008.

**Read.** The +0.080 is a real win but comes from the *richer feature set* (speed/shear/divergence/
geometry), NOT from clot-awareness: accurate (GT) flow beats clot-blind (kine) flow by only +0.008,
and that is the corrector's ceiling. Two structural reasons: (1) the flow features are **static**
(one representative time) so they cannot represent the corrector's *dynamic* diversion as the clot
grows; (2) stagnation localization is largely **redundant with geometry** (`z_kin`/SDF) the model
already encodes. **Actions:** (a) promote the flow-aware teacher as the new baseline regardless of
the corrector; (b) if the corrector must matter, make the flow features **dynamic** -- recompute the
speed/shear/divergence channels each rollout step from the current corrector-coupled flow -- since
the static upper bound is already only +0.008, expectations should be modest.

**2026-06-21 -- Latent leash (the input leash) implemented.** Diagnosis of the +0.008 ceiling: even
with flow channels present, the teacher is free to lean 100% on the (clot-blind) `z_kin` latent and
ignore the flow features ("latent dominance" / causal confusion). Fix = **latent dropout** during
training so the model is forced to learn backup weights on the explicit flow:
- `species_pushforward_continuous.py`: `species_latent_dropout_p()` (`SPECIES_LATENT_DROPOUT`, train
  only) + `maybe_drop_latent(base_feats, model, training)` zeros the first `kin_latent_dim` columns
  (the z_kin slice; sdf+flow untouched) with prob `p`, resampled per unrolled step. Applied in both
  `predict_continuous_step_delta` and the `unroll_continuous_loss` step loop. No-op at eval/deploy.
- Trainer sets `model.kin_latent_dim` (true z_kin width, now returned by `build_band_base_features`
  as `latent_dim`) and `model.latent_dropout_p`; persists `latent_dropout` into meta.
- Verification knob: `flow_feats_ablate()` (`SPECIES_FLOW_FEATS_ABLATE=1`) zeros the flow block while
  preserving its width. A leashed teacher's baseline F1 should **drop** under ablation; a
  latent-dominant teacher is unaffected (that is the whole tell).
- Launcher: `scripts/go_species_flow_aware.ps1 -LatentDropout 0.5` -> separate
  `outputs/biochem/biochem_gnn/flow_aware_leashed/...` (does not clobber the un-leashed ckpt).
- Tests: `test_species_flow_feats.py` (`maybe_drop_latent` zeros z_kin / identity at eval/off,
  `flow_feats_ablate` flag).

**2026-06-21 -- Trap C: time-varying flow features (dynamic) + gate.** The flow channel was static
(one representative time), so it cannot represent the corrector's *dynamic* diversion as the clot
grows -- the static gt-flow upper bound is only +0.008. Fix = recompute the flow block per rollout
step from the time-`t` velocity:
- `species_pushforward_gnn.py`: `flow_feats_dynamic()` (`SPECIES_FLOW_FEATS_DYNAMIC=1`); refactor
  extracts `_flow_feats_from_uv` + `_flow_feats_series_from_y` (per-time block `[n_t, n_band, 5]`
  from `data.y[t][:, 0:2]` -- GT in train/gate, per-step coupled velocity at deploy). When dynamic,
  `build_band_base_features` returns `flow_series` + `flow_cols=(start,width)`.
- `species_pushforward_continuous.py`: `splice_dynamic_flow(base_feats, flow_series, flow_cols, ti)`
  replaces the flow block with the time-`ti` slice; applied (current-state time) in the training
  `unroll_continuous_loss`, the val `rollout_continuous_states`/`eval_continuous_window`, and the
  deploy rollout. `predict_continuous_step_delta` gains `flow_time_index` (decoupled from the retired
  temporal gate's `time_index`). `SpeciesGnnRolloutStatic` carries `flow_series`/`flow_cols`; trainer
  persists `flow_dynamic` and the loader re-enables it at deploy.
- Gate: `compare_coupled_mat_rollout --gt-flow-dynamic` (frozen z_kin, no corrector, per-step GT
  flow). Run on a dynamic-trained teacher; **delta vs static `--gt-flow` = the temporal-sharpening
  headroom** (Trap C's ceiling). Launcher: `go_species_flow_aware.ps1 -DynamicFlow`.
- Tests: `flow_feats_dynamic` flag, time-varying `_flow_feats_series_from_y`, `splice_dynamic_flow`
  (replace/clamp/no-op).

**Path A/B reconciliation.** The user's "Hybrid Rollout Controller" (Path A macro re-solve + Path B
per-step local corrector) is **already** implemented by `ClotAwareFlow` (`src/inference/
corrector_coupling.py`): node-count + growth-factor trigger -> full DEQ re-solve via `MU_PRIOR`
injection (Path A), else local-tiled corrector diversion every step (Path B). We deliberately did
NOT swap the working `MU_PRIOR` re-solve for the proposed `SDF=0` occlusion trick: the kine model
reads viscosity through `MU_PRIOR`, not "SDF=0 == wall", so the occlusion trick is unvalidated and
risks an OOD latent. The missing lever was the **leash**, now added.

**2026-06-22 -- Run #6 (full gate ladder, leashed p=0.5 + dynamic teacher): corrector path is a dead
end for Mat localization; the leash BACKFIRED.** Trained a fresh flow-aware teacher (latent dropout
0.5, dynamic flow, 75 ep, `outputs/biochem/biochem_gnn/flow_aware_leashed_dynamic/...`,
`best_score=0.700`) then ran every diagnostic rung on p007 (Mat-band Dice @ t200). Table (ref = the
flow-active reference baseline):

| rung | F1 | vs ref | reading |
|---|---|---|---|
| ref baseline (flow on, kine) | 0.621 | -- | reference B |
| **leash check: flow ABLATED** | **0.671** | **+0.050** | zeroing flow IMPROVES -> leash backfired |
| #5  static GT ceiling | 0.630 | +0.009 | perfect static flow barely helps |
| #5c dynamic GT ceiling | 0.626 | +0.005 | no temporal headroom (Trap C ~0) |
| #5b oracle-mu (true clot loc) | 0.584 | -0.038 | approx flow at TRUE clot hurts |
| #6a corrector, frozen z_kin | 0.588 | -0.033 | corrector diversion hurts |
| #6b corrector + z_kin re-solve | 0.467 | -0.154 | clot-aware DEQ re-solve catastrophic |

**Read.** Three findings, all negative for the corrector path:
1. **Leash backfired.** Ablating the flow block (zeroing it) *raised* baseline F1 0.621 -> 0.671. The
   leash (`SPECIES_LATENT_DROPOUT=0.5`) forced the teacher to lean on a flow channel that, at the
   static-final representation, is *net noise* on this metric: no-flow (0.671) > perfect GT flow
   (0.630) > kine flow (0.621). Latent dominance was the *correct* equilibrium here; the leash
   pushed the model off it. **Drop the leash.**
2. **Flow is a near-zero lever for Mat.** Perfect GT flow ceiling is only +0.009 static / +0.005
   dynamic over ref. Stagnation localization is largely redundant with the geometry the teacher
   already encodes (`z_kin`/SDF). There is no ROI to chase with a better corrector.
3. **Every coupled variant regresses** (oracle-mu -0.038, corrector -0.033), and the **z_kin
   re-solve is catastrophic** (-0.154): the clot-aware re-solve drives `max|div|_nd` 0.23 -> 0.49
   (far out of the corrector's micro-clot training band) and regenerates a latent the teacher never
   saw. **Keep `BIOCHEM_KINE_RESOLVE_ON_CLOT=0` always.**

**Verdict.** The corrector coupling does not improve Mat-band species localization for this teacher
and metric, even with oracle clot location or oracle flow. The +0.080 "win" from Run #5 was the
richer *feature set* (speed/shear/divergence/geometry), not clot-awareness -- and pairing those
features with the leash now makes them harmful. **Recommended next:** (a) retire the corrector
coupling for Mat localization (it is a confirmed dead end on this probe); (b) retrain the deploy
teacher **unleashed**, and likely **without the flow block** (or keep it un-leashed only if a clean
A/B on the deploy clot-F1 metric -- not this band-Dice probe -- shows it does not hurt); (c) caveat
(Trap D): this gate scores Mat-band Dice at a fixed threshold, NOT the headline swept deploy clot
F1, so confirm the "drop flow features" decision with a real `deploy_ab_eval` before committing.

**GPU OOM fix (resolved).** The `corrector_resolve` rung previously OOM'd on the 4 GiB card (two
RGP-DEQ solves back-to-back: Path-A re-solve + flow-feature solve). Fixed without going to CPU:
free the coupler's kine model before `prepare_species_gnn_rollout_static` loads its own
(`compare_coupled_mat_rollout`), `empty_cache()` + **retry on GPU** before any CPU fallback in
`_resolve_flow_uv` / `ClotAwareFlow.resolve_full`, and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
(Windows-safe, set in `go_clot_flow_gate_ladder.ps1`). The standalone ladder then completed all
seven rungs; the unified `go_clot_flow_gate_full.ps1` crashed only because it ran the pre-fix code.

---

## 2026-08-31 -- THE ORACLE GATE: the corrector's ceiling, measured

Asked before rebuilding anything: **if the flow model reported the clot->flow blockage
perfectly, would `clot_ml_0` improve?**  The corrector's whole value is bounded by that
answer, so it was measured first.

### The instrument

`physics_wall_model.oracle_blockage` -- a `blockage` callable that at rollout step `i` reads
the **ground-truth** clot occupancy at pack time `i`, applies the measured gelation collapse
`sr <- sr * 0.1226` there (`GELATION_SR_RATIO`, MODEL_REVIEW 9e.4), re-differentiates with the
SAME MLS operator the consumer uses, and re-evaluates the gate.  Perfect clot localisation,
correct shear response, no model error, no stencil mismatch.  NOT deploy-legal by construction.

Wired into `clot_ml.features.build_features` behind `CLOT_ML_ORACLE_BLOCKAGE` (off by default,
bit-identical when off).  It moves exactly 9 channels: `log_mat_phys`, `onset_phys`,
`log_mat_owner`, `log_mat_adv`, `log_mat_adv_n`, `log_mat_off_est`, `log_src_reach`,
`att_adv`, `att_reach` -- the physics backbone and everything downstream of it.

### Result 1 -- the mechanism is REAL and the metric SEES it

Physics backbone masks alone, final frame, GT flow, deploy metric of record (`guiding`):

| vessel | gt_n | base_n | orc_n | baseF1 | orcF1 | baseDS | orcDS | dDS |
|---|---|---|---|---|---|---|---|---|
| wound_patient001 | 94 | 63 | 67 | 0.7134 | 0.7702 | 0.7642 | 0.7941 | +0.0299 |
| wound_patient002 | 58 | 27 | 28 | 0.6353 | 0.6512 | 0.6891 | 0.7264 | +0.0373 |
| wound_patient003 | 254 | 171 | 210 | 0.8047 | 0.9052 | 0.8127 | 0.9203 | +0.1076 |
| patient012 | 96 | 41 | 87 | 0.5985 | 0.9508 | 0.7107 | 0.9618 | +0.2511 |
| patient020 | 110 | 55 | 94 | 0.6667 | 0.9216 | 0.6904 | 0.9205 | +0.2301 |
| patient032 | 193 | 174 | 191 | 0.9155 | 0.9688 | 0.9120 | 0.9751 | +0.0632 |
| patient041 | 113 | 55 | 104 | 0.6548 | 0.9585 | 0.7408 | 0.9689 | +0.2281 |
| patient044 | 163 | 61 | 145 | 0.5446 | 0.9416 | 0.6633 | 0.9538 | +0.2905 |
| **MEAN** | | | | **0.6917** | **0.8835** | **0.7479** | **0.9026** | **+0.1547** |

**+0.155 on the deploy metric, 8/8 positive**, eight times the +/-0.024 wall noise floor.  The
node counts show the mechanism directly: `patient044` commits 61 nodes open-loop against a GT of
163, and 145 with the loop closed.  The open loop under-recruits by ~60% and the shear collapse
recovers almost exactly the missing nodes.  **This is C3' confirmed** -- one measured constant,
no corrector, against the corrector's own -3.5%.

### Result 2 -- and it buys `clot_ml_0` NOTHING

Same oracle, end to end through `scripts/eval_clot_ml_0.py`:

| vessel | v0 wall off | v0 wall on | v0 off off | v0 off on |
|---|---|---|---|---|
| patient012 | 0.9837 | 0.9713 | 0.9125 | 0.8984 |
| patient020 | 0.9879 | 0.9818 | 0.4911 | 0.5281 |
| patient032 | 0.9940 | 0.9920 | 0.8117 | 0.8168 |
| patient041 | 0.9943 | 0.9871 | 0.9603 | 0.9415 |
| patient044 | 0.9885 | 0.9840 | 0.9276 | 0.9299 |
| **MEAN** | **0.9897** | **0.9832** | **0.8224** | **0.8230** |

**-0.0065 wall / +0.0006 off-wall**, inside the noise floor.

### The read, and it is NOT "the metric cannot see it"

The ordering is the whole finding:

```
0.748   physics backbone, open loop
0.903   physics backbone + ORACLE closed loop      (+0.155)
0.990   clot_ml_0, open loop                      (+0.087 ABOVE the oracle backbone)
```

`clot_ml_0` already sits **0.087 above the oracle-corrected backbone**.  The network recovers
the entire closed-loop gain on its own, from t=0 geometry and shear, and the oracle hands it
information it already has -- while costing distribution shift, since v0 was trained on
open-loop features.  The corrector's achievable contribution lies *below* where the learned
model already is.

**Scope of the conclusion.**  This bounds the corrector on the **non-wound cohort under GT t=0
flow**, where v0 is saturated at 0.984-0.994 and the headroom to a perfect score is 0.010 --
less than the noise floor.  On that population the question is unanswerable by any experiment,
a retrain included.  Two regimes are NOT covered and remain open:
* **The wound vessels**, the only ones with headroom left (v0 wall 0.71-0.90).  There the result
  is mixed rather than null: `wound_patient003` +0.032 v0 wall / +0.019 `w_reg` (and the
  `clot_gnn_v5w` baseline +0.037 on both), `wound_patient002` -0.013, `wound_patient001` flat.
  n=3 -- a hint, not a result.  This is the population MODEL_REVIEW 2.4 nominated.
* **Predicted flow**, where v0 collapses to wall 0.586 / off 0.350 and ~0.37 of headroom opens
  up.  The oracle has never been run there.  Note the fix in that regime is surrogate `sr`/`dsrx`
  accuracy (a converged FEM solve scores as GT does), not a diversion corrector.

### What this retires

The bugged `LocalKinematicCorrector` should not be repaired, retrained, or replaced **for
`clot_ml_0` on this population**.  Measured on `patient001`, sweeping `delta_mu` over 100x
(0.068 -> 6.8 Pa.s) moves wall `sr/sr0` only 0.979 -> 0.960, non-monotone, while injecting
`max|du|_nd` of 0.13-0.32 -- a large velocity perturbation carrying no information about the
occlusion driving it.  Root causes, all structural: it was selected on interior velocity relL2
(which correlates -0.030 with the deploy F1 drop); its Patch Factory training BC prescribes a
moving lid at `y=H` (`patch_factory_comsol.py:51`), so a patch physically **cannot stall**; and
a 3-layer 5-hop GAT on a k-hop subgraph cannot represent flux redistribution, which is elliptic.
`data/processed/cfd_results_patch_factory_v2` is also no longer on disk, so any retrain of that
design is a COMSOL regeneration campaign.

Reproduce: `CLOT_ML_ORACLE_BLOCKAGE=1 python scripts/eval_clot_ml_0.py`.
Artifacts: `outputs/diag_oracle_blockage_{off,on}.json`.

---

## 2026-08-31 -- TIER 0/1/1.5: retaining the module honestly

The oracle gate above says the corrector does not pay for `clot_ml_0` today.  It is retained
because **severe occlusion** is a regime nothing else in the stack covers.  That makes one
thing indefensible: it was trained at 3-10% blockage and is claimed for the opposite end.
Three pieces of work close that, none of which is a retrain.

### Tier 0 -- the model card (`scripts/diag_corrector_characterization.py`)

Panel A sweeps `delta_mu` over 4 decades on `patient001` (40 occluded wall nodes, base
`sr` 83 1/s, `lss` 25):

```
 dmu[Pa.s]    learned      prior   max|du|_nd  gate(learn)  gate(prior)
    0.0068     0.9834     0.9332       0.1976         shut         shut
    0.0680     0.9794     0.5828       0.1790         shut         shut
    0.3400     0.9891     0.2184       0.1573         shut        FIRES
    0.6800     0.9944     0.1226       0.1352         shut        FIRES
    3.4000     0.9831     0.0272       0.1820         shut        FIRES
    6.8000     0.9596     0.0138       0.3227         shut        FIRES
   68.0000     0.9752     0.0014       2.9593         shut        FIRES

   learned: response span 0.0348 over 10000x of dmu, monotone=False   <- FAILS the claim
   prior:   response span 0.9318, monotone=True
```

Panel B reads the training domain **out of the generator**, so the validity statement cannot
drift from the code: `clot_mu` 0.1-10 Pa.s, `shear_rate` 50-5000 1/s, blockage 3-10% of channel
height, and the prescribed-lid top BC that makes stalling impossible.  Panel C states what is
and is not validated.  `outputs/diag_corrector_characterization.json`.

### Tier 1 -- the Delta-mu response, imposed instead of learned

`coupled_shear_gnn.shear_attenuation` / `composed_wall_shear`:

```
A(dmu) = 1 / (1 + dmu / DELTA_MU_HALF)        DELTA_MU_HALF = 0.0950 Pa.s
```

`A(0) = 1` exactly, `A -> 0` at solid occlusion, monotone throughout.  The constant is not
tuned: it is the solution of `A(0.68) = 0.1226`, the measured gelation collapse, so the law
reproduces the project's own anchor by construction.  The learned head becomes an OPTIONAL
additive residual, off by default.  Ten property tests in
`src/tests/test_local_corrector_properties.py` pin all of it, including the constant-vs-anchor
identity, so an edit to either fails loudly.

### Tier 1.5 -- validation in the claimed regime, with FEM as the oracle

`solve_local_t0_flow` now accepts `delta_mu_nodal_si`, so a clot can be injected as a
high-viscosity region -- the Patch Factory's own constitutive picture -- but on the real vessel
geometry under its real fixed-flux inlet BC, at ANY occlusion fraction.
`scripts/diag_corrector_severe_occlusion.py`, 5 vessels x 5 fractions, `clot_mu` 0.68 Pa.s,
scored on `sr/sr0` (the gate's own quantity) against the FEM solve at the same occlusion:

```
MEAN |error| in sr/sr0, 25 cases
   base (do nothing)   0.6297
   learned corrector   0.6841      <- WORSE THAN DOING NOTHING
   analytic prior      0.3270      <- ~2x better than either
```

**Three findings, and the third is the important one.**

1. **The learned corrector is worse than the null arm** in the regime it is retained for.  It
   is not merely uninformative there; applying it costs accuracy.  It must not ship as-is.
2. **The prior roughly halves the error** and is best exactly where its anchor sits (frac
   0.05-0.2: |err| 0.006-0.16), degrading away from it (frac 0.6: 0.18-0.96).  That is the
   expected behaviour of a one-point anchor and should be reported as such, not as a fit.
3. **THE CLOT->SHEAR MAP IS NOT MONOTONE IN OCCLUSION, and neither arm can express it.**  The
   FEM ratio RISES with occlusion fraction in **5 of 5 vessels**:

```
   frac        0.05    0.20    0.40    0.60
   patient001  0.252   0.237   0.439   0.467
   patient005  0.074   0.076   0.244   0.300
   patient008  0.053   0.702   1.565   1.084      <- exceeds 1.0: shear INCREASES
   patient010  0.037   0.116   0.362   0.406
   patient011  0.217   0.285   0.496   0.527
```

   At low occlusion viscous shielding dominates and wall shear collapses.  As the clot fills
   the lumen, **flux redistribution accelerates the residual channel** and pushes the ratio
   back up -- past 1.0 on `patient008`, i.e. a partial occlusion RAISES wall shear at the
   throat.  This is the elliptic, global effect a k-hop local operator cannot represent, and
   the analytic prior (a pure function of `dmu`) cannot represent it either.  **Any replacement
   must take occlusion fraction / residual lumen width as an input.**  This is the concrete
   design requirement Tier 2 would be built against.

**Caveats, stated because they bound the numbers above.**  Gate agreement reads `prior 1.00 /
learned 0.60` over 25 cases, but FEM fires in 25/25 and `patient008`/`patient010` sit at base
`sr` 2.6/6.5 1/s -- already below `lss` with no clot at all -- so that statistic is partly
trivial and should not be quoted alone.  Spatial correlation with the FEM field is weak and
often negative for BOTH arms (-0.86 to +0.85): they get the magnitude, not the pattern.  The
`frac` parameterisation saturates between 0.6 and 0.8 (identical rows), so the tested range is
really 0.05-0.6.  Solve cost is 2-186 s per case, not the ~5 s of the clot-free solve --
severe occlusion is a harder nonlinear problem.

### Where this leaves the module

Publishable as: *a local operator whose Delta-mu response is analytic and anchored on a
measured constant, whose learned component is characterised and disabled, with a stated
validity domain and one honest evaluation in the severe-occlusion regime showing what it still
cannot do.*  That is a defensible retained module.  It is NOT "a working severe-occlusion
corrector", and the tables above are what stop anyone claiming so.

---

## 2026-08-31 -- TIER 2 (physics-informed rebuild): NEGATIVE, and why

Tier 1.5 gave a concrete design requirement: the operator must read occlusion geometry, not
just `delta_mu`.  Tier 2 built exactly that and it does not work.  This section is the record of
why, because the failure is more useful than the module would have been.

### The corpus (no COMSOL)

`scripts/build_corrector_pi_corpus.py` -> `outputs/pi_corpus`: **96 FEM cases across 12
vessels, 0 failures, 53 minutes**, generated by injecting a clot as a high-viscosity region via
the new `solve_local_t0_flow(delta_mu_nodal_si=...)`.  Occlusion swept to 75% and `mu` to
3 Pa.s -- far past the Patch Factory's 3-10% cap.  Regenerable for free, which makes every
number below reproducible.

### The operator

`src/core_physics/pi_wall_shear.py`:

```
sr_pred / sr0  =  A(dmu) * (h0/h_eff)^p * exp(eps_theta(x))
```

with `A` the Tier 1 anchored attenuation, `h_eff` the hydraulic lumen (below), `p` initialised
at Poiseuille 2, and `eps` a `tanh`-bounded residual.  Untrained it IS the closed form, and OOD
it degrades to physics rather than extrapolating -- both pinned by tests.

**A real correction found on the way.**  A constant flux exponent is wrong: regressing
`log(sr/sr0)` on `log(h)` inside `dmu` terciles gives slopes **-0.218 / -0.588 / -2.073**.
Poiseuille is recovered only where the occlusion is stiff; a soft gel barely redirects flux.
`hydraulic_h` fixes this by blocking only the SOLID fraction of the lumen
(`B = 1 - A`, `h_eff = 1 - (1-h)*B`), adding no free parameter and keeping both limits exact.
Recorded in `python -m src.tools.diagnostics pi-flux-interaction`.

### The result: LOVO by vessel, 12 folds

```
      arm   MAE log  MAEratio/case  corr log   corr sr
     null    1.4759         1.3345       nan     0.239
    prior    1.3690         0.9848    -0.071     0.094
phys_geom    1.3834         1.1502    -0.016     0.092
 phys_hyd    1.3169         1.1175     0.012     0.117
     full    1.0882         0.9903     0.021     0.142
```

`full` cuts log error 26% below null and 21% below the prior, on 10/12 folds.  **It is still a
failure**, for three reasons that the log column hides:

1. **`corr_log` is ~0 for every arm** (-0.071 to +0.021).  There is NO spatial skill.  The gain
   is entirely per-case scale, not knowing which nodes shear more.
2. **`corr_sr` is best for the null arm** (0.239).  Every model arm correlates with the true
   wall-shear field *worse than doing nothing*.
3. **On the case-level metric -- the only one comparable to Tier 1.5 -- the one-line Tier 1
   prior wins** (0.985 vs `full` 0.990; both physics arms worse at 1.12-1.15).  All of Tier 2
   does not beat `A(dmu)`.

**Two diagnostics say the mechanism is misspecified**, by the criterion set before the run:
* Learned physics parameters drifted off their physical values: `p` 2.0 -> 1.20,
  `delta_mu_half` 0.095 -> 0.21 (2.2x).  The module docstring called that "a signal the
  mechanism is wrong, not a free parameter to be quietly absorbed."
* Closed-form fit gives `p ~ 3.1`; gradient descent gives `p ~ 1.20` on the same data.  A 2.6x
  disagreement between two routes to one parameter is an identifiability failure, not a fit.

**And the flux signal weakened as vessels were added**: `corr(log h, log ratio)` went
-0.196 (n=4) -> -0.191 (n=6) -> **-0.124 (n=12)**, with 5 of 12 vessels the WRONG SIGN,
including both wounds (+1.33, +2.70).  An early strong reading on 15 per-case medians at a
single viscosity did not survive per-node scoring across the full sweep.  That early number
should not be quoted.

### It does not divert flow either -- the qualitative claim also fails

The module's original claim was geometric, not quantitative: *flow reroutes over and around a
clot*.  `scripts/diag_corrector_diversion_field.py` decomposes the diversion against FEM truth
into direction and magnitude, on `patient001` at 50% occlusion:

```
   cos(du_corr, du_fem)   -0.142        (energy weighted; -0.147 median)
   |du| ratio              0.000
   max |du|  FEM           0.359 m/s
   max |du|  corrector     0.020 m/s     18x too small at PEAK
   median |du| at live nodes:  FEM 5.4e-2 m/s   corrector 7.3e-9 m/s
```

The corrector's perturbation is not merely small, it is **in the wrong places**: ~zero at the
nodes where the real diversion lives, and anti-correlated in direction where it is nonzero.
`outputs/reports/figures/kinematics/diversion_patient001_frac50.png` shows it -- the corrector
panel is visually identical to the clot-free base while FEM forms a clear jet through the
residual lumen.  **The corrector cannot be used even as a qualitative picture of rerouting.**

*(Methodological note, kept because it would otherwise be repeated: the first version of that
figure seeded the clot at `wall_i[len(wall_i)//2]`, an arbitrary node ordering, which on
patient001 landed ON THE INLET.  A clot on a Dirichlet boundary is pinned by the BC and cannot
reroute anything.  `pick_mid_vessel_seed` now chooses by geometry.  The `pi_corpus` above used
random wall seeds and so contains an unknown fraction of such degenerate cases -- a caveat on
every Tier 2 number here, and the first thing to fix if it is ever revisited.)*

### The conclusion worth keeping

Wall-shear redistribution around a clot is set by the GLOBAL flow solution -- it is elliptic.
A model built from LOCAL per-node features cannot represent it, and that is now demonstrated for
two unrelated architectures: the original k-hop GATv2 and a physics-structured readout with an
anchored analytic backbone.  The flux mechanism is real but lives at CASE level (correlation
-0.554 on case medians), i.e. it is close to one scalar per clot, which is not something a
per-node operator recovers.

**If anyone revisits this**, the only lead supported by evidence is predicting a per-case
scalar (a burden-level shear multiplier) rather than a per-node field -- and fixing the
inlet-seeded cases first.

### What IS useful, and it is not the model

The FEM oracle answers the question the corrector was built for, exactly.
`scripts/viz_occlusion_flow_sweep.py` sweeps occlusion on a real vessel and shows the flow
profile change directly: the jet forming through the residual lumen, shear collapsing under the
clot (SHIELDING) while overshooting at its shoulders (ACCELERATION), and the non-monotone
`sr/sr0` that no `delta_mu`-only law can express.  That is the figure this module was supposed
to produce, and the solver produces it.

### The occlusion sweep, as numbers (`scripts/viz_occlusion_flow_sweep.py`)

`patient001`, clot `mu` 0.68 Pa.s, FEM, wall nodes under the clot vs the whole near-clot band:

```
  frac  n_clot  sr med clot    ratio  sr max band  vs clot-free
  0.20     322         1.34    0.051        53.75         1.052
  0.40     652         2.46    0.093        65.68         1.286
  0.60     890         5.28    0.200        79.16         1.550
  0.80     890         5.28    0.200        79.16         1.550
```

Read the two columns against each other -- they move in OPPOSITE directions and that is the
whole point:

* **Under the clot, shear collapses** to 1.3-5.3 1/s against a clot-free ~25 1/s, i.e. ratio
  0.05-0.20, well below `lss` -- the low-shear gate fires, as the deposition law expects.
* **At the shoulders, shear OVERSHOOTS.**  Peak band shear rises 1.05x -> 1.55x the clot-free
  maximum, and the ratio profile peaks at **1.45 just upstream** of the clot at 80% occlusion
  and **1.37 just downstream** at 20%.  Flux displaced by the occlusion has to go somewhere and
  it accelerates past the shoulders.
* **The collapse gets WEAKER as the clot deepens** (ratio 0.051 -> 0.200 from 20% to 60%),
  because the acceleration term grows faster than the shielding term.  This is
  [[clot-shear-map-is-non-monotone]] visible as a profile rather than a scalar, and it is
  exactly what a `delta_mu`-only law -- including the shipped `GELATION_SR_RATIO` blockage --
  cannot produce.

Caveat: the 0.60 and 0.80 rows are identical because the clot mask saturates once it fills the
local column, so the effective range swept is 0.05-0.6, not 0.05-0.8.

Figures: `outputs/reports/figures/kinematics/occlusion_sweep_patient001.png` (sweep) and
`diversion_patient001_frac50.png` (corrector vs FEM diversion).  Data:
`outputs/viz_occlusion_sweep_patient001.json`.

**Three method bugs were fixed building these, all commented in place**: clot seeds chosen by
NODE ORDERING can land on the inlet, where the Dirichlet BC pins velocity and the case is
degenerate (`pick_mid_vessel_seed` now chooses by geometry); straight-line arclength collapses
on a curved vessel and mixes both wall sides (replaced by a wall-subgraph geodesic); and the P2
meshes alternate vertex and mid-side nodes along the boundary, which reads as a sawtooth rather
than noise (rolling median).
