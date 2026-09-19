# Experiments — the run queue

Ranked by **what each run would settle**, not by cost. A run that hardens a leg already believed
beats a run that adds a number nobody will quote.

Every entry states: the leg it serves, the command, roughly what it costs, and — the part that
matters — **what result would change the story**. A run whose every outcome leaves the story
identical should be cut, not queued.

---

## E1g — the NON-LEAKING corrector  ·  **DONE 2026-09-12. THE ANSWER: coupling is worth nothing**

> **The uncontaminated result, and the only coupling number the paper may quote.**
> Corrector driven by the rollout's own committed set (`gelation_wake`, `occ = mat >= crit`),
> never by GT. Same 36 vessels, 3 folds, arm A4.
>
> | domain | uncoupled | self-driven coupling | delta |
> |---|---|---|---|
> | wall | 0.9494 | 0.9495 | **+0.0000** |
> | off-wall | 0.8293 | 0.7981 | −0.0312 |
>
> **Zero at the wall to four decimal places**, inside the uncoupled arm's seed noise (0.0009).
> Not inert, though: 10 of 36 vessels improve and the largest single-vessel move is 0.0373 — it
> moves vessels and nets to nothing.
>
> **Leakage gate PASSED** (this is what makes it usable): `mat_phys` alone recovers the wall label
> at F1 0.8252, against 0.7950 uncoupled and 0.9609 for the GT oracle.
>
> **This settles Leg 2 on honest evidence.** Solving once is not a cost compromise — a closed
> clot→flow loop driven the only way a deployable one could be buys nothing measurable.
>
> **Two caveats, both in STORY 2.1g.** (1) **One healthy seed pair** — the wake arm's seed-0 run
> collapsed a fold, so this rests on seed 1; a second clean replicate is wanted before freeze.
> (2) ~~The fold-collapse pathology is a coupled-arm pattern~~ **RETRACTED 2026-09-13**: by fold
> cut signature the uncoupled `cpl_open_s3` collapsed too (off-wall 0.690). Collapses are a
> readout-stability problem present with and without coupling (uncoupled 1/3, FEM-coupled 3/18).

### Reproduce

```bash
CLOT_ML_ORACLE_BLOCKAGE="wake" python scripts/build_clot_ml_cache.py --flow fem --out outputs/clot_ml_cache_fem_wake
CLOT_ML_ORACLE_BLOCKAGE="wake" python scripts/build_clot_ml_cache_v4.py --flow fem     --src outputs/clot_ml_cache_fem_wake --out outputs/clot_ml_cache_v5_fem_wake
python scripts/run_phase9_cv.py --tag cpl_wake --cache v5_fem_wake --arm A4 --folds 3 --seeds 1
python scripts/eval_strict.py --tags cpl_wake --cache v5_fem_wake --metric severity --save outputs/deployclot/cpl_wake.json
```

**Second clean seed pair DONE** (open_s3/wake_s3): wall +0.0100 against pair 1's −0.0010, mean
+0.0045, inside the 0.0137 seed null. Off-wall came back **+0.1452** against pair 1's −0.0300 — the two
disagree in sign and the open arm's off-wall null is 0.1417, so off-wall is not established either
way. (Re-scored 2026-09-16 under the mirror-branch rule; before it the wall pairs were +0.0000 /
+0.0002 and the off-wall pairs −0.0312 / +0.1360 — same conclusion.)

> **SCOPE CORRECTION — E1g did NOT test flow coupling, and the docs now say so.** The gelation
> wake rescales `sr`/`dsrx` locally for the *gate computation inside the ODE*. Measured against
> the cache it moves **5 of 55 feature columns** (meaningfully: `log_mat_phys`, `log_mat_owner`,
> `onset_phys` — all one trajectory). **`u`, `v`, `sr`, `spd` and the t=0 gate are unchanged to
> ~1e-7**, i.e. build nondeterminism. So the +0.0001 says *the ODE-gate channel adds nothing*,
> not *coupling adds nothing*.

---

## E1i — how much coupling is needed  ·  **DONE 2026-09-13**

> Six GT-free FEM-coupled arms, 36 vessels, A4, 3 seeds each. Healthy coupled runs vs healthy
> uncoupled runs (2 of 3), degenerate runs screened by one cut-signature rule on BOTH sides.
> Figure `figures/coupling_frequency.{pdf,png}` (`plot_coupling_frequency.py`), data
> `data/coupling_frequency.json`, per-arm `outputs/deployclot/coupling_freq_<stem>.json`.
>
> | arm | re-solves/vessel | wall ±2SE | off-wall ±2SE | degenerate runs |
> |---|---|---|---|---|
> | final flow only (`femfinal`) | 0.8 | −0.0009 ±0.0046 | +0.0126 ±0.041 | 1/3 |
> | every 64 steps (`fem@64`) | 2.7 | −0.0117 ±0.0046 | −0.0408 ±0.041 | 1/3 |
> | every 20 new clot nodes (`fem@1+20`) | 2.9 | +0.0029 ±0.0046 | −0.0004 ±0.041 | 1/3 |
> | every 16 steps (`fem@16`) | 7.2 | +0.0016 ±0.0042 | +0.0084 ±0.037 | 0/3 |
> | every 5 new clot nodes (`fem@1+5`) | 8.4 | +0.0005 ±0.0042 | −0.0543 ±0.037 | 0/3 |
> | every new clot = every step (`fem@1`) | 24.9 | +0.0022 ±0.0042 | +0.0171 ±0.037 | 0/3 |
> | *uncoupled* | 0 | — | — | *1/3 (`open_s3`)* |
>
> **Coupling changes nothing, at any frequency.** Wall within ±0.012, off-wall within ±0.055, no
> dose-response, largest move negative. Three of 12 arm×domain cells sit just past 2 SE, all
> negative (every-64-steps at the wall and, by 0.0002, off-wall; every-5-new-nodes off-wall) —
> no more than 12 comparisons at 2 SE produce by chance. (Re-scored 2026-09-16 under the
> mirror-branch rule, `src/clot_ml/mirror_branch.py`.)
>
> **RETRACTED:** the first version of this table (seed-paired) reported off-wall +0.05 to +0.08
> with p 0.005–0.015 for four arms, and "0 of 3 uncoupled collapses". Every arm's pair 3 was
> measured against the degenerate `cpl_open_s3` (pair-3 delta +0.114 ± 0.038 across arms; pairs 1–2
> −0.006). See STORY 2.1i.
>
> Build cost per vessel (measured, contended CPU): 27 s final-only → 509 s every step.
> `fem@1` needed a 4-rung solver fallback on comsol012 (`v0.solve_fem_velocity_nd`).

## E1h — an actual FEM re-solve  ·  **DONE 2026-09-12 — see STORY 2.1i**

> **RESULT (36 vessels, A4, stride 16, physical Δμ 0.5135 Pa·s, GT-free), corrected 2026-09-13:**
> wall **+0.0016 ± 0.0021** SE, off-wall **+0.0084 ± 0.0185** SE, healthy runs vs healthy runs —
> no effect. *The originally reported off-wall +0.0517 (paired p 0.008) is RETRACTED: its pair 3
> was measured against the degenerate `cpl_open_s3`.* No degenerate run in 3. Leakage gate F1
> 0.8893 vs 0.8215 uncoupled / 0.9742 oracle. Coupled build 113 s/vessel mean vs ~15 s.
> Artifacts: `coupling_fem_resolve.json`, `coupling_leakage_ladder.json`, `cpl_femcpl{,_s2,_s3}`.
>
> **THE BLOCKER WAS TWO BUGS, NOT STIFFNESS.** (1) P2 interpolation of the clot step drove the
> quadrature viscosity to −0.127 Pa·s; clipped at 0 (`local_fem_solver.py`), the solve converges
> at 0.51/0.68/2.0 Pa·s. The old 0.10 Pa·s diagnostic (Jaccard 0.2609 etc.) ran on the same bug and
> is retracted; at physical Δμ it is rel-L2 0.0912, Jaccard 0.4096, firing set roughly doubling.
> (2) Warm-starting from the previous coupled field hits a Picard limit cycle and returned a field
> 0.77 u_ref wrong with only a warning; `solve_fem_velocity_nd(require_converged=True)` now seeds
> analytic, caps the warm attempt at 90 iterations, retries cold and raises. The earlier note that
> "warm start makes the stiff solve worse" was bug (1).
>
> ```bash
> CLOT_ML_ORACLE_BLOCKAGE="fem@16" python scripts/build_clot_ml_cache.py --flow fem --out outputs/clot_ml_cache_fem_femcpl
> CLOT_ML_ORACLE_BLOCKAGE="fem@16" python scripts/build_clot_ml_cache_v4.py --flow fem --src outputs/clot_ml_cache_fem_femcpl --out outputs/clot_ml_cache_v5_fem_femcpl
> python scripts/run_phase9_cv.py --tag cpl_femcpl --cache v5_fem_femcpl --arm A4 --folds 3 --seeds 1 --seed-offset 0   # _s2: 1, _s3: 2
> python scripts/eval_strict.py --tags cpl_femcpl --cache v5_fem_femcpl --metric severity --save outputs/deployclot/cpl_femcpl.json
> python scripts/publication/diag_coupling_arms.py --arm femcpl --out outputs/deployclot/coupling_fem_resolve.json
> python scripts/publication/diag_coupling_leakage.py --caches uncoupled=v5_fem_open wake=v5_fem_wake fem_resolve=v5_fem_femcpl c16_oracle=v5_fem_c16 every_oracle=v5_fem_closed --out outputs/deployclot/coupling_leakage_ladder.json
> ```
>
> **What would move it next:** off-wall is the only open question. More seeds shrink the
> across-seed null; E4's missing meshes are not needed (the arm uses the pack meshes already), but
> more clotted vessels are. The history below is kept for context.

### Pre-fix status (kept)

> **What was run, and what it found.** `scripts/diag_coupled_flow_delta.py` (a local-only probe) solves each vessel
> twice — clot-free, and with a clot viscosity elevation on the clotted nodes — and compares what
> the deposition law reads.
>
> **The gate churns ~74%.** Velocity rel-L2 0.0699, firing-set size nearly unchanged (44→43,
> 59→50, 50→77), **gate Jaccard 0.2609**, 183 wall nodes switching state across 3 vessels.
> Size is stable; identity is not — §2.4's mechanism, measured directly.
>
> **This reframes the whole leg.** The wake arm's +0.0001 (E1g) came from a corrector that never
> touched `u`/`v`/`sr`. There IS physical content in coupling; our proxy could not carry it.
>
> **THE BLOCKER IS NUMERICAL, NOT COMPUTATIONAL.** The solver **diverges** at Δμ = 0.68 Pa·s
> (|v|max 226 vs 0.43) and COMSOL's own `mu1` step is ~0.51 — inside the divergent regime. The
> measurement above ran at **0.10 Pa·s**, ~5× below physical, so it is a LOWER bound. Also found:
> warm-starting the stiff solve from the clot-free field makes it *worse* (relative step 1.00 vs
> 0.227 cold) — the opposite of its effect on the clot-free problem.
>
> **To finish this properly, in order:** (1) make the clot-loaded solve converge at ~0.5 Pa·s —
> continuation in Δμ, stronger stabilisation, or a Newton step are the obvious candidates;
> (2) re-run this diagnostic at physical viscosity; (3) only then rebuild features from the
> re-solved field and retrain. Step 1 is the real work and it is solver engineering, not ML.

### Original scoping (kept)

**Everything to date has tested proxies.** The oracle arms leaked the label (E1f). The wake arm is
clean but touches only the ODE gate (above). **Nobody has re-solved the Navier–Stokes problem
against the clot-laden viscosity field** — the thing the user proposed, the thing E1d priced at
1.95×–4.96×, and the only version that updates `u`, `v`, `sr`, `spd` and every flow-derived
feature.

**What it needs:** at stride ~16, rebuild the viscosity field from the rollout's current `Mat`
(`mu1(Mat)` steps at `viscosity_mat_crit`, exactly as COMSOL does), re-run
`local_fem_solver`, and re-derive the flow features from the new field. The pieces exist —
`solve_fem_into_pack`, the stride grammar, the 3-fold protocol — but no call site chains them.

**Why it is worth the GPU-hours:** it is the only route to a defensible sentence about coupling.
Today the honest claim is "a gate-level correction buys nothing"; a reader will reasonably ask
whether re-solving the flow would, and we cannot answer.

**Cost, from E1d:** 1.95× runtime at stride 16, 4.96× at every deploy step — the figures that do
**not** apply to the wake proxy but do apply here.

### Original scoping (kept)

**The replacement E1f called for, and it already existed in the tree.**
`src/core_physics/gelation_wake.make_gelation_wake_blockage` closes the clot→flow loop from the
rollout's **own** committed set (`mat >= crit`) and never reads ground truth. It was wired into
`features.build_features` so a non-leaking coupled cache can be built:
`CLOT_ML_ORACLE_BLOCKAGE="wake[@<every>]"` (the variable is named ORACLE_ for history and the env
surface is frozen; **this arm is not an oracle**).

**IT PASSES THE LEAKAGE GATE — this is the load-bearing check, run it first on any future arm.**
Can `mat_phys` alone recover the wall label?

| arm | best F1 | AUC | verdict |
|---|---|---|---|
| uncoupled | 0.6905 | 0.7825 | clean |
| **wake (self-driven)** | **0.7322** | **0.8155** | **clean** |
| oracle (GT-driven) | 0.9613 | 0.9982 | **LEAKS** |

A modest lift consistent with real physics, not label copying — and it genuinely couples (23–47
nodes of `mat_phys` move per vessel). The v4 builder's `w_phys` diagnostic agrees: 0.5867
uncoupled → **0.6199 wake** → 0.7336 oracle.

**DONE and on disk:**

| artifact | state |
|---|---|
| `features.py` wake branch | wired, trial-verified |
| `outputs/clot_ml_cache_{fem,v5_fem}_wake` | 36 vessels, matches the other arms' vessel set exactly |
| `outputs/phase9_scores/cpl_wake{,_s2}.npz` | **both seeds trained** |

**REMAINING — evaluation and comparison only, no GPU needed:**

```bash
python scripts/eval_strict.py --tags cpl_wake    --cache v5_fem_wake --metric severity --save outputs/deployclot/cpl_wake.json
python scripts/eval_strict.py --tags cpl_wake_s2 --cache v5_fem_wake --metric severity --save outputs/deployclot/cpl_wake_s2.json
```

Then pair against `cpl_open{,_s2}` exactly as E1c did — seed-averaged delta, and **judge it against
the seed null**, which is where the stride comparison went wrong. Write
`outputs/deployclot/coupling_selfdriven.json` and add `cplself.*` ledger rows.

**This is the number the project has been trying to get all along:** the first estimate of what
clot→flow coupling is worth that is not contaminated by the answer. Whatever it says — including
"nothing" — it is the one that can go in the paper.

> **Do not quote any earlier coupling figure as its preview.** E1/E1b/E1c/E1e are all contaminated
> (E1f) and their +0.0343 tells us nothing about what this will show.

---

## E1f — THE ORACLE INSTRUMENT LEAKS THE LABEL  ·  **found 2026-09-12; invalidates E1/E1b/E1c/E1e**

> **What went wrong, and it is upstream of four experiments.** `oracle_blockage` lowers shear
> exactly where the GT clot is, so the coupled `mat_phys` encodes the answer — for held-out
> vessels too, since each vessel's features come from its own GT. **That one feature, thresholded,
> recovers the wall label at F1 0.9609 / AUC 0.9966**, against 0.7950 / 0.8610 uncoupled
> (`outputs/deployclot/coupling_leakage.json`).
>
> **Consequence.** Every coupling number measured through it is contaminated: E1's null, E1b's
> ceiling, E1c's +0.0343, E1e's stride comparison. The +0.0343 is mostly the model reading a
> leaked label, not a flow-physics gain. **The value of coupling is UNKNOWN, not small** — and the
> docs now say so rather than quoting a bound.
>
> **Why care could not have fixed it.** A real coupled solve is driven by the model's PREDICTED
> clot: its feedback carries the model's own errors and leaks nothing. The oracle swaps in truth,
> which perfects the correction AND injects the answer at once. Inseparable by construction.
>
> **THE REPLACEMENT EXPERIMENT — now the project's highest-value open item.** Drive the re-solve
> from the model's own clot estimate at stride ~16. Cheap: the stride grammar
> (`CLOT_ML_ORACLE_BLOCKAGE="1@16"`), the caches, and the 3-fold protocol all exist; what is
> needed is a blockage callable that reads the rollout's current `Mat` instead of
> `gt_clot_phi_at_time`.
>
> **The lesson worth keeping.** An "oracle upper bound" is only a bound if the oracle injects
> *capability*, not *answers*. Before trusting one, check whether a single oracle-derived feature
> predicts the label on its own — that check takes minutes and would have caught this four
> experiments ago.

---

## E1 — back the oracle closed loop with an artifact  ·  **DONE 2026-09-12 (result CONTAMINATED, see E1f)**

> **RAN.** `--flow fem --every 4`, n=8 intact + 6 injured →
> `outputs/deployclot/closed_loop_oracle.json`, figure `closed_loop_oracle.{pdf,png}`, six ledger
> rows now PASS.
>
> **The result did NOT match the transcribed numbers, and the story changed.** Wall is flat
> (+0.0002, n=8; 5 of 8 vessels bit-identical) and the wound lumen is unchanged to the last digit
> — that half held. **Off-wall it is −0.1604 on all three vessels that have off-wall GT**, not the
> +0.0143 the note claimed. Most likely a train/test mismatch (the readout was fitted on
> open-loop features), which this experiment cannot separate from a statement about coupling.
> See STORY 2.1, which now says so. Two traps recorded there: the n=3 permutation floor of
> p=0.25, and the fact that a killed background run leaves a silently partial artifact.

**Why it is first.** Leg 2 is a chain — *the closed loop is flat, therefore one solve suffices,
therefore the ceiling is 7.9%, therefore choose on accuracy and FEM wins.* §2.1 is the first link,
and it rests on numbers transcribed by hand from a note: no artifact, no claim ids, six STALE
ledger rows. Until this runs, the leg's premise is unpublishable and everything downstream of it
reads as assertion.

```bash
python scripts/eval_closed_loop_oracle.py --cohort --out outputs/deployclot/closed_loop_oracle.json
PYTHONPATH=. python scripts/publication/verify_claims.py --paper docs/publication/STORY.md
```

**Cost.** Two full cohort evaluation passes (open-loop and `CLOT_ML_ORACLE_BLOCKAGE=1`), sharing
one interpreter and one set of loaded packs.

**What would change the story.** If the deltas come back *outside* the noise floor, the chain
breaks at its first link: per-step coupling would be worth something, the flow stage would no
longer be a single 7.9% solve, and Leg 2 would have to be rebuilt around the ceiling alone. If
they land inside, as the hand-computed run says, §2.1 is quotable and [FIGURES.md](FIGURES.md) 2.1
becomes drawable.

**Known trap.** `CLOT_ML_ORACLE_BLOCKAGE` was silently hardcoded off by the 2026-09-04 env
cleanup and the break was invisible because blockage-on and blockage-off produced *identical*
output rather than erroring. Fixed 2026-09-07. **Assert the two arms differ before trusting the
diff** — if they are bit-identical, the flag is off again, not the physics.

---

## E2 — give the wound half of Leg 0 claim ids  ·  **BLOCKING Leg 0.2**

**Why.** Leg 0 now claims the tool works on *injured and intact* vessels — so the wound numbers
are load-bearing for the headline, not a side leg. The n=6 LOVO table is real and reproducible and
currently quoted nowhere a machine can check.

```bash
python scripts/train_wound_rate.py --flow fem     # writes outputs/clot_ml/wound_rate_fem/lovo.json
python scripts/eval_wound_complement.py
python scripts/eval_wound_ab_pair.py --model clot_ml_0 --flow fem
```

Then add ledger rows to `scripts/publication/verify_claims.py` for: the three-arm LOVO table
(physics / two-constant / per-node), the held-out `w_reg` and `w_lum`, and `G_pre` / `G_post`
with their fold spreads.

**What would change the story.** If the per-node `WoundRateNet` ever *wins* LOVO at a larger n,
the wound stops being Leg 1's thesis on a new geometry class ("physics wins where the network has
no signal") and becomes "the network needed more data" — materially weaker and much more ordinary.
It has lost at n=3 and n=6; the next wound simulations are the test that matters.

---

## E2b — give BATC quotable numbers  ·  **DONE 2026-09-12**

> **RAN.** `scripts/publication/diag_batc_decomposition.py` →
> `data/batc_decomposition.json`, five ledger rows PASS.
>
> **It does not reproduce DEPLOYCLOT §0.4's decomposition, and the story now says so.** On the
> wound set the knobs compound to **+0.0668** (`w_reg`), not +0.192, with two substantive
> disagreements: shape weight dominates tolerance (0.052 vs 0.027, where DEPLOYCLOT has them
> tied), and **β is NEGATIVE** (−0.0195) against the +0.002 claimed — F₁ penalises a
> high-precision prediction that F₀.₅ rewarded. This is the same effect on a **different set**
> (wound domains, not deploy-cohort off-wall), so it neither confirms nor refutes +0.192; that
> number is still unverified on its own cohort, which needs a deploy-cohort prediction dump.

**Why.** Leg 0 now argues BATC as a designed instrument, and [FIGURES.md](FIGURES.md) 0.3 draws it.
The definition is tested and the +0.192 knob-by-knob decomposition is stated in `DEPLOYCLOT.md`
§0.4 — but as prose, with no artifact and no ledger rows. A metric the paper *argues for* cannot
have unverifiable numbers attached.

**What to produce:** a small artifact recording, on one fixed set of predictions and masks, each
knob's contribution (tolerance 2→4, shape weight 0.5→0.2, the graces, β 0.5→1.0) and the compounded
total, plus the burden-vs-recall curve behind the figure's second panel. Then ledger rows for each.

**What would change the story.** If the decomposition does not reproduce at +0.192, the "never
quote one against the other" rule keeps its force but loses its size — and the figure's annotation
changes.

---

## E3 — re-run the novelty triage before submission  ·  **process rule, not optional**

**Why.** The Phase 0 triage missed a paper published in May 2026 in a journal on our own target
venue list, because it searched the ML-thrombosis and mesh-GNN-hemodynamics literatures
separately and the near neighbour sits in the intersection. That is the failure that would have
surfaced at review.

**Protocol.** Search the **intersection** explicitly. PubMed and Europe PMC as well as arXiv and
Google Scholar. Re-run immediately before submission, not once at project start.

**Last pass: 2026-09-11** — found Jeken-Rico et al., *npj Digital Medicine* 9:212 (2026), a
physics-constrained GNN for real-time aneurysm hemodynamics that generalizes to unseen geometry
with no fine-tuning. Not a threat (it predicts hemodynamics, not a clot field, and has no
thresholded downstream consumer) but it **must be cited** — see [RELATED_WORK.md](RELATED_WORK.md).

---

## E4 — the cheapest hardening available: export the missing meshes

**Why.** 22 deploy vessels have no `.nas` mesh, so the FEM-vs-GT comparison runs on the 8 that do.
Exporting the rest takes the surrogate-requirement result from n=4–5 to n≈30 and settles the
`dsrx`-correlation discrepancy. **This is the single cheapest thing that would harden Leg 2.**

**What would change the story.** A larger n could move the FEM-vs-GT gap (currently δ −0.036,
p=0.022 on the paired n=27) enough to change how Leg 2b frames the classical solve's own
accuracy.

---

## E1b — the FAIR coupling test  ·  **DONE 2026-09-12**

> **The experiment E1 could not be.** E1 scored an oracle-coupled field with a readout tuned on
> open-loop features, so its −0.1604 off-wall confounded "coupling does not help" with "our
> readout cannot consume this field". `scripts/eval_coupling_ceiling.py` gives **each arm its own
> perfect readout** — rank by that arm's own score field, sweep the committed set size, keep the
> best — and compares ceiling to ceiling.
>
> **Result: the ceilings are the same.** wall +0.0036 ⟨coupling.wall_delta⟩, off-wall −0.0072
> ⟨coupling.off_delta⟩, n=3, signs disagreeing between domains. A perfectly coupled feature set
> does not rank clot nodes better than the uncoupled one. So E1's off-wall regression **was**
> tuning mismatch, and — the stronger statement — **building a coupled-native readout would not
> pay.**
>
> **A methodological trap worth keeping.** The first version swept a quantile grid and reported
> ceilings *below* the scores the real pipeline achieves. That is impossible for a true ceiling
> and was proof the family was wrong: `comsol005` has 4 GT positives among 9623 lumen nodes, so
> no global threshold can express the answer. Top-k fixed it.
>
> **What it still cannot settle:** feature sets under a perfect readout, not architectures. A
> network trained from scratch on closed-loop features might extract signal the shipped score
> field does not expose. n=3, and the ceilings are not deployable numbers.

---

## E1c — train a network FOR perfect coupling  ·  **DONE 2026-09-12, and it moved the story**

> **The experiment E1b named as its own limit.** Two feature caches differing only by
> `CLOT_ML_ORACLE_BLOCKAGE`, the same arm trained from scratch on each, same 36 vessels, same 3
> folds (membership verified identical), twice each on disjoint seeds to measure the seed null.
>
> **Wall: +0.0343, exceeds noise, reproduces across seeds.** E1b's ranking test said the coupled
> field carried no extra reachable signal *for the shipped representation*; a network trained on
> it does extract wall signal. **E1b's stated limit was real and it bit** — the honest lesson is
> that a "ceiling" measured through one fixed representation is a ceiling for that
> representation, not for the information.
>
> **Off-wall: not established, and the interesting number is the variance.** The −0.0910 sits
> inside the closed arm's own seed noise (0.1234), which is **9×** the open arm's. Coupled
> training buys instability off-wall, not accuracy.
>
> **The oracle is the instrument, not the only route.** It reads GT clot at t>0, so a model
> trained on it is not deployable — but **coupling itself is perfectly attainable**: re-solve the
> FEM each step against the current clot's viscosity field. E1d prices that. So +0.0343 is the
> ceiling of an ACHIEVABLE thing, which is a stronger and more useful statement than "oracle
> only", and the argument becomes an exchange rate rather than a possibility claim.
>
> **Controls that mattered.** (1) The pre-existing open v3 cache was checked against a fresh
> flag-off rebuild: all differences are ~1e-8 (FEM nondeterminism), against an O(1) relative
> blockage effect on `mat_phys` — so the caches differ by the flag and nothing else. (2) SEALED
> (comsol007/013/031/043) is excluded by the builder, correctly, leaving 36.
>
> **Open:** nothing tests a *deployable* corrector. The space between +0.0343 (oracle) and E1's
> 0.0000 (naive bolt-on) is unmeasured.

```bash
CLOT_ML_ORACLE_BLOCKAGE=1 python scripts/build_clot_ml_cache.py --flow fem --out outputs/clot_ml_cache_fem_closed
CLOT_ML_ORACLE_BLOCKAGE=1 python scripts/build_clot_ml_cache_v4.py --flow fem     --src outputs/clot_ml_cache_fem_closed --out outputs/clot_ml_cache_v5_fem_closed
python scripts/run_phase9_cv.py --tag cpl_closed --cache v5_fem_closed --arm A4 --folds 3 --seeds 1
```

---

## E1d — what coupling would cost  ·  **DONE 2026-09-12 (arithmetic on measured timings)**

> **Corrects a framing error.** Earlier drafts leaned on "the oracle is not deployable" as if that
> disposed of coupling. It does not: re-solving the FEM at each step needs no ground truth, only
> time. Priced from `timing.json` (one solve 4.66 s, deploy total 58.9 s, 51 steps on the deploy
> grid):
>
> | re-solve | solves | runtime | vs now | vs COMSOL |
> |---|---|---|---|---|
> | once (shipped) | 1 | 58.9 s | 1× | 2,934× |
> | every 16th step | 13 | 114.8 s | 1.95× | 1,505× |
> | every deploy step | 51 | 291.9 s | 4.96× | 592× |
> | every step, full grid | 201 | 991 s | 16.83× | 174× |
>
> **Affordability is NOT the argument** — fully coupled still answers in five minutes at ~600×
> COMSOL. The argument is the exchange rate: ~5× runtime for a +0.0343 wall *ceiling* that a real
> solve would only partly realise, with nothing established off-wall. `k × fem` is a lower bound
> (timed on a clot-free field; re-featurising uncounted), which strengthens the judgement.
>
> **THE OPEN EXPERIMENT, and it is now Leg 2's main risk.** Nobody has measured a point between
> "solve once" and "solve every step". Every-16th costs 1.95×; if it captured even half the
> +0.0343 the trade would look very different. Measuring one intermediate point is the highest-value
> run left in this leg.

---

## E1e — the every-16th-step coupling point  ·  **DONE 2026-09-12: stride is a COST lever, not an accuracy one**

> **Stride-16 reaches the every-step wall gain at 40% of the cost:** +0.0368 at 1.95× against
> +0.0343 at 4.96×.
>
> **A FIRST READING OF THIS SAID "COARSE DOMINATES FINE". THAT WAS WRONG AND IS RETRACTED.**
> Against the seed null the k16-vs-every differences are **inside noise**: wall +0.0025 (noise
> 0.0069), off-wall +0.1154 (noise 0.1234). The apparent off-wall reversal rides entirely on the
> every-step arm's off-wall seed instability.
>
> **Why stride barely matters, measured in the features rather than asserted.** Across the 36
> vessels' `mat_phys`: correlation with the label is 0.7831 uncoupled, **0.9515 at stride 16,
> 0.9556 at every step**; ignited wall fraction 0.1027 / 0.1313 / 0.1330. The whole jump is
> uncoupled → coupled. Refining stride 16 → 1 adds almost nothing, so there is no "jitter" for a
> coarse update to avoid, and the transient-chasing story first offered here is **not supported by
> the features and should not be repeated.** The clot field evolves slowly enough that a 16-step
> update resolves it.
>
> **The usable rule is a cost rule:** if you couple, couple coarsely — finer is not measurably
> better and costs 2.5× more.
>
> **1 of 3 k16 runs failed and is excluded from the headline, with the number shown anyway.**
> Seed 0 had a fold-level readout collapse — all 12 fold-0 held-out vessels at 0.001–0.29, from a
> degenerate cut (2nd `resid` parameter 0.05 vs 0.95–0.98 healthy). Including it: wall −0.0637,
> spread 0.3020. No open/closed run (2/2 each) failed this way.
>
> **This retires 2.1d's "not worth it yet" as stated** — that priced the wrong variant. STORY
> 2.1e now carries the narrowed claim.
>
> **THE NEXT EXPERIMENT, now the highest-value one in this leg: the DEPLOYABLE version.** Every
> coupling number so far is oracle-driven (GT clot). Drive the re-solve from the model's own clot
> estimate at stride 16 and measure what survives. Also unresolved: the 1-in-3 collapse, and n=36
> / 3 folds / 2–3 seeds is a hint by this project's own standard.

### Artifacts and how to reproduce

| artifact | what |
|---|---|
| `outputs/deployclot/coupling_stride.json` | the comparison, healthy and contaminated figures both |
| `outputs/clot_ml_cache_{fem,v5_fem}_c16` | stride-16 caches, 36 vessels |
| `outputs/phase9_scores/cpl_c16{,_s2,_s3}.npz` | three seeds |

```bash
CLOT_ML_ORACLE_BLOCKAGE="1@16" python scripts/build_clot_ml_cache.py --flow fem --out outputs/clot_ml_cache_fem_c16
CLOT_ML_ORACLE_BLOCKAGE="1@16" python scripts/build_clot_ml_cache_v4.py --flow fem     --src outputs/clot_ml_cache_fem_c16 --out outputs/clot_ml_cache_v5_fem_c16
python scripts/run_phase9_cv.py --tag cpl_c16 --cache v5_fem_c16 --arm A4 --folds 3 --seeds 1
```

**Operational notes that cost time here.** (1) Process checks must use
`Where-Object { $_.ProcessName -like 'python*' }` — `-Name python` silently matches nothing
because the interpreter is `python3.13`, which let overlapping GPU runs be launched. (2) A CV run
needs ~9.5 min; an inner `timeout 592` kills it ~20 s short, so omit the inner timeout and let the
harness background it. (3) `nvidia-smi` lists the user's Crusader Kings III — never sweep the GPU
process list.

### Original scoping (kept)

**The question.** E1c measured the ceiling of coupling at EVERY step (+0.0343 wall); E1d priced
it at 4.96x runtime. Re-solving every 16th step costs only **1.95x** — so: how much of the
+0.0343 does it capture? If most of it, the exchange rate changes and Leg 2's judgement with it.

**DONE and on disk (no need to redo):**

| artifact | state |
|---|---|
| `src/clot_ml/features.py` stride grammar | **committed to the working tree.** `CLOT_ML_ORACLE_BLOCKAGE` now parses `"<ratio>[@<every>]"` — `"1@16"` re-evaluates the coupling every 16th step. Rides on the EXISTING variable so `test_env_registry.py`'s frozen surface is not widened |
| `outputs/clot_ml_cache_fem_c16` | 36 vessels, stride-16 v3 features |
| `outputs/clot_ml_cache_v5_fem_c16` | 36 vessels, v4 layer |
| `outputs/clot_ml_cache_v5_fem_open` / `_closed` | the two comparison arms, already built |
| `outputs/phase9_scores/cpl_open{,_s2}.npz`, `cpl_closed{,_s2}.npz` | trained, 2 seeds each |

**REMAINING — two training runs and an eval:**

```bash
python scripts/run_phase9_cv.py --tag cpl_c16    --cache v5_fem_c16 --arm A4 --folds 3 --seeds 1
python scripts/run_phase9_cv.py --tag cpl_c16_s2 --cache v5_fem_c16 --arm A4 --folds 3 --seeds 1 --seed-offset 1
python scripts/eval_strict.py --tags cpl_c16    --cache v5_fem_c16 --metric severity --save outputs/deployclot/cpl_c16.json
python scripts/eval_strict.py --tags cpl_c16_s2 --cache v5_fem_c16 --metric severity --save outputs/deployclot/cpl_c16_s2.json
```

Then extend `outputs/deployclot/coupling_trained.json` with the k16 arm and add `cpl16.*` ledger
rows. **The comparison to report is `(k16 - open) / (every-step - open)` at the wall** — the
fraction of the +0.0343 ceiling that 1.95x buys.

**A HINT ALREADY IN HAND, and it is worth stating because it predicts the answer.** The v4
builder's own `w_phys` correlation diagnostic reads **open +0.5867, k16 +0.7287, every-step
+0.7336** — stride-16 sits at ~97% of the way from uncoupled to fully coupled *at the feature
level*. If that carries through training, most of the ceiling is available at 1.95x rather than
4.96x, and Leg 2.1d's "not worth it yet" needs revisiting rather than restating.

> **WHY IT PAUSED, and the operational lesson.** The `cpl_c16` run was killed after 30.8 min
> against the ~9.0 min identical runs take. Cause: **every process check this session used
> `Get-Process -Name python`, but the interpreter is named `python3.13`**, so the checks
> reported "nothing running" while runs were live, and overlapping GPU jobs were launched.
> **Check with `Where-Object { $_.ProcessName -like 'python*' }` or `nvidia-smi`, never
> `-Name python`.** Also: `nvidia-smi` on this box lists a Crusader Kings III process — it is
> the user's game, not ours; never kill by sweeping the GPU list.

---

## E5 — the two flow-coupling regimes never measured

Both are open questions from the old notes, both cheap relative to what they settle.

* **`--cohort` under GT flow.** The oracle-gate conclusion rests on 8 vessels whose baseline
  (0.9844) sits well above the cohort-wide ~0.92, so the saturation argument may be an artifact of
  an easy subset. **Either hardens Leg 2a or qualifies it.**
* **The oracle under `--flow pred`.** The only regime where flow coupling is still plausibly live
  — deploy score collapses to wall 0.586 / off 0.350, i.e. ~0.37 of headroom. **Never run.** If
  coupling pays anywhere, it pays here, and Leg 2a should say so rather than be silent about the
  one regime it did not test.

---

## E6 — timing and reproducibility debts

* **Time a COMSOL t=0 flow solve properly.** The ~45 s figure is recollection. With a real log it
  becomes a second, separate comparison (stage-level: COMSOL vs FEM vs RGP-DEQ) distinct from the
  pipeline-level 2,934×. Without one it stays out of the manuscript.
* **Restore or deliberately retire the `preflight_validation` generator.** Two claims currently
  resolve against artifacts that can no longer be regenerated, which violates
  [EVIDENCE.md](EVIDENCE.md)'s own rule. A copy exists under
  `dist/LocalFEMSolver-Predict-win64-1.0/scripts/`.

---

## Running conventions on this machine

Learned the hard way; each has cost a bad artifact at least once.

* **This shell is Git-Bash on Windows.** No `pkill`/`pgrep`. Use PowerShell for process work.
* **`CUDA_VISIBLE_DEVICES=""` does not disable CUDA here** — it leaves CUDA "available" with 0
  devices, and the failure is silent. Check device count, not availability.
* **Never `git stash` to produce a benchmark baseline.** A concurrent writer makes the pop
  conflict. Build baselines as scratchpad copies instead.
* **Prefer extending an existing script over forking a variant.** This repo has already paid for
  script sprawl once; the env-registry guard exists to keep the configuration surface shrinking.

---

## E7 — make the wound and intact halves of Leg 0 comparable  ·  **OPTION 2 DONE 2026-09-12**

> **RAN, via the recommended cheap path.** `scripts/eval_wound_complement.py` gained
> `--flow`, `--lovo`, `--save-series` and `--save-summary`; the n=6 FEM leave-one-vessel-out run
> wrote `data/wound_series_fem.npz` + `data/wound_lovo_fem.json`, and
> `plot_wound_temporal.py` draws it. Five ledger rows now PASS.
>
> **It also caught an unbacked claim.** STORY quoted held-out `w_reg` 0.9270 / `w_lum` 0.8611;
> those numbers exist in the repo **only as hardcoded literals in `promote_clot_ml_0.py`**, and
> describe a different arm. The measured values are **0.8789 / 0.7970**. STORY 0.2 now carries the
> measured pair and a note not to requote the old one.
>
> **Option 1 (a true wound OOF) remains open** — the five-blocker chain below is unchanged, and
> the protocol difference is now printed on the figure instead of being hidden.

**Why, and it is a claim problem before it is a figure problem.** Leg 0 says the tool works on
injured *and* intact vessels. Those two halves are currently evidenced by **different protocols**:

| half | protocol | artifact |
|---|---|---|
| intact | strict nested CV, out-of-fold trajectories, 27 vessels | `outputs/publication/data/clot_ml_0_oof_series.npz` |
| injured | leave-one-vessel-out over the complement's two scalars, GNN base fixed, n=6 | `outputs/clot_ml/wound_rate_fem/lovo.json` |

**The OOF archive contains zero wound vessels** (verified 2026-09-11). So the headline result
figure cannot put a wound row beside an intact row without implying a comparability that does not
exist.

### The reframe that came out of tracing this (2026-09-11)

**The wound vessels are already held out in a STRONGER sense than the intact ones** — and the
docs did not say so. The `v5_fem` feature cache carries **40 vessels and not one wound pack**, so
no wound vessel has ever entered GNN training. The intact vessels are held out *fold-wise*; the
wound vessels are unseen *entirely*, and the only fitted quantities touching them are the
complement's two scalars, which are already LOVO'd.

So the gap is **not** that wound numbers lack held-out status. It is that they come from a
different protocol, so they are not directly comparable, and there is no per-timestep trajectory
archive for [FIGURES.md](FIGURES.md) 0.4. Say it that way in Leg 0 — the current phrasing
undersells the wound evidence.

### The actual dependency chain for a true wound OOF (traced, not guessed)

A wound OOF is **not a flag on an eval script.** In order:

| # | blocker | evidence |
|---|---|---|
| 1 | **No wound features in any cache.** The CV pool is built from the cache (`run_phase9_cv.py:176`), so wound vessels cannot enter until a cache carries them | `load_cache("v5_fem")` → 40 vessels, zero wound |
| 2 | **`classes_for` does not cover wound packs**, and the pool is filtered by it (`run_phase9_cv.py:180`) | `classes_for` returns 40, zero wound |
| 3 | **`eval_strict_temporal.py` excludes wound packs outright**, and says so in its own header | lines 62–70 |
| 4 | **Two mask helpers would be silently wrong.** `_lag_masks` and `_owner_lag_masks` use a bare `~wall`, which is only correct *because* no wound is in the cohort. `src/tests/test_eval_domains.py` fires the moment one enters — the test is the tripwire, and those are the call sites to fix | lines 66–70 |
| 5 | **No existing CV arm has a wound in its pool**, so a new `outputs/phase9_scores/<tag>.npz` must be produced by an actual training run | scanned every arm, zero wound |

**Only after 1–5 does `--save-oof-series` emit wound trajectories.** Steps 1 and 5 each cost a
real run; step 4 is a correctness fix that must land before any wound number from this path is
trusted.

**Two ways out, and they are not equivalent.**

1. **Do 1–5 and get a true wound OOF** — comparable to the intact table, one figure for both.
   Expensive, and step 4 is a genuine correctness prerequisite.
2. **Emit held-out wound trajectories from the deploy path with LOVO constants** — cheap, uses the
   existing `eval_wound_complement.py` machinery, unblocks 0.4's wound row and gives new numbers,
   but the protocol label on the figure must say "leave-one-vessel-out, GNN never trained on any
   wound vessel" rather than "out-of-fold". **Recommended first**, because it is honest, it is
   days cheaper, and per the reframe above it is arguably the stronger claim anyway.

**What would change the story.** If a wound OOF series cannot be produced at n=6 without leaking
(the complement's scalars are fitted on so few vessels that a truly held-out trajectory may be
unstable), then option 2 is forced, and Leg 0.2 should say plainly that the injured half is
evidenced leave-one-out rather than out-of-fold.
