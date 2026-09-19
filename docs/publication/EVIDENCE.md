# Evidence — how every number is checked

**The rule, and it has teeth:** a number without a claim id has not been re-derived from an
artifact and **does not go in a draft, a figure caption, a talk, or an abstract.** Four claims
were retracted in one session because point estimates were quoted without their intervals, or
because a vacuous run produced a clean-looking artifact. A prose rule cannot catch that. The
checker can, because it reads the artifact rather than the paragraph.

```bash
PYTHONPATH=. python scripts/publication/verify_claims.py
PYTHONPATH=. python scripts/publication/verify_claims.py --paper docs/publication/STORY.md
```

Exit code = FAIL + STALE count, so a release step can gate on it.

---

## 1. How the ledger works

A row is `(claim id, where it is stated, the value stated, the artifact path that produces it,
tolerance, note)`. Three outcomes:

| | |
|---|---|
| **PASS** | the doc's number is what the artifact says |
| **FAIL** | the doc's number is **not** what the artifact says — fix one of them before quoting |
| **STALE** | the artifact is missing, so the claim currently rests on nothing re-derivable |

**A tolerance is not a fudge factor.** It is the rounding the doc itself uses: "0.9127" checks to
5e-5, "~2,934×" to 1.0, "~73%" to 0.01. If a doc rounds harder, widen the row and say so. Never
widen a tolerance to make a mismatch go away.

**Add the row before you quote the number**, not after. The `--paper` mode also scans a doc for
`⟨claim-id⟩` tags and fails on any that do not resolve, so a tag you invent without a row is
caught rather than silently trusted.

---

## 2. Claim families, and what each leg currently rests on

| family | leg | artifact | state |
|---|---|---|---|
| `timing.*` | 0, 2.2 | `outputs/publication/data/timing.json` | BACKED |
| `t4.*` | 0 | `outputs/publication/data/table4_kfold.json` | BACKED |
| `sealed.*` | 0 | `outputs/deployclot/eval_sealed.json` | BACKED — **spent, never re-read** |
| `pvlf.*` | 1.1, 1.2 | `outputs/ablation/physics_vs_learned_fem.json` | BACKED |
| `mgn.*` | 1.2 | `outputs/ablation/physics_vs_learned_mgn.json` | BACKED |
| `rgp.*` | 2.3 | `outputs/deployclot/crossfit5_vs_shipped.json` | BACKED |
| `rgpdeq.*`, `fem.*`, `flowreq.*` | 2.3, 2.4 | `outputs/publication/data/flow_requirement.json`, `outputs/deployclot/fem_flow_audit.json` | BACKED |
| `c0.*` | 1.1 | `outputs/deployclot/c0_ablation_paired.json` | BACKED |
| `wc.*`, `odet.*`, `tmp.*` | 1.1 | `outputs/ablation/temporal_*.json` | BACKED |
| `closedloop.*` | 2.1 | `outputs/deployclot/closed_loop_oracle.json` | **BACKED** — E1 run 2026-09-12 |
| `wound.lovo.*`, `wound.base.*` | 0.2 | `outputs/publication/data/wound_lovo_fem.json` | **BACKED** — E7 option 2, n=6 FEM LOVO, 2026-09-12 |
| **BATC definition + the +0.192 decomposition** | **0.3** | `src/clot_ml/severity_metric.py`, `src/tests/test_severity_metric.py` | **UNBACKED as quotable numbers** — the knob-by-knob decomposition is stated in `DEPLOYCLOT.md` §0.4 but has no ledger row |

---

## 3. Conventions that change numbers

These are not pedantry. Each one has already produced a wrong published-looking number in this
project at least once.

**One metric, two settings, never mixed in one table.** **BATC** (Burden-Adjusted Thrombus
Concordance) is the reported score everywhere; **BATC₀** is the unadjusted setting and appears in
supplementary tables only. They are one function — `src/clot_ml/severity_metric.py`, where
`BATC = DEFAULT` and `BATC_0 = LEGACY` — and **BATC₀ reproduces `evaluate.domain_score` exactly**
(verified to 0.00e+00 over 13 vessel-domains, 2026-09-03).

| | BATC (reported) | BATC₀ (unadjusted) |
|---|---|---|
| tolerance `k` | 4 hops | 2 hops |
| detection `β` | 1.0 (F₁) | 0.5 (F₀.₅) |
| shape weight `w` | 0.2 | 0.5 |
| miss grace | 15 nodes, ≤25% of \|G\| | none |
| FP grace | 6 nodes, ≤15% of \|P\| | none |
| CLI | `--metric severity` (default) | `--metric legacy` |

BATC₀ is stricter on all four axes at once, compounding to **+0.192** off-wall on the *same
predictions*: tolerance +0.066, shape weight +0.066, graces +0.041, β +0.002. **None of that is a
model difference.** Quoting one against the other once manufactured a SEALED "off-wall collapse"
of 0.22 that did not exist. Every table must name its setting; `eval_clot_ml_0.py` now returns
both on every call and the pointer records both.

Internal names (`severity`, `guiding`) still appear in code and older docs. Neither reads as a
metric in an abstract — **the manuscript says BATC.**

**Two noise floors, with different scopes. Do not interchange them.**

| floor | scope | value |
|---|---|---|
| deploy cohort, config spread of one arm | wall / off-wall | ±0.024 / ±0.074 |
| deploy cohort, *per-vessel* spread | wall / off-wall | median 0.042 / 0.112 |
| `table4_kfold` config floor | wall / off-wall | 0.0037 ⟨t4.noise_floor.wall⟩ / 0.0432 ⟨t4.noise_floor.off⟩ |

**Never a mean without its paired interval.** A point estimate inside a noise band is not a
directional result. Say "inside noise", not "slightly helps" or "slightly hurts" — the oracle
closed-loop comparison has already flipped sign between runs inside the same band.

**n < 10 vessels at one operating point is a hint, not a finding.** This project has retracted
several such numbers, most recently an n=1 tolerance threshold that did not survive n=3.

**State the sign convention beside any correlation.** A correlation without one, or a domain mean
without saying which vessels entered it, is not a reportable number. Both failures have happened
here.

**Mirror-branch vessels are scored against their mirrored ground truth (2026-09-16).** On a
mirror-symmetric vessel the steady flow is bistable, and our FEM solve can land on the mirror image
of COMSOL's (a Coanda pitchfork). Of 54 packs, 5 meshes are symmetric and 2 flip —
`comsol045`/`comsol046`, whose solved flow is 0.08/0.04 rel-L2 from COMSOL's *reflected* field and
0.53/0.67 from COMSOL's as given. Their clot is scored against the reflected label. The branch is
decided from the flow, never from the clot score, and applies to evaluation labels only; training
targets are COMSOL's. Rule `src/clot_ml/mirror_branch.py`, registry `configs/mirror_branch.json`,
guard `src/tests/test_mirror_branch.py`. **Exception:** GT-oracle coupling arms (`cpl_closed*`,
`cpl_c16*`) feed COMSOL's own clot into their features, so they and the uncoupled runs they are
compared with are scored on COMSOL's labels (`eval_strict --labels comsol`). What it moved: Table 4
stenosis wall 0.7721 → 0.8608 (the "2.3-SEM stenosis gap" is now 1.3 SEM), Fig 1.1 shipped wall
0.9356 → 0.9481, RGP-DEQ off-wall −0.1075 → −0.1185, the C0 ablation off-wall +0.1312 → +0.1318, the
temporal final-time wall 0.9479 → 0.9614, and split-vs-shipped wall p 0.043 → 0.054, which is no
longer significant. Every re-scorable artifact was regenerated (117 ledger rows moved, no
significance label flipped except split-vs-shipped); `verify_claims.PRE_MIRROR_RULE` lists any
artifact still scored before the rule, and is empty.

**Other measurement traps that must appear in Methods:**

* Clot placement must be chosen by geometry, not node ordering — a clot seeded on the inlet is
  pinned by the Dirichlet BC and cannot reroute anything. Such a case looks real and is not.
* Anchor meshes are quadratic (`MeshTri2`): vertices and mid-side nodes alternate along the
  boundary and carry different shear, so raw wall profiles are a sawtooth between two interleaved
  curves, not noise.
* The gelation constant `sr/sr0 = 0.1226` is anchored on wound vessels **at gelation only**. On a
  synthetic severe-occlusion sweep the case-median ratio spans 0.004–19.7. Report it as a measured
  constant with a stated validity domain, never as a blockage law.

---

## 4. Retractions

Kept visible so they cannot be silently re-quoted. **Read this list before writing any number
from memory.**

**The big one — every priority claim.** "First field-level surrogate of a continuum multi-species
thrombosis model", "first-of-kind mesh-agnostic graph surrogate", "nobody learns the clot field"
in the present tense. All retracted 2026-09-09 on discovery of Pelissier, Meliga & Hachem (2026),
*Comput Biol Med* 208:111649 (DOI 10.1016/j.compbiomed.2026.111649, PMID 41905249, open at
[hal-05670914](https://hal.science/hal-05670914v1)). See STORY.md (local only) "What we must never
claim".

**The process failure behind it, which is the one that would have surfaced at review.** The Phase
0 novelty triage searched the ML-thrombosis and mesh-GNN-hemodynamics literatures *separately*;
the near neighbour sits in the intersection and was caught by neither. **Rule for any future
triage: search the intersection explicitly, search PubMed/Europe PMC as well as arXiv and Google
Scholar, and re-run the pass immediately before submission.**

| retracted | why | use instead |
|---|---|---|
| "flux term flips wall-shear correlation to +0.554" | 15 per-case medians at one viscosity | −0.124 on the full 12-vessel per-node corpus, 5/12 wrong sign |
| "`GELATION_SR_RATIO` is a shipped blockage law" | it is not shipped — diagnostic only, off by default | nothing; C3' was never built |
| "Wall F1 holds to ~5% velocity error, falls to zero by 8%" | measured on `comsol010` alone | the three curves; `comsol005` has no tolerance window at all |
| §2 correlation signs `+0.613 / −0.395 / …` | magnitudes reproduce, signs were on no stated convention | the 2026-09-02 regeneration, with the convention stated |
| "FEM sits inside noise of GT (0.710 vs 0.705)" | n=5 vessels with a `.nas` mesh | paired n=27: GT 0.970 vs FEM 0.935, δ −0.036, p=0.022 — a real gap |
| "RGP-DEQ costs −0.35 on the deployed score" | old non-cross-fit arm, n=4–5, against GT flow | cross-fit K=5: n.s. at the wall, −0.1185 off-wall (p=0.0005) |
| "Wall generalization holds across class, 0.83–0.95" | `comsol045/046/047` were misclassified | corrected table: stenosis wall 0.8608 (n=5) with comsol045/046 scored on their flow branch (2026-09-16); 1.3 SEM below baseline, no longer a significant gap |
| "C0 takes off-wall 0.5812 → 0.7078" | retired model, GT cache | shipped generation: 0.7121 → 0.8439, δ +0.1318, CI [0.055, 0.215], P<0.001, n=20 (mirror-branch rule, 2026-09-16; was 0.7046 → 0.8358) |
| "OOF archive is GT-flow, subtract 0.03–0.04" | the GT labelling was a bug, since fixed and the archive rebuilt | Table 4 is already the deployed configuration; apply nothing |
| "5/5 detection, 0/33 false alarms" as a live claim | pre-cross-fit arm | live claim is the **false-alarm rate: 0 in 37 on the deployed path**; detection stays valid as history |
| "empty-gate r = +0.745, rel-L2 +0.029" | pre-cross-fit arm | shipped arm has 0/37 empty gates (no variance to correlate); rel-L2 is +0.224 |
| Table 5 SEALED walls `0.9489 / 0.9925 / 0.9884 / 1.0000` | do not match the actual sealed read; cause never diagnosed | comsol007 0.8968, comsol013 0.9786, comsol031 0.9533, comsol043 1.0000, mean 0.9572. **Do not re-run SEALED to check — it is spent.** |
| **BATC recall `0.690` for the 150-burden / 100-found case** (`DEPLOYCLOT.md` §0.3 and the `severity_metric.py` module docstring) | written when `tau_abs = 5`; the shipped `BATC` has `tau_abs = 15`, so the denominator is 135, not 145 | **0.741.** The 15/10 and 4/1 rows are unaffected — the `rho` cap binds before `tau_abs` there. Found 2026-09-11 by generating the BATC figure from `severity_from_counts` rather than transcribing the table |
| **Oracle closed loop `+0.0093` wall / `+0.0143` off-wall (n=5, GT flow)** | hand-computed from two console runs, never saved as an artifact; the off-wall half does not reproduce | **+0.0002 wall (n=8) / −0.1604 off-wall (n=3)**, measured on the deployed FEM flow, `outputs/deployclot/closed_loop_oracle.json`. The wall result survives; the off-wall sign is opposite and large. See STORY 2.1 for the train/test-mismatch reading and the n=3 permutation floor. Found 2026-09-12 by running E1 |
| **Wound held-out `w_reg 0.9270` / `w_lum 0.8611` (n=6)** | exists in the repo **only as hardcoded literals** in `scripts/promote_clot_ml_0.py`'s `lovo_held_out` block; no artifact produces them and the shipped manifest has no `scores_wound` key. Also a *different arm* (`clot_ml_0` with chemistry replace+depth) | **0.8789 / 0.7970** for the v4 + two-regime complement, FEM flow, LOVO, n=6 — `outputs/publication/data/wound_lovo_fem.json`. Do not reconcile the two; they measure different arms. Found 2026-09-12 by giving the claim a ledger row |

---

## 5. Known unbacked numbers — do not promote these without a run

* **COMSOL flow solve at t=0 ≈ 45 s.** Recollection, not a logged artifact. If confirmed it is a
  *second*, separate comparison from the pipeline-level 2,934×: at the t=0 stage alone, COMSOL
  (~45 s) vs local FEM (4.66 s ⟨timing.fem_flow_s⟩) vs RGP-DEQ (0.667 s ⟨timing.rgp_flow_s⟩). Do
  not conflate with `comsol_reference_hours` (48 h), which is the full coupled pipeline.
* **`preflight_validation` generator is gitignored**, so two claims resolve against artifacts that
  can no longer be regenerated — which violates this file's own rule. Restore the script (a copy
  exists under `dist/LocalFEMSolver-Predict-win64-1.0/scripts/`) or record the removal
  deliberately.
* **Everything in Leg 4.** The n=6 wound LOVO table is real and reproducible but has no ledger
  rows. [EXPERIMENTS.md](EXPERIMENTS.md) E2.
