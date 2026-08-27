# THE SEALED SET IS SPLIT — read before touching any of these 8 vessels

Decided 2026-08-21. This is a durable protocol change, not a one-off script argument.
Any future session — human or agent — must read this before opening, scoring, or
otherwise looking at `patient001/007/010/013/014/031/042/043`.

---

> ## AMENDMENT, 2026-08-22 — VIZ_HALF IS RELEASED INTO TRAINING
>
> **`patient001`, `patient010`, `patient014`, `patient042` are no longer held out.** They
> are in `WALL_COHORT_V2_TRAIN` and may be trained on, tuned on, and selected on like any
> other pool vessel. Rule 2 below (*"must never be used to select, tune, or choose between
> models"*) is **superseded for these four**. `patient042` is a DEV vessel, so it joins
> DEV-train.
>
> **FINAL_HALF is unchanged**: `patient007`, `patient013`, `patient031`, `patient043` stay
> closed, and rules 3–5 apply to them in full. `WALL_COHORT_V2_GENERALIZATION` is now those
> four names only.
>
> **What this costs, stated plainly.** VIZ_HALF was the project's only genuinely held-out
> evidence, and it is where both diagnosed wall failure modes live —
> `patient001` cut placement (band occupancy z = +4.44, cut-gap +0.239) and `patient042`
> ranking (AUC z = −12.95). Once they are trained on, **those two diagnoses can no longer be
> confirmed out-of-sample.** The offsetting reason to do it: `patient001`'s 13.5% band
> occupancy is the failure regime the 19-vessel pool never visits, and having it *inside*
> the fit is the only way the readout/calibration work can see that regime at all. This was
> a deliberate trade, not an erosion.
>
> **Provenance of earlier artifacts is preserved.** Anything promoted before 2026-08-22 —
> `clot_gnn_v3`, `clot_gnn_v4`, `clot_gnn_v4w` — was fitted without these four. Tests that
> assert "SEALED never seen" for those artifacts must compare against
> `WALL_COHORT_V2_SEALED_PRE_20260822` (the frozen 8-vessel tuple), **not** the live
> `SEALED`, or they pass for a reason unrelated to the artifact.
>
> **Also changed the same day:** the 8 clot-free vessels
> (`patient017/022/023/026/027/030/033/034`, `maxMat = 0`, empty GT, T = 201) were admitted
> to training and to false-positive scoring as `WALL_COHORT_V2_CLOT_FREE`. They were
> previously neither pool nor SEALED. They carry no recall and must never enter a
> recall-bearing mean — see `docs/MODEL_REVIEW_2026-08-22.md` §8b.

---

## Why this exists

Every phase doc in this project (PHASE3 through PHASE10) held one 8-vessel SEALED set —
`WALL_COHORT_V2_GENERALIZATION` in `src/core_physics/wall_cohort_splits.py` — closed,
with the explicit rule "spend once, on a frozen configuration, at the very end of the
whole project." Visualizing `clot_gnn_v4` created real pressure to open it just to look,
on the reasoning "the model is already frozen, so looking can't leak into training." That
reasoning is incomplete: seeing a SEALED score can still steer *which future model gets
built next*, even with zero retraining involved. See the discussion preserved in the
session that produced this doc for the full argument.

**The compromise struck:** split SEALED in half. One half is permanently released for
visualization — genuinely held-out evidence, safe to look at repeatedly, never spent
because it was never a decision input. The other half stays exactly as closed as SEALED
has always been, reserved for the one true final read of the whole project.

## The split

Method: sort the 8 vessels by ID, alternate assignment starting with VIZ. Deterministic,
disclosed before execution, not chosen by outcome — nobody picked based on which vessels
"look good."

```
sorted:  001  007  010  013  014  031  042  043
assign:  VIZ  FINAL VIZ FINAL VIZ FINAL VIZ FINAL
```

| | vessels | geometry class | status |
|---|---|---|---|
| **VIZ_HALF** | `patient001`, `patient010`, `patient014`, `patient042` | 042 = stenosis (priority); rest baseline | **open, permanently, for visualization only** |
| **FINAL_HALF** | `patient007`, `patient013`, `patient031`, `patient043` | 043 = aneurysm (priority); rest baseline | **closed — the project's one remaining true final read** |

One priority-class vessel landed on each side (042 stenosis in VIZ, 043 aneurysm in
FINAL) — not engineered, just where the alternating assignment put them.

## The rules, going forward

1. **VIZ_HALF may be opened, scored, and shown in any visualization**, for any model,
   repeatedly, going forward. It is genuinely held out (never trained on by any
   `clot_gnn` version), so a model's score on it is real evidence of generalization —
   more informative per-vessel than the training-pool "in-sample" vessels every viz has
   been forced to use so far.
2. **VIZ_HALF must never be used to select, tune, or choose between models or
   configurations.** Looking is fine; using what you saw to decide "ship config A over
   config B" is exactly the leak this whole discipline exists to prevent. If a choice
   needs to be made, make it on FIT/DEV or the strict CV protocol, then look at VIZ_HALF
   afterward, not before.
3. **FINAL_HALF is untouched.** Same rule SEALED always had: never opened, never
   scored, never peeked at "just to check monotonicity" — that carve-out in
   `docs/PHASE9_ML.md`/`docs/PHASE10_V4.md` for `patient001`'s mask-growth smoke test
   predates this split and should not be read as precedent for touching FINAL_HALF now
   that `patient001` itself is VIZ_HALF and fair game.
4. **This split applies project-wide, not per-model.** It was spent once, here, for
   `clot_gnn_v4`'s viz. It is not re-decided for v5, v6, or whatever comes next — VIZ_HALF
   stays open, FINAL_HALF stays closed, for the rest of the project.
5. **When the project is actually done** — no more architecture changes, no more
   configs to choose between — FINAL_HALF is the one score that gets read, once, and
   reported as the project's real answer to "does this generalize."

## Reference implementation

`scripts/gen_v4_temporal_data.py`'s `VIZ_HALF_SEALED` / `FINAL_HALF_SEALED` constants and
its assertion guard are the canonical, enforced version of this split — copy that pattern
into any future viz rather than re-deriving the vessel lists by hand.
