"""Wall-cohort vessel lists (FIT / DEV / SEALED / CLOT_FREE).

Canonical home for the tuples that used to live at the bottom of
``mat_growth_simple.py``.  Active scoring and split helpers should import from here
(or from ``wall_cohort_splits``) rather than pulling in the mat-growth leg registry.

See ``docs/WALL_MODEL_PLAN.md`` section 21 and ``docs/SEALED_SPLIT.md``.
"""
from __future__ import annotations

# 2026-08-22: VIZ_HALF released from SEALED into TRAIN (docs/SEALED_SPLIT.md).
WALL_COHORT_V2_VIZ_RELEASED: tuple[str, ...] = (
    "comsol001",
    "comsol010",
    "comsol014",
    "comsol042",
)

#: 2026-09-02: the final synthetic corpus landed -- `comsol045`-`comsol048` (full horizon,
#: clot-carrying) and `comsol038` (full horizon, empty GT).  `comsol047` is the SECOND
#: non-SEALED aneurysm, which is what `src/clot_ml/geometry_splits.py` says the protocol was
#: missing: with one, no split could train on an aneurysm and measure a different one.
#: `comsol048` is the no-wound half of the matched A/B pair whose wound half is
#: `wound_comsol005` -- same outline to 0.0000 median wall-node distance, remeshed.
WALL_COHORT_V3_ADDED: tuple[str, ...] = (
    "comsol045",
    "comsol046",
    "comsol047",
    "comsol048",
)

#: The paired counterfactual: identical vessel outline, wound and no-wound.  Held for the
#: A/B read; both halves are scored by the fold model that held `comsol048` out, and no
#: wound pack is ever in the GNN training pool, so neither half is in-sample.
WOUND_AB_PAIR: tuple[str, str] = ("wound_comsol005", "comsol048")

#: Every wound simulation, in run order.  T < `wall_cohort_splits.MIN_T` on all six, so they
#: are NOT in the GNN training pool -- a truncated horizon is a different label quantity
#: (docs/PHASE6_RESULTS.md 6.2).  They fit the wound complement (leave-one-vessel-out) and
#: they are the unified artifact's held-out wound evaluation.
WOUND_COHORT: tuple[str, ...] = (
    "wound_comsol001",
    "wound_comsol002",
    "wound_comsol003",
    "wound_comsol004",
    "wound_comsol005",
    "wound_comsol006",
)

#: The three vessels the wound complement was validated leave-one-vessel-out on
#: (``wound_lovo.n_vessels = 3`` in the locked ``clot_ml_0`` manifest).  The two
#: gate scalars are refit on all of :data:`WOUND_COHORT`; only the LOVO evidence
#: is restricted to these.
#:
#: This lived as a bare tuple in ``scripts/eval_wound_complement.py`` while the
#: three promotion scripts used the full cohort, so nothing connected the two and
#: the difference read as a mistake rather than a decision.
WOUND_LOVO_COHORT: tuple[str, ...] = WOUND_COHORT[:3]

WALL_COHORT_V2_TRAIN: tuple[str, ...] = (
    "comsol003",
    "comsol004",
    "comsol005",
    "comsol006",
    "comsol008",
    "comsol009",
    "comsol011",
    "comsol012",
    "comsol015",
    "comsol016",
    "comsol018",
    "comsol019",
    "comsol020",
    "comsol021",
    "comsol024",
    "comsol025",
    "comsol028",
    "comsol029",
    "comsol032",
    "comsol035",
    "comsol036",
    "comsol037",
    "comsol039",
    "comsol040",
    "comsol041",
    "comsol044",
) + WALL_COHORT_V2_VIZ_RELEASED + WALL_COHORT_V3_ADDED

# FINAL_HALF only since 2026-08-22.
WALL_COHORT_V2_GENERALIZATION: tuple[str, ...] = (
    "comsol007",
    "comsol013",
    "comsol031",
    "comsol043",
)

WALL_COHORT_V2_SEALED_PRE_20260822: tuple[str, ...] = (
    "comsol001",
    "comsol007",
    "comsol010",
    "comsol013",
    "comsol014",
    "comsol031",
    "comsol042",
    "comsol043",
)

WALL_COHORT_V2_CLOT_FREE: tuple[str, ...] = (
    "comsol017",
    "comsol022",
    "comsol023",
    "comsol026",
    "comsol027",
    "comsol030",
    "comsol033",
    "comsol034",
    "comsol038",
)

WALL_COHORT_V2_DEV: tuple[str, ...] = (
    "comsol039",
    "comsol040",
    "comsol041",
    "comsol042",
    "comsol043",
    "comsol044",
)

WALL_COHORT_V2_DEV_HOLDOUT: tuple[str, ...] = ("comsol043",)

WALL_COHORT_V2_DEV_TRAIN: tuple[str, ...] = tuple(
    n for n in WALL_COHORT_V2_DEV if n not in WALL_COHORT_V2_DEV_HOLDOUT
)

WALL_COHORT_V2_FIT: tuple[str, ...] = tuple(
    n for n in WALL_COHORT_V2_TRAIN if n not in WALL_COHORT_V2_DEV_TRAIN
)
