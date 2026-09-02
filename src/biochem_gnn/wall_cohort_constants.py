"""Wall-cohort vessel lists (FIT / DEV / SEALED / CLOT_FREE).

Canonical home for the tuples that used to live at the bottom of
``mat_growth_simple.py``.  Active scoring and split helpers should import from here
(or from ``wall_cohort_splits``) rather than pulling in the mat-growth leg registry.

See ``docs/WALL_MODEL_PLAN.md`` section 21 and ``docs/SEALED_SPLIT.md``.
"""
from __future__ import annotations

# 2026-08-22: VIZ_HALF released from SEALED into TRAIN (docs/SEALED_SPLIT.md).
WALL_COHORT_V2_VIZ_RELEASED: tuple[str, ...] = (
    "patient001",
    "patient010",
    "patient014",
    "patient042",
)

WALL_COHORT_V2_TRAIN: tuple[str, ...] = (
    "patient003",
    "patient004",
    "patient005",
    "patient006",
    "patient008",
    "patient009",
    "patient011",
    "patient012",
    "patient015",
    "patient016",
    "patient018",
    "patient019",
    "patient020",
    "patient021",
    "patient024",
    "patient025",
    "patient028",
    "patient029",
    "patient032",
    "patient035",
    "patient036",
    "patient037",
    "patient039",
    "patient040",
    "patient041",
    "patient044",
) + WALL_COHORT_V2_VIZ_RELEASED

# FINAL_HALF only since 2026-08-22.
WALL_COHORT_V2_GENERALIZATION: tuple[str, ...] = (
    "patient007",
    "patient013",
    "patient031",
    "patient043",
)

WALL_COHORT_V2_SEALED_PRE_20260822: tuple[str, ...] = (
    "patient001",
    "patient007",
    "patient010",
    "patient013",
    "patient014",
    "patient031",
    "patient042",
    "patient043",
)

WALL_COHORT_V2_CLOT_FREE: tuple[str, ...] = (
    "patient017",
    "patient022",
    "patient023",
    "patient026",
    "patient027",
    "patient030",
    "patient033",
    "patient034",
)

WALL_COHORT_V2_DEV: tuple[str, ...] = (
    "patient039",
    "patient040",
    "patient041",
    "patient042",
    "patient043",
    "patient044",
)

WALL_COHORT_V2_DEV_HOLDOUT: tuple[str, ...] = ("patient043",)

WALL_COHORT_V2_DEV_TRAIN: tuple[str, ...] = tuple(
    n for n in WALL_COHORT_V2_DEV if n not in WALL_COHORT_V2_DEV_HOLDOUT
)

WALL_COHORT_V2_FIT: tuple[str, ...] = tuple(
    n for n in WALL_COHORT_V2_TRAIN if n not in WALL_COHORT_V2_DEV_TRAIN
)
