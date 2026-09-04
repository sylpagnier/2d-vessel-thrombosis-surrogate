"""FIT / DEV / SEALED splits for the wall-cohort physics model.

This is the protocol used by ``scripts/eval_wall_protocol.py`` and
``docs/WALL_MODEL_PLAN.md`` §21.1. Phase-7/8 evals that average
``WALL_COHORT_V2_TRAIN`` (27, or the 19 eligible full-horizon clot-carrying subset)
are mixing FIT with DEV, and ``comsol020`` is a FIT vessel -- not a holdout.

    FIT        TRAIN minus DEV-train
    DEV        039, 040, 041, 042, 044 -- selection only, never fitted
    SEALED     WALL_COHORT_V2_GENERALIZATION -- never tune, spend once
    CLOT_FREE  empty-GT vessels -- trainable, false-positive scoring only

**Changed 2026-08-22** (docs/SEALED_SPLIT.md, docs/MODEL_REVIEW_2026-08-22.md 8b):

* **VIZ_HALF (001/010/014/042) released from SEALED into TRAIN.**  SEALED is now
  FINAL_HALF only (007/013/031/043), and 042 becomes a DEV-train vessel.  Artifacts
  fitted before this date were fitted without them, so their "SEALED never seen"
  provenance still reads true against the 8-vessel SEALED of the time.
* **The 8 clot-free vessels are no longer dropped.**  Truncated (T<150) vessels remain a
  different *quantity* and are still dropped everywhere (PHASE6_RESULTS 6.2), but empty-GT
  is not the same case: those vessels carry no recall, yet they carry real evidence about
  FALSE POSITIVES.  They are excluded from any recall-bearing mean and scored through
  ``severity_components``' empty-GT branch instead.

The wall-gen small cohort in AGENTS.md (train 005/006/010/023/002, val=020) is a different
stack; do not mix it into these numbers.
"""
from __future__ import annotations

from collections import defaultdict

# These tuples are OWNED by wall_cohort_constants; mat_growth_simple only
# re-exported them, and importing them from there pulled a 236 KB retired
# module into the shipped clot_ml import closure.
from src.biochem_gnn.wall_cohort_constants import (
    WALL_COHORT_V2_CLOT_FREE,
    WALL_COHORT_V2_DEV,
    WALL_COHORT_V2_DEV_HOLDOUT,
    WALL_COHORT_V2_DEV_TRAIN,
    WALL_COHORT_V2_FIT,
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
    WALL_COHORT_V2_VIZ_RELEASED,
)

MIN_T = 150

FIT = WALL_COHORT_V2_FIT
DEV = WALL_COHORT_V2_DEV_TRAIN
SEALED = WALL_COHORT_V2_GENERALIZATION

#: Empty-GT vessels.  A separate SPLIT-level category, not a fourth split: they are eligible
#: for training and for false-positive scoring, and must never enter a recall-bearing mean.
CLOT_FREE = WALL_COHORT_V2_CLOT_FREE

#: Released from SEALED on 2026-08-22 (docs/SEALED_SPLIT.md).  Kept as a named constant so a
#: reader can tell which TRAIN vessels were sealed for the artifacts fitted before that date.
VIZ_RELEASED = WALL_COHORT_V2_VIZ_RELEASED


def split_of(anchor: str) -> str:
    if anchor in SEALED:
        return "sealed"
    if anchor in CLOT_FREE:
        return "clot_free"
    if anchor in DEV:
        return "dev"
    if anchor in WALL_COHORT_V2_TRAIN:
        return "fit"
    return "other"


def assert_disjoint() -> None:
    fit, dev, sealed = set(FIT), set(DEV), set(SEALED)
    free = set(CLOT_FREE)
    assert not (fit & dev), fit & dev
    assert not (fit & sealed), fit & sealed
    assert not (dev & sealed), dev & sealed
    assert not (free & (fit | dev | sealed)), free & (fit | dev | sealed)
    assert set(DEV) == set(WALL_COHORT_V2_DEV) - set(WALL_COHORT_V2_DEV_HOLDOUT)
    assert set(WALL_COHORT_V2_DEV_HOLDOUT) <= set(SEALED)
    assert set(FIT) | set(DEV) == set(WALL_COHORT_V2_TRAIN)
    # the release moved VIZ_HALF out of SEALED and into TRAIN, in both directions
    assert not (set(VIZ_RELEASED) & sealed), set(VIZ_RELEASED) & sealed
    assert set(VIZ_RELEASED) <= set(WALL_COHORT_V2_TRAIN)


def bucket(anchors) -> dict[str, list[str]]:
    out = defaultdict(list)
    for a in anchors:
        out[split_of(a)].append(a)
    return dict(out)


def mean_by_split(scores: dict[str, float]) -> dict[str, dict]:
    """``scores`` maps anchor -> scalar.  Sealed is returned but must not drive selection."""
    acc: dict[str, list[float]] = {"fit": [], "dev": [], "sealed": [], "clot_free": [],
                                   "other": []}
    for a, s in scores.items():
        if s != s:  # nan
            continue
        acc[split_of(a)].append(float(s))
    out = {}
    for k, vs in acc.items():
        out[k] = dict(n=len(vs), mean=(float(sum(vs) / len(vs)) if vs else None))
    return out


