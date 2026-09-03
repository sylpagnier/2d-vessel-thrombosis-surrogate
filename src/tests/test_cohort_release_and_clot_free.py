"""The 2026-08-22 cohort changes: VIZ_HALF released, clot-free vessels admitted.

Two decisions, recorded in `docs/SEALED_SPLIT.md` and `docs/MODEL_REVIEW_2026-08-22.md` 8b:

  * **VIZ_HALF (001/010/014/042) moved from SEALED into TRAIN.**  SEALED is FINAL_HALF only
    (007/013/031/043).  Provenance tests on artifacts promoted BEFORE that date must assert
    against `WALL_COHORT_V2_SEALED_PRE_20260822`, or they silently stop testing anything.
  * **The 8 clot-free vessels are admitted** to training and to false-positive scoring.
    They carry no recall, so they must never enter a recall-bearing mean; they are scored
    through `severity_components`' empty-GT branch instead.

The empty-GT branch already had the requested shape (nothing -> 1.0, a few -> down a bit,
many -> tanks); what changed is that `SeverityScorer.score` can now return it instead of NaN.
The NaN default is load-bearing at the DOMAIN level and is pinned here too.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.biochem_gnn.mat_growth_simple import (
    WALL_COHORT_V2_CLOT_FREE,
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_SEALED_PRE_20260822,
    WALL_COHORT_V2_TRAIN,
    WALL_COHORT_V2_VIZ_RELEASED,
)
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer
from src.core_physics.wall_cohort_splits import (
    CLOT_FREE, DEV, FIT, SEALED, assert_disjoint, mean_by_split, split_of,
)


# --------------------------------------------------------------------------- cohort
def test_viz_half_is_released_into_train_and_out_of_sealed():
    assert_disjoint()
    assert set(WALL_COHORT_V2_VIZ_RELEASED) <= set(WALL_COHORT_V2_TRAIN)
    assert not (set(WALL_COHORT_V2_VIZ_RELEASED) & set(WALL_COHORT_V2_GENERALIZATION))
    assert set(SEALED) == {"patient007", "patient013", "patient031", "patient043"}
    # 042 is a DEV vessel, so releasing it makes it DEV-train rather than FIT
    assert split_of("patient042") == "dev"
    for a in ("patient001", "patient010", "patient014"):
        assert split_of(a) == "fit"


def test_the_historical_sealed_constant_is_frozen():
    """Pre-release artifacts are judged against this; it must never track the live set."""
    assert len(WALL_COHORT_V2_SEALED_PRE_20260822) == 8
    assert set(WALL_COHORT_V2_GENERALIZATION) < set(WALL_COHORT_V2_SEALED_PRE_20260822)
    assert set(WALL_COHORT_V2_VIZ_RELEASED) < set(WALL_COHORT_V2_SEALED_PRE_20260822)


def test_clot_free_vessels_are_their_own_category():
    # 9 since 2026-09-03: `patient038` joined on the corpus rebuild -- its GT `Mat` is
    # identically zero at every frame (`outputs/mat_field_cache_fem/patient038.npz`, gtmax
    # 0.00), so it is a clot-free vessel and not a FIT one that happens to score well.
    assert len(CLOT_FREE) == 9
    assert CLOT_FREE == WALL_COHORT_V2_CLOT_FREE
    assert not (set(CLOT_FREE) & (set(FIT) | set(DEV) | set(SEALED)))
    for a in CLOT_FREE:
        assert split_of(a) == "clot_free"


def test_mean_by_split_reports_clot_free_separately():
    scores = {"patient020": 0.9, "patient041": 0.8, "patient017": 1.0, "patient022": 0.5}
    m = mean_by_split(scores)
    assert m["clot_free"]["n"] == 2
    assert m["clot_free"]["mean"] == pytest.approx(0.75)
    assert m["fit"]["n"] == 1 and m["dev"]["n"] == 1


# --------------------------------------------------------------------------- scoring
def _chain_scorer(n=40, gt=None):
    ei = np.stack([np.arange(n - 1), np.arange(1, n)])
    g = np.zeros(n, dtype=bool) if gt is None else gt
    return SeverityScorer(ei, g, n, DEFAULT), n


def test_empty_gt_scores_one_when_nothing_is_committed():
    sc, n = _chain_scorer()
    pred = np.zeros(n, dtype=bool)
    assert sc.score(pred, empty_gt="score") == pytest.approx(1.0)


def test_empty_gt_degrades_gently_then_tanks():
    """The requested shape: a few false positives cost a little, many cost almost everything."""
    sc, n = _chain_scorer()
    out = []
    for k in (0, 1, 2, 4, 8, 16, 32):
        pred = np.zeros(n, dtype=bool)
        pred[:k] = True
        out.append(sc.score(pred, empty_gt="score"))
    assert out[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(out, out[1:])), out      # strictly decreasing
    assert out[1] > 0.85, out                                  # one FP is a small cost
    assert out[4] == pytest.approx(0.5, abs=1e-9)              # 8 == empty_gt_fp_tol
    assert out[-1] < 0.25, out                                 # 32 FPs tanks it


def test_nan_remains_the_default_so_domain_means_are_unchanged():
    """6 of the clot-carrying vessels have no OFF-WALL GT; folding them in would silently
    redefine every off-wall number in the project."""
    sc, n = _chain_scorer()
    pred = np.zeros(n, dtype=bool)
    pred[:3] = True
    assert np.isnan(sc.score(pred))
    assert np.isnan(sc.score(pred, empty_gt="nan"))
    assert not np.isnan(sc.score(pred, empty_gt="score"))


def test_non_empty_gt_is_unaffected_by_the_flag():
    n = 40
    gt = np.zeros(n, dtype=bool)
    gt[10:20] = True
    sc, _ = _chain_scorer(n, gt)
    pred = np.zeros(n, dtype=bool)
    pred[11:19] = True
    assert sc.score(pred, empty_gt="nan") == sc.score(pred, empty_gt="score")


def test_bad_empty_gt_value_is_rejected():
    sc, n = _chain_scorer()
    with pytest.raises(ValueError, match="empty_gt"):
        sc.score(np.zeros(n, dtype=bool), empty_gt="zero")
