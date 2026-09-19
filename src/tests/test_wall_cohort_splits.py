"""FIT / DEV / SEALED are disjoint and match the wall-cohort protocol."""

from src.biochem_gnn.wall_cohort_constants import (
    WALL_COHORT_V2_DEV_HOLDOUT,
    WALL_COHORT_V2_DEV_TRAIN,
    WALL_COHORT_V2_FIT,
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
)
from src.core_physics.wall_cohort_splits import (
    DEV, FIT, SEALED, assert_disjoint, split_of,
)


def test_splits_match_published_cohort_constants():
    assert_disjoint()
    assert FIT == WALL_COHORT_V2_FIT
    assert DEV == WALL_COHORT_V2_DEV_TRAIN
    assert SEALED == WALL_COHORT_V2_GENERALIZATION
    # 042 left SEALED on 2026-08-22 and is a DEV vessel, so it joins DEV-train
    assert DEV == ("comsol039", "comsol040", "comsol041", "comsol042", "comsol044")
    assert "comsol020" in FIT
    assert "comsol020" not in DEV
    assert "comsol020" not in SEALED
    assert set(WALL_COHORT_V2_DEV_HOLDOUT) <= set(SEALED)
    assert set(FIT) | set(DEV) == set(WALL_COHORT_V2_TRAIN)


def test_split_of_never_puts_sealed_in_fit_or_dev():
    for a in WALL_COHORT_V2_GENERALIZATION:
        assert split_of(a) == "sealed"
    for a in DEV:
        assert split_of(a) == "dev"
    assert split_of("comsol020") == "fit"
    assert split_of("comsol041") == "dev"
    assert split_of("comsol042") == "dev"
    assert split_of("comsol043") == "sealed"
