"""Freeze the ablation ladder, because a silent hole in it makes every arm wrong.

Two failure modes are worth a test and neither announces itself:

  1. **An unpartitioned feature column.**  `apply_arm` zeroes what an arm does not keep, so a
     column missing from `FEATURE_GROUPS` is silently KEPT in every arm -- A0 would quietly
     be "geometry plus whatever was added last week" and the ladder would read as a null.
  2. **`--arm A4` not being the shipped model.**  Every delta is measured against A4; if the
     reference arm perturbed anything at all, the whole table is measured against the wrong
     zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.clot_ml.ablation import (
    ARCH_ARMS, ARMS, FEATURE_GROUPS, GROUP_ORDER, apply_arm, columns_for, zeroed_columns,
)


def test_groups_are_a_partition():
    """No column in two groups, and the nesting order names every group."""
    seen: set[str] = set()
    for g, cols in FEATURE_GROUPS.items():
        assert len(set(cols)) == len(cols), f"{g} repeats a column"
        assert not (seen & set(cols)), f"{g} overlaps an earlier group: {seen & set(cols)}"
        seen |= set(cols)
    assert set(GROUP_ORDER) == set(FEATURE_GROUPS)


def test_arms_are_nested_and_A4_is_everything():
    """The A ladder must be nested, or an arm-to-arm delta is not attributable."""
    ladder = ["A0", "A1", "A1p", "A2", "A3", "A4"]
    for lo, hi in zip(ladder, ladder[1:]):
        assert set(columns_for(lo)) < set(columns_for(hi)), f"{lo} is not inside {hi}"
    assert set(columns_for("A4")) == {c for g in FEATURE_GROUPS.values() for c in g}
    # A1_pure is A1p's features with the architecture doors shut -- same columns, so the
    # A1p/A1_pure delta isolates the architecture and nothing else.
    assert set(columns_for("A1_pure")) == set(columns_for("A1p"))
    # every B arm is measured at FULL conditioning, or its delta mixes two causes
    for b in [a for a in ARMS if a.startswith("B_")]:
        assert set(columns_for(b)) == set(columns_for("A4")), b


def test_A4_changes_nothing():
    """The reference arm must be a no-op in BOTH ladders."""
    assert ARCH_ARMS.get("A4", {}) == {}
    cols = list(columns_for("A4"))
    assert zeroed_columns("A4", cols) == []
    cache = {"v": {"cols": np.array(cols), "X": np.arange(3 * len(cols), dtype=np.float32)
                   .reshape(3, len(cols))}}
    before = cache["v"]["X"].copy()
    assert apply_arm(cache, "A4", verbose=False) == []
    assert np.array_equal(cache["v"]["X"], before)


def test_unpartitioned_column_is_fatal():
    """The one failure that would make the ladder wrong without looking wrong."""
    with pytest.raises(ValueError, match="FEATURE_GROUPS"):
        zeroed_columns("A0", list(columns_for("A4")) + ["a_channel_added_later"])


def test_apply_arm_holds_dropped_columns_constant():
    cols = list(columns_for("A4"))
    rng = np.random.default_rng(0)
    cache = {"v%d" % i: {"cols": np.array(cols),
                         "X": rng.normal(size=(5, len(cols))).astype(np.float32)}
             for i in range(3)}
    drop = apply_arm(cache, "A0", verbose=False)
    keep_idx = [cols.index(c) for c in columns_for("A0")]
    drop_idx = [cols.index(c) for c in drop]
    assert len(keep_idx) + len(drop_idx) == len(cols)
    for S in cache.values():
        assert np.all(S["X"][:, drop_idx] == 0.0)
        assert np.isfinite(S["X"][:, keep_idx]).all()
    # A CONSTANT column carries no information whatever the constant is -- that is the whole
    # ablation semantics.  `train_one` standardizes with the fold's own mu/sd and clamps a
    # zero sd to 1, so the standardized column is exactly 0.0, which is what the plan asks
    # for ("zero the ablated columns AFTER normalization; do not remove them").
    X = np.concatenate([S["X"] for S in cache.values()])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    assert np.all(((X - mu) / sd)[:, drop_idx] == 0.0)


def test_arch_switches_are_defaults_off():
    """Every switch an arm can flip must default to the shipped behaviour in the model."""
    import inspect

    from src.clot_ml.gnn import ClotGNN, rollout, to_device

    assert inspect.signature(ClotGNN.__init__).parameters["phys_base"].default is True
    assert inspect.signature(rollout).parameters["phys_seed"].default is True
    assert inspect.signature(to_device).parameters["iso"].default is False
