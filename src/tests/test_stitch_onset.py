"""Cover for ``stitch_onset`` -- the schedule for wall nodes the surface ODE never ignites.

The shipped convention hands every such node one constant (the median igniter onset). That
constant is measurably in the wrong place and has no spread; see the function's docstring for
the cohort numbers. These tests pin the contract, not the fitted values.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core_physics.physics_wall_model import (
    STITCH_OFFSET, STITCH_SPREAD, stitch_onset,
)


def _case():
    onset = np.array([10.0, 20.0, 999.0, 999.0, 999.0, 999.0])
    ignited = np.array([True, True, False, False, False, False])
    node_set = np.ones(6, dtype=bool)
    sr = np.array([5.0, 5.0, 40.0, 10.0, 30.0, 20.0])
    return onset, ignited, node_set, sr, 100


def test_igniters_are_never_touched():
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T)
    assert np.array_equal(out[ign], onset[ign]), "the ODE's own crossings must survive intact"


def test_stitch_nodes_are_ordered_by_shear():
    """Low t=0 shear commits first -- that ordering is the whole reason the spread helps."""
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T)
    st = np.flatnonzero(~ign)
    assert list(np.argsort(out[st])) == list(np.argsort(sr[st])), "onset order must follow sr"


def test_centre_sits_later_than_the_median_igniter():
    """The measured defect is that the constant fires these nodes far too early."""
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T, spread=0.0)
    assert np.allclose(out[~ign], np.median(onset[ign]) + STITCH_OFFSET * T)
    assert out[~ign].min() > np.median(onset[ign])


def test_zero_spread_reduces_to_a_shifted_constant():
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T, spread=0.0)
    assert len(np.unique(out[~ign])) == 1


def test_output_stays_inside_the_horizon():
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T, offset=0.9, spread=2.0)
    assert out[~ign].min() >= 0.0
    assert out[~ign].max() <= T - 1


def test_no_igniters_is_a_noop():
    onset, ign, sel, sr, T = _case()
    none = np.zeros_like(ign)
    assert np.array_equal(stitch_onset(onset, none, sel, sr, T), onset)


def test_all_ignited_is_a_noop():
    onset, ign, sel, sr, T = _case()
    allign = np.ones_like(ign)
    assert np.array_equal(stitch_onset(onset, allign, sel, sr, T), onset)


def test_only_nodes_in_the_set_are_scheduled():
    onset, ign, sel, sr, T = _case()
    sel = sel.copy()
    sel[5] = False
    out = stitch_onset(onset, ign, sel, sr, T)
    assert out[5] == onset[5], "a node outside the set must not be rescheduled"


@pytest.mark.parametrize("offset,spread", [(STITCH_OFFSET, STITCH_SPREAD), (0.35, 0.8)])
def test_defaults_and_the_alternate_fold_choice_both_behave(offset, spread):
    onset, ign, sel, sr, T = _case()
    out = stitch_onset(onset, ign, sel, sr, T, offset=offset, spread=spread)
    assert np.isfinite(out).all()
    assert (out[~ign] >= 0).all()
