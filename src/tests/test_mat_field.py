"""Pins for the v6 learned ``Mat`` field (:mod:`src.clot_ml.mat_field`).

These cover the properties the off-wall rule depends on, NOT the score -- the score is a
research number and lives in docs/WOUND_PROGRESS.md.  What must not drift silently:

* the time sampler always keeps the final frame, which is the one the deploy score reads;
* the crossing target is exactly the bar the shipped rule asks about (``crit / off_att``);
* the shells the rule walks are disjoint, off-solid, and start from the SHIPPED first shell;
* the model is the physics at initialisation -- ``head_reg`` is a zero-init residual on the
  ODE's own ``Mat``, so an untrained v6 reproduces the ODE exactly rather than noise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.clot_ml.mat_field import (
    EXTRA_CHANNELS, OFF_ATT, MatFieldConfig, crossing_target, extra_channels, make_model,
    sample_time_indices,
)


def test_time_sampler_always_keeps_the_final_frame():
    # the final frame is what the deploy score reads; a rounding accident must never drop it
    for T in (1, 2, 3, 7, 16, 17, 29, 45, 71, 129, 201):
        idx = sample_time_indices(T, 16)
        assert idx[0] == 0
        assert idx[-1] == T - 1
        assert np.all(np.diff(idx) > 0)
        assert idx.max() < T


def test_time_sampler_returns_every_frame_when_the_horizon_is_short():
    # comsol011 has T=45 and comsol003 T=29; a vessel shorter than the budget is dense
    assert np.array_equal(sample_time_indices(9, 16), np.arange(9))


def test_crossing_target_is_exactly_the_bar_the_offwall_rule_asks_about():
    """The rule commits when ``off_att * Mat_owner >= crit``, i.e. ``Mat >= crit/off_att``.

    The classifier is trained on that same inequality so no second threshold has to be
    fitted downstream -- which is the whole reason v6 predicts a crossing and not a
    magnitude (003's ``Mat`` p90 is the dataset maximum; see the module docstring).
    """
    bar = 1.0 / OFF_ATT
    mat_over_crit = np.array([0.0, 1.0, bar - 1e-6, bar, bar + 1.0, 100.0])
    entry = {"gt_t": np.log1p(mat_over_crit)[None, :]}
    got = crossing_target(entry, 0, OFF_ATT)
    assert np.array_equal(got, np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_crossing_target_follows_a_changed_attenuation():
    entry = {"gt_t": np.log1p(np.array([3.0, 5.0]))[None, :]}
    assert np.array_equal(crossing_target(entry, 0, 0.25), np.array([0.0, 1.0], np.float32))
    assert np.array_equal(crossing_target(entry, 0, 0.5), np.array([1.0, 1.0], np.float32))


def _toy_entry(n=6, k=2):
    return {
        "ode_t": np.tile(np.linspace(0.0, 2.0, n, dtype=np.float32), (k, 1)),
        "gt_t": np.zeros((k, n), dtype=np.float32),
        "town": np.array([-1, 0, 0, 1, 1, 2], dtype=np.int64),
        "t_idx": np.array([0, 10], dtype=np.int64),
        "T": np.int64(11),
    }


def test_extra_channels_shape_and_owner_lookup():
    e = _toy_entry()
    ex = extra_channels(e, 1, torch.device("cpu")).numpy()
    assert ex.shape == (6, len(EXTRA_CHANNELS))
    assert ex[:, 0] == pytest.approx(1.0)          # t_frac at the last sampled time
    assert ex[:, 1] == pytest.approx(e["ode_t"][1])
    # an ownerless node (`town < 0`) contributes 0 rather than indexing node -1
    assert ex[0, 2] == pytest.approx(0.0)
    assert ex[1, 2] == pytest.approx(e["ode_t"][1][0])


def test_untrained_v6_is_exactly_the_ode():
    """``head_reg`` is zero-init, so v6 STARTS as the physics and training only adds to it.

    This is what makes the arm safe to compare against the shipped ODE: a v6 that has
    learned nothing is not noise, it is the baseline.  It is also the property that made the
    failure on ``wound_comsol003`` legible -- the predicted ``Mat`` p90 came back at 2.00x
    crit against the ODE's own 1.96x, which is how we knew the residual had collapsed
    rather than mislearned (docs/WOUND_PROGRESS.md 17).
    """
    torch.manual_seed(0)
    model = make_model(in_dim=7, edim=6, cfg=MatFieldConfig(dim=16, layers=2, drop=0.0))
    model.eval()
    n, ei = 5, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    with torch.no_grad():
        _, reg = model(torch.randn(n, 7), ei, torch.randn(4, 6),
                       torch.rand(4, 1), torch.rand(4, 1), base,
                       extra=torch.randn(n, len(EXTRA_CHANNELS)))
    assert torch.allclose(reg, base, atol=1e-6)


def test_extra_dim_matches_the_declared_channel_list():
    m = make_model(in_dim=7, edim=6, cfg=MatFieldConfig(dim=16, layers=1))
    assert m.extra_dim == len(EXTRA_CHANNELS)


def test_solid_shells_are_disjoint_and_off_solid():
    """Depth is what buys 003 its headroom (0.7897 -> 0.9240 with an oracle field), so the
    rings must not overlap -- an overlapping ring would let one node be judged at two
    different bars, and shell 1 must stay the SHIPPED ``first_corner_shell``."""
    from src.core_physics.physics_lumen_model import solid_boundary_shells

    # a path graph 0-1-2-...-9 with node 0 solid; even hops are the corner shells
    n = 10
    ei = np.array([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                   [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    solid = np.zeros(n, bool)
    solid[0] = True
    shipped = np.zeros(n, bool)
    shipped[1] = True                      # pretend `first_corner_shell` chose node 1
    pos = np.stack([np.arange(n), np.zeros(n)], 1).astype(np.float64)
    town = np.full(n, -1, dtype=np.int64)
    shells, owner = solid_boundary_shells(pos, solid, ei, shell1=shipped, town=town, max_depth=3)

    assert np.array_equal(shells[0], shipped)          # shell 1 is the shipped set verbatim
    for sh in shells:
        assert not (sh & solid).any()
    for i in range(len(shells)):
        for j in range(i + 1, len(shells)):
            assert not (shells[i] & shells[j]).any()
    assert (owner[~solid] == 0).all()                  # every node owned by the one solid node
