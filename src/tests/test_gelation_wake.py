"""Pins on the gelation-wake closed loop (src/core_physics/gelation_wake.py).

The load-bearing properties are the SAFETY ones: a positive feedback on the deposition gate
must be a provable no-op where nothing has gelled, must never extrapolate its kernel past
the data, and must not let a node close its own gate after committing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core_physics.gelation_wake import (
    GELLED_SR_RATIO, WAKE_LOAD_AMP, WAKE_LOAD_KNOTS, WAKE_MAX_HOPS, wake_amplitude,
)


def test_wake_amplitude_is_one_at_zero_load():
    """No committed tissue anywhere means the t=0 shear is untouched."""
    assert wake_amplitude(np.zeros(5)) == pytest.approx(np.ones(5))


def test_wake_amplitude_is_monotone_decreasing():
    load = np.linspace(0.0, 20.0, 400)
    amp = wake_amplitude(load)
    assert np.all(np.diff(amp) <= 1e-12)


def test_wake_amplitude_is_clamped_at_both_ends_and_never_extrapolates():
    """The kernel runs between two MEASUREMENTS and past neither.

    Above the last knot it holds ``GELLED_SR_RATIO`` -- the measured per-node step -- rather
    than continuing a slope into a regime with no data, which is how a feedback like this
    runs away.
    """
    amp = wake_amplitude(np.array([WAKE_LOAD_KNOTS[-1], 1e3, 1e9]))
    assert amp == pytest.approx(np.full(3, GELLED_SR_RATIO))
    assert wake_amplitude(np.array([-5.0])) == pytest.approx([WAKE_LOAD_AMP[0]])
    assert float(wake_amplitude(np.array([1e9]))[0]) >= GELLED_SR_RATIO


def test_knot_table_is_well_formed():
    assert len(WAKE_LOAD_KNOTS) == len(WAKE_LOAD_AMP)
    assert list(WAKE_LOAD_KNOTS) == sorted(WAKE_LOAD_KNOTS)
    assert WAKE_LOAD_AMP[0] == 1.0
    assert WAKE_LOAD_AMP[-1] == GELLED_SR_RATIO
    assert WAKE_MAX_HOPS >= 1


def _toy(n_wall: int = 6):
    """A tiny path graph whose nodes are all wall, with a stub `data` and `fields`."""
    import torch

    class _D:
        pass

    d = _D()
    d.num_nodes = n_wall
    ei = np.array([[i for i in range(n_wall - 1)] + [i + 1 for i in range(n_wall - 1)],
                   [i + 1 for i in range(n_wall - 1)] + [i for i in range(n_wall - 1)]])
    d.edge_index = torch.tensor(ei)
    return d


def test_blockage_is_bit_identical_to_gate0_when_nothing_has_gelled():
    """The regression check: a vessel the model never ignites is untouched."""
    from src.config import BiochemConfig
    from src.core_physics.gelation_wake import make_gelation_wake_blockage
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    d = _toy()
    wall = np.ones(d.num_nodes, dtype=bool)
    f = T0Fields(sr=np.full(d.num_nodes, 200.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_gelation_wake_blockage(d, bio, f, wall)
    gate0 = np.full(d.num_nodes, 0.7)
    out = blk(np.zeros(d.num_nodes), gate0, 0)
    assert out is gate0 or np.array_equal(out, gate0)


def test_committed_node_never_loses_its_gate():
    """`mu1` has fired: a committed node is clot, and the ODE must stay monotone."""
    from src.config import BiochemConfig
    from src.core_physics.gelation_wake import make_gelation_wake_blockage
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d = _toy()
    wall = np.ones(d.num_nodes, dtype=bool)
    f = T0Fields(sr=np.full(d.num_nodes, 200.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_gelation_wake_blockage(d, bio, f, wall)
    gate0 = np.full(d.num_nodes, 0.9)
    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g = blk(mat, gate0, 1)
    assert g[0] >= gate0[0] - 1e-12


def test_wake_lowers_shear_for_neighbours_of_committed_tissue():
    """The whole point: committed tissue must be able to open a neighbour's low-shear gate."""
    from src.config import BiochemConfig
    from src.core_physics.gelation_wake import make_gelation_wake_blockage
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d = _toy(n_wall=4)
    wall = np.ones(d.num_nodes, dtype=bool)
    # sr just above lss, so a modest wake is enough to trip the low-shear branch
    f = T0Fields(sr=np.full(d.num_nodes, float(bio.lss) * 1.02), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_gelation_wake_blockage(d, bio, f, wall)
    gate0 = np.zeros(d.num_nodes)
    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g = blk(mat, gate0, 1)
    assert g[1] > 0.0, "a neighbour of committed tissue should gain a gate"


# ---------------------------------------------------------------------------
# The wound-regime readout and the wound Mat source (docs/WOUND_PROGRESS.md 14.6/14.7)
# ---------------------------------------------------------------------------
def test_deposition_gate_is_a_noop_without_a_wound_mask():
    """The `srf2` term may only appear where COMSOL put one."""
    import torch

    from src.core_physics.physics_wall_model import deposition_gate

    class _D:
        pass

    d = _D()
    d.num_nodes = 4
    d.mask_wall = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

    class _F:
        gate = np.array([0.5, 2.0, 9.0, 9.0])

    want = _F.gate * np.array([1, 1, 0, 0], dtype=bool)
    assert np.array_equal(deposition_gate(d, _F), want)
    d.mask_wound = torch.zeros(4, dtype=torch.bool)          # present but empty
    assert np.array_equal(deposition_gate(d, _F), want)


def test_deposition_gate_gives_the_wound_the_UNGATED_prefactor():
    """`srf2` is `srf1` with both shear gates deleted, so the wound's multiplier is 1 --
    it does NOT inherit the healthy wall's gated value, and it is not zero either."""
    import torch

    from src.core_physics.physics_wall_model import (
        WOUND_UNGATED_PREFACTOR, deposition_gate,
    )

    class _D:
        pass

    d = _D()
    d.num_nodes = 4
    d.mask_wall = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    d.mask_wound = torch.tensor([0, 0, 1, 0], dtype=torch.bool)

    class _F:
        gate = np.array([0.5, 2.0, 0.0, 0.0])

    g = deposition_gate(d, _F)
    assert g[2] == WOUND_UNGATED_PREFACTOR
    assert list(g[:2]) == [0.5, 2.0]
    assert g[3] == 0.0
    # and the switch reproduces the old healthy-wall-only field exactly
    assert np.array_equal(deposition_gate(d, _F, wound_source=False),
                          _F.gate * np.array([1, 1, 0, 0], dtype=bool))


def test_wound_regime_readout_cannot_fire_without_wound_nodes():
    """The rank readout buys wound vessels wall +0.075 / far +0.21 and costs the cohort
    -0.022 / -0.162, so it MUST be unreachable on a pack with no wound."""
    from src.clot_ml.locked import _committed_set_v4

    n = 6
    wall = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    S = {"wall": wall, "solid": wall, "phys_mask": np.ones(n, bool),
         "edge_index": np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])}
    sc = np.linspace(0.1, 0.99, n)
    temporal = {"wall_spec": {"kind": "cohort_cut", "t": 0.5},
                "off_spec": {"kind": "cohort_cut", "t": 0.5}}
    base = _committed_set_v4(S, sc, temporal)
    with_spec = _committed_set_v4(
        S, sc, dict(temporal, wound_spec={"kind": "cohort_cut", "t": 0.0}))
    assert np.array_equal(base, with_spec), "wound_spec fired on a pack with no wound"
