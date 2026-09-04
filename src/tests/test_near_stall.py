"""Pins on the near-field stall blockage (src/core_physics/near_stall.py).

Safety properties first: nothing gelled is a no-op; a committed node keeps its gate;
the wound's ungated prefactor is not rewritten; stall=False on the ODE is the shipped
clock.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core_physics.gelation_wake import GELLED_SR_RATIO
from src.core_physics.near_stall import STALL_HOPS, make_near_stall_blockage, near_stall_amplitude


def test_amplitude_is_one_beyond_the_stencil_and_mu1_inside():
    h = np.array([0.0, 1.0, 4.0, 5.0, 99.0])
    amp = near_stall_amplitude(h, hops_cut=4)
    assert amp[:3] == pytest.approx(np.full(3, GELLED_SR_RATIO))
    assert amp[3:] == pytest.approx(np.ones(2))


def _toy(n: int = 8, wound_at: int | None = None):
    class _D:
        pass

    d = _D()
    d.num_nodes = n
    ei = np.array([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                   [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    d.edge_index = torch.tensor(ei)
    wall = np.ones(n, dtype=bool)
    wound = np.zeros(n, dtype=bool)
    if wound_at is not None:
        wall[wound_at] = False
        wound[wound_at] = True
    d.mask_wall = torch.tensor(wall)
    d.mask_wound = torch.tensor(wound)
    return d, wall, wound


def test_blockage_is_gate0_when_nothing_has_gelled():
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    d, wall, _ = _toy()
    f = T0Fields(sr=np.full(d.num_nodes, 200.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=4)
    gate0 = np.full(d.num_nodes, 0.7)
    out = blk(np.zeros(d.num_nodes), gate0, 0)
    assert out is gate0 or np.array_equal(out, gate0)


def test_committed_node_never_loses_its_gate():
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d, wall, _ = _toy()
    f = T0Fields(sr=np.full(d.num_nodes, 200.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2)
    gate0 = np.full(d.num_nodes, 0.9)
    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g = blk(mat, gate0, 1)
    assert g[0] >= gate0[0] - 1e-12


def test_neighbour_inside_stencil_opens_low_shear_gate():
    """A high-sr wall node next to committed solid must be able to drop below lss."""
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d, wall, _ = _toy(n=8)
    f = T0Fields(sr=np.full(d.num_nodes, 120.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2)
    gate0 = np.zeros(d.num_nodes)
    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g = blk(mat, gate0, 1)
    assert g[1] > 0.0 and g[2] > 0.0
    assert g[4] == 0.0 and g[7] == 0.0, "beyond the stencil the t=0 gate is unchanged"


def test_wound_keeps_ungated_prefactor_when_stall_rewrites_the_wall():
    """srf2 is already ungated; stall must not replace it with stalled srf1."""
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields, WOUND_UNGATED_PREFACTOR

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d, wall, wound = _toy(n=6, wound_at=0)
    f = T0Fields(sr=np.full(d.num_nodes, 120.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2)
    gate0 = np.zeros(d.num_nodes)
    gate0[wound] = WOUND_UNGATED_PREFACTOR
    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g = blk(mat, gate0, 1)
    assert g[0] == pytest.approx(WOUND_UNGATED_PREFACTOR)
    assert g[1] > 0.0, "healthy wall next to a gelled wound must stall"


def test_stall_hops_default_is_one_corner_shell():
    assert STALL_HOPS == 2


def test_scale_dsrx_defaults_off():
    import inspect
    sig = inspect.signature(make_near_stall_blockage)
    assert sig.parameters["scale_dsrx"].default is False
    assert sig.parameters["seed_wound"].default is True


def test_seed_wound_opens_hop_shell_with_zero_mat():
    """The injured patch is a stall source from t=0; occupancy must not wait for Mat>=crit."""
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    d, wall, _ = _toy(n=8, wound_at=0)
    f = T0Fields(sr=np.full(d.num_nodes, 120.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2, seed_wound=True)
    gate0 = np.zeros(d.num_nodes)
    g = blk(np.zeros(d.num_nodes), gate0, 0)
    assert g[1] > 0.0 and g[2] > 0.0
    assert g[4] == 0.0, "beyond one corner shell the t=0 gate is unchanged"


def test_without_seed_wound_zero_mat_is_noop_even_with_a_wound_mask():
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    d, wall, _ = _toy(n=8, wound_at=0)
    f = T0Fields(sr=np.full(d.num_nodes, 120.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2, seed_wound=False)
    gate0 = np.full(d.num_nodes, 0.7)
    out = blk(np.zeros(d.num_nodes), gate0, 0)
    assert out is gate0 or np.array_equal(out, gate0)


def test_nothing_gelled_without_a_wound_attribute_is_still_a_noop():
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields

    bio = BiochemConfig(phase="biochem")
    d, wall, _ = _toy()
    del d.mask_wound
    f = T0Fields(sr=np.full(d.num_nodes, 200.0), dsrx=np.zeros(d.num_nodes),
                 gate_low=None, gate_sep=None, gate=None)
    blk = make_near_stall_blockage(d, bio, f, wall=wall, hops=2)
    gate0 = np.full(d.num_nodes, 0.7)
    out = blk(np.zeros(d.num_nodes), gate0, 0)
    assert out is gate0 or np.array_equal(out, gate0)


def test_default_keeps_A_gate_that_scaling_dsrx_would_close():
    """001 A-gates die if dsrx is multiplied by mu1 (sgt is large and negative)."""
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import T0Fields, gate_from_shear

    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    d, wall, _ = _toy(n=8)
    # sr high enough that even mu1*sr stays above lss (B never opens).
    sr = np.full(d.num_nodes, 500.0)
    # dsrx < sgt_cgs unscaled, but dsrx * GELLED_SR_RATIO is not -- A closes if scaled.
    dsrx = np.full(d.num_nodes, -1000.0)
    f = T0Fields(sr=sr, dsrx=dsrx, gate_low=None, gate_sep=None, gate=None)
    gate0 = gate_from_shear(sr, dsrx, bio, wall=wall)
    assert gate0[1] > 0.0, "fixture: node 1 must be an A-gate at t=0"

    mat = np.zeros(d.num_nodes)
    mat[0] = 2.0 * crit
    g_off = make_near_stall_blockage(
        d, bio, f, wall=wall, hops=2, scale_dsrx=False)(mat, gate0, 1)
    g_on = make_near_stall_blockage(
        d, bio, f, wall=wall, hops=2, scale_dsrx=True)(mat, gate0, 1)
    assert g_off[1] > 0.0, "unscaled dsrx must keep the A-gate inside the stencil"
    assert g_on[1] == 0.0, "scaling dsrx by mu1 must be what closes this A-gate"


def test_ode_trajectory_rejects_wake_and_stall_together():
    from src.clot_ml.temporal import ode_trajectory
    from src.config import BiochemConfig

    class _D:
        pass

    with pytest.raises(ValueError, match="at most one"):
        ode_trajectory(_D(), BiochemConfig(phase="biochem"), wake=True, stall=True)


def _load_pack(stem: str):
    from pathlib import Path

    p = Path("data/processed/graphs_biochem_anchors") / f"{stem}.pt"
    if not p.exists():
        pytest.skip(f"{stem} not on disk")
    return torch.load(p, map_location="cpu", weights_only=False)


def _ungated_stall_extra(data):
    from src.clot_ml.temporal import ode_trajectory
    from src.config import BiochemConfig
    from src.core_physics.physics_wall_model import t0_flow_fields

    bio = BiochemConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    ung = wall & (np.asarray(f.gate) * wall <= 0)
    traj, _ = ode_trajectory(data, bio, flow="gt", stall=True)
    crit = float(bio.viscosity_mat_crit)
    return ung & (np.asarray(traj)[-1] >= crit), wall


def test_seeded_stall_stays_small_and_net_positive_on_wound_001():
    """001's near-wound wall is already gated, so the union must stay a trickle.

    It is NOT zero, which an earlier revision of this test asserted.  Measured at the shipped
    ``STALL_HOPS = 2`` with the wound acting as a ``Mat`` source (WOUND_PROGRESS 14.6), the
    union opens **8** t=0-ungated wall nodes on 001, 6 of them GT clot.  The property worth
    pinning is that it does not spray: a handful of nodes, majority true.  `wound_comsol003`
    is where the mechanism has to earn its place, and that is the next test.
    """
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    data = _load_pack("wound_comsol001")
    extra, wall = _ungated_stall_extra(data)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, PhysicsConfig(phase="biochem")).numpy() > 0.5
    n_fp = int((extra & wall & ~gt).sum())
    assert int(extra.sum()) <= 20
    assert n_fp <= 4
    assert int((extra & wall & gt).sum()) > n_fp


def test_seeded_stall_opens_ungated_wall_on_wound_003():
    """The 003 blinds live on t=0-ungated wall; hops=2 + wound seed must open a subset."""
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    data = _load_pack("wound_comsol003")
    extra, wall = _ungated_stall_extra(data)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, PhysicsConfig(phase="biochem")).numpy() > 0.5
    assert int(extra.sum()) >= 5
    assert int((extra & wall & ~gt).sum()) == 0, "hops=2 union is only legal at 0 wall FP"
