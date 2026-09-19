"""The real clot->flow coupling: a Navier-Stokes re-solve against COMSOL's `mu1` viscosity step.

Two things broke the E1h experiment (docs/publication/EXPERIMENTS.md) and each gets a guard:

  * the clot step, interpolated to quadrature points on the P2 basis, rang NEGATIVE (-0.127 Pa.s
    against a 0.0035 Carreau floor), so the solve diverged at the physical 0.51 Pa.s and the
    experiment ran for two days at 5x below it;
  * seeded from its own previous coupled field, the Picard iteration fell into a limit cycle and
    returned a field 0.77 u_ref off the converged one, with only a warning.

And the arm's defining property -- it reads the rollout's OWN committed set, never ground truth
-- is pinned, because that is the whole difference between it and the oracle arm that leaked.
"""
import warnings

import numpy as np
import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.utils.paths import anchor_meshes_dir, anchor_packs_dir

PACK = anchor_packs_dir() / "comsol001.pt"
MESH = anchor_meshes_dir() / "comsol001.nas"


def _pack():
    if not PACK.is_file() or not MESH.is_file():
        pytest.skip("comsol001 pack or mesh not found")
    data = torch.load(PACK, map_location="cpu", weights_only=False)
    data.graph_stem = "comsol001"
    return data


def test_physical_clot_viscosity_converges():
    """COMSOL's own step on a wall band must converge to a bounded field, not blow up."""
    from src.core_physics.local_fem_solver import solve_local_t0_flow
    from src.core_physics.physics_wall_model import clot_delta_mu_si

    data = _pack()
    bio = BiochemConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().numpy()
    band = np.zeros_like(wall)
    band[np.flatnonzero(wall)[: max(20, wall.sum() // 4)]] = True
    kw = dict(max_iters=300, tol=1e-9, verbose=False)
    clean = solve_local_t0_flow(str(MESH), data, PhysicsConfig(), **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        loaded = solve_local_t0_flow(str(MESH), data, PhysicsConfig(),
                                     delta_mu_nodal_si=np.where(band, clot_delta_mu_si(bio), 0.0),
                                     **kw)
    assert not any("did not converge" in str(w.message) for w in caught)
    assert float(np.abs(loaded).max()) < 3.0 * float(np.abs(clean).max())
    assert not np.allclose(loaded, clean), "a clot load that moves nothing is a dead toggle"


def test_clot_delta_mu_is_the_comsol_step():
    from src.core_physics.physics_wall_model import clot_delta_mu_si

    assert clot_delta_mu_si(BiochemConfig(phase="biochem")) == pytest.approx(0.5135, abs=5e-3)


def _stub_blockage(monkeypatch, *, every=4, **kw):
    """The blockage with the FEM swapped for a stub that records what it was asked to solve."""
    import src.clot_ml.v0 as v0
    import src.core_physics.t0_mu_physics as t0mu
    from src.core_physics.physics_wall_model import fem_resolve_blockage, t0_flow_fields

    data = _pack()
    bio = BiochemConfig(phase="biochem")
    f0 = t0_flow_fields(data, bio, flow_source="gt")
    wall = data.mask_wall.reshape(-1).bool().numpy()
    calls = []

    def fake_solve(d, *, delta_mu_nodal_si=None, require_converged=False, **kw):
        assert require_converged, "the coupled arm must never accept an unconverged field"
        calls.append(np.asarray(delta_mu_nodal_si).copy())
        return np.stack([f0.u * 0.5, f0.v * 0.5], axis=1)

    def no_gt(*a, **k):
        raise AssertionError("the FEM-coupled arm read ground-truth clot")

    monkeypatch.setattr(v0, "solve_fem_velocity_nd", fake_solve)
    monkeypatch.setattr(t0mu, "gt_clot_phi_at_time", no_gt)
    blk = fem_resolve_blockage(data, bio, f0, wall, every=every, flow="gt", **kw)
    gate0 = f0.gate * wall
    return blk, gate0, calls, float(bio.viscosity_mat_crit), np.flatnonzero(wall)


def test_nothing_committed_is_bit_identical_and_solves_nothing(monkeypatch):
    blk, gate0, calls, crit, _ = _stub_blockage(monkeypatch)
    mat = np.zeros(gate0.shape[0])
    for i in range(10):
        assert blk(mat, gate0, i) is gate0
    assert calls == []


def test_resolves_on_cadence_and_only_when_the_committed_set_changes(monkeypatch):
    blk, gate0, calls, crit, widx = _stub_blockage(monkeypatch, every=4)
    mat = np.zeros(gate0.shape[0])
    mat[widx[:3]] = 2 * crit
    blk(mat, gate0, 0)
    assert len(calls) == 1 and int((calls[0] > 0).sum()) == 3
    mat[widx[3]] = 2 * crit
    blk(mat, gate0, 2)                  # inside the stride: held
    assert len(calls) == 1
    blk(mat, gate0, 4)                  # stride elapsed, set grew: re-solve
    assert len(calls) == 2
    blk(mat, gate0, 8)                  # stride elapsed, set unchanged: exact to skip
    assert len(calls) == 2


def test_committed_nodes_keep_their_gate(monkeypatch):
    blk, gate0, calls, crit, widx = _stub_blockage(monkeypatch, every=1)
    lit = widx[gate0[widx] > 0][:5]
    if lit.size == 0:
        pytest.skip("no gated wall nodes on this pack")
    mat = np.zeros(gate0.shape[0])
    mat[lit] = 2 * crit
    g = blk(mat, gate0, 0)
    assert np.all(g[lit] >= gate0[lit])


def test_min_new_waits_for_enough_new_clot(monkeypatch):
    blk, gate0, calls, crit, widx = _stub_blockage(monkeypatch, every=1, min_new=5)
    mat = np.zeros(gate0.shape[0])
    mat[widx[:3]] = 2 * crit
    assert blk(mat, gate0, 0) is gate0 and calls == []     # 3 < 5: not yet
    mat[widx[3:5]] = 2 * crit
    blk(mat, gate0, 1)                                      # 5 new: solve
    assert len(calls) == 1
    mat[widx[5:9]] = 2 * crit
    blk(mat, gate0, 2)                                      # 4 more since the solve: hold
    assert len(calls) == 1
    mat[widx[9]] = 2 * crit
    blk(mat, gate0, 3)                                      # 5 more: solve
    assert len(calls) == 2


def test_features_only_arm_never_regates_and_solves_once(monkeypatch):
    blk, gate0, calls, crit, widx = _stub_blockage(monkeypatch, every=1, update_gate=False)
    mat = np.zeros(gate0.shape[0])
    for i in range(6):
        mat[widx[i]] = 2 * crit
        assert blk(mat, gate0, i) is gate0
    assert calls == []
    blk.final_velocity(mat)
    assert len(calls) == 1


def test_coupling_grammar_parses():
    import re

    pat = r"fem(?:@(\d+))?(?:\+(\d+))?|femfinal"
    assert re.fullmatch(pat, "fem@1+5").groups() == ("1", "5")
    assert re.fullmatch(pat, "fem+20").groups() == (None, "20")
    assert re.fullmatch(pat, "femfinal")
    assert not re.fullmatch(pat, "wake@16")


def test_final_velocity_reflects_the_final_committed_set(monkeypatch):
    blk, gate0, calls, crit, widx = _stub_blockage(monkeypatch, every=100)
    mat = np.zeros(gate0.shape[0])
    mat[widx[:2]] = 2 * crit
    blk(mat, gate0, 0)
    mat[widx[2:6]] = 2 * crit           # grows after the last re-solve
    blk.final_velocity(mat)
    assert len(calls) == 2 and int((calls[-1] > 0).sum()) == 6
