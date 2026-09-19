"""The two guarantees the solenoidal projection has to keep, and their order of precedence.

`precache_rgp_deq.py --solenoidal` projects the RGP-DEQ's velocity residual onto the discretely
divergence-free fields, because the off-wall feature block transports material along the flow
and a transport solve on a field that is not solenoidal creates and destroys the material the
equation conserves (RGP_DEQ_REPAIR_PLAN.md §18.12).

It has to do that **without moving the wall**.  The first implementation did not: the
projection uses natural conditions, so `B^T y` is nonzero on the wall and the corrected field
left it at 3-9e-3 of the field's peak.  That is small, and it is not harmless --
`test_deployable_flow_fixes.py::test_wall_destination_edges_carry_no_direction_under_either_flow`
failed on it, because a wall node with velocity gives every edge pointing into it a direction
the clot ensemble never saw at training time.

So the precedence is fixed: **no-slip is exact, divergence is minimised subject to it.**  These
tests pin that order, and the measurement that says it is nearly free.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core_physics.solenoidal_residual import divergence, project_solenoidal


def _operators(n: int = 24, seed: int = 0):
    """A small structured graph and its MLS gradient operators."""
    from src.core_physics.mls_gradient import build_mls_gradient

    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 0.25, 8))
    pos = np.stack([xs.ravel(), ys.ravel()], axis=1)
    pos = pos + rng.normal(0.0, 1e-3, pos.shape)          # break the exact lattice symmetry

    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=0.075, output_type="ndarray")
    ei = np.concatenate([pairs.T, pairs.T[::-1]], axis=1)
    Dx, Dy = build_mls_gradient(pos, ei, hops=2)
    wall = (pos[:, 1] <= pos[:, 1].min() + 1e-9) | (pos[:, 1] >= pos[:, 1].max() - 1e-9)
    return pos, Dx, Dy, wall


@pytest.fixture(scope="module")
def setup():
    pos, Dx, Dy, wall = _operators()
    rng = np.random.default_rng(1)
    # A residual shaped like the real one: smooth, small, and already ~0 on the wall.
    env = np.clip(np.minimum(pos[:, 1] - pos[:, 1].min(), pos[:, 1].max() - pos[:, 1]), 0, None)
    env = env / (env.max() + 1e-30)
    delta = rng.normal(0.0, 1.0, (pos.shape[0], 2)) * env[:, None] * 0.01
    return pos, Dx, Dy, wall, delta


def test_no_slip_is_exact_after_projection(setup):
    """The wall is zeroed, bit-for-bit -- not "small", which is what the first version gave."""
    _, Dx, Dy, wall, delta = setup
    out, info = project_solenoidal(delta, Dx, Dy, wall=wall)
    assert np.array_equal(out[wall], np.zeros_like(out[wall]))
    assert info["wall_leak"] == 0.0


def test_the_raw_projection_really_does_move_the_wall(setup):
    """Guard the guard: if this ever stops being true the zeroing is dead code."""
    _, Dx, Dy, wall, delta = setup
    out, info = project_solenoidal(delta, Dx, Dy, wall=wall)
    assert info["wall_leak_raw"] > 0.0, (
        "the natural-condition projection left the wall untouched, so the explicit zeroing "
        "above is no longer earning its place -- re-derive before deleting it")


def test_divergence_falls_and_no_slip_costs_almost_nothing(setup):
    """Enforcing no-slip must not give back the divergence the projection exists to remove."""
    _, Dx, Dy, wall, delta = setup
    free, _ = project_solenoidal(delta, Dx, Dy, wall=None)
    pinned, info = project_solenoidal(delta, Dx, Dy, wall=wall)

    d0 = float(np.abs(divergence(delta, Dx, Dy)).mean())
    d_free = float(np.abs(divergence(free, Dx, Dy)).mean())
    d_pin = float(np.abs(divergence(pinned, Dx, Dy)).mean())
    assert d_free < 0.25 * d0, f"projection did not reduce divergence ({d_free:.3g} vs {d0:.3g})"
    assert d_pin < 0.5 * d0, "pinning the wall gave the divergence back"
    assert info["div_after"] == pytest.approx(d_pin, rel=1e-9)


def test_projection_keeps_most_of_the_residual(setup):
    """It is a projection, not a suppressor: the head's correction must largely survive."""
    _, Dx, Dy, wall, delta = setup
    out, info = project_solenoidal(delta, Dx, Dy, wall=wall)
    kept = np.linalg.norm(out) / (np.linalg.norm(delta) + 1e-30)
    assert 0.5 < kept <= 1.0, f"projection kept only {kept:.2f} of the residual"
    assert info["moved"] < 0.75


def test_shape_is_validated(setup):
    _, Dx, Dy, _, delta = setup
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        project_solenoidal(delta[:, 0], Dx, Dy)
