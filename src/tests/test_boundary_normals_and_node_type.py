"""The two dead channel groups, fixed 2026-08-22.

`wall_normal` was identically zero at every wall node on every pack (the Gmsh line-cell
branch never ran -- COMSOL exports only `triangle6`), and `node_type_0..3` was a literal
`torch.zeros((N, 4))`.  See `docs/MODEL_REVIEW_2026-08-22.md` 6.5 and WOUND_PROGRESS 8.

The normals are now fitted from the graph, so they need no mesh at all -- which is what makes
them applicable to the three wound packs whose COMSOL exports are gone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data_gen.lib.mesh_wls import boundary_normals_from_graph, node_type_one_hot

PACKS = Path(__file__).resolve().parents[2] / "data/processed/graphs_biochem_anchors"


def _channel(nx=21, ny=5, length=2.0):
    """Straight channel, walls at y=0 and y=1; inward normals are +y and -y."""
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, 1.0, ny)
    pos = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    idx = lambda i, j: j * nx + i  # noqa: E731
    ei = []
    for j in range(ny):
        ei += [(idx(i, j), idx(i + 1, j)) for i in range(nx - 1)]
    for i in range(nx):
        ei += [(idx(i, j), idx(i, j + 1)) for j in range(ny - 1)]
    solid = np.zeros(len(pos), dtype=bool)
    solid[[idx(i, 0) for i in range(nx)]] = True
    solid[[idx(i, ny - 1) for i in range(nx)]] = True
    return pos, np.array(ei).T, solid, idx, nx, ny


# --------------------------------------------------------------------------- normals
def test_normals_point_into_the_lumen_on_a_straight_channel():
    pos, ei, solid, idx, nx, ny = _channel()
    N = boundary_normals_from_graph(pos, solid, ei)
    bottom = [idx(i, 0) for i in range(1, nx - 1)]
    top = [idx(i, ny - 1) for i in range(1, nx - 1)]
    assert np.allclose(N[bottom], np.array([0.0, 1.0]), atol=1e-9)
    assert np.allclose(N[top], np.array([0.0, -1.0]), atol=1e-9)


def test_normals_are_unit_on_solid_and_exactly_zero_elsewhere():
    pos, ei, solid, *_ = _channel()
    N = boundary_normals_from_graph(pos, solid, ei)
    assert np.allclose(np.linalg.norm(N[solid], axis=1), 1.0)
    assert np.all(N[~solid] == 0.0)


def test_isolated_solid_nodes_still_get_a_normal():
    """Degree-0 nodes in the solid subgraph exist on real packs (12 of 539 on patient008)."""
    pos, ei, solid, idx, nx, ny = _channel()
    keep = ~(solid[ei[0]] & solid[ei[1]] &
             ((ei[0] == idx(10, 0)) | (ei[1] == idx(10, 0))))
    N = boundary_normals_from_graph(pos, solid, ei[:, keep])
    assert np.linalg.norm(N[idx(10, 0)]) == pytest.approx(1.0)
    assert N[idx(10, 0)][1] > 0.9          # still points into the lumen


def test_orientation_follows_the_supplied_targets():
    pos, ei, solid, idx, nx, ny = _channel()
    flipped = np.tile(np.array([0.0, -5.0]), (len(pos), 1))   # target below both walls
    N = boundary_normals_from_graph(pos, solid, ei, orient_targets=flipped)
    assert N[idx(10, 0)][1] < 0 and N[idx(10, ny - 1)][1] < 0


def test_empty_solid_mask_is_a_no_op():
    pos, ei, solid, *_ = _channel()
    N = boundary_normals_from_graph(pos, np.zeros(len(pos), dtype=bool), ei)
    assert np.all(N == 0.0)


# --------------------------------------------------------------------------- node_type
def test_one_hot_is_exactly_one_per_node_with_inlet_priority():
    n = 10
    solid = torch.zeros(n, dtype=torch.bool)
    solid[:6] = True
    inlet = torch.zeros(n, dtype=torch.bool)
    inlet[0] = True                     # also solid -> inlet wins
    outlet = torch.zeros(n, dtype=torch.bool)
    outlet[1] = True                    # also solid -> outlet wins
    oh = node_type_one_hot(solid, inlet, outlet)
    assert oh.shape == (n, 4)
    assert torch.all(oh.sum(dim=1) == 1)
    assert oh[0].tolist() == [0, 0, 1, 0]
    assert oh[1].tolist() == [0, 0, 0, 1]
    assert oh[2].tolist() == [0, 1, 0, 0]      # solid
    assert oh[9].tolist() == [1, 0, 0, 0]      # interior


# --------------------------------------------------------------------------- on-disk
@pytest.mark.parametrize("stem", ["patient020", "wound_patient001"])
def test_repaired_packs_carry_real_normals_and_one_hots(stem):
    """End-to-end: after `scripts/repair_pack_wall_normals.py` these must not be zero."""
    p = PACKS / f"{stem}.pt"
    if not p.exists():
        pytest.skip(f"{stem} pack not present")
    d = torch.load(p, map_location="cpu", weights_only=False)
    ch = d.x_channel_names.split(",")
    solid = d.mask_wall.reshape(-1).bool().numpy()
    w = getattr(d, "mask_wound", None)
    if torch.is_tensor(w) and w.numel():
        solid = solid | w.reshape(-1).bool().numpy()
    nx_ = d.x[:, ch.index("wall_normal_x")].numpy()
    ny_ = d.x[:, ch.index("wall_normal_y")].numpy()
    mag = np.hypot(nx_, ny_)[solid]
    oh = d.x[:, [ch.index("node_type_%d" % k) for k in range(4)]].numpy()
    if mag.max() == 0.0 and oh.sum() == 0.0:
        pytest.skip("pack not repaired yet -- run scripts/repair_pack_wall_normals.py")
    # UNIT length, not merely non-zero.  A `> 0.99` assertion passes at |n| = 2.0, which is
    # exactly what a double-applied delta produces -- running the repair twice once left
    # every normal at 2.0 and no test noticed.  Pin the magnitude both ways.
    assert np.allclose(mag, 1.0, atol=1e-5), (
        "wall normals must be UNIT at every solid node; got median %.4f, max %.4f "
        "(|n| ~ 2 means the repair delta was applied twice)"
        % (float(np.median(mag)), float(mag.max())))
    assert (oh.sum(axis=1) == 1).all(), "node_type must be a strict one-hot"
    assert oh[:, 1].sum() > 0, "some node must be labelled solid"


@pytest.mark.parametrize("stem", ["patient020", "wound_patient001"])
def test_repair_is_idempotent(stem):
    """Re-running the repair must change nothing -- see the |n| = 2.0 incident above."""
    import sys
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    if not (PACKS / f"{stem}.pt").exists():
        pytest.skip(f"{stem} pack not present")
    from repair_pack_wall_normals import report

    r = report(stem)
    if not r["was_repaired"]:
        pytest.skip("pack not repaired yet")
    assert not r["moved"], (
        "re-running the repair on an already-repaired pack must be a no-op, but it would "
        "move %s" % sorted(r["moved"]))
