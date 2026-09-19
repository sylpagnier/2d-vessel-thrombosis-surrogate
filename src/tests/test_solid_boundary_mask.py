"""The wound is wall for geometry, and a separate label only for the deposition law.

Regression cover for the encoding bug where every wall-derived feature (SDF, wall normal,
hydraulic width, no-slip, WSS masking) was measured against ``mask_wall`` alone.  Because the
extract carves ``mask_wall`` disjoint from ``mask_wound`` (COMSOL ``dif1`` vs ``sel1``), that
put injured nodes 0.11-0.32 diameters into the lumen and dragged the interior nodes above the
wound with them.
"""

from __future__ import annotations
from src.utils.paths import anchor_packs_dir

import numpy as np
import torch

from src.data_gen.lib.mesh_wls import solid_boundary_mask


def test_union_is_wall_plus_wound():
    wall = torch.tensor([True, True, False, False])
    wound = torch.tensor([False, False, True, False])
    assert solid_boundary_mask(wall, wound).tolist() == [True, True, True, False]


def test_nowound_packs_are_untouched():
    """An empty or absent wound mask must return the wall mask unchanged."""
    wall = torch.tensor([True, False, True])
    assert torch.equal(solid_boundary_mask(wall, None), wall)
    assert torch.equal(solid_boundary_mask(wall, torch.zeros(0, dtype=torch.bool)), wall)
    empty = torch.zeros(3, dtype=torch.bool)
    assert torch.equal(solid_boundary_mask(wall, empty), wall)


def test_wound_node_sdf_is_zero_not_distance_to_nearest_healthy_wall():
    """The bug, reproduced on a straight channel with a wound cut into the lower wall.

    Measured against ``mask_wall`` alone, a node in the middle of the wound reports the
    distance to the nearest un-wounded wall node -- half the wound length -- instead of 0.
    """
    from scipy.spatial import cKDTree

    xs = np.linspace(0.0, 10.0, 21)
    lower = np.stack([xs, np.zeros_like(xs)], axis=1)
    upper = np.stack([xs, np.ones_like(xs)], axis=1)
    pos = np.concatenate([lower, upper], axis=0)

    n = pos.shape[0]
    wound = torch.zeros(n, dtype=torch.bool)
    wound[8:13] = True  # a 2.0-long patch mid-way along the lower wall
    wall = torch.ones(n, dtype=torch.bool) & ~wound

    def sdf(mask):
        idx = np.where(mask.numpy())[0]
        return cKDTree(pos[idx]).query(pos)[0]

    wall_only = sdf(wall)
    solid = sdf(solid_boundary_mask(wall, wound))

    mid = 10  # centre of the wound patch
    assert wall_only[mid] > 0.9, "expected the bug: wound node reads as lumen"
    assert solid[mid] == 0.0, "wound node must sit on the boundary"
    assert np.allclose(solid[wall.numpy()], 0.0), "healthy wall nodes stay at zero"


def test_repaired_wound_packs_encode_the_wound_as_wall():
    """End-to-end: on-disk wound packs must carry SDF 0 on the injured segment."""
    from pathlib import Path

    root = anchor_packs_dir()
    packs = sorted(root.glob("wound_comsol*.pt"))
    if not packs:
        import pytest

        pytest.skip("no wound packs on disk")

    for p in packs:
        data = torch.load(p, map_location="cpu", weights_only=False)
        names = data.x_channel_names.split(",")
        sdf = data.x[:, names.index("sdf_nd")].numpy()
        w = data.mask_wound.numpy()
        wl = data.mask_wall.numpy()
        assert w.any(), f"{p.name}: wound pack with an empty wound mask"
        assert not (w & wl).any(), f"{p.name}: wall and wound must stay disjoint labels"
        # 1e-6 is the clamp floor in the builder, not a tolerance on a real distance.
        assert sdf[w].max() <= 1e-5, (
            f"{p.name}: wound nodes encoded {sdf[w].max():.4f} diameters into the lumen; "
            "re-run scripts/repair_wound_pack_geometry.py"
        )
        assert sdf[wl].max() <= 1e-5, f"{p.name}: healthy wall SDF is not zero"


# --------------------------------------------------------------------------------------
# The clot-ML feature path.  WOUND_PROGRESS 6 wired the union through the pack BUILDERS
# (`src/data_gen/lib/*`), so the stored `sdf_nd` above is correct -- while
# `src/clot_ml/features.py` went on re-deriving its own geometry from `mask_wall` alone and
# re-introduced the same bug one layer up (MODEL_REVIEW_2026-08-22 5b.3).  These assertions
# are what the end-to-end test above was missing.
# --------------------------------------------------------------------------------------


def test_solid_boundary_nodes_reads_a_pack_the_same_way_as_the_mask():
    """The accessor, on the shapes packs actually store (trailing singleton dims)."""
    from src.data_gen.lib.mesh_wls import solid_boundary_nodes

    class _P:
        pass

    p = _P()
    p.mask_wall = torch.tensor([[True], [False], [False], [False]])
    p.mask_wound = torch.tensor([[False], [True], [False], [False]])
    assert solid_boundary_nodes(p).tolist() == [True, True, False, False]

    del p.mask_wound                       # a no-wound pack carries no attribute at all
    assert solid_boundary_nodes(p).tolist() == [True, False, False, False]


def test_clot_ml_features_encode_the_wound_as_boundary():
    """`build_sample` on a real wound pack: the injured segment must read as wall.

    Before the fix these three channels read `is_wall` 0, `hop_wall` 8.7 (max 12) and
    `dist_wall_edges` 10.92 at the wound -- the wound encoded as open lumen eleven edge
    lengths off the wall, with a distant healthy node as its `owner`.
    """
    from pathlib import Path

    root = anchor_packs_dir()
    p = root / "wound_comsol001.pt"
    if not p.exists():
        import pytest

        pytest.skip("wound_comsol001 pack not present")

    from src.clot_ml.locked import build_sample
    from src.config import BiochemConfig, PhysicsConfig

    data = torch.load(p, map_location="cpu", weights_only=False)
    S = build_sample(data, BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem"),
                     flow="gt", variant="v4")
    cols = [str(c) for c in S["cols"]]
    w = data.mask_wound.reshape(-1).bool().numpy()
    wall = data.mask_wall.reshape(-1).bool().numpy()

    assert S["solid"][w].all(), "the wound is missing from the sample's solid mask"
    assert S["solid"][wall].all(), "the healthy wall is missing from the solid mask"
    assert np.array_equal(S["wall"], wall), (
        "`wall` must stay the healthy-wall LABEL -- it selects the gated `srf1` deposition "
        "law and (for now) the eval domain; only the geometry takes the union")

    for ch, want in (("is_wall", 1.0), ("hop_wall", 0.0), ("dist_wall_edges", 0.0)):
        got = S["X"][:, cols.index(ch)][w]
        assert np.allclose(got, want, atol=1e-5), (
            "%s reads %.4f at the wound, expected %.1f -- clot-ML geometry is still "
            "measured against mask_wall alone" % (ch, float(got.mean()), want))

    # a wound node's nearest solid node is itself, so it owns itself -- it used to be
    # owned by a healthy wall node carrying no `Mat`, several diameters away
    assert np.array_equal(S["owner"][w], np.flatnonzero(w)), "wound nodes must own themselves"


def test_transport_boundary_falls_back_to_wall_without_a_solid_key():
    """A v3 cache built before this change carries no `solid`; that must be a no-op."""
    from src.clot_ml.features_v4 import new_channels

    xs = np.linspace(0.0, 10.0, 21)
    pos = np.concatenate([np.stack([xs, np.zeros_like(xs)], 1),
                          np.stack([xs, np.ones_like(xs)], 1)], 0)
    n = len(pos)
    ei = np.array([[i, i + 1] for i in range(20)] + [[21 + i, 22 + i] for i in range(20)]).T
    wall = np.zeros(n, dtype=bool)
    wall[[0, 1, 2, 21, 22, 23]] = True
    u = np.ones(n)
    v = np.zeros(n)
    S = dict(wall=wall, edge_index=ei, owner=np.zeros(n, np.int64), pos=pos.astype(np.float32),
             u=u.astype(np.float32), v=v.astype(np.float32),
             mat_phys=np.ones(n, np.float32))
    ind = np.ones(n)
    a = new_channels(S, ind, ind, ind, 1.0)
    b = new_channels(dict(S, solid=wall), ind, ind, ind, 1.0)
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{k} moved with solid == wall"
