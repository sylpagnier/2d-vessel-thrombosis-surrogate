"""Pins for the unified ``clot_ml_v0`` stack.

The properties that must not drift:

* replace+depth is a magnitude rule on owner Mat, not a union with the GNN off-wall set;
* shell 1 of ``solid_boundary_shells`` is the shipped first corner shell;
* a manifest without extra keys reproduces the documented defaults;
* on a pack with no wound the predictor is the base GNN (promotion gate, skip if missing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.clot_ml.v0 import (
    ClotMlV0Config, DA_SCALE_AUTO, REPLACE_ATT, REPLACE_DEPTH,
    REPLACE_SCOPE_ALL_LUMEN, REPLACE_SCOPE_WOUND_REGION, REPLACE_SCOPES,
    replace_depth_mask,
)
from src.core_physics.physics_lumen_model import solid_boundary_shells

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")
LOCKED = Path("outputs/clot_ml/locked")


def test_config_defaults_match_the_documented_stack():
    cfg = ClotMlV0Config()
    assert cfg.base_model == "clot_gnn_v5w"
    assert cfg.da_scale_auto == DA_SCALE_AUTO == 123.0
    assert cfg.replace_att == REPLACE_ATT == 0.23
    assert cfg.replace_depth == REPLACE_DEPTH == 3
    assert cfg.replace_scope == REPLACE_SCOPE_ALL_LUMEN
    assert cfg.ap_renewal_scale == 1.0
    assert cfg.washout is True
    assert cfg.ap_residual is None


def test_config_from_manifest_ignores_unknown_keys():
    cfg = ClotMlV0Config.from_manifest({
        "base_model": "clot_gnn_v6w",
        "v0": {"replace_att": 0.16, "not_a_field": 1},
    })
    assert cfg.base_model == "clot_gnn_v6w"
    assert cfg.replace_att == 0.16
    assert cfg.replace_depth == REPLACE_DEPTH
    assert cfg.replace_scope == REPLACE_SCOPE_ALL_LUMEN


def test_replace_scopes_are_explicit_and_closed():
    assert REPLACE_SCOPE_ALL_LUMEN in REPLACE_SCOPES
    assert REPLACE_SCOPE_WOUND_REGION in REPLACE_SCOPES


def test_replace_depth_does_not_union_a_seed():
    """A false-positive seed off-wall must not survive -- union is a measured loss."""
    n = 6
    mat = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    owner = np.array([0, 0, 0, 0, 0, 0])
    shells = [
        np.array([False, True, False, False, False, False]),
        np.array([False, False, True, False, False, False]),
        np.array([False, False, False, True, False, False]),
    ]
    crit = 1.0
    got = replace_depth_mask(mat, shells, owner, crit=crit, att=0.23, depth=3)
    assert not got[0]
    assert got[1]                       # 10 >= 1/0.23 ~ 4.3
    assert not got[2]                   # 10 < 1/0.23**2 ~ 18.9
    assert not got[3]                   # 10 < 1/0.23**3 ~ 82
    assert not got[4] and not got[5]


def test_solid_boundary_shells_keep_shipped_shell1_and_stay_disjoint():
    n = 10
    ei = np.array([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                   [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    solid = np.zeros(n, bool)
    solid[0] = True
    shipped = np.zeros(n, bool)
    shipped[1] = True
    pos = np.stack([np.arange(n), np.zeros(n)], 1).astype(np.float64)
    shells, owner = solid_boundary_shells(
        pos, solid, ei, shell1=shipped, town=np.full(n, -1, np.int64), max_depth=3)
    assert np.array_equal(shells[0], shipped)
    for sh in shells:
        assert not (sh & solid).any()
    for i in range(len(shells)):
        for j in range(i + 1, len(shells)):
            assert not (shells[i] & shells[j]).any()
    assert (owner[~solid] == 0).all()


def test_v0_is_the_base_gnn_on_a_nowound_pack():
    nowound = GRAPH_DIR / "patient012.pt"
    man_path = LOCKED / "clot_ml_v0" / "manifest.json"
    if not nowound.exists() or not man_path.exists():
        pytest.skip("clot_ml_v0 or patient012 not in this checkout")
    import json
    man = json.loads(man_path.read_text())
    if man.get("kind") != "unified_v0" or "v0" not in man:
        pytest.skip("clot_ml_v0 is a stale stub; run scripts/promote_clot_ml_v0.py")
    from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound
    from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_v0
    from src.clot_ml.wound import has_wound

    data = torch.load(nowound, map_location="cpu", weights_only=False)
    assert not has_wound(data)
    T = int(data.y.shape[0])
    times = [0, T // 2, T - 1]
    bundle = load_v0_bundle()
    base = predict_temporal_v4_wound(bundle["base"], data, times, flow="gt")
    got = predict_clot_ml_v0(bundle, data, times, flow="gt")
    assert np.array_equal(base["mask"], got["mask"])
    for ti in times:
        assert np.array_equal(base["series"][int(ti)], got["series"][int(ti)])
