"""Pins for the unified ``clot_ml_0`` stack.

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
    # DEFAULT CHANGED 2026-09-03, `all_lumen` -> `wound_region`.  Measured leave-one-vessel-out
    # over the six-vessel wound cohort on deploy flow (`scripts/eval_replace_scope.py`): the two
    # scopes are identical to four decimals on wall, wound region and wound lumen, and the far
    # field reads 0.0817 against 0.2448 -- `wound_patient004/005/006` go from exactly 0.0000 to
    # 0.356 / 0.179 / 0.164.  All six folds pick `wound_region`.  Artifacts promoted before that
    # date pin their own scope in their manifest; `clot_ml_v0_chem_legacy`, the one that never
    # recorded one, was pinned to `all_lumen` explicitly so this default could move.
    assert cfg.replace_scope == REPLACE_SCOPE_WOUND_REGION
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
    # a manifest that names no scope inherits the CURRENT default -- see the note above
    assert cfg.replace_scope == REPLACE_SCOPE_WOUND_REGION


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


def test_att_beta_defaults_off_so_the_shipped_rule_is_unchanged():
    """`att_beta = 0` must reproduce the cohort constant exactly.

    The shear-modulated attenuation is measured (DEPLOYCLOT 18) but not promoted; what has to
    hold is that an artifact which never mentions it behaves as it always did.
    """
    from src.clot_ml.v0 import shear_attenuation

    assert ClotMlV0Config().att_beta == 0.0
    assert ClotMlV0Config.from_manifest({"v0": {}}).att_beta == 0.0
    sr = np.array([1.0, 10.0, 100.0])
    wall = np.array([True, True, False])
    assert np.array_equal(shear_attenuation(sr, wall, 0.23, 0.0), np.full(3, 0.23))


def test_shear_attenuation_falls_with_shear_and_is_clipped():
    """Higher local shear must mean a SHORTER reach: a thinner boundary layer carries less
    surface material into the lumen, which is the whole physical content of the knob."""
    from src.clot_ml.v0 import ATT_CLIP, shear_attenuation

    sr = np.array([1.0, 5.5, 10.0, 1e6])
    wall = np.array([True, False, True, False])   # sr_ref = median(1, 10) = 5.5
    a = shear_attenuation(sr, wall, 0.23, 1.0)
    assert a[0] > a[1] > a[2] > a[3]
    assert a[1] == pytest.approx(0.23)            # at the reference, beta is inert
    assert a[3] == pytest.approx(ATT_CLIP[0])     # clipped, never zero
    assert (a >= ATT_CLIP[0]).all() and (a <= ATT_CLIP[1]).all()


def test_per_node_attenuation_reduces_to_the_scalar_rule():
    """A constant per-node ``att`` must be bit-identical to passing the scalar.

    The per-node form exists so the depth rule can carry a LOCAL attenuation -- `Mat` is made
    at the surface and has to survive convection to reach depth, which is a property of the
    flow and not a cohort constant (`scripts/diag_wound_offwall_attenuation.py`).  Nothing
    ships with a non-constant field yet, so what must be pinned is that turning the knob off
    leaves the shipped rule untouched.
    """
    mat = np.array([10.0, 0.0, 0.0, 0.0])
    owner = np.zeros(4, dtype=np.int64)
    shells = [np.array([False, True, False, False]),
              np.array([False, False, True, False])]
    scalar = replace_depth_mask(mat, shells, owner, crit=1.0, att=0.23, depth=2)
    vector = replace_depth_mask(mat, shells, owner, crit=1.0,
                                att=np.full(4, 0.23), depth=2)
    assert np.array_equal(scalar, vector)


def test_per_node_attenuation_is_read_at_the_committed_node():
    """The bar belongs to the node being judged, not to its wall owner.

    A shell-1 node whose own attenuation is high commits on a wall ``Mat`` its neighbour with
    a low attenuation would not -- that is the whole content of making it local.
    """
    mat = np.array([5.0, 0.0, 0.0])
    owner = np.zeros(3, dtype=np.int64)
    shells = [np.array([False, True, True])]
    att = np.array([0.5, 0.5, 0.1])          # node 1 needs 2x crit, node 2 needs 10x
    got = replace_depth_mask(mat, shells, owner, crit=1.0, att=att, depth=1)
    assert got[1] and not got[2]


def test_per_node_attenuation_rejects_a_wrong_length_field():
    with pytest.raises(ValueError):
        replace_depth_mask(np.zeros(4), [np.ones(4, bool)], np.zeros(4, np.int64),
                           crit=1.0, att=np.full(3, 0.23), depth=1)


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


def test_the_two_clot_metrics_are_not_interchangeable():
    """`domain_score` and `SeverityScorer` are DIFFERENT numbers for the same prediction.

    Both are legitimate and both ship: the strictly-nested CV tables report severity, and
    `scripts/eval_clot_ml_0.py` -- and therefore the SEALED read -- reports the deploy
    metric.  Off-wall they run 0.19-0.22 apart on this cohort, and quoting one against the
    other manufactured a "SEALED off-wall gap" of 0.22 that does not exist
    (docs/DEPLOYCLOT.md 22).  This pins that they are distinct, so nobody reads a difference
    between them as a difference between vessels.
    """
    import torch as _t

    from src.clot_ml.evaluate import domain_score
    from src.clot_ml.severity_metric import DEFAULT, SeverityScorer

    n = 12
    ei = _t.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                    [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    wall = np.zeros(n, bool)
    wall[:4] = True
    dom = ~wall
    gt = np.zeros(n, bool)
    gt[5:9] = True
    pred = np.zeros(n, bool)
    pred[5:7] = True                      # half the GT, no false positives
    d = domain_score(pred, gt, ei, dom, wall)
    sv = SeverityScorer(ei.numpy(), gt, n, DEFAULT).score(pred & dom, dom)
    assert 0.0 <= d <= 1.0 and 0.0 <= sv <= 1.0
    assert d != pytest.approx(sv, abs=1e-6), (
        "the deploy and severity metrics returned the same value -- if they have genuinely "
        "been unified, delete this test and the both-metric reporting it guards")


def test_nowound_scoring_reports_both_metrics():
    """`eval_clot_ml_0._score_nowound` must emit both, so a read cannot carry only one.

    The SEALED read carried the deploy metric alone, which is what allowed it to be quoted
    against a cross-validated severity number for two sessions running.
    """
    import torch as _t

    from scripts.eval_clot_ml_0 import _score_nowound

    n = 12
    ei = _t.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                    [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]])
    wall = np.zeros(n, bool)
    wall[:4] = True
    solid = wall.copy()
    gt = np.zeros(n, bool)
    gt[5:9] = True
    pred = np.zeros(n, bool)
    pred[5:7] = True
    S = {"wall": wall, "solid": solid, "edge_index": ei.numpy()}
    got = _score_nowound(pred, gt, ei, S)
    assert {"wall", "off", "wall_sev", "off_sev"} <= set(got)
    # the off-wall pair must actually be two different measurements of the same mask
    assert got["off"] == got["off"]
    assert got["off_sev"] == got["off_sev"]
    assert got["off"] != pytest.approx(got["off_sev"], abs=1e-6)


def test_a_production_artifact_is_refused_by_the_scorer():
    """The PRODUCTION family trains on the whole corpus, SEALED included, so no number
    computed on any vessel estimates anything.  The two families are identical in shape --
    same composition, same code path -- so the boundary needs a machine-checkable marker,
    not a naming convention (docs/PUBLICATION_PLAN.md 12).

    `metrics_invalid` is stamped on the temporal_v4 manifest at the BOTTOM of the chain, so
    the walker has to find it through the wound and unified_v0 wrappers above it.
    """
    from src.clot_ml.v0 import metrics_invalid_reason

    base = {"manifest": {"name": "X", "metrics_invalid": True,
                         "metrics_invalid_reason": "trained with --include-sealed"}}
    wound = {"base": base, "manifest": {"name": "Xw"}}
    v0 = {"base": wound, "manifest": {"name": "X0"}}
    assert metrics_invalid_reason(v0) == "trained with --include-sealed"
    assert metrics_invalid_reason(wound) == "trained with --include-sealed"
    # a validated chain, and a chain with no manifests at all, are both scoreable
    ok = {"base": {"base": {"manifest": {"name": "V", "metrics_invalid": False}},
                   "manifest": {"name": "Vw"}}, "manifest": {"name": "V0"}}
    assert metrics_invalid_reason(ok) is None
    assert metrics_invalid_reason({}) is None


def test_the_shipped_artifact_is_scoreable():
    """Whatever the pointer names must NOT be a production build -- the deployed default is
    the validated one, and the production artifact is fetched by explicit name only."""
    from src.clot_ml.v0 import LOCKED, POINTER, load_v0_bundle, metrics_invalid_reason

    if not POINTER.exists():
        pytest.skip("no locked pointer in this checkout")
    import json

    name = json.loads(POINTER.read_text()).get("name", "")
    if not (LOCKED / name).is_dir():
        pytest.skip("pointed artifact absent")
    assert metrics_invalid_reason(load_v0_bundle()) is None


def test_the_default_artifact_follows_the_locked_pointer():
    """Promoting and repointing must change what a caller with no explicit name loads.

    It did not, until 2026-09-03: `clot_ml_0` is not a directory, so `_locked_root` fell
    through to the `clot_ml_v0` legacy fallback and the pointer was decorative for every
    default caller -- including `CustomerDeployPipeline`, which asks for `clot_ml_0` by a
    module constant and was being served an artifact built on `clot_gnn_v6`.
    """
    import json

    from src.clot_ml.v0 import (
        KIND, LOCKED, POINTER, pointer_v0_name, resolve_clot_ml_name,
    )

    if not POINTER.exists():
        pytest.skip("no locked pointer in this checkout")
    ptr = json.loads(POINTER.read_text())
    if ptr.get("kind") != KIND:
        pytest.skip("pointer names a non-unified_v0 generation")
    name = ptr["name"]
    if not (LOCKED / name).is_dir():
        pytest.skip("pointed artifact is not present in this checkout")
    assert pointer_v0_name() == name
    assert resolve_clot_ml_name() == name
    assert resolve_clot_ml_name("clot_ml_0") == name


def test_an_explicit_artifact_id_is_honoured_verbatim():
    """Pinned comparisons against a NAMED past generation must not follow the pointer."""
    from src.clot_ml.v0 import resolve_clot_ml_name

    for n in ("DeployClot_0", "clot_ml_v0_chem_legacy", "some_future_artifact"):
        assert resolve_clot_ml_name(n) == n


def test_resolution_survives_a_missing_pointer(tmp_path, monkeypatch):
    """An ordinary checkout with no promoted artifact keeps the compiled-in default."""
    from src.clot_ml import v0 as v0mod

    monkeypatch.setattr(v0mod, "POINTER", tmp_path / "absent.json")
    assert v0mod.pointer_v0_name() is None
    assert v0mod.resolve_clot_ml_name() == v0mod.DEFAULT_NAME

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(v0mod, "POINTER", bad)
    assert v0mod.pointer_v0_name() is None

    other = tmp_path / "other.json"
    other.write_text('{"kind": "temporal_v4", "name": "DeployClot"}', encoding="utf-8")
    monkeypatch.setattr(v0mod, "POINTER", other)
    assert v0mod.pointer_v0_name() is None


def test_v0_is_the_base_gnn_on_a_nowound_pack():
    nowound = GRAPH_DIR / "patient012.pt"
    # Resolve the POINTER, not a hardcoded artifact name.  This gate used to look for
    # `clot_ml_0`/`clot_ml_v0`, so once the shipped artifact was renamed (DeployClot_0,
    # then DeployClot_1) the test SKIPPED -- while `load_v0_bundle()` below happily loaded
    # the pointer's artifact.  A skip that looks like a pass is how a no-op guarantee stops
    # being checked, and this one is quoted on the pointer itself as the reason the SEALED
    # read carries over.
    import json

    from src.clot_ml.v0 import resolve_clot_ml_name

    try:
        man_path = LOCKED / resolve_clot_ml_name() / "manifest.json"
    except Exception:  # noqa: BLE001
        man_path = LOCKED / "clot_ml_0" / "manifest.json"
    if not man_path.exists():
        man_path = LOCKED / "clot_ml_v0" / "manifest.json"
    if not nowound.exists() or not man_path.exists():
        pytest.skip("no unified_v0 artifact or patient012 in this checkout")
    man = json.loads(man_path.read_text())
    if man.get("kind") != "unified_v0" or "v0" not in man:
        pytest.skip("clot_ml_0 is a stale stub; run scripts/promote_clot_ml_0.py")
    from src.clot_ml.locked import predict_temporal_v4_wound
    from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0
    from src.clot_ml.wound import has_wound

    data = torch.load(nowound, map_location="cpu", weights_only=False)
    assert not has_wound(data)
    T = int(data.y.shape[0])
    times = [0, T // 2, T - 1]
    bundle = load_v0_bundle()
    base = predict_temporal_v4_wound(bundle["base"], data, times, flow="gt")
    got = predict_clot_ml_0(bundle, data, times, flow="gt")
    assert np.array_equal(base["mask"], got["mask"])
    for ti in times:
        assert np.array_equal(base["series"][int(ti)], got["series"][int(ti)])
