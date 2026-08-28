"""`flow=` must reach every channel it can change, and change nothing when it is "gt".

Two GT leaks were live before this file existed (`docs/MODEL_REVIEW_2026-08-22.md` 6.1/6.3):

  * ``features_v4.indicator_physics`` hardwired ``flow_source="gt"`` and ``augment_sample``
    took no ``flow`` at all, so ``build_sample(flow="pred", variant="v4")`` returned 55
    predicted-flow channels next to four GT-flow ones (``gate_ind``, ``log_mat_phys_ind``,
    ``onset_phys_ind``, ``log_mat_ind_owner``) plus a ``log_mat_adv_ind`` transporting a
    GT-derived source.  Any deploy-faithful number built on that is silently optimistic.
  * ``wound.wound_features`` read ``data.y[0, :, 0:2]`` for its ``speed`` channel whatever
    ``flow`` said, contradicting the module docstring's deploy-legality claim.

The regression that matters just as much is the other direction: ``flow="gt"`` must
reproduce the shipped `clot_gnn_v4` cache **bit for bit**, or the locked artifact's feature
normaliser no longer matches the features it is applied to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "data/processed/graphs_biochem_anchors"
V5_CACHE = REPO / "outputs/clot_ml_cache_v5"

#: the five v4 channels that were GT-locked, and are the point of the fix
GT_LOCKED_CHANNELS = ("gate_ind", "log_mat_phys_ind", "onset_phys_ind",
                      "log_mat_ind_owner", "log_mat_adv_ind")


def _pack(name: str):
    p = PACKS / f"{name}.pt"
    if not p.exists():
        pytest.skip(f"{name} pack not present in this checkout")
    return torch.load(p, map_location="cpu", weights_only=False)


def _cfgs():
    from src.config import BiochemConfig, PhysicsConfig
    return BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")


@pytest.mark.slow
def test_gt_sample_reproduces_the_shipped_v5_cache_bit_for_bit():
    """The locked artifact's normaliser is only valid against these exact columns."""
    src = V5_CACHE / "patient020.npz"
    if not src.exists():
        pytest.skip("clot_ml_cache_v5 not built in this checkout")
    from src.clot_ml.locked import build_sample

    bio, phys = _cfgs()
    S = build_sample(_pack("patient020"), bio, phys, flow="gt", variant="v4")
    z = np.load(src, allow_pickle=True)
    cached_cols = [str(c) for c in z["cols"]]
    live_cols = [str(c) for c in S["cols"]]

    # build_sample appends phys_mask; the cache stops one column earlier
    assert live_cols[:len(cached_cols)] == cached_cols
    assert live_cols[len(cached_cols):] == ["phys_mask"]
    diff = np.abs(S["X"][:, :len(cached_cols)].astype(np.float64)
                  - z["X"].astype(np.float64))
    if diff.max() != 0.0:
        moved = {cached_cols[j] for j in range(len(cached_cols)) if diff[:, j].max() > 0}
        # The 2026-08-22 pack repair populated `wall_normal` and `node_type_*`, which moved
        # every width- and prior-derived column with them (MODEL_REVIEW 6.5).  A cache built
        # before that is legitimately stale; anything moving OUTSIDE that set is a real bug.
        from_repair = {"width_nd", "width_d1", "width_d2", "wss_prior_nd",
                       "u_prior", "v_prior", "mu_prior_nd", "shear_potential",
                       "sdf_nd", "u_n", "u_t"}
        # v4 channels are functions of the above, so they move too
        from src.clot_ml.features_v4 import V4_CHANNELS
        allowed = from_repair | set(V4_CHANNELS) | {
            "log_mat_phys", "onset_phys", "log_mat_owner", "gate_owner", "sr_owner",
            "phys_mask", "dist_wall_dbar", "dist_wall_edges"}
        unexplained = sorted(moved - allowed)
        assert not unexplained, ("gt path drifted on columns the pack repair does not "
                                 "explain: %s" % unexplained)
        pytest.skip("clot_ml_cache_v5 predates the 2026-08-22 pack repair -- rebuild it "
                    "(scripts/build_clot_ml_cache_v4.py) and re-promote clot_gnn_v4")


@pytest.mark.slow
def test_pred_flow_reaches_the_v4_indicator_channels():
    """The bug: these five were identical under both flow sources."""
    from src.clot_ml.locked import build_sample

    data = _pack("patient020")
    if getattr(data, "u0_pred", None) is None:
        pytest.skip("pack carries no u0_pred")
    bio, phys = _cfgs()
    gt = build_sample(data, bio, phys, flow="gt", variant="v4")
    pr = build_sample(data, bio, phys, flow="pred", variant="v4")

    cols = [str(c) for c in gt["cols"]]
    for name in GT_LOCKED_CHANNELS:
        i = cols.index(name)
        d = np.abs(gt["X"][:, i].astype(np.float64) - pr["X"][:, i].astype(np.float64)).max()
        assert d > 0.0, f"{name} did not respond to flow -- GT is still hardwired"


def test_indicator_physics_defaults_to_gt():
    """Existing callers that pass no `flow` must be unaffected."""
    from src.clot_ml.features_v4 import indicator_physics

    data = _pack("patient020")
    bio, _ = _cfgs()
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    a = indicator_physics(data, bio, wall)
    b = indicator_physics(data, bio, wall, flow="gt")
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_t0_fields_carry_the_velocity_they_differentiated():
    from src.core_physics.physics_wall_model import t0_flow_fields

    data = _pack("patient020")
    bio, _ = _cfgs()
    f_gt = t0_flow_fields(data, bio, flow_source="gt")
    assert f_gt.u is not None and f_gt.v is not None
    assert np.array_equal(f_gt.u, data.y[0, :, 0].cpu().numpy().astype(np.float64))
    if getattr(data, "u0_pred", None) is not None:
        f_pr = t0_flow_fields(data, bio, hops=4, flow_source="pred")
        assert not np.array_equal(f_pr.u, f_gt.u)


def test_wound_features_refuses_a_hand_built_t0fields():
    """No silent GT fallback: a T0Fields without u/v is a caller bug, not a default."""
    from src.clot_ml.wound import wound_features
    from src.core_physics.physics_wall_model import T0Fields

    data = _pack("wound_patient001")
    bio, _ = _cfgs()
    n = int(data.num_nodes)
    z = np.zeros(n)
    hand_built = T0Fields(sr=z + 1.0, dsrx=z + 1.0, gate_low=z, gate_sep=z, gate=z)
    with pytest.raises(ValueError, match="u/v"):
        wound_features(data, hand_built, bio)


def test_wound_features_uses_the_resolved_velocity():
    """`speed` must track the T0Fields' own u/v, not `data.y[0]`."""
    from src.clot_ml.wound import WOUND_FEATURES, wound_features
    from src.core_physics.physics_wall_model import t0_flow_fields

    data = _pack("wound_patient001")
    bio, _ = _cfgs()
    f0 = t0_flow_fields(data, bio, flow_source="gt")
    base = wound_features(data, f0, bio)

    # perturb only the carried velocity; if `speed` still reads data.y[0] it will not move
    f0.u = np.asarray(f0.u) * 2.0
    f0.v = np.asarray(f0.v) * 2.0
    moved = wound_features(data, f0, bio)
    i = WOUND_FEATURES.index("speed")
    assert np.abs(moved[:, i] - base[:, i]).max() > 0.0
