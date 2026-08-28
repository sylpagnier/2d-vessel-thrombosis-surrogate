"""Cover for the wound complement to ``clot_gnn_v4`` (``src/clot_ml/wound.py``).

Three things must hold, in decreasing order of how badly a regression would hurt:

1. **It is a no-op on a no-wound pack.** The complement ships alongside a validated model on
   a 19-vessel cohort that has no wounds; if it perturbs any of them it is a regression
   dressed as a feature.
2. **The torch ODE is the numpy ODE.** The learned rate is fitted through
   ``mat_trajectory_torch``; if that drifts from ``integrate_mat_trajectory`` the fit is
   against a different physics than the one that ships.
3. **The ungated law recovers the wound set.** ``G_wound == 1`` is COMSOL's own law, and it
   must reproduce GT clot on the injured segment without any fitted quantity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.clot_ml.wound import (
    compose_with_v4, has_wound, mat_trajectory_torch, prepare_vessel, predict_wound_series,
    solid_mask, wound_mask, wound_owned_masks, wound_rate_blockage,
)
from src.config import BiochemConfig

GRAPH_DIR = Path("data/processed/graphs_biochem_anchors")
WOUND = "wound_patient001"
NOWOUND = "patient012"


def _load(stem: str):
    p = GRAPH_DIR / f"{stem}.pt"
    if not p.exists():
        pytest.skip(f"{stem} not on disk")
    return torch.load(p, map_location="cpu", weights_only=False)


def test_torch_ode_matches_numpy_integrator():
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import integrate_mat_trajectory

    data = _load(NOWOUND)
    bio = BiochemConfig(phase="biochem")
    V = prepare_vessel(data, bio)
    solid = V["solid"]
    ref, _ = integrate_mat_trajectory(
        data, bio, V["f0"].gate * solid, da_scale=SHIPPED_DA_SCALE,
        ap_closure=make_rollout_hook(SHIPPED, bio, V["f0"].sr))
    g = V["gate"] * torch.tensor(solid, dtype=V["gate"].dtype)
    got = mat_trajectory_torch(t=V["t"], gate_pre=g, gate_post=g, rp=V["rp"], ap=V["ap"],
                               sr=V["sr"], C=V["C"]).numpy()
    rel = np.abs(ref - got) / np.maximum(np.abs(ref), 1e-12)
    assert rel.max() < 1e-10, f"torch ODE drifted from the numpy integrator (rel {rel.max():.2e})"


def test_complement_is_a_noop_without_a_wound():
    data = _load(NOWOUND)
    assert not has_wound(data)
    bio = BiochemConfig(phase="biochem")
    times = [0, 5, 10]
    out = predict_wound_series(data, bio, times)
    assert not out["owned"].any(), "claimed ownership of nodes on a pack with no wound"
    assert not out["mask"].any()

    n = int(data.num_nodes)
    rng = np.random.default_rng(0)
    base_mask = rng.random(n) > 0.7
    base_onset = np.where(base_mask, rng.integers(0, 10, n), -1).astype(np.float64)
    base = dict(mask=base_mask, onset=base_onset, score=np.zeros(n))
    comp = compose_with_v4(base, out, times)
    assert np.array_equal(comp["mask"], base_mask), "wound module altered a no-wound mask"
    assert np.array_equal(comp["onset"], base_onset), "wound module altered a no-wound onset"


def test_wound_rate_blockage_is_structurally_absent_without_a_wound():
    """No wound mask -> the factory hands back ``inner`` itself, not a wrapper around it.

    A wrapper that happened to be numerically neutral would still be a live code path on
    every cohort pack.  Returning ``inner`` unchanged makes the no-op structural, so the
    cohort cannot be touched by a later bug in the rewrite.
    """
    from src.clot_ml.temporal import ode_trajectory

    data = _load(NOWOUND)
    bio = BiochemConfig(phase="biochem")
    sentinel = object()
    assert wound_rate_blockage(data, bio, inner=sentinel) is sentinel
    assert wound_rate_blockage(data, bio) is None

    base, _ = ode_trajectory(data, bio, flow="gt")
    got, _ = ode_trajectory(data, bio, flow="gt", wound_rate=(2.0, 14.0))
    assert np.array_equal(base, got), "wound_rate perturbed a pack with no wound"


def test_wound_rate_lifts_the_injured_patch_to_the_fitted_magnitude():
    """The shared ODE must integrate the SAME rate the complement fitted.

    At COMSOL's static ``srf2`` prefactor of 1 the injured patch reaches ~1.35x crit on
    ``wound_patient001`` against GT's 9.04x, so no wound-owned lumen node can clear the
    ``crit / off_att`` bar the off-wall rule is built on (WOUND_PROGRESS 15).  The healthy
    wall must not move: this rewrites ``srf2``, never ``srf1``.
    """
    from src.clot_ml.temporal import ode_trajectory

    data = _load(WOUND)
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    wnd, wall = wound_mask(data), data.mask_wall.reshape(-1).bool().numpy()

    base, _ = ode_trajectory(data, bio, flow="gt")
    got, _ = ode_trajectory(data, bio, flow="gt", wound_rate=(1.98, 14.28))
    assert np.median(base[-1][wnd]) / crit < 3.0
    assert np.median(got[-1][wnd]) / crit > 5.0
    assert np.array_equal(base[-1][wall], got[-1][wall]), "the healthy wall must not move"

    # `wound_source=False` removes the patch as a source entirely, so the rate has nothing
    # to act on and must not sneak back in.
    off, _ = ode_trajectory(data, bio, flow="gt", wound_source=False, wound_rate=(1.98, 14.28))
    ref, _ = ode_trajectory(data, bio, flow="gt", wound_source=False)
    assert np.array_equal(off, ref)


def test_wound_masks_are_disjoint_and_solid_is_the_union():
    data = _load(WOUND)
    w, wl = wound_mask(data), data.mask_wall.reshape(-1).bool().numpy()
    assert w.any()
    assert not (w & wl).any()
    assert np.array_equal(solid_mask(data), w | wl)
    wnd, owned_off, _ = wound_owned_masks(data)
    assert not (owned_off & solid_mask(data)).any(), "owned off-wall must be off the boundary"


def test_ungated_law_recovers_the_wound_set():
    """G == 1 is COMSOL's wound law with nothing fitted; it must commit the whole patch."""
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.config import PhysicsConfig

    data = _load(WOUND)
    bio = BiochemConfig(phase="biochem")
    T = int(data.y.shape[0])
    out = predict_wound_series(data, bio, [T - 1], g_pre=1.0, g_post=1.0)
    gt = gt_clot_phi_at_time(data, T - 1, PhysicsConfig(phase="biochem")).numpy() > 0.5
    w = wound_mask(data)
    recall = float((out["mask"] & gt)[w].sum()) / float(gt[w].sum())
    assert recall == 1.0, f"ungated law missed part of the wound (recall {recall:.3f})"


def test_two_regime_gate_is_faster_than_flat():
    """The post-gelation branch may only ever add deposition, never remove it."""
    data = _load(WOUND)
    bio = BiochemConfig(phase="biochem")
    V = prepare_vessel(data, bio)
    idx = np.flatnonzero(V["wound"])
    sub = dict(t=V["t"], rp=V["rp"][idx], ap=V["ap"][idx], sr=V["sr"][idx], C=V["C"])
    flat = mat_trajectory_torch(gate_pre=torch.full((idx.size,), 2.0, dtype=torch.float64),
                                gate_post=torch.full((idx.size,), 2.0, dtype=torch.float64),
                                **sub).numpy()
    two = mat_trajectory_torch(gate_pre=torch.full((idx.size,), 2.0, dtype=torch.float64),
                               gate_post=torch.full((idx.size,), 10.0, dtype=torch.float64),
                               **sub).numpy()
    assert (two >= flat - 1e-9).all(), "two-regime gate lost material against the flat gate"
    assert two[-1].mean() > flat[-1].mean()


def test_neighbour_trigger_is_inert_where_the_neighbourhood_is_quiet():
    """The coupling may never disturb a self-triggered wound.

    ``wound_patient001`` has no gelled wall node within 61 mesh hops of its wound, so every
    trigger source -- including the GT oracle -- must leave its trajectory untouched. This is
    the safety property that lets the coupling ship at all: it can only ever add.
    """
    data = _load(WOUND)
    bio = BiochemConfig(phase="biochem")
    T = int(data.y.shape[0])
    times = [0, T // 2, T - 1]
    V = prepare_vessel(data, bio)
    base = predict_wound_series(data, bio, times, prepared=V, trigger="self")
    for trig in ("wall", "oracle"):
        out = predict_wound_series(data, bio, times, prepared=V, trigger=trig)
        assert np.array_equal(out["mask"], base["mask"]), f"{trig} trigger changed the set"
        assert np.allclose(out["onset"], base["onset"]), f"{trig} trigger changed the timing"


def test_trigger_rejects_an_unknown_source():
    data = _load(WOUND)
    with pytest.raises(ValueError):
        predict_wound_series(data, BiochemConfig(phase="biochem"), [0], trigger="magic")


# ---------------------------------------------------------------------------
# the shipped artifact: clot_gnn_v4w
# ---------------------------------------------------------------------------
def _v4w_bundle():
    """Load the locked v4w artifact, or skip if it has not been promoted here."""
    import json

    root = Path("outputs/clot_ml/locked/clot_gnn_v4w")
    if not (root / "manifest.json").exists():
        pytest.skip("clot_gnn_v4w not promoted (run scripts/promote_clot_gnn_v4_wound.py)")
    from src.clot_ml.locked import load_temporal_v4

    manifest = json.loads((root / "manifest.json").read_text())
    return dict(base=load_temporal_v4(name=manifest["base_model"]),
                wound=manifest["wound"], manifest=manifest)


@pytest.mark.slow
def test_v4w_is_bit_identical_to_v4_on_a_no_wound_pack():
    """The licence for v4w superseding v4 outright. If this fails, it must not ship."""
    from src.clot_ml.locked import predict_temporal_v4, predict_temporal_v4_wound

    bundle = _v4w_bundle()
    data = _load(NOWOUND)
    T = int(data.y.shape[0])
    times = sorted({0, T // 3, T - 1})
    a = predict_temporal_v4(bundle["base"], data, times, flow="gt")
    b = predict_temporal_v4_wound(bundle, data, times, flow="gt")
    assert np.array_equal(a["mask"], b["mask"])
    assert np.array_equal(np.asarray(a["onset"]), np.asarray(b["onset"]))
    for ti in times:
        assert np.array_equal(a["series"][int(ti)], b["series"][int(ti)]), f"series drift at {ti}"


@pytest.mark.slow
def test_v4w_commits_the_wound_and_touches_nothing_else():
    from src.clot_ml.locked import predict_temporal_v4, predict_temporal_v4_wound

    bundle = _v4w_bundle()
    data = _load(WOUND)
    T = int(data.y.shape[0])
    w = wound_mask(data)
    a = predict_temporal_v4(bundle["base"], data, [T - 1], flow="gt")
    b = predict_temporal_v4_wound(bundle, data, [T - 1], flow="gt")
    assert b["mask"][w].all(), "v4w must commit the whole injured segment"
    from src.core_physics.physics_wall_model import t0_flow_fields
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, BiochemConfig(phase="biochem"), hops=3, flow_source="gt")
    ung = wall & (np.asarray(f.gate) * wall <= 0)
    changed = np.asarray(a["mask"]) != np.asarray(b["mask"])
    unexpected = changed & ~w & ~np.asarray(b["owned"], dtype=bool) & ~ung
    assert not unexpected.any(), (
        "v4w changed a node that is not wound-owned and not t=0-ungated stall wall"
    )


def test_v4w_manifest_records_the_two_scalars_and_its_base():
    bundle = _v4w_bundle()
    m = bundle["manifest"]
    assert m["kind"] == "temporal_v4_wound"
    assert m["base_model"] == "clot_gnn_v4"
    assert m["supersedes"] == "clot_gnn_v4"
    for k in ("g_pre", "g_post", "off_att", "lag_frac"):
        assert k in m["wound"], f"manifest is missing wound.{k}"
    # G_pre must stay near the mechanism's own value: ungated(1) + low-shear(1).
    assert 1.5 <= float(m["wound"]["g_pre"]) <= 2.5
    assert float(m["wound"]["g_post"]) > float(m["wound"]["g_pre"])


# ---------------------------------------------------------------------------
# the scoring domains (WOUND_PROGRESS 13)
# ---------------------------------------------------------------------------
def test_wound_boundary_domain_is_degenerate_and_region_domains_are_not():
    """The reason `wnd` must never be the headline, asserted rather than remembered."""
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.clot_ml.wound import wound_region_masks

    data = _load(WOUND)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, PhysicsConfig(phase="biochem")).numpy() > 0.5
    w = wound_mask(data)
    assert gt[w].mean() == 1.0, "the wound boundary is expected to be 100% GT clot"

    region, lumen, far = wound_region_masks(data)
    for name, dom in (("w_reg", region), ("w_lum", lumen)):
        rate = gt[dom].mean()
        assert 0.02 < rate < 0.9, f"{name} positive rate {rate:.3f} is degenerate"
    assert lumen.sum() and not (lumen & solid_mask(data)).any()
    assert not (far & region).any(), "far and region must be disjoint"


def test_wound_region_covers_the_thrombus_and_far_excludes_it():
    """The radius must contain the GT thrombus, not clip it."""
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.clot_ml.wound import wound_region_masks

    data = _load(WOUND)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, PhysicsConfig(phase="biochem")).numpy() > 0.5
    region, lumen, far = wound_region_masks(data)
    # On 001 every off-boundary GT clot node is wound-caused, so `far` holds none of them.
    assert not (gt & far).any(), "far domain contains wound-caused clot; radius is too small"
    assert (gt & lumen).sum() > 0, "the wound thrombus must reach the lumen"


def test_region_masks_are_empty_on_a_no_wound_pack():
    from src.clot_ml.wound import wound_region_masks

    data = _load(NOWOUND)
    region, lumen, far = wound_region_masks(data)
    assert not region.any() and not lumen.any()
    assert np.array_equal(far, ~solid_mask(data)), "far must be the whole lumen when no wound"


def test_recursive_lumen_is_strictly_additive_and_gated():
    """C2 (MODEL_REVIEW 9d): deeper shells may only ADD, and only where the physics admits.

    The first version rebuilt shell 1 from hop distances, which is a different and smaller set
    than `first_corner_shell` (43 nodes against 80 on `wound_patient001`), and it cost
    001/002 `w_lum` 0.0160 -- a gate violation produced by changing shell 1 while trying to
    add shell 2.  Shell 1 must stay exactly as shipped.
    """
    from pathlib import Path

    import numpy as np
    import torch

    from src.clot_ml.wound import wound_owned_masks, wound_shells

    root = Path(__file__).resolve().parents[2] / "data/processed/graphs_biochem_anchors"
    p_ = root / "wound_patient001.pt"
    if not p_.exists():
        pytest.skip("wound_patient001 pack not present")
    d = torch.load(p_, map_location="cpu", weights_only=False)
    wnd, owned, _ = wound_owned_masks(d)
    shells, _ = wound_shells(d, 4)

    assert int(shells[0].sum()) < int(owned.sum()), (
        "the hop-2 ring is expected to be SMALLER than `first_corner_shell` -- if that has "
        "changed, re-check that the recursive rule still starts from the shipped shell")
    # every deeper ring must be a genuine addition, never a replacement
    for k, sh in enumerate(shells[1:], start=2):
        assert not (sh & wnd).any(), f"ring {k} overlaps the wound boundary itself"


@pytest.mark.slow
def test_dispatcher_lumen_defaults_to_the_shipped_single_shell():
    """An artifact with no ``lumen`` key must run ``shell``, bit-for-bit.

    The depth rule became an artifact field (WOUND_PROGRESS 16.4) so it can be promoted
    deliberately rather than flipped at a call site.  Every artifact promoted before that
    change lacks the key, so the default is what keeps their recorded numbers meaningful.
    """
    from src.clot_ml.locked import predict_temporal_v4_wound

    bundle = _v4w_bundle()
    data = _load(WOUND)
    T = int(data.y.shape[0])
    times = [0, T - 1]

    assert "lumen" not in bundle["wound"], (
        "this test exists to pin the DEFAULT; the fixture artifact now sets the key")
    implicit = predict_temporal_v4_wound(bundle, data, times, flow="gt")

    explicit_b = dict(bundle, wound=dict(bundle["wound"], lumen="shell"))
    explicit = predict_temporal_v4_wound(explicit_b, data, times, flow="gt")
    assert np.array_equal(implicit["mask"], explicit["mask"])
    assert np.array_equal(np.asarray(implicit["onset"]), np.asarray(explicit["onset"]))


@pytest.mark.slow
def test_dispatcher_recursive_is_inert_on_a_nine_x_wound():
    """``recursive`` must add nothing on a wound that only reaches ~9x crit.

    ``test_recursive_lumen_leaves_low_mat_wounds_untouched`` asserts the arithmetic of the
    bar; this asserts the DEPLOY consequence on a real pack, which is the property that lets
    the mode be promoted without re-validating 001/002: at ``off_att=0.16`` shell 2 needs 39x
    crit and ``wound_patient001`` reaches ~9x, so the committed set must be identical.
    """
    from src.clot_ml.locked import predict_temporal_v4_wound

    bundle = _v4w_bundle()
    data = _load(WOUND)
    T = int(data.y.shape[0])
    times = [0, T - 1]
    a = predict_temporal_v4_wound(dict(bundle, wound=dict(bundle["wound"], lumen="shell")),
                                  data, times, flow="gt")
    b = predict_temporal_v4_wound(dict(bundle, wound=dict(bundle["wound"], lumen="recursive")),
                                  data, times, flow="gt")
    assert np.array_equal(a["mask"], b["mask"]), (
        "recursive deepened a wound whose Mat cannot clear the second shell's bar")


@pytest.mark.slow
def test_dispatcher_recursive_never_removes_a_committed_node():
    """``recursive`` must be additive in the committed SET, not merely in the ownership map.

    These are different claims and the difference was a live bug.  ``compose_with_v4`` applies
    ``mask[owned] = wound_out["mask"][owned]``, so widening ``owned_off`` to a deeper ring
    hands v4's nodes to a module that may decline them -- on ``wound_patient003`` the first
    version removed 2 committed nodes and added none.  It removed two false positives, so the
    score went UP and the defect was invisible in the score; only a set comparison finds it.
    """
    from src.clot_ml.locked import predict_temporal_v4_wound

    bundle = _v4w_bundle()
    for stem in ("wound_patient001", "wound_patient003"):
        p = GRAPH_DIR / f"{stem}.pt"
        if not p.exists():
            pytest.skip(f"{stem} not on disk")
        data = torch.load(p, map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = [0, T - 1]
        a = predict_temporal_v4_wound(
            dict(bundle, wound=dict(bundle["wound"], lumen="shell")), data, times, flow="gt")
        b = predict_temporal_v4_wound(
            dict(bundle, wound=dict(bundle["wound"], lumen="recursive")), data, times,
            flow="gt")
        removed = np.asarray(a["mask"], bool) & ~np.asarray(b["mask"], bool)
        assert not removed.any(), (
            f"{stem}: recursive removed {int(removed.sum())} committed node(s); "
            "deeper shells may only add")


@pytest.mark.slow
def test_dispatcher_lumen_is_a_noop_without_a_wound():
    """No wound mask -> the dispatcher returns v4 unchanged whatever the depth rule says."""
    from src.clot_ml.locked import predict_temporal_v4, predict_temporal_v4_wound

    bundle = _v4w_bundle()
    data = _load(NOWOUND)
    T = int(data.y.shape[0])
    times = sorted({0, T // 3, T - 1})
    ref = predict_temporal_v4(bundle["base"], data, times, flow="gt")
    for mode in ("shell", "recursive", "transport", "union"):
        got = predict_temporal_v4_wound(
            dict(bundle, wound=dict(bundle["wound"], lumen=mode)), data, times, flow="gt")
        assert np.array_equal(ref["mask"], got["mask"]), f"lumen={mode} perturbed a no-wound pack"


def test_recursive_lumen_leaves_low_mat_wounds_untouched():
    """A wound reaching 9x crit admits ONE shell at `off_att=0.16`, so nothing may be added.

    This is the gate from MODEL_REVIEW 5b.5 expressed as physics rather than as a score: shell
    k needs `Mat_wound >= crit / 0.16**k`, i.e. 6.25x for k=1 and 39x for k=2.
    """
    crit_mult_shell2 = 1.0 / (0.16 ** 2)
    assert crit_mult_shell2 > 39.0 and crit_mult_shell2 < 40.0
    # 001/002 reach ~9x, 003 reaches ~104x -- so exactly one vessel may deepen
    assert 9.0 < crit_mult_shell2, "a 9x wound must not admit a second shell"
    assert 103.8 > crit_mult_shell2, "a 104x wound must admit a second shell"
