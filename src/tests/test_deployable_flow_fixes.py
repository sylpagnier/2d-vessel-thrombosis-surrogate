"""The three 2026-08-23 fixes that stand between GT t=0 flow and RGP-DEQ.

Each one is a single line of behaviour with a measured justification, and each one is easy
to undo by accident, so each gets a pin here:

  1. ``clot_ml.gnn.edge_features`` normalised velocity by ``|f| + 1e-9``.  COMSOL's no-slip
     wall is EXACTLY zero, so the locked ensembles were trained with wall nodes receiving no
     anisotropic messages; RGP-DEQ's wall speed is ~5e-6 (its hard BC is
     ``uv_prior + sdf*uvp`` and neither term is exactly zero there), which the old floor
     turned into a unit direction vector on every wall-destination edge.
  2. ``utils.kinematics_inference`` now clamps the width priors into the range the Stage-A
     checkpoint was trained on before every solve -- 34 of 52 packs carry ``|width_d2|`` up
     to 1.8e5 from a rank-deficient WLS row at P2 mid-side nodes.
  3. ``clot_ml.features.build_features`` differentiates predicted flow on a 6-hop stencil.
     At 4 the wall shear GRADIENT -- the gate's dominant argument -- is anti-correlated with
     COMSOL on 3 of 10 vessels.

Full derivation: the 2026-08-23 flow-swap investigation (``outputs/diag_flow_*.json``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _pack(name: str):
    p = PACKS / f"{name}.pt"
    if not p.exists():
        pytest.skip(f"{name} pack not present in this checkout")
    return torch.load(p, map_location="cpu", weights_only=False)


def _geometry(data):
    from src.clot_ml.gnn import edge_features

    ei = data.edge_index.detach().cpu().numpy()
    pos = (data.siren_pos if hasattr(data, "siren_pos") else data.x[:, :2])
    pos = pos.detach().cpu().numpy().astype(np.float64)
    h = float(np.median(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)))
    return edge_features, ei, pos, h


def _uv(data, flow: str):
    if flow == "pred":
        if getattr(data, "u0_pred", None) is None:
            pytest.skip("pack carries no u0_pred")
        return (data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64),
                data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64))
    return (data.y[0, :, 0].detach().cpu().numpy().astype(np.float64),
            data.y[0, :, 1].detach().cpu().numpy().astype(np.float64))


# --------------------------------------------------------------------------------------
# 1. the flow-direction deadband
# --------------------------------------------------------------------------------------

def test_gt_flow_puts_no_node_inside_the_deadband():
    """Why the deadband is safe: under GT the band is EMPTY, so the trained regime is intact.

    COMSOL's field is exactly 0.0 at the wall and O(1) in the lumen -- there is nothing in
    between for a relative floor to catch.  If this ever fails, the deadband has started
    changing GT features and the locked normaliser no longer matches its own inputs.
    """
    from src.clot_ml.gnn import FLOW_DIR_DEADBAND

    for name in ("patient020", "patient001"):
        data = _pack(name)
        u, v = _uv(data, "gt")
        spd = np.hypot(u, v)
        inside = (spd > 0.0) & (spd < FLOW_DIR_DEADBAND * float(np.median(spd)))
        assert not inside.any(), f"{name}: {int(inside.sum())} GT nodes fell in the deadband"


def test_wall_destination_edges_carry_no_direction_under_either_flow():
    """The point of the fix: the predicted field must reproduce GT's behaviour at the wall.

    ``to_device`` reads ``ea[:, 4]`` (``cos_d``) for ``w_up``/``w_dn``, so a non-zero cosine
    on an edge pointing INTO a wall node is aggregation the ensemble never saw at training
    time.  Before the deadband this read 0.0000 under GT and ~0.70 under RGP-DEQ.
    """
    data = _pack("patient020")
    edge_features, ei, pos, h = _geometry(data)
    wall_dst = data.mask_wall.reshape(-1).bool().cpu().numpy()[ei[1]]
    assert wall_dst.any()

    for flow in ("gt", "pred"):
        u, v = _uv(data, flow)
        cos_d = edge_features(pos, ei, u, v, h)[:, 4]
        assert np.abs(cos_d[wall_dst]).max() == 0.0, (
            f"{flow}: wall-destination edges carry a direction ("
            f"max |cos_d| = {np.abs(cos_d[wall_dst]).max():.3g})")


def test_the_deadband_leaves_resolved_lumen_flow_alone():
    """It must silence noise, not the signal: interior cosines stay the analytic value."""
    data = _pack("patient020")
    edge_features, ei, pos, h = _geometry(data)
    u, v = _uv(data, "gt")
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    interior = ~(wall[ei[0]] | wall[ei[1]])

    ea = edge_features(pos, ei, u, v, h)
    d = pos[ei[1]] - pos[ei[0]]
    dh = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    fd = np.stack([u[ei[1]], v[ei[1]]], 1)
    want = (dh * fd).sum(1) / (np.linalg.norm(fd, axis=1) + 1e-9)
    assert np.abs(ea[interior, 4] - want[interior]).max() < 1e-5


# --------------------------------------------------------------------------------------
# 2. the width-prior clamp
# --------------------------------------------------------------------------------------

def test_width_priors_are_clamped_for_the_solve_and_restored_after():
    from src.config import NodeFeat
    from src.utils.kinematics_inference import WIDTH_D1_MAX, WIDTH_D2_MAX, clamped_width_priors

    data = _pack("patient020")
    before = data.x
    d2_before = float(before[:, NodeFeat.WIDTH_D2].abs().max())
    if d2_before <= WIDTH_D2_MAX:
        pytest.skip("patient020 no longer carries out-of-range width priors")

    with clamped_width_priors(data) as g:
        # float32 rounds the clamp bound up in the last bit, hence the tolerance
        assert float(g.x[:, NodeFeat.WIDTH_D1].abs().max()) <= WIDTH_D1_MAX + 1e-3
        assert float(g.x[:, NodeFeat.WIDTH_D2].abs().max()) <= WIDTH_D2_MAX + 1e-3
        # everything else is untouched
        assert torch.equal(g.x[:, :NodeFeat.WIDTH_ND.start], before[:, :NodeFeat.WIDTH_ND.start])
        assert torch.equal(g.x[:, NodeFeat.WIDTH_ND], before[:, NodeFeat.WIDTH_ND])

    assert data.x.data_ptr() == before.data_ptr(), "the caller's own tensor must come back"
    assert float(data.x[:, NodeFeat.WIDTH_D2].abs().max()) == d2_before


def test_the_clamp_is_an_exact_no_op_when_the_priors_are_already_in_range():
    """A vessel inside the training range -- every kinematics vessel -- must not be copied.

    Note this is stricter than the cohort: even the 18 corner-edge packs trip the ``d1``
    bound (patient001 reads 6.90 against 4.14), and clamping them is measurably inert
    (rel L2 0.130 -> 0.131).  What must hold is that an in-range input is passed through
    untouched, so the operation cannot perturb a vessel it has nothing to fix.
    """
    from src.config import NodeFeat
    from src.utils.kinematics_inference import clamped_width_priors

    data = _pack("patient001")
    x = data.x.clone()
    x[:, NodeFeat.WIDTH_D1] = 0.5
    x[:, NodeFeat.WIDTH_D2] = 7.0
    data.x = x
    with clamped_width_priors(data) as g:
        assert g.x.data_ptr() == x.data_ptr()


def test_the_clamp_survives_a_pack_without_width_channels():
    from src.utils.kinematics_inference import clamped_width_priors

    data = _pack("patient020")
    data.x = data.x[:, :6].contiguous()
    with clamped_width_priors(data) as g:
        assert g.x.shape[1] == 6


# --------------------------------------------------------------------------------------
# 3. the 6-hop stencil on predicted flow
# --------------------------------------------------------------------------------------

class _StencilProbe(Exception):
    """Aborts build_features once the stencil width is known -- the rest costs ~25 s."""


def _first_stencil(flow: str) -> int:
    import src.core_physics.mls_gradient as mls
    # Import every module that binds `build_mls_gradient` at module scope BEFORE patching.
    # `build_features` imports `physics_wall_model` lazily, so patching first would let that
    # module capture the probe permanently and leak it into every later test in the file.
    import src.core_physics.physics_wall_model as pwm
    from src.clot_ml.features import build_features
    from src.config import BiochemConfig, PhysicsConfig

    data = _pack("patient020")
    if flow == "pred" and getattr(data, "u0_pred", None) is None:
        pytest.skip("pack carries no u0_pred")
    seen: list[int] = []
    holders = [mls, pwm]
    saved = [m.build_mls_gradient for m in holders]

    def probe(pos, ei, hops=3, **kw):
        seen.append(int(hops))
        raise _StencilProbe

    for m in holders:
        m.build_mls_gradient = probe
    try:
        with pytest.raises(_StencilProbe):
            build_features(data, BiochemConfig(phase="biochem"),
                           PhysicsConfig(phase="biochem"), flow=flow)
    finally:
        for m, fn in zip(holders, saved):
            m.build_mls_gradient = fn
    return seen[0]


def test_predicted_flow_is_differentiated_on_a_six_hop_stencil():
    """hops=4 is inside the sign-flip band for wall `dsrx` (corr +0.24; 0.96 at 6)."""
    assert _first_stencil("pred") == 6


def test_gt_flow_keeps_its_three_hop_stencil():
    """The locked artifacts' features are GT-derived -- this must not move."""
    assert _first_stencil("gt") == 3


# --------------------------------------------------------------------------------------
# 4. the predicted-flow dsrx amplitude correction
# --------------------------------------------------------------------------------------

def test_pred_dsrx_is_scaled_and_gt_is_not():
    """`sgt` is a physical constant; the discrete `dsrx` it gates on is stencil-dependent.

    GT (hops=3) is the convention `sgt` was fitted against, so it must stay untouched no
    matter what the gain is set to; the surrogate (hops=6) must carry it exactly.
    FEM (hops=3) must also stay untouched: it is a converged field, not a surrogate.
    """
    import src.core_physics.physics_wall_model as pwm

    data = _pack("patient020")
    if getattr(data, "u0_pred", None) is None:
        pytest.skip("pack carries no u0_pred")
    from src.config import BiochemConfig
    bio = BiochemConfig(phase="biochem")

    real = pwm.PRED_DSRX_GAIN
    try:
        pwm.PRED_DSRX_GAIN = 1.0
        raw_pred = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").dsrx.copy()
        raw_gt = pwm.t0_flow_fields(data, bio, flow_source="gt").dsrx.copy()
        raw_fem = pwm.t0_flow_fields(data, bio, hops=3, flow_source="fem").dsrx.copy()
        pwm.PRED_DSRX_GAIN = 7.0
        scaled_pred = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").dsrx
        scaled_gt = pwm.t0_flow_fields(data, bio, flow_source="gt").dsrx
        scaled_fem = pwm.t0_flow_fields(data, bio, hops=3, flow_source="fem").dsrx
    finally:
        pwm.PRED_DSRX_GAIN = real

    assert np.allclose(scaled_pred, 7.0 * raw_pred, rtol=1e-6, atol=0)
    assert np.array_equal(scaled_gt, raw_gt), "the GT convention must not move"
    assert np.array_equal(scaled_fem, raw_fem), "FEM must not be scaled (no surrogate deficit)"
    assert 2.0 < real < 4.0, "gain left outside the FIT/DEV bracket it was fitted in"


def test_the_gain_reaches_the_gate_branch():
    """It has to move `gate_sep`, or it is decoration -- that branch is what it exists for."""
    import src.core_physics.physics_wall_model as pwm

    data = _pack("patient012")
    if getattr(data, "u0_pred", None) is None:
        pytest.skip("pack carries no u0_pred")
    from src.config import BiochemConfig
    bio = BiochemConfig(phase="biochem")

    real = pwm.PRED_DSRX_GAIN
    try:
        pwm.PRED_DSRX_GAIN = 1.0
        off = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").gate_sep.sum()
        pwm.PRED_DSRX_GAIN = real
        on = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").gate_sep.sum()
    finally:
        pwm.PRED_DSRX_GAIN = real
    assert on > off, f"gate_sep did not open with the gain ({off} -> {on})"
