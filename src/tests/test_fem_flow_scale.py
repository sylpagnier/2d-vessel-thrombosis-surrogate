"""FEM flow source scale invariants (2026-09-01 fix for D1/D2/D3).

Two things that must stay true forever:

  1. FEM and GT produce the same dsrx scale when u0_pred == GT velocity.
     This is the invariant that would have caught D3 (PRED_DSRX_GAIN applied to a
     converged field that carries no surrogate deficit).

  2. The DEQ/pred arm's gain is unchanged -- PRED_DSRX_GAIN still reaches it so the
     surrogate's numbers are bit-identical after the split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.paths import anchor_packs_dir, get_project_root

REPO = get_project_root()
PACKS = anchor_packs_dir()


def _pack(name: str):
    p = PACKS / f"{name}.pt"
    if not p.exists():
        pytest.skip(f"{name} pack not present in this checkout")
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# 1. Oracle invariant: fem == gt when u0_pred IS the GT field
# ---------------------------------------------------------------------------

def test_fem_and_gt_on_same_scale_when_u0_pred_is_gt():
    """flow='fem' must land on exactly the same dsrx scale as flow='gt'.

    Set u0_pred to the GT field so both arms read the same velocity; any
    remaining difference is purely the gain/stencil treatment, which must be
    zero for FEM.  This would have caught D3 (FEM receiving PRED_DSRX_GAIN).
    """
    import src.core_physics.physics_wall_model as pwm
    from src.config import BiochemConfig

    data = _pack("comsol005")
    bio = BiochemConfig(phase="biochem")

    data.u0_pred = data.y[0, :, 0].clone()
    data.v0_pred = data.y[0, :, 1].clone()

    gt_fields = pwm.t0_flow_fields(data, bio, hops=3, flow_source="gt")
    fem_fields = pwm.t0_flow_fields(data, bio, hops=3, flow_source="fem")

    assert np.allclose(fem_fields.dsrx, gt_fields.dsrx, rtol=1e-6, atol=0), (
        "FEM dsrx diverged from GT even with identical velocity -- "
        "a gain > 1.0 was applied to the fem branch"
    )
    assert np.allclose(fem_fields.sr, gt_fields.sr, rtol=1e-6, atol=0), (
        "FEM sr diverged from GT even with identical velocity"
    )


def test_fem_scale_invariant_across_vessels():
    """Spot-check the oracle invariant on comsol020."""
    import src.core_physics.physics_wall_model as pwm
    from src.config import BiochemConfig

    data = _pack("comsol020")
    bio = BiochemConfig(phase="biochem")

    data.u0_pred = data.y[0, :, 0].clone()
    data.v0_pred = data.y[0, :, 1].clone()

    gt_f = pwm.t0_flow_fields(data, bio, hops=3, flow_source="gt")
    fem_f = pwm.t0_flow_fields(data, bio, hops=3, flow_source="fem")

    assert np.allclose(fem_f.dsrx, gt_f.dsrx, rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# 2. DEQ regression guard: pred arm gain is unchanged
# ---------------------------------------------------------------------------

def test_deq_pred_gain_still_reaches_pred_branch():
    """PRED_DSRX_GAIN must still reach flow_source='pred' after the fem split.

    The surrogate arm numbers must be bit-identical to pre-fix: mutating
    PRED_DSRX_GAIN changes scaled_pred proportionally and does NOT touch fem.
    """
    import src.core_physics.physics_wall_model as pwm
    from src.config import BiochemConfig

    data = _pack("comsol020")
    if getattr(data, "u0_pred", None) is None:
        pytest.skip("pack carries no u0_pred")
    bio = BiochemConfig(phase="biochem")

    data_fem = torch.load(PACKS / "comsol020.pt", map_location="cpu", weights_only=False)
    data_fem.u0_pred = data_fem.y[0, :, 0].clone()
    data_fem.v0_pred = data_fem.y[0, :, 1].clone()

    real = pwm.PRED_DSRX_GAIN
    try:
        pwm.PRED_DSRX_GAIN = 1.0
        base_pred = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").dsrx.copy()
        base_fem = pwm.t0_flow_fields(data_fem, bio, hops=3, flow_source="fem").dsrx.copy()
        pwm.PRED_DSRX_GAIN = 5.0
        scaled_pred = pwm.t0_flow_fields(data, bio, hops=6, flow_source="pred").dsrx
        scaled_fem = pwm.t0_flow_fields(data_fem, bio, hops=3, flow_source="fem").dsrx
    finally:
        pwm.PRED_DSRX_GAIN = real

    assert np.allclose(scaled_pred, 5.0 * base_pred, rtol=1e-6, atol=0), (
        "PRED_DSRX_GAIN is no longer reaching the pred branch"
    )
    assert np.array_equal(scaled_fem, base_fem), (
        "PRED_DSRX_GAIN change leaked into the fem branch"
    )
    assert 2.0 < real < 4.0, f"PRED_DSRX_GAIN={real} outside FIT/DEV bracket [2.56, 3.00]"
