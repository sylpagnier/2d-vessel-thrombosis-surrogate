"""Selection metrics for Stage-A: score the model on what the clot stack consumes.

RGP_DEQ_REPAIR_PLAN.md T7.  Velocity rel-L2 is the wrong selection metric and this project has
paid for that twice: the width fix halved rel-L2 and made the frozen clot model *worse*
(`DEPLOY_FLOW_PLAN.md` §3), and the current surrogate posts a competitive rel-L2 (0.197 vs the
analytic prior's 0.188) while halving wall `dsrx` correlation (0.316 vs 0.644).

`clot_ml.features.build_features` never reads velocity.  It reads `sr`, `dsrx`, `dsry`, `vort`,
`div` and the hard gate `(sr < lss) + (dsrx < sgt)`, all inside a wall band.  So selection is
ordered: **gate union Jaccard first, `dsrx` correlation as a diagnostic, rel-L2 as a
tie-break.**
"""

from __future__ import annotations

import numpy as np
import torch


#: ``PRED_DSRX_GAIN`` bundles two unrelated factors and only one of them belongs in a metric.
#: The consumer differentiates a PREDICTED field at ``hops=6`` and GT at ``hops=3``; the wider
#: stencil attenuates the wall ``dsrx`` by ~2.2x on its own, and the shipped 3.0 also carries
#: the OLD surrogate's ~1.35x under-resolution because it was least-squares fitted against it.
#: Holding a fitted constant fixed while retraining the thing it was fitted to makes the metric
#: reward a model that stays under-resolved -- measured, a PERFECT field reads gate Jaccard
#: 0.835 median on the deploy packs at gain 3.0 against 0.941 at the stencil-only gain.
#: ``"stencil"`` derives the ratio per vessel from the ground truth's own two stencils, which
#: is legitimate for a metric (it has the labels) and never for deployment (it does not).
GAIN_STENCIL = "stencil"


def _mls_ops(data, pos, ei, hops: int):
    """``build_mls_gradient`` is a per-node Python loop; the mesh never moves, so cache it."""
    from src.core_physics.mls_gradient import build_mls_gradient

    key = f"_sel_mls_h{int(hops)}"
    got = getattr(data, key, None)
    if got is not None:
        return got
    ops = build_mls_gradient(pos, ei, hops=hops)
    try:
        setattr(data, key, ops)
    except Exception:
        pass
    return ops


def wall_shear_selection_metrics(pred_uv, data, *, hops_pred: int = 6, hops_gt: int = 3,
                                 gain: float | str | None = None, bio_cfg=None) -> dict:
    """`dsrx` correlation / scale and gate agreement for one vessel, against its own GT.

    Stencils deliberately differ: `build_features` uses ``hops=6`` on a predicted field and
    ``hops=3`` on GT, because a wide stencil is needed to keep a surrogate's second derivative
    from being its own sign flip (`DEPLOY_FLOW_PLAN.md` §1c).  Reproducing that exactly here is
    the point -- a selection metric computed on a different operator than the consumer uses is
    measuring a different quantity.

    **The stencil asymmetry gives the metric a ceiling below 1.0, and it is per-vessel.**  Even
    at the ideal gain, feeding the ground-truth velocity in as ``pred_uv`` reads a gate Jaccard
    of 0.53-1.00 across the deploy packs: on some vessels the two stencils simply disagree about
    which nodes cross the threshold, and no flow model can fix that.  Averaging raw Jaccard over
    a cohort therefore mixes model quality with the metric's own defect, so this returns
    ``gate_jaccard_ceiling`` (the same quantity computed with ``pred := GT``) and
    ``gate_jaccard_frac`` (the fraction of the achievable agreement the model actually captured).
    **Select on the fraction.**
    """
    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import node_positions, shear_rate_2d
    from src.core_physics.physics_wall_model import PRED_DSRX_GAIN

    y = getattr(data, "y", None)
    if y is None or not hasattr(data, "mask_wall"):
        return {}
    bio = bio_cfg or BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM

    ei = data.edge_index.detach().cpu().numpy()
    pos = node_positions(data)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    wall = data.mask_wall.reshape(-1).bool().detach().cpu().numpy()
    if wall.sum() < 5:
        return {}

    Dx_g, Dy_g = _mls_ops(data, pos, ei, hops_gt)
    Dx_p, Dy_p = _mls_ops(data, pos, ei, hops_pred)

    def fields(u, v, Dx, Dy):
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        return sr, (Dx @ sr) / (d_bar * M_TO_CM)

    yv = y[0] if y.dim() == 3 else y
    ug = yv[:, 0].double().detach().cpu().numpy()
    vg = yv[:, 1].double().detach().cpu().numpy()
    up = pred_uv[:, 0].double().detach().cpu().numpy()
    vp = pred_uv[:, 1].double().detach().cpu().numpy()

    sr_g, dx_g = fields(ug, vg, Dx_g, Dy_g)          # GT through the consumer's GT stencil
    sr_p, dx_p0 = fields(up, vp, Dx_p, Dy_p)         # pred through the consumer's pred stencil
    sr_c, dx_c0 = fields(ug, vg, Dx_p, Dy_p)         # GT through the PRED stencil -> the ceiling

    if isinstance(gain, str) and gain.strip().lower() == GAIN_STENCIL:
        g = float(np.std(dx_g[wall]) / (np.std(dx_c0[wall]) + 1e-30))
    else:
        g = PRED_DSRX_GAIN if gain is None else float(gain)
    dx_p = dx_p0 * g
    dx_c = dx_c0 * g

    gate_g = (sr_g < lss) | (dx_g < sgt)
    gate_p = (sr_p < lss) | (dx_p < sgt)
    gate_c = (sr_c < lss) | (dx_c < sgt)

    def jac(a, b):
        union = float((a[wall] | b[wall]).sum())
        return float((a[wall] & b[wall]).sum()) / union if union else float("nan")

    a, b = dx_p[wall], dx_g[wall]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    j_pred, j_ceil = jac(gate_p, gate_g), jac(gate_c, gate_g)
    return {
        "dsrx_corr": corr,
        "dsrx_scale": float(np.std(a) / (np.std(b) + 1e-30)),
        "sr_scale": float(np.std(sr_p[wall]) / (np.std(sr_g[wall]) + 1e-30)),
        "gate_jaccard": j_pred,
        "gate_jaccard_ceiling": j_ceil,
        "gate_jaccard_frac": (j_pred / j_ceil) if (np.isfinite(j_ceil) and j_ceil > 1e-9)
        else float("nan"),
        "gain": float(g),
        "gate_fire_ratio": float(gate_p[wall].sum()) / max(float(gate_g[wall].sum()), 1.0),
    }


def selection_score(dsrx_corr: float, gate_jaccard: float, rel_l2: float) -> float:
    """One scalar for ``val_comp``-style comparison.  Lower is better.

    **Gate union Jaccard dominates, not `dsrx` correlation.**  That ordering is measured, and it
    reverses both this function's first version and `DEPLOY_FLOW_PLAN.md` §3's stated ordering.
    Against the locked clot ensemble's own per-node oracle-F1 (12 vessels x 2 flow arms,
    `scratch/diag_selection_vs_clot.py`):

    ```
                                       pearson   spearman
    gate_jaccard vs oracle-F1           +0.918     +0.904
    dsrx_corr    vs oracle-F1           +0.431     +0.555
    gate_jaccard vs F1 drop vs GT       +0.905     +0.889
    dsrx_corr    vs F1 drop vs GT       +0.304     +0.392

    within the analytic arm alone (removes the between-arm effect):
    gate_jaccard vs oracle-F1           +0.765     +0.797
    dsrx_corr    vs oracle-F1           -0.073     -0.126   <- no relationship at all
    ```

    `dsrx` correlation is kept with a small weight because it is a useful *diagnostic* -- it is
    what the gate is built from, so it explains failures -- but it is not what to select on.
    Within a single flow arm it carries no information about the downstream outcome.

    Pass ``gate_jaccard_frac`` rather than the raw Jaccard where it is available: the raw number
    carries a per-vessel ceiling of 0.53-1.00 that belongs to the metric, not to the model.
    """
    c = 0.0 if not np.isfinite(dsrx_corr) else float(dsrx_corr)
    j = 0.0 if not np.isfinite(gate_jaccard) else float(gate_jaccard)
    r = 1.0 if not np.isfinite(rel_l2) else float(rel_l2)
    return 2.0 * (1.0 - j) + 0.3 * (1.0 - c) + 0.05 * r


__all__ = [
    "wall_gate_health","GAIN_STENCIL", "selection_score", "wall_shear_selection_metrics"]

def wall_gate_health(pred_uv, data, *, hops_pred: int = 3, hops_gt: int = 3, bio_cfg=None) -> dict:
    """Does the WALL gate survive this prediction, and does the low shear tail reach the cut?

    The deploy collapse does not run through gate Jaccard.  `clot_ml`'s `physics_mask` seeds
    from ``(gate > 0) & wall``; when the surrogate's wall shear never drops below ``lss`` that
    seed is EMPTY, the mask empties, and thirteen physics/advection channels go identically
    zero -- measured on 7 of 30 deploy packs, and worth -0.97 wall F1 on patient010.  So the
    quantity to watch during training is not agreement, it is whether the gate fires at all.

    Returns ``fire_gt`` / ``fire_pred`` (share of wall nodes firing), ``empty`` (the pred gate
    is empty where GT's is not -- the failure), and ``p05_ratio`` (predicted 5th-percentile
    wall shear over the truth's; >1 means the tail is compressed upward, the root cause).
    """
    import numpy as np

    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import shear_rate_2d

    bio = bio_cfg or BiochemConfig(phase="biochem")
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    y = data.y[0] if data.y.dim() == 3 else data.y
    wall = data.mask_wall.reshape(-1).bool().detach().cpu().numpy()
    if wall.sum() < 5:
        return {}
    u_ref = float(data.u_ref.reshape(-1)[0]); d_bar = float(data.d_bar.reshape(-1)[0])
    from src.core_physics.mls_gradient import node_positions

    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx_g, Dy_g = _mls_ops(data, pos, ei, hops_gt)
    Dx_p, Dy_p = _mls_ops(data, pos, ei, hops_pred)

    def fields(u, v, Dx, Dy):
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        return sr, (Dx @ sr) / (d_bar * M_TO_CM)

    ug = y[:, 0].double().detach().cpu().numpy(); vg = y[:, 1].double().detach().cpu().numpy()
    up = pred_uv[:, 0].double().detach().cpu().numpy(); vp = pred_uv[:, 1].double().detach().cpu().numpy()
    sg, dg = fields(ug, vg, Dx_g, Dy_g)
    sp, dp = fields(up, vp, Dx_p, Dy_p)
    gg = ((sg < lss) | (dg < sgt))[wall]
    gp = ((sp < lss) | (dp < sgt))[wall]
    p05g = float(np.percentile(sg[wall], 5)); p05p = float(np.percentile(sp[wall], 5))
    return {
        "fire_gt": float(gg.mean()), "fire_pred": float(gp.mean()),
        "empty": float(bool(gg.any() and not gp.any())),
        "p05_ratio": p05p / max(p05g, 1e-9),
        "sr_min_ratio": float(sp[wall].min()) / max(float(sg[wall].min()), 1e-9),
    }
