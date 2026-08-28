"""Selection metrics for Stage-A: score the model on what the clot stack consumes.

RGP_DEQ_REPAIR_PLAN.md T7.  Velocity rel-L2 is the wrong selection metric and this project has
paid for that twice: the width fix halved rel-L2 and made the frozen clot model *worse*
(`DEPLOY_FLOW_PLAN.md` §3), and the current surrogate posts a competitive rel-L2 (0.197 vs the
analytic prior's 0.188) while halving wall `dsrx` correlation (0.316 vs 0.644).

`clot_ml.features.build_features` never reads velocity.  It reads `sr`, `dsrx`, `dsry`, `vort`,
`div` and the hard gate `(sr < lss) + (dsrx < sgt)`, all inside a wall band.  So selection is
ordered: **wall `dsrx` correlation, then gate union Jaccard, then rel-L2 as a tie-break.**
"""

from __future__ import annotations

import numpy as np
import torch


def wall_shear_selection_metrics(pred_uv, data, *, hops_pred: int = 6, hops_gt: int = 3,
                                 gain: float | None = None, bio_cfg=None) -> dict:
    """`dsrx` correlation / scale and gate agreement for one vessel, against its own GT.

    Stencils deliberately differ: `build_features` uses ``hops=6`` on a predicted field and
    ``hops=3`` on GT, because a wide stencil is needed to keep a surrogate's second derivative
    from being its own sign flip (`DEPLOY_FLOW_PLAN.md` §1c).  Reproducing that exactly here is
    the point -- a selection metric computed on a different operator than the consumer uses is
    measuring a different quantity.
    """
    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d
    from src.core_physics.physics_wall_model import PRED_DSRX_GAIN

    y = getattr(data, "y", None)
    if y is None or not hasattr(data, "mask_wall"):
        return {}
    bio = bio_cfg or BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    g = PRED_DSRX_GAIN if gain is None else float(gain)

    ei = data.edge_index.detach().cpu().numpy()
    pos = node_positions(data)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    wall = data.mask_wall.reshape(-1).bool().detach().cpu().numpy()
    if wall.sum() < 5:
        return {}

    def fields(u, v, hops, gg):
        Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        return sr, (Dx @ sr) / (d_bar * M_TO_CM) * gg

    yv = y[0] if y.dim() == 3 else y
    ug = yv[:, 0].double().detach().cpu().numpy()
    vg = yv[:, 1].double().detach().cpu().numpy()
    up = pred_uv[:, 0].double().detach().cpu().numpy()
    vp = pred_uv[:, 1].double().detach().cpu().numpy()

    sr_g, dx_g = fields(ug, vg, hops_gt, 1.0)
    sr_p, dx_p = fields(up, vp, hops_pred, g)
    gate_g = (sr_g < lss) | (dx_g < sgt)
    gate_p = (sr_p < lss) | (dx_p < sgt)

    a, b = dx_p[wall], dx_g[wall]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    union = float((gate_p[wall] | gate_g[wall]).sum())
    return {
        "dsrx_corr": corr,
        "dsrx_scale": float(np.std(a) / (np.std(b) + 1e-30)),
        "sr_scale": float(np.std(sr_p[wall]) / (np.std(sr_g[wall]) + 1e-30)),
        "gate_jaccard": float((gate_p[wall] & gate_g[wall]).sum()) / union if union else float("nan"),
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
    """
    c = 0.0 if not np.isfinite(dsrx_corr) else float(dsrx_corr)
    j = 0.0 if not np.isfinite(gate_jaccard) else float(gate_jaccard)
    r = 1.0 if not np.isfinite(rel_l2) else float(rel_l2)
    return 2.0 * (1.0 - j) + 0.3 * (1.0 - c) + 0.05 * r


__all__ = ["selection_score", "wall_shear_selection_metrics"]
