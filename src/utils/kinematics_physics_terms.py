"""
Physics loss **terms** for unified Kine phase kinematics training.

``train_kinematics_predictor.compute_step_loss`` calls this so validation tests exercise
the **exact** same code path as training (no duplicated derivative stacks).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.config import PredChannels
from src.core_physics.physics_kernels import PhysicsKernels, scatter_add
from src.utils.anchor_mask import anchor_node_mask
from src.utils.rheology import compute_shear_rate


#: Hops from the wall that count as "near-wall" for the supervised data term.
#: `clot_ml.features.build_features` reads `sr`, `dsrx`, `dsry`, `vort`, `div` and the
#: deposition gate inside a 3-hop band, so this is the region the downstream actually consumes.
WALL_BAND_HOPS = 3


def wall_band_mask(data, hops: int = WALL_BAND_HOPS) -> torch.Tensor:
    """Nodes within ``hops`` graph hops of a wall node.  Cached on ``data``.

    RGP_DEQ_REPAIR_PLAN.md B10.  The supervised data term used to weight ``data.mask_wall``
    itself, where COMSOL's velocity is exactly 0 AND the hard BC pins the prediction to
    ``uv_prior`` -- so the x2 "boundary weight" doubled a term that is zero by construction.
    The band one to three hops in is where wall shear is actually determined, and it carried
    weight 1.0 in an absolute MSE that it contributes 0.7-4.3% of.
    """
    key = f"_wall_band_mask_h{int(hops)}"
    cached = getattr(data, key, None)
    if cached is not None and torch.is_tensor(cached) and cached.numel() == int(data.num_nodes):
        return cached
    mw = getattr(data, "mask_wall", None)
    n = int(data.num_nodes)
    if mw is None:
        return torch.zeros(n, dtype=torch.bool, device=data.x.device)
    band = mw.reshape(-1).bool().clone()
    row, col = data.edge_index
    for _ in range(max(0, int(hops))):
        # index_add, not `grown[row] = band[col]`: plain indexed assignment with duplicate
        # indices is last-write-wins, so a node would be admitted only when its LAST incident
        # edge happened to point into the band -- a random subsample of the true dilation.
        acc = torch.zeros(n, dtype=torch.float32, device=band.device)
        acc.index_add_(0, row, band[col].to(torch.float32))
        band = band | (acc > 0)
    try:
        setattr(data, key, band)
    except Exception:
        pass
    return band


def _sdf_edge_gradient_proxy(data) -> torch.Tensor:
    """Mean |ΔSDF| over outgoing edges per node (geometry-variation proxy). Shape ``[N]``."""
    sdf = data.x[:, 2]
    row, col = data.edge_index
    n = int(data.num_nodes)
    diff = (sdf[row] - sdf[col]).abs()
    sum_d = scatter_add(diff, row, dim_size=n)
    cnt = scatter_add(torch.ones_like(diff), row, dim_size=n)
    return sum_d / (cnt + 1e-8)


def compute_anchor_kinematic_importance(
    data,
    node_is_anchor: torch.Tensor,
    *,
    mode: str = "uniform",
    sdf_wall_beta: float = 2.0,
    sdf_wall_tau: float = 0.12,
    sdf_grad_beta: float = 1.0,
    shear_true_alpha: float = 1.0,
    kernels: Optional[PhysicsKernels] = None,
    props=None,
) -> Optional[torch.Tensor]:
    """Return per-anchor-node positive weights, or ``None`` for uniform weighting."""
    mode = (mode or "uniform").strip().lower()
    if mode == "uniform" or node_is_anchor is None or int(node_is_anchor.sum().item()) == 0:
        return None
    if mode == "sdf_wall":
        sdf_a = data.x[node_is_anchor, 2].abs()
        return 1.0 + float(sdf_wall_beta) * torch.exp(-sdf_a / float(sdf_wall_tau))
    if mode == "sdf_grad":
        g = _sdf_edge_gradient_proxy(data)
        ga = g[node_is_anchor]
        med = torch.median(ga) if ga.numel() else ga.new_tensor(1.0)
        med = torch.clamp(med, min=1e-6)
        return 1.0 + float(sdf_grad_beta) * (ga / med)
    if mode == "shear_true":
        if kernels is None:
            return None
        # shear_true requires dense CFD labels across the graph. If this graph has sparse
        # anchor labeling, avoid derivative-based weighting to prevent boundary label leakage.
        if hasattr(data, "is_anchor") and torch.is_tensor(data.is_anchor):
            labeled = data.is_anchor.view(-1).bool()
            if int(labeled.sum().item()) < int(labeled.numel()):
                return None
        if props is None:
            props = kernels._get_geometric_props(data)
        # Ground-truth kinematic gradients from labels (u, v) to emphasize accelerated/sheared zones.
        c_u_true = kernels._compute_derivatives(data.y[:, PredChannels.U:PredChannels.U + 1], props)
        c_v_true = kernels._compute_derivatives(data.y[:, PredChannels.V:PredChannels.V + 1], props)
        u_x_true = c_u_true[:, 0, 0]
        u_y_true = c_u_true[:, 1, 0]
        v_x_true = c_v_true[:, 0, 0]
        v_y_true = c_v_true[:, 1, 0]
        gamma_dot_true = compute_shear_rate(u_x_true, u_y_true, v_x_true, v_y_true, eps=1e-6)
        gamma_anchor = gamma_dot_true[node_is_anchor]
        gamma_mean = torch.clamp(gamma_anchor.mean(), min=1e-6)
        return 1.0 + float(shear_true_alpha) * (gamma_anchor / gamma_mean)
    return None


def boundary_weighted_mse(
    pred_uvp: torch.Tensor,
    true_uvp: torch.Tensor,
    node_is_anchor: torch.Tensor,
    wall_mask: Optional[torch.Tensor] = None,
    wall_weight: float = 2.0,
    p_weight: float = 1.0,
    anchor_importance: Optional[torch.Tensor] = None,
    relative: bool = False,
) -> torch.Tensor:
    """Supervised kinematic loss on anchor nodes (Kine phase anchor regime).

    ``anchor_importance`` — optional positive weights per anchor node (e.g. SDF-based explorer).
    ``wall_mask`` — nodes receiving ``wall_weight``.  Callers should pass the wall *band*
    (:func:`wall_band_mask`), not the bare wall vertices; see B10 in RGP_DEQ_REPAIR_PLAN.md.
    ``relative`` — divide the squared error by the graph's own mean square (D3).
    """
    if node_is_anchor is None or int(node_is_anchor.sum().item()) == 0:
        return pred_uvp.sum() * 0.0
    p = pred_uvp[node_is_anchor, PredChannels.KINEMATICS]
    y = true_uvp[node_is_anchor, PredChannels.KINEMATICS]
    e = (p - y) ** 2
    if relative:
        # RGP_DEQ_REPAIR_PLAN.md D3.  An ABSOLUTE MSE on (u, v, p) is owned by the fast core:
        # the 3-hop wall band is 5-9% of the nodes but 0.06-1.5% of the field's squared
        # magnitude, and the wall band's share of the actual squared error runs 0.7-4.3%.
        # Normalising by the graph's own mean square puts every vessel on one scale, which
        # matters because `u_ref` differs 2x across the cohort and the loss is summed over a
        # mixed synthetic + clinical sampler.
        denom = (y**2).mean().clamp(min=1e-12)
        e = e / denom
    wp = float(p_weight)
    channel_weights = e.new_tensor([1.0, 1.0, wp]).view(1, 3)
    active_channels = torch.clamp(channel_weights.sum(), min=1e-12)
    if wp != 1.0:
        e = e * channel_weights
    if anchor_importance is None:
        if wall_mask is None:
            return e.sum() / (e.shape[0] * active_channels + 1e-12)
        wm = wall_mask[node_is_anchor].view(-1, 1).float()
        w = 1.0 + (max(float(wall_weight), 1.0) - 1.0) * wm
        return (e * w).sum() / (w.sum() * active_channels + 1e-12)
    if wall_mask is None:
        w = torch.ones_like(e[:, :1])
    else:
        wm = wall_mask[node_is_anchor].view(-1, 1).float()
        w = 1.0 + (max(float(wall_weight), 1.0) - 1.0) * wm
    imp = anchor_importance.view(-1, 1).to(device=e.device, dtype=e.dtype)
    w = w * imp
    return (e * w).sum() / (w.sum() * active_channels + 1e-12)


def corner_view(data):
    """A P1 corner view of a P2-elevated graph, cached on it.  ``None`` for a native graph.

    **Why the shear terms need one.**  ``elevate_to_p2`` fabricates every mid-side label as the
    mean of its two corners, so an elevated label field is piecewise-linear along each half-edge
    *by construction* -- and ``dsrx``, the argument of the gate branch that decides deployment,
    is a second derivative of exactly that field.  Measured on the deploy packs, holding the
    operator and the node set fixed (decimate a native-P2 pack to corners, re-elevate it by
    interpolation, differentiate both the same way):

    ```
    arm                       wall sr_med   wall dsrx_sd   sep-branch share of firing nodes
    native P2 (COMSOL)               91.3          398.5         0.0% (p041: 65.5%)
    corner P1                        72.5          161.9         0.0% (p041: 57.1%)
    re-elevated by interpolation     65.0           64.1         0.0% (p041: 47.5%)
    ```

    6.2x of the wall ``dsrx`` spread is destroyed by the interpolation alone.  On the synthetic
    corpus's own labels the same step halves it (300 -> 152) and takes the ``dsrx < sgt`` branch
    from firing on 21/40 vessels to 17/40 and from 0.5% of wall nodes to 0.0% -- against 5.6% at
    deployment, where **50.8% of firing wall nodes fire through that branch alone**.

    So supervising `sr` / `dsrx` / the gate on the elevated graph asks the model to reproduce an
    interpolation artifact.  On the corner view the labels are COMSOL's own.  The topology the
    MODEL sees is unchanged -- this only moves where the shear terms are evaluated.
    """
    ei1 = getattr(data, "p1_edge_index", None)
    n1 = getattr(data, "p1_num_nodes", None)
    if ei1 is None or n1 is None:
        return None
    n1 = int(n1)
    cached = getattr(data, "_corner_view", None)
    if cached is not None:
        return cached
    view = data.__class__()
    view.x = data.x[:n1]
    view.edge_index = ei1
    view.num_nodes = n1
    y = getattr(data, "y", None)
    if not torch.is_tensor(y) or y.shape[0] < n1:
        return None
    view.y = y[:n1]
    for name in ("mask_inlet", "mask_outlet", "mask_wall", "mask_wound"):
        m = getattr(data, name, None)
        if torch.is_tensor(m) and m.reshape(-1).numel() >= n1:
            setattr(view, name, m.reshape(-1)[:n1])
    for name in ("u_ref", "d_bar", "graph_stem"):
        v = getattr(data, name, None)
        if v is not None:
            setattr(view, name, v)
    if not torch.is_tensor(getattr(view, "mask_wall", None)):
        return None
    try:
        setattr(data, "_corner_view", view)
    except Exception:
        pass
    return view


def _corner_hops(hops: int) -> int:
    """Hop count on the corner view that spans the same distance as ``hops`` on P2."""
    import os

    raw = os.environ.get("KINEMATICS_BAND_CORNER_HOPS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(round(hops / 2.0)))


def _band_dsrx_absolute() -> bool:
    """``KINEMATICS_BAND_DSRX_ABS`` -- off by default.  See the note in `wall_band_shear_losses`."""
    import os

    return os.environ.get("KINEMATICS_BAND_DSRX_ABS", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _band_on_corners() -> bool:
    """``KINEMATICS_BAND_ON_CORNERS`` -- OFF by default.  See :func:`corner_view`.

    Off until it is measured on gate Jaccard, so an unset environment reproduces the previous
    runs exactly and the arm that turns it on differs in one thing.
    """
    import os

    return os.environ.get("KINEMATICS_BAND_ON_CORNERS", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _boundary_data_weight(default: float) -> float:
    """``KINEMATICS_BAND_DATA_WEIGHT`` -- how much the DATA term favours the wall band.

    Deployment reads `u`,`v` only in the wall band, differentiates them twice there and
    thresholds; nothing else the model emits is read.  A near-uniform data term therefore
    spends most of its gradient where the readout never looks, and the measured symptom is a
    model with healthy bulk rel-L2 (0.12) and wall `sr` correlation of only 0.413.  The
    shipped 2.0 is the historical value.
    """
    import os

    raw = os.environ.get("KINEMATICS_BAND_DATA_WEIGHT", "").strip()
    if not raw:
        return float(default)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(default)


def _wall_sr_tail_loss(data, band, sr_pr, sr_gt, s_scale):
    """Penalise the surrogate's inability to reach the LOW TAIL of wall shear rate.

    Root cause of the deploy-flow collapse, measured over 30 deploy packs: the surrogate gets
    the median wall `sr` right (87.3 against 89.9) and compresses the distribution
    (IQR ratio 0.62), so its MINIMUM wall shear is 31.5 where the truth reaches 9.3 -- and the
    deposition gate cuts at 25.  On 7 of 30 vessels no wall node crosses the cut, the wall
    gate is empty, `physics_mask` seeds from it and empties, and THIRTEEN downstream feature
    channels go identically zero.  That is the -0.97 on patient010, not accumulated field error.

    An L2 loss on velocity is exactly what produces that compression, and nothing in the
    objective looked at the tail.  This is a one-sided penalty on OVER-predicting shear where
    the truth is low: zero wherever the model is at or below the label, quadratic above it,
    restricted to the band where the gate is actually decided.  Normalised by the label spread
    so it sits on the other band terms' scale.  Off unless `KINEMATICS_TAIL_WEIGHT` is set.
    """
    import os as _os

    from src.config import BiochemConfig

    lss = float(BiochemConfig(phase="biochem").lss)
    u_ref = float(data.u_ref.reshape(-1)[0]) if hasattr(data, "u_ref") else 1.0
    d_bar = float(data.d_bar.reshape(-1)[0]) if hasattr(data, "d_bar") else 1.0
    k = u_ref / max(d_bar, 1e-12)                      # nd shear -> 1/s

    g = sr_gt[band] * k
    q = sr_pr[band] * k
    lo = g < (lss * float(_os.environ.get("KINEMATICS_TAIL_BAND_MULT", "2.0") or 2.0))
    if int(lo.sum()) < 2:
        return sr_pr.sum() * 0.0
    # SYMMETRIC, deliberately.  A one-sided penalty was the first instinct -- the surrogate
    # looked like it could not reach low shear -- but per-vessel the tail is wrong in BOTH
    # directions: measured, patient005's predicted minimum wall shear is 14.5x the truth's and
    # patient010's is 0.018x.  The cohort median hid that.  What matters is that the tail is
    # WRONG, not which way.
    return ((q[lo] - g[lo]) ** 2).mean() / (s_scale * k).clamp(min=1e-6) ** 2


def wall_band_shear_losses(
    pred, data, kernels: PhysicsKernels, *, props=None, hops: int = WALL_BAND_HOPS,
    node_is_anchor: Optional[torch.Tensor] = None,
):
    """Supervise the two quantities the clot stack actually consumes, where it consumes them.

    RGP_DEQ_REPAIR_PLAN.md D2.  ``clot_ml.features.build_features`` never reads velocity -- it
    reads ``sr`` and ``dsrx`` inside a wall band and thresholds them into the deposition gate.
    Stage-A was scored on velocity rel-L2 and supervised on wall *vertices* only, and the
    result is a model that moves shear AMPLITUDE without preserving shear STRUCTURE.  Measured
    on the wall under repaired analytic priors (`scratch/diag_wall_shear_authority.py`):

    ```
    vessel      field                 sr corr   sr scale   dsrx corr   dsrx scale
    patient020  analytic prior only     0.568      0.104       0.612        0.060
    patient020  RGP-DEQ                 0.703      0.506       0.228        0.387
    patient001  analytic prior only     0.806      0.410       0.962        0.146
    patient001  RGP-DEQ                 0.552      1.165       0.826        0.332
    ```

    So the hard BC does NOT deny the model wall-shear authority -- ``d/dn(sdf * uvp) = uvp`` at
    the wall, and the scale column shows it exercising that authority.  It simply was never
    asked to get the structure right, and on 2 of 4 vessels it makes the correlation *worse*
    than the closed-form prior it started from.

    Both terms are normalised by the ground truth's own spread on the band, which is what makes
    them scale-sensitive: an under-scaled prediction (every ``scale`` above is < 1, which is
    what ``PRED_DSRX_GAIN = 3.0`` exists to patch downstream) cannot reduce this loss by
    shrinking further.
    """
    zero = pred.sum() * 0.0
    if (not hasattr(data, "y")) or (data.y is None) or (data.y.shape[1] <= PredChannels.V):
        return zero, zero, zero, zero, zero

    # Evaluate on the CORNER subgraph of an elevated graph: its labels are COMSOL's own, where
    # the elevated graph's mid-side labels are an interpolation whose second derivative is an
    # artifact of the interpolation (see `corner_view` for the measurement).  `pred` is sliced
    # rather than re-run -- the model still sees, and is still differentiated through, the full
    # P2 graph.
    view = corner_view(data) if _band_on_corners() else None
    if view is not None:
        n1 = int(view.num_nodes)
        node_is_anchor = None if node_is_anchor is None else node_is_anchor.view(-1)[:n1]
        data, pred, props = view, pred[:n1], None
        # A hop is not the same length on the two graphs.  P2 edges are corner<->mid-side only,
        # so 3 hops reaches 1.5 ELEMENTS; on the corner graph 3 hops reaches 3.  Left alone this
        # arm would change two things at once -- whose labels are supervised AND how wide a band
        # -- so the corner hop count is halved to keep the physical width comparable.
        hops = _corner_hops(hops)

    band = wall_band_mask(data, hops).view(-1)
    if node_is_anchor is not None:
        band = band & node_is_anchor.view(-1).bool()
    if int(band.sum()) < 2:
        return zero, zero, zero, zero, zero
    if props is None:
        props = kernels._get_geometric_props(data)

    def _sr(field_u, field_v):
        cu = kernels._compute_derivatives(field_u, props)
        cv = kernels._compute_derivatives(field_v, props)
        return compute_shear_rate(cu[:, 0, 0], cu[:, 1, 0], cv[:, 0, 0], cv[:, 1, 0], eps=1e-6)

    sr_pr = _sr(pred[:, PredChannels.U:PredChannels.U + 1], pred[:, PredChannels.V:PredChannels.V + 1])
    sr_gt = _sr(data.y[:, PredChannels.U:PredChannels.U + 1], data.y[:, PredChannels.V:PredChannels.V + 1])

    s = sr_gt[band].std().clamp(min=1e-8)
    l_sr = F.mse_loss(sr_pr[band] / s, sr_gt[band] / s)

    # `dsrx` in `build_features` is the plain x-derivative of the shear rate.  This uses the
    # kernels' WLS operator rather than `build_mls_gradient(hops=6)` (which is numpy and not
    # differentiable), so the two are not literally the same call.  Measured, they agree on
    # STRUCTURE and differ only in amplitude -- wall-node correlation 0.966-0.999 for `dsrx`
    # and 0.986-0.998 for `sr`, at a scale ratio of 1.8-2.6x, which is the known stencil-width
    # attenuation.  Both terms below are normalised by the ground truth's own spread, so the
    # amplitude difference cancels and what is optimised is the structure the consumer reads.
    dsr_pr = kernels._compute_derivatives(sr_pr.unsqueeze(1), props)[:, 0, 0]
    dsr_gt = kernels._compute_derivatives(sr_gt.unsqueeze(1), props)[:, 0, 0]
    sd = dsr_gt[band].std().clamp(min=1e-8)
    if _band_dsrx_absolute():
        # ABSOLUTE, in units of the gate threshold the consumer actually compares against.
        #
        # The spread-normalised form is invariant to exactly the thing that is wrong.  Wall
        # `dsrx` AMPLITUDE is what separates a useful model from the analytic prior --
        # prior 0.07, overfit-on-one-vessel 0.955, every trained arm 0.067-0.149 -- and a loss
        # divided by the ground truth's own spread cannot see it: a prediction at 0.13x the
        # right amplitude scores the same as one at 1.0x if the shapes match.  `sgt` is a fixed
        # physical constant, so dividing by it keeps every vessel on one scale AND leaves the
        # loss sensitive to scale.
        #
        # Expect this to drive the model to the CORPUS's amplitude, which is 0.13x deployment's
        # (§16.5) -- that is the point: it separates "the objective cannot see amplitude" from
        # "the labels do not have it".
        from src.clot_ml.features import M_TO_CM
        from src.config import BiochemConfig

        _sgt = abs(float(BiochemConfig(phase="biochem").sgt) / M_TO_CM)
        u_ref = float(data.u_ref.reshape(-1)[0]) if hasattr(data, "u_ref") else 1.0
        d_bar = float(data.d_bar.reshape(-1)[0]) if hasattr(data, "d_bar") else 1.0
        k_dx = (u_ref / max(d_bar, 1e-12)) / (max(d_bar, 1e-12) * M_TO_CM)
        l_dsrx = F.mse_loss(dsr_pr[band] * k_dx / _sgt, dsr_gt[band] * k_dx / _sgt)
    else:
        l_dsrx = F.mse_loss(dsr_pr[band] / sd, dsr_gt[band] / sd)
    l_gate = _soft_gate_bce(data, sr_pr, sr_gt, dsr_pr, dsr_gt, band, s, sd)
    l_floor = _band_shear_floor(data, kernels, props, band, sr_pr, sr_gt, dsr_pr, dsr_gt, s, sd)
    l_tail = _wall_sr_tail_loss(data, band, sr_pr, sr_gt, s)
    return l_sr, l_dsrx, l_gate, l_floor, l_tail


def _band_shear_floor(data, kernels, props, band, sr_pr, sr_gt, dsr_pr, dsr_gt, s, sd):
    """Make the analytic prior a floor **in the wall shear channel**, not only in velocity.

    T6's `prior_floor_loss` hinges on ``|u - y|``, and the surrogate clears it comfortably while
    still being *worse than its own input* on the quantity the clot gate reads.  Measured on the
    8 strided deploy selection packs under deploy-legal priors:

    ```
    arm                gateJ% of ceiling   wall dsrx corr   rel-L2
    analytic prior              32.5           +0.559        0.159
    RGP-DEQ (shipped)           24.4           +0.073        0.195
    ```

    The trained surrogate is 8 points BEHIND the closed-form field it is handed, and its wall
    `dsrx` correlation is essentially zero.  Nothing in the objective noticed, because the only
    one-sided term looked at velocity.  This is the same hinge on the two band quantities,
    normalised by the ground truth's own spread so it is on the band terms' scale:

        L = relu(|sr_pred - sr_gt|^2 - |sr_prior - sr_gt|^2) / std(sr_gt)^2   + the same for dsrx

    Zero wherever the model beats the prior, positive only where it is worse.  The prior's own
    shear fields depend on nothing that changes during training, so they are cached per graph.
    """
    if not _band_shear_floor_enabled():
        return sr_pr.sum() * 0.0
    prior = data.x[:, 11:13]
    key = "_band_prior_shear"
    got = getattr(data, key, None)
    if got is None:
        with torch.no_grad():
            cu = kernels._compute_derivatives(prior[:, 0:1], props)
            cv = kernels._compute_derivatives(prior[:, 1:2], props)
            sr0 = compute_shear_rate(cu[:, 0, 0], cu[:, 1, 0], cv[:, 0, 0], cv[:, 1, 0], eps=1e-6)
            dsr0 = kernels._compute_derivatives(sr0.unsqueeze(1), props)[:, 0, 0]
        got = (sr0.detach(), dsr0.detach())
        try:
            setattr(data, key, got)
        except Exception:
            pass
    sr0, dsr0 = got
    e_m = (sr_pr[band] - sr_gt[band]) ** 2
    e_p = (sr0[band] - sr_gt[band]) ** 2
    out = torch.relu(e_m - e_p).mean() / (s**2)
    e_m = (dsr_pr[band] - dsr_gt[band]) ** 2
    e_p = (dsr0[band] - dsr_gt[band]) ** 2
    return out + torch.relu(e_m - e_p).mean() / (sd**2)


def _band_shear_floor_enabled() -> bool:
    """``KINEMATICS_BAND_SHEAR_FLOOR`` -- off by default until measured on gate Jaccard."""
    import os

    return os.environ.get("KINEMATICS_BAND_SHEAR_FLOOR", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _gate_tau_mult() -> float:
    """``KINEMATICS_GATE_TAU_MULT`` -- soft-gate temperature as a fraction of the GT spread."""
    import os

    raw = os.environ.get("KINEMATICS_GATE_TAU_MULT", "").strip()
    if not raw:
        return 0.1
    try:
        return max(1e-3, float(raw))
    except ValueError:
        return 0.1


def _gate_neg_weight() -> float:
    """``KINEMATICS_GATE_NEG_WEIGHT`` -- cost of a spurious firing node vs a missed one."""
    import os

    raw = os.environ.get("KINEMATICS_GATE_NEG_WEIGHT", "").strip()
    if not raw:
        return 1.0
    try:
        return max(1e-3, float(raw))
    except ValueError:
        return 1.0


def _soft_gate_bce(data, sr_pr, sr_gt, dsr_pr, dsr_gt, band, s_scale, d_scale):
    """Differentiable agreement with the GT deposition gate.

    RGP_DEQ_REPAIR_PLAN.md §10.3/§11.  Gate union Jaccard is the ONLY Stage-A metric measured
    to predict the clot model's own oracle-F1 (+0.918 pooled, +0.765 within a single flow arm,
    against `dsrx` correlation's -0.073) -- yet nothing in the objective optimised it.  `sr` and
    `dsrx` were supervised as continuous fields, which is a proxy for the threshold crossing
    that actually matters.

    The shipped gate is `(sr < lss) | (dsrx < sgt)`.  Its soft form is the complement of "both
    branches stay off":

        p_fire = 1 - (1 - sigma((lss - sr)/tau_s)) * (1 - sigma((sgt - dsrx)/tau_d))

    with temperatures set from the ground truth's own spread, so the sharpness follows the
    vessel rather than a hand-picked constant.  The target is the HARD GT gate, computed with
    the same operator -- verified well-posed: under the WLS operator the GT gate fires on
    10.1 / 36.0 / 10.1 % of wall nodes on patient020 / 001 / 041, identical to the shipped
    3-hop MLS convention to one decimal.
    """
    from src.clot_ml.features import M_TO_CM
    from src.config import BiochemConfig

    bio = BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM

    u_ref = float(data.u_ref.reshape(-1)[0]) if hasattr(data, "u_ref") else 1.0
    d_bar = float(data.d_bar.reshape(-1)[0]) if hasattr(data, "d_bar") else 1.0
    k_sr = u_ref / max(d_bar, 1e-12)
    k_dx = k_sr / (max(d_bar, 1e-12) * M_TO_CM)

    # Temperature as a fraction of the GT spread.  At the original 0.1 the term is very nearly
    # gradient-free: the gate's `dsrx` cut is ABSOLUTE (-750 1/(s*cm)) and sits ~1 sigma out, so
    # a 0.1-sigma temperature puts it ~10 tau from the prediction.  Measured over the 25 deploy
    # selection packs with the analytic prior standing in for the model (its dsrx scale, 0.092,
    # matches the trained model's 0.105-0.186):
    #
    #     tau_mult   band nodes with |z|>3   median z   mean sigmoid slope
    #        0.1              94.7%           -10.63          0.0063
    #        0.5              26.7%            -2.13          0.1039
    #        1.0               4.0%            -1.06          0.1746
    #
    # The `dsrx` GAIN barely moves this (88.3% still saturated at the stencil gain), so the
    # temperature is the binding quantity, not the amplitude calibration.  Default keeps the
    # historical 0.1; set KINEMATICS_GATE_TAU_MULT to open the gradient.
    _tm = _gate_tau_mult()
    tau_s = (s_scale * k_sr * _tm).clamp(min=1e-6)
    tau_d = (d_scale * k_dx * _tm).clamp(min=1e-6)

    off_low = torch.sigmoid((lss - sr_pr[band] * k_sr) / tau_s)
    off_sep = torch.sigmoid((sgt - dsr_pr[band] * k_dx) / tau_d)
    p_fire = 1.0 - (1.0 - off_low) * (1.0 - off_sep)

    with torch.no_grad():
        tgt = ((sr_gt[band] * k_sr < lss) | (dsr_gt[band] * k_dx < sgt)).to(p_fire.dtype)
    p_fire = p_fire.clamp(1e-6, 1 - 1e-6)

    # The gate error is ONE-SIDED.  Measured over the 25 deploy selection packs, the union
    # Jaccard loses 0.366 to SPURIOUS firing against 0.046 to missed firing -- the field does
    # not miss firing nodes, it invents them, at 8x the rate, and worst on the `sr < lss` branch
    # (per-branch Jaccard 0.060, firing on 7.1% of wall nodes against the truth's 3.5%).  A
    # symmetric BCE spends equal effort on the error that is not happening.  `w_neg` > 1 prices
    # a false positive above a false negative; 1.0 keeps the historical symmetric loss.
    w_neg = _gate_neg_weight()
    if w_neg == 1.0:
        return F.binary_cross_entropy(p_fire, tgt)
    w = tgt + (1.0 - tgt) * w_neg
    return F.binary_cross_entropy(p_fire, tgt, weight=w)


# ---------------------------------------------------------------------------
# PDE label floors -- the hinge for l_cont / l_mom
# ---------------------------------------------------------------------------
#: Node attributes carrying the per-node label residual floors.  Plain node-level tensors so
#: PyG collates them with the rest of the graph.
PDE_FLOOR_CONT = "pde_floor_cont"
PDE_FLOOR_MOM = "pde_floor_mom"


def compute_pde_floors(data, kernels: PhysicsKernels, *, re_ref=None, re_scale=None):
    """Per-node PDE residual of **the COMSOL labels themselves**.

    ``l_cont`` and ``l_mom`` ask the model to satisfy the discrete strong form.  The labels do
    not satisfy it either -- they are an FEM solution read onto a graph and differentiated by a
    5-term WLS stencil -- so the terms as written ask the model to be *more* PDE-consistent than
    the data it is simultaneously being fit to.  On this corpus that conflict is not a rounding
    detail.  Measured on 211 solved vessels with ``pred = y``:

    * ``l_cont`` median 2.5e-04 but p90 1.1e-01, max 2.2e-01 -- at the training weight of 100
      that is a residual of **22** on COMSOL's own answer.
    * It is not solve error.  The scale-free ratio ``rms(div u) / rms(|u_x| + |v_y|)`` has median
      **1.9%**, and the residual is extraordinarily concentrated: the single worst node carries a
      median **10%** of a vessel's total squared divergence and the worst 1% of nodes carry
      **70%**.  Those nodes sit at ``sdf_nd ~ 0.047`` -- the first interior ring off the wall,
      where the WLS stencil is one-sided and the profile curvature is highest.  Dropping three
      rings collapses ``l_cont`` by **113-257x** on the worst vessels, to the same 5e-04 the
      smooth ones already sit at.
    * It therefore scales with geometry, not with quality: spearman **+0.82** against stenosis
      ratio and **+0.72** against peak nd velocity, but only **+0.10** against the scale-free
      divergence ratio.

    Penalising that pulls the model away from COMSOL hardest on exactly the severe-stenosis
    vessels this cohort was generated to teach.  The fix is the same one-sided hinge used by
    :func:`prior_floor_loss`: penalise the residual only *in excess of* what the labels achieve,

        L = mean( relu( |r_pred|^2 - |r_gt|^2 ) )

    which is zero where the model is as PDE-consistent as the data and positive only where it is
    worse.  The floor is a property of the graph and the labels, so it is computed once.

    Returns ``None`` when the graph carries no usable labels (an unsolved vessel, whose ``y`` is
    an all-zero placeholder) -- there is no floor to grant, and the plain term applies.
    """
    y = getattr(data, "y", None)
    if y is None or y.dim() != 2 or y.shape[1] <= PredChannels.MU_EFF_ND:
        return None
    if float(y[:, PredChannels.U:PredChannels.V + 1].abs().max()) <= 0.0:
        return None  # unsolved vessel: zero placeholder labels, no floor to grant

    with torch.no_grad():
        props = kernels._get_geometric_props(data)
        c_u = kernels._compute_derivatives(y[:, PredChannels.U:PredChannels.U + 1], props)
        c_v = kernels._compute_derivatives(y[:, PredChannels.V:PredChannels.V + 1], props)
        div = c_u[:, 0, 0] + c_v[:, 1, 0]
        mom_sq = kernels.navier_stokes_residual(
            y, data, props=props, re_ref=re_ref, re_scale=re_scale, return_field=True
        )
    return {
        PDE_FLOOR_CONT: (div ** 2).detach().reshape(-1),
        PDE_FLOOR_MOM: mom_sq.detach().reshape(-1),
    }


def attach_pde_floors(data, kernels: PhysicsKernels, **kw) -> bool:
    """Attach :func:`compute_pde_floors` to ``data`` in place.  True if a floor was attached."""
    floors = compute_pde_floors(data, kernels, **kw)
    if floors is None:
        return False
    for name, val in floors.items():
        setattr(data, name, val)
    return True


def prior_floor_loss(pred, data, *, node_is_anchor: Optional[torch.Tensor] = None):
    """Penalise the model **only where it is worse than the prior it was handed**.

    RGP_DEQ_REPAIR_PLAN.md T6.  The DEQ is already residual -- ``u = uv_prior + sdf * uvp`` --
    but ``uvp`` is unconstrained, so nothing stops the model from being worse than its own
    input.  Measured, it usually is: on **45 of 52 packs** the cached prediction is farther from
    COMSOL than the prior it started from (median 0.141 against 0.025), and under deploy-legal
    priors the surrogate *lowers* wall ``dsrx`` correlation from 0.644 to 0.316.

    A plain shrinkage penalty on ``uvp`` would fight the model everywhere, including where it is
    right.  A one-sided hinge on the squared error does not:

        L = mean( relu( |pred - y|^2 - |prior - y|^2 ) )

    It is exactly zero wherever the model beats the prior and grows only where it does not, so
    the analytic prior becomes a performance **floor** rather than merely a starting point.
    That is the robustness property the deployable arm needs: a retrain can then only improve on
    a closed-form baseline that is already competitive (§1g), never regress below it.

    Normalised by the graph's own mean square so it is on the same scale as the relative data
    term (D3) and comparable across vessels.
    """
    zero = pred.sum() * 0.0
    y = getattr(data, "y", None)
    if y is None or y.shape[1] <= PredChannels.V:
        return zero
    uv_cols = slice(PredChannels.U, PredChannels.V + 1)
    prior = data.x[:, 11:13]
    if prior.shape != y[:, uv_cols].shape:
        return zero
    m = node_is_anchor
    if m is None:
        m = torch.ones(int(data.num_nodes), dtype=torch.bool, device=pred.device)
    m = m.reshape(-1).bool()
    if not bool(m.any()):
        return zero
    tgt = y[m][:, uv_cols]
    e_model = ((pred[m][:, uv_cols] - tgt) ** 2).sum(dim=1)
    e_prior = ((prior[m] - tgt) ** 2).sum(dim=1)
    denom = (tgt**2).mean().clamp(min=1e-12)
    return torch.relu(e_model - e_prior).mean() / denom


def compute_kinematics_physics_terms(
    pred: torch.Tensor,
    data,
    kernels: PhysicsKernels,
    *,
    phase: str = "kinematics",
    boundary_data_weight: float = 2.0,   # see `_boundary_data_weight`, env-overridable
    carreau_n: Optional[float] = None,
    distillation: bool = False,
    kine_p_weight: float = 1.0,
    anchor_kine_importance: Optional[torch.Tensor] = None,
    re_ref: Optional[float] = None,
    re_scale: Optional[float] = None,
    wall_band_hops: int = WALL_BAND_HOPS,
    relative_data_loss: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute every scalar physics term used across Kine phase training (except DEQ ``jac_loss``).

    ``phase`` is kept for naming consistency with config; Kine phase now uses unified kinematics physics.
    """
    props = kernels._get_geometric_props(data)
    l_wss = kernels.wall_shear_stress_loss(pred, data, props=props)

    z = pred.sum() * 0.0
    node_is_anchor = anchor_node_mask(data)

    l_data_kine = z.clone()
    l_data_mu = z.clone()
    if node_is_anchor is not None and int(node_is_anchor.sum().item()) > 0:
        # B10: the BAND, not the bare wall vertices.  On the wall itself GT `u` is exactly 0
        # and the hard BC pins `pred = uv_prior`, so weighting it changes nothing.
        wall_mask = (
            wall_band_mask(data, wall_band_hops)
            if wall_band_hops > 0
            else getattr(data, "mask_wall", None)
        )
        l_data_kine = boundary_weighted_mse(
            pred,
            data.y,
            node_is_anchor,
            wall_mask=wall_mask,
            wall_weight=_boundary_data_weight(boundary_data_weight),
            p_weight=kine_p_weight,
            anchor_importance=anchor_kine_importance,
            relative=relative_data_loss,
        )
        if not distillation:
            l_data_mu = F.mse_loss(
                pred[node_is_anchor, PredChannels.MU_EFF_ND],
                data.y[node_is_anchor, PredChannels.MU_EFF_ND],
            )

    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)

    if distillation:
        l_mom = z
        l_cont = z
        l_shear_grad = z
        l_band_sr = z
        l_band_dsrx = z
        l_band_gate = z
        l_band_floor = z
        l_band_tail = z
        l_rheo = kernels.rheology_loss(pred, data, props=props, carreau_n=carreau_n)
    else:
        # One-sided hinge against the labels' own PDE residual when a floor is attached
        # (`compute_pde_floors`); the plain term otherwise, so every other caller is unchanged.
        floor_mom = getattr(data, PDE_FLOOR_MOM, None)
        floor_cont = getattr(data, PDE_FLOOR_CONT, None)
        l_mom = kernels.navier_stokes_residual(
            pred, data, props=props, re_ref=re_ref, re_scale=re_scale, floor=floor_mom
        )
        c_u = kernels._compute_derivatives(pred[:, PredChannels.U:PredChannels.U + 1], props)
        c_v = kernels._compute_derivatives(pred[:, PredChannels.V:PredChannels.V + 1], props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        l_cont = kernels.continuity_loss(du_ij, data=data, floor=floor_cont)
        l_rheo = kernels.rheology_loss(pred, data, props=props, carreau_n=carreau_n)
        l_shear_grad = kernels.wall_shear_gradient_loss(pred, data, props=props)
        l_band_sr, l_band_dsrx, l_band_gate, l_band_floor, l_band_tail = wall_band_shear_losses(
            pred, data, kernels, props=props, hops=wall_band_hops,
            node_is_anchor=node_is_anchor,
        )

    return {
        "l_wss": l_wss,
        "l_shear_grad": l_shear_grad,
        "l_band_sr": l_band_sr,
        "l_band_dsrx": l_band_dsrx,
        "l_band_gate": l_band_gate,
        "l_band_floor": l_band_floor,
        "l_band_tail": l_band_tail,
        "l_prior_floor": prior_floor_loss(pred, data, node_is_anchor=node_is_anchor),
        "l_data_kine": l_data_kine,
        "l_data_mu": l_data_mu,
        "l_mom": l_mom,
        "l_cont": l_cont,
        "l_bc": l_bc,
        "l_io": l_io,
        "l_rheo": l_rheo,
    }


__all__ = [
    "WALL_BAND_HOPS",
    "PDE_FLOOR_CONT",
    "PDE_FLOOR_MOM",
    "compute_pde_floors",
    "attach_pde_floors",
    "corner_view",
    "wall_band_mask",
    "wall_band_shear_losses",
    "prior_floor_loss",
    "boundary_weighted_mse",
    "compute_anchor_kinematic_importance",
    "compute_kinematics_physics_terms",
]
