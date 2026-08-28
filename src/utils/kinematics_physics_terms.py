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
        return zero, zero, zero
    band = wall_band_mask(data, hops).view(-1)
    if node_is_anchor is not None:
        band = band & node_is_anchor.view(-1).bool()
    if int(band.sum()) < 2:
        return zero, zero, zero
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
    l_dsrx = F.mse_loss(dsr_pr[band] / sd, dsr_gt[band] / sd)
    l_gate = _soft_gate_bce(data, sr_pr, sr_gt, dsr_pr, dsr_gt, band, s, sd)
    return l_sr, l_dsrx, l_gate


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

    tau_s = (s_scale * k_sr * 0.1).clamp(min=1e-6)
    tau_d = (d_scale * k_dx * 0.1).clamp(min=1e-6)

    off_low = torch.sigmoid((lss - sr_pr[band] * k_sr) / tau_s)
    off_sep = torch.sigmoid((sgt - dsr_pr[band] * k_dx) / tau_d)
    p_fire = 1.0 - (1.0 - off_low) * (1.0 - off_sep)

    with torch.no_grad():
        tgt = ((sr_gt[band] * k_sr < lss) | (dsr_gt[band] * k_dx < sgt)).to(p_fire.dtype)
    return F.binary_cross_entropy(p_fire.clamp(1e-6, 1 - 1e-6), tgt)


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
    boundary_data_weight: float = 2.0,
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
            wall_weight=boundary_data_weight,
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
        l_band_sr, l_band_dsrx, l_band_gate = wall_band_shear_losses(
            pred, data, kernels, props=props, hops=wall_band_hops,
            node_is_anchor=node_is_anchor,
        )

    return {
        "l_wss": l_wss,
        "l_shear_grad": l_shear_grad,
        "l_band_sr": l_band_sr,
        "l_band_dsrx": l_band_dsrx,
        "l_band_gate": l_band_gate,
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
    "wall_band_mask",
    "wall_band_shear_losses",
    "prior_floor_loss",
    "boundary_weighted_mse",
    "compute_anchor_kinematic_importance",
    "compute_kinematics_physics_terms",
]
