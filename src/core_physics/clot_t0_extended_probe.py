"""Extended t=0 vs t_final feature sweep: graph x, biochem BC, topology, flow derivs.

Finds deployable signals (strong @ t=0) vs oracle-only signals (strong @ t_final only).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.mls_gradient import graph_gradient_operators
from src.core_physics.clot_anchor_survey import _graph_props, discover_anchor_paths
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_kinematics_fields import (
    adjacent_band_mask,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, gt_neg_dgamma_dx_phys, sdf_nd_from_data
from src.core_physics.clot_t0_pattern_probe import (
    _binary_auc,
    _decile_rule_metrics,
    _hop_distance_from_seed,
    _wall_mask,
    build_t0_feature_table,
)
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.utils.channel_schema import BIO_X_SCHEMA, BIO_Y_SCHEMA, KINE_X_SCHEMA, X_SCHEMAS, Y_SCHEMAS
from src.utils.rheology import compute_shear_rate


def _x_kine_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x") or data.x is None:
        return None
    schema = getattr(data, "x_schema", None)
    if schema != KINE_X_SCHEMA:
        return None
    try:
        idx = X_SCHEMAS[KINE_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x[:, idx].to(device=device, dtype=torch.float32)


def _x_biochem_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x_biochem") or data.x_biochem is None:
        return None
    schema = getattr(data, "x_biochem_schema", None)
    if schema != BIO_X_SCHEMA:
        return None
    try:
        idx = X_SCHEMAS[BIO_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x_biochem[:, idx].to(device=device, dtype=torch.float32)


def _y_channel(y_slice: torch.Tensor, name: str) -> torch.Tensor | None:
    try:
        idx = Y_SCHEMAS[BIO_Y_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return y_slice[:, idx]


def _flow_derivatives(
    data,
    u: torch.Tensor,
    v: torch.Tensor,
    props: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    u = u.reshape(-1).float()
    v = v.reshape(-1).float()
    G_x, G_y = graph_gradient_operators(data, device=u.device, dtype=u.dtype)
    du_dx = torch.sparse.mm(G_x, u.unsqueeze(1)).squeeze(1)
    du_dy = torch.sparse.mm(G_y, u.unsqueeze(1)).squeeze(1)
    dv_dx = torch.sparse.mm(G_x, v.unsqueeze(1)).squeeze(1)
    dv_dy = torch.sparse.mm(G_y, v.unsqueeze(1)).squeeze(1)
    gamma = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy)
    u_ref = props["u_ref"].reshape(-1).clamp(min=1e-8)
    d_bar = props["d_bar"].reshape(-1).clamp(min=1e-8)
    scale = u_ref / d_bar
    return {
        "div_uv": du_dx + dv_dy,
        "vorticity": dv_dx - du_dy,
        "gamma_dot_nd": gamma,
        "gamma_si_raw": gamma * scale,
        "du_dx": du_dx * scale,
        "du_dy": du_dy * scale,
        "dv_dx": dv_dx * scale,
        "dv_dy": dv_dy * scale,
        "speed_nd": torch.sqrt(u * u + v * v),
    }


def build_feature_table_at_time(
    data,
    time_index: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[str, tuple[torch.Tensor, str, str]]:
    """Return feature -> (values, group, source)."""
    n = int(data.num_nodes)
    ti = int(time_index)
    y = data.y[ti].to(device=device, dtype=torch.float32)
    u = y[:, 0]
    v = y[:, 1]
    props = _graph_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio_cfg)
    neg_dx = gt_neg_dgamma_dx_phys(data, ti, bio_cfg, device)
    sdf = sdf_nd_from_data(data, device, n)
    wall = _wall_mask(data, device, n)
    ei = data.edge_index.to(device=device)
    derivs = _flow_derivatives(data, u, v, props)
    u_ref = props["u_ref"].reshape(-1).clamp(min=1e-8)

    out: dict[str, tuple[torch.Tensor, str, str]] = {}

    def put(key: str, val: torch.Tensor, group: str, source: str) -> None:
        if val is None:
            return
        out[key] = (val.reshape(-1).float(), group, source)

    # Core kinematic / prior (time-varying)
    for key, val, group, source in (
        ("prior_score", prior, "prior", "computed"),
        ("dgamma_dx", fields.dgamma_dx_phys, "shear_grad", "computed"),
        ("dgamma_dy", fields.dgamma_dy_phys, "shear_grad", "computed"),
        ("neg_dgamma_dx", neg_dx, "shear_grad", "computed"),
        ("dshear_ds", fields.dshear_ds_phys, "shear_grad", "computed"),
        # COMSOL's actual separation-gate input. `dshear_ds` above is the streamwise
        # derivative, which no-slip pins to exactly 0 at every wall node.
        ("dgamma_dx_si", fields.dgamma_dx_si, "shear_grad", "computed"),
        ("is_separation_dx", fields.is_separation_dx, "shear_grad", "computed"),
        ("gamma_si", fields.gamma_si, "flow", "computed"),
        ("flux_path_dx", fields.flux_path_dx, "prior", "computed"),
        ("flux_stag", fields.flux_stag, "prior", "computed"),
        ("flux_path_stream", fields.flux_path_stream, "prior", "computed"),
        ("is_low_shear", fields.is_low_shear, "flow", "computed"),
        ("is_separation", fields.is_separation_stream, "flow", "computed"),
        ("wall_proximity", fields.wall_proximity, "geometry", "computed"),
        ("adjacent_band", fields.adjacent_band.float(), "geometry", "computed"),
    ):
        put(key, val, group, source)

    put("vel_mag_si", derivs["speed_nd"] * u_ref, "flow", "computed")
    put("p_t", y[:, 2], "flow", "y_slice")
    if y.shape[1] >= 16:
        for ch in Y_SCHEMAS[BIO_Y_SCHEMA].channels[4:]:
            put(f"y_{ch}@t", _y_channel(y, ch), "species", "y_slice")

    for key, val in derivs.items():
        put(key, val, "flow_derived", "computed")

    # Static graph / kine x
    put("sdf_nd", sdf, "geometry", "data.x")
    put("on_wall", wall.float(), "geometry", "mask_wall")
    put("hop_from_wall", _hop_distance_from_seed(wall, ei).float(), "topology", "graph")

    if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        inlet = data.mask_inlet.view(-1).to(device).bool()
        put("hop_from_inlet", _hop_distance_from_seed(inlet, ei).float(), "topology", "graph")
        put("on_inlet", inlet.float(), "topology", "mask_inlet")
    if hasattr(data, "mask_outlet") and data.mask_outlet is not None:
        outlet = data.mask_outlet.view(-1).to(device).bool()
        put("hop_from_outlet", _hop_distance_from_seed(outlet, ei).float(), "topology", "graph")
        put("on_outlet", outlet.float(), "topology", "mask_outlet")

    deg = torch.zeros(n, device=device)
    if ei.numel():
        deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1], device=device))
        deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1], device=device))
    put("graph_degree", deg, "topology", "edge_index")

    wnx = _x_kine_channel(data, "wall_normal_x", device)
    wny = _x_kine_channel(data, "wall_normal_y", device)
    if wnx is not None and wny is not None:
        flow_wall_align = torch.abs(u * wnx + v * wny) / derivs["speed_nd"].clamp(min=1e-8)
        put("flow_wall_alignment", flow_wall_align, "flow_derived", "computed")
        put("wall_normal_x", wnx, "geometry", "data.x")
        put("wall_normal_y", wny, "geometry", "data.x")

    u_pr = _x_kine_channel(data, "u_prior", device)
    v_pr = _x_kine_channel(data, "v_prior", device)
    if u_pr is not None and v_pr is not None:
        put("u_prior", u_pr, "kine_x", "data.x")
        put("v_prior", v_pr, "kine_x", "data.x")
        put("speed_mismatch_nd", torch.sqrt((u - u_pr) ** 2 + (v - v_pr) ** 2), "kine_x", "computed")

    for ch in X_SCHEMAS[KINE_X_SCHEMA].channels:
        val = _x_kine_channel(data, ch, device)
        if val is None:
            continue
        key = ch if ch in ("sdf_nd", "on_wall") else f"kine_x_{ch}"
        if ch == "sdf_nd":
            continue  # already set
        put(key, val, "kine_x", "data.x")

    for ch in X_SCHEMAS[BIO_X_SCHEMA].channels:
        val = _x_biochem_channel(data, ch, device)
        if val is not None:
            put(f"bio_x_{ch}", val, "bio_x", "data.x_biochem")

    # wss from bio y channel 4 is wss_nd in kine schema - bio y has u,v,p,mu then species
    # Fix wss: use kine y if 5ch else skip
    return out


@dataclass
class ExtendedFeatureRow:
    anchor: str
    feature: str
    group: str
    source: str
    auc_t0: float
    auc_tfinal: float
    delta_auc: float
    decile_rec_t0: float
    decile_rec_tfinal: float
    clot_mean_t0: float
    non_mean_t0: float
    higher_is_risk: bool


@dataclass
class ComboRuleRow:
    anchor: str
    rule: str
    f1_t0: float
    prec: float
    rec: float


@dataclass
class ExtendedProbeReport:
    anchor: str
    n_ceiling: int
    n_clot: int
    graph_attrs: dict[str, bool]
    rows: list[ExtendedFeatureRow] = field(default_factory=list)
    combos: list[ComboRuleRow] = field(default_factory=list)


