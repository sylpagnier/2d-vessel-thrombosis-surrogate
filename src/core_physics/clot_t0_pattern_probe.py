"""Exploratory probe: t=0 kinematics vs GT clot @ t_final inside deploy ceiling mask.

Goal: find interpretable rules (shear grad, stagnation, geometry) that separate
clot from non-clot nodes **without using GT at inference** — only for offline
pattern discovery on anchor graphs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_anchor_survey import _graph_props, discover_anchor_paths
from src.core_physics.clot_growth_masks import (
    clot_ceiling_hops,
    graph_dilate_hops,
    resolve_ceiling_mask,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields, score_clot_risk_from_fields
from src.core_physics.clot_phi_simple import clot_phi_thresh_si, gt_neg_dgamma_dx_phys, sdf_nd_from_data
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat
from src.utils.channel_schema import KINE_X_SCHEMA, X_SCHEMAS


def _wall_mask(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        return data.mask_wall.view(-1).to(device=device).bool()
    return torch.zeros(n, dtype=torch.bool, device=device)


def _hop_distance_from_seed(seed: torch.Tensor, edge_index: torch.Tensor, max_hops: int = 64) -> torch.Tensor:
    """BFS hop count from seed nodes; unreachable -> max_hops+1."""
    n = int(seed.numel())
    dist = torch.full((n,), max_hops + 1, dtype=torch.long)
    if not bool(seed.any().item()):
        return dist
    dist[seed] = 0
    active = seed.clone()
    ei = edge_index
    for h in range(max_hops):
        nxt = graph_dilate_hops(active, ei, 1) & ~active
        if not bool(nxt.any().item()):
            break
        dist[nxt] = h + 1
        active = active | nxt
    return dist


def _x_channel(data, name: str, device: torch.device) -> torch.Tensor | None:
    if not hasattr(data, "x") or data.x is None or not torch.is_tensor(data.x):
        return None
    if getattr(data, "x_schema", None) != KINE_X_SCHEMA:
        if name == "sdf_nd" and data.x.shape[1] > 2:
            return data.x[:, 2].to(device=device, dtype=torch.float32)
        return None
    try:
        idx = X_SCHEMAS[KINE_X_SCHEMA].channels.index(name)
    except ValueError:
        return None
    return data.x[:, idx].to(device=device, dtype=torch.float32)


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    higher_is_risk: bool
    group: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("neg_dgamma_dx", "-d(gamma)/dx @ t0 [1/s/m]", True, "shear_grad"),
    FeatureSpec("dgamma_dx", "d(gamma)/dx @ t0", False, "shear_grad"),
    FeatureSpec("dgamma_dy", "d(gamma)/dy @ t0", False, "shear_grad"),
    FeatureSpec("dshear_ds", "streamwise d(gamma)/ds", False, "shear_grad"),
    FeatureSpec("gamma_si", "shear rate gamma [1/s]", False, "flow"),
    FeatureSpec("vel_mag_si", "speed |u,v| [m/s]", False, "flow"),
    FeatureSpec("flux_path_dx", "adverse dx flux (prior)", True, "prior"),
    FeatureSpec("flux_stag", "stagnation flux", True, "prior"),
    FeatureSpec("flux_path_stream", "stream separation flux", True, "prior"),
    FeatureSpec("prior_score", "comsol_hybrid prior", True, "prior"),
    FeatureSpec("is_low_shear", "low-shear sigmoid", True, "flow"),
    FeatureSpec("is_separation", "stream separation sigmoid", True, "flow"),
    FeatureSpec("sdf_nd", "wall distance sdf", False, "geometry"),
    FeatureSpec("wall_proximity", "exp(-sdf/lambda)", True, "geometry"),
    FeatureSpec("on_wall", "mask_wall", True, "geometry"),
    FeatureSpec("hop_from_wall", "graph hops from wall", False, "geometry"),
    FeatureSpec("hop_from_t0_dgamma", "hops from t0 dgamma strip", False, "geometry"),
    FeatureSpec("width_nd", "hydraulic width", False, "geometry"),
    FeatureSpec("width_d1", "d(width)/ds", False, "geometry"),
    FeatureSpec("width_d2", "d2(width)/ds2 (curvature proxy)", True, "geometry"),
    FeatureSpec("wss_prior_nd", "WSS prior", False, "flow"),
    FeatureSpec("log10_gamma", "log10(gamma)", False, "flow"),
    FeatureSpec("log1p_neg_dx", "log1p(-d(gamma)/dx)+", True, "shear_grad"),
    FeatureSpec("fi_t0", "FI species @ t0", True, "species"),
    FeatureSpec("mat_t0", "Mat species @ t0", True, "species"),
)


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUC; scores higher => positive."""
    s = scores.detach().cpu().float().reshape(-1)
    y = labels.detach().cpu().float().reshape(-1)
    n_pos = int(y.sum())
    n_neg = int((1.0 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s, descending=False)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    sum_pos_ranks = float(ranks[y > 0.5].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _decile_rule_metrics(
    risk: torch.Tensor,
    labels: torch.Tensor,
    *,
    frac: float = 0.10,
) -> dict[str, float]:
    """Flag top ``frac`` risk nodes; report precision/recall vs GT clot in mask."""
    n = int(labels.numel())
    n_pos = int(labels.sum())
    if n_pos == 0:
        return {"prec": float("nan"), "rec": float("nan"), "n_flag": 0.0}
    k = max(int(math.ceil(frac * n)), 1)
    k = min(k, n)
    _, idx = torch.topk(risk, k)
    pred = torch.zeros(n, dtype=torch.bool)
    pred[idx] = True
    tp = int((pred & labels).sum())
    return {
        "prec": tp / max(int(pred.sum()), 1),
        "rec": tp / max(n_pos, 1),
        "n_flag": float(k),
    }


@dataclass
class FeatureProbeRow:
    anchor: str
    feature: str
    label: str
    group: str
    n_mask: int
    n_clot: int
    clot_frac: float
    clot_mean: float
    non_mean: float
    delta_mean: float
    auc: float
    decile_prec: float
    decile_rec: float


@dataclass
class RuleProbeRow:
    anchor: str
    rule: str
    n_flag: int
    prec: float
    rec: float
    f1: float


@dataclass
class AnchorPatternReport:
    anchor: str
    n_nodes: int
    n_ceiling: int
    n_clot_ceiling: int
    clot_recall_in_ceiling: float
    t_final_s: float
    feature_rows: list[FeatureProbeRow] = field(default_factory=list)
    rule_rows: list[RuleProbeRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_t0_feature_table(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    t0_index: int = 0,
) -> dict[str, torch.Tensor]:
    """All probe features at t=0 (deploy-visible inputs)."""
    n = int(data.num_nodes)
    y0 = data.y[int(t0_index)].to(device=device, dtype=torch.float32)
    u = y0[:, 0]
    v = y0[:, 1]
    props = _graph_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior, _, _ = score_clot_risk_from_fields(fields, bio_cfg)
    neg_dx = gt_neg_dgamma_dx_phys(data, int(t0_index), bio_cfg, device)
    sdf = sdf_nd_from_data(data, device, n)
    wall = _wall_mask(data, device, n)
    ei = data.edge_index.to(device=device)
    hop_wall = _hop_distance_from_seed(wall, ei).float()
    t0_strip = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
    hop_t0 = _hop_distance_from_seed(t0_strip, ei).float()
    vel_mag_si = torch.sqrt(u * u + v * v) * props["u_ref"].reshape(-1).clamp(min=1e-8)

    out: dict[str, torch.Tensor] = {
        "neg_dgamma_dx": neg_dx,
        "dgamma_dx": fields.dgamma_dx_phys,
        "dgamma_dy": fields.dgamma_dy_phys,
        "dshear_ds": fields.dshear_ds_phys,
        "gamma_si": fields.gamma_si,
        "vel_mag_si": vel_mag_si,
        "flux_path_dx": fields.flux_path_dx,
        "flux_stag": fields.flux_stag,
        "flux_path_stream": fields.flux_path_stream,
        "prior_score": prior,
        "is_low_shear": fields.is_low_shear,
        "is_separation": fields.is_separation_stream,
        "sdf_nd": sdf,
        "wall_proximity": fields.wall_proximity,
        "on_wall": wall.float(),
        "hop_from_wall": hop_wall,
        "hop_from_t0_dgamma": hop_t0,
        "log10_gamma": torch.log10(fields.gamma_si.clamp(min=1e-6)),
        "log1p_neg_dx": torch.log1p(neg_dx.clamp(min=0.0)),
        "fi_t0": y0[:, 12],
        "mat_t0": y0[:, 15],
    }
    for ch in ("width_nd", "width_d1", "width_d2", "wss_prior_nd"):
        xch = _x_channel(data, ch, device)
        if xch is not None:
            out[ch] = xch
    return out


