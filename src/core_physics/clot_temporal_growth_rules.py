"""Time-varying rule-based clot phi: progressive commit, incubation, neighbor growth."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import graph_dilate_hops, resolve_ceiling_mask
from src.core_physics.clot_phi_simple import (
    ClotPriorRuleConfig,
    _anchor_flow_props,
    _hop_distance_from_seed,
    _top_frac_mask,
    _wall_mask_from_data,
    clot_prior_score_flat,
    log_blend_mu_eff_si,
    predict_phi_prior_rule,
    prior_rule_config_from_env,
    project_deploy_mu_with_support,
    sdf_nd_from_data,
)
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.core_physics.clot_localized_spatial import (
    LocalizedSpatialConfig,
    blend_species_into_risk,
    build_eligible_pool,
    build_localized_static_support,
    normalize_risk_per_wall_half,
    resolve_species_time_index,
    segment_topk_mask,
)
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics

_temporal_pred_uv: tuple[torch.Tensor, torch.Tensor] | None = None
_temporal_pred_uv_key: int | None = None
_temporal_kine_model = None


def temporal_vel_source() -> str:
    """``gt`` = COMSOL [u,v] on anchor; ``kinematics`` = steady GINO-DEQ on mesh."""
    raw = (
        os.environ.get("CLOT_TEMPORAL_VEL_SOURCE")
        or os.environ.get("CLOT_PHI_VEL_SOURCE")
        or "gt"
    ).strip().lower()
    if raw in ("kin", "kinematics", "deq", "gino", "pred"):
        return "kinematics"
    if raw in ("coupled", "mu_coupled", "feedback", "5b"):
        return "coupled"
    return "gt"


def reset_temporal_kinematics_cache() -> None:
    """Clear cached steady GINO-DEQ uv (tests / multi-anchor sweeps)."""
    global _temporal_pred_uv, _temporal_pred_uv_key, _temporal_kine_model
    _temporal_pred_uv = None
    _temporal_pred_uv_key = None
    _temporal_kine_model = None


def _temporal_graph_cache_key(data) -> tuple[int, int, int]:
    """Stable cache key per graph (``id(data)`` alone can collide after GC)."""
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def _resolve_uv_for_temporal_risk(
    data,
    t_in: int,
    device: torch.device,
    vel_source: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow [u,v] ND for shear-risk features (steady pred when deploy mode)."""
    ti = max(0, min(int(t_in), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    u_gt = y[:, 0]
    v_gt = y[:, 1]
    vel_source = vel_source or temporal_vel_source()
    if vel_source == "gt":
        return u_gt, v_gt

    if vel_source == "coupled":
        from src.core_physics.clot_coupled_rollout import get_coupled_uv

        coupled = get_coupled_uv(data, device)
        if coupled is not None:
            return coupled

    global _temporal_pred_uv, _temporal_pred_uv_key, _temporal_kine_model
    key = _temporal_graph_cache_key(data)
    if _temporal_pred_uv is None or _temporal_pred_uv_key != key:
        from src.core_physics.clot_phi_rollout import clot_phi_kine_teacher_forcing
        from src.utils.kinematics_inference import (
            load_kinematics_predictor,
            predict_kinematics,
            resolve_kinematics_checkpoint,
        )

        ckpt = (os.environ.get("CLOT_PHI_KINE_CKPT") or "").strip()
        if not ckpt:
            ckpt = str(resolve_kinematics_checkpoint())
        if _temporal_kine_model is None:
            _temporal_kine_model = load_kinematics_predictor(
                ckpt,
                device,
                phys_cfg=PhysicsConfig(phase="kinematics"),
            )
        batch = data.to(device)
        with torch.no_grad():
            pred = predict_kinematics(_temporal_kine_model, batch)
        u_p = pred[:, 0]
        v_p = pred[:, 1]
        tf = clot_phi_kine_teacher_forcing()
        if tf >= 1.0:
            u_p, v_p = u_gt, v_gt
        elif tf > 0.0:
            u_p = (1.0 - tf) * u_p + tf * u_gt
            v_p = (1.0 - tf) * v_p + tf * v_gt
        _temporal_pred_uv = (u_p, v_p)
        _temporal_pred_uv_key = key
    return _temporal_pred_uv


@dataclass(frozen=True)
class TemporalGrowthRuleConfig:
    """Composable temporal commit policy on top of spatial risk."""

    name: str
    kind: str
    spatial_rule: ClotPriorRuleConfig | None = None
    localized: LocalizedSpatialConfig | None = None
    risk_flow_time: int = 0
    start_frac: float = 0.05
    end_frac: float = 0.22
    power: float = 1.5
    onset_spread: float = 0.55
    min_onset_frac: float = 0.05
    seed_frac: float = 0.08
    hop_per_step: int = 1
    risk_floor_quantile: float = 0.45
    neighbor_risk_q: float = 0.40
    global_onset_frac: float = 0.0
    promotion_boost: float = 1.0
    accum_gain: float = 0.25
    accum_threshold: float = 1.2
    accum_split_wall: float = 0.80
    accum_split_lumen: float = 0.03

    def describe(self) -> str:
        parts = [self.kind]
        if self.kind == "progressive_topk":
            hi = min(0.95, float(self.end_frac) * float(self.promotion_boost))
            parts.append(f"{self.start_frac:.2f}->{hi:.2f}^{self.power:.1f}")
        elif self.kind == "threshold_accum":
            parts.append(
                f"g={self.accum_gain:.2f}_Y={self.accum_threshold:.2f}"
                f"_sw={self.accum_split_wall:.2f}_sl={self.accum_split_lumen:.2f}"
            )
        elif self.kind == "ranked_onset":
            parts.append(f"spread={self.onset_spread:.2f}")
        elif self.kind == "hop_growth":
            parts.append(f"seed={self.seed_frac:.2f}")
        elif self.kind == "neighbor_ac":
            parts.append(f"seed={self.seed_frac:.2f}_nb={self.neighbor_risk_q:.2f}")
        elif self.kind == "static_spatial":
            parts.append("instant")
        if self.localized is not None:
            parts.append(f"loc={self.localized.describe()}")
        if self.global_onset_frac > 0:
            parts.append(f"offset={self.global_onset_frac:.2f}")
        if float(self.promotion_boost) > 1.0 + 1e-6:
            parts.append(f"boost={self.promotion_boost:.2f}")
        return "|".join(parts)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return float(default)
    return float(raw)


INC40_BASELINE_RULE_NAME = "loc_prog_both_t20_s0_ndx25_inc40"


def _time_frac_at_index(data, time_index: int) -> float:
    from src.core_physics.clot_continuous_time import time_frac_for_rollout

    return time_frac_for_rollout(data, int(time_index), clamp_unit=False)


def compute_spatial_risk_score(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    t_in: int,
    ceiling: torch.Tensor,
    spatial_rule: ClotPriorRuleConfig | None = None,
) -> torch.Tensor:
    n = int(data.num_nodes)
    u, v = _resolve_uv_for_temporal_risk(data, t_in, device, vel_source=vel_source)
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)
    stag = fields.flux_stag.reshape(-1)
    neg_dx = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
    wall = _wall_mask_from_data(data, device, n)
    hop = _hop_distance_from_seed(wall, data.edge_index.to(device)).float()
    dx = fields.flux_path_dx_raw.reshape(-1)

    pool = ceiling.reshape(-1).bool()
    rule = spatial_rule
    if rule and rule.rank_sdf_max_nd is not None:
        sdf = sdf_nd_from_data(data, device, n)
        pool = pool & (sdf <= float(rule.rank_sdf_max_nd))
    if rule and rule.skip_inlet_quantile is not None:
        if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
            inlet = data.mask_inlet.view(-1).to(device).bool()
            if int(inlet.numel()) == n and bool(inlet.any().item()):
                hin = _hop_distance_from_seed(inlet, data.edge_index.to(device)).float()
                eligible = pool & (hin > 0)
                if bool(eligible.any().item()):
                    thr = torch.quantile(hin[eligible], float(rule.skip_inlet_quantile))
                    pool = pool & (hin >= thr)

    def _norm(v: torch.Tensor) -> torch.Tensor:
        if not bool(pool.any().item()):
            return torch.zeros(n, device=device)
        return (v - v[pool].min()) / (v[pool].max() - v[pool].min() + 1e-12)

    score = 0.40 * _norm(prior) + 0.35 * _norm(stag) + 0.25 * _norm(neg_dx)
    if rule and rule.rank_tie_break:
        score = score + 1e-6 * (_norm(dx) + _norm(-hop))
    return score.clamp(0, 1) * pool.float()


def _vessel_span_nd(data) -> float:
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] >= 2:
        xy = data.x[:, :2].detach().float()
        span = xy.max(dim=0).values - xy.min(dim=0).values
        return float(torch.linalg.vector_norm(span))
    return 0.01


def _effective_low_shear_thresh_si(bio_cfg: BiochemConfig, loc: LocalizedSpatialConfig) -> float:
    if float(loc.low_shear_thresh_si) > 0:
        return float(loc.low_shear_thresh_si)
    return float(bio_cfg.lss)


def compute_localized_risk_score(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    t_in: int,
    ceiling: torch.Tensor,
    pool: torch.Tensor,
    spatial_rule: ClotPriorRuleConfig | None,
    loc: LocalizedSpatialConfig,
) -> torch.Tensor:
    """Risk with tunable shear channels, then optional per-half renormalization."""
    n = int(data.num_nodes)
    u, v = _resolve_uv_for_temporal_risk(data, t_in, device, vel_source=vel_source)
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)
    stag_legacy = fields.flux_stag.reshape(-1)
    neg_dx = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
    sep_stream = fields.flux_path_stream.reshape(-1)
    grad_mag = torch.sqrt(
        fields.dgamma_dx_phys.reshape(-1) ** 2 + fields.dgamma_dy_phys.reshape(-1) ** 2
    )
    lss = _effective_low_shear_thresh_si(bio_cfg, loc)
    T_ls = max(float(bio_cfg.soft_step_T_low_shear) * float(bio_cfg.soft_step_T_scale), 1e-6)
    low_shear = torch.sigmoid(((lss - fields.gamma_si.reshape(-1)) / T_ls).clamp(-50.0, 50.0))
    vel_mag_si = torch.sqrt(u ** 2 + v ** 2) * props["u_ref"].to(device=device).reshape(-1)
    u_ref_safe = props["u_ref"].to(device=device).reshape(-1).clamp(min=1e-8)
    residence = torch.exp(-(vel_mag_si / u_ref_safe).clamp(min=0.0, max=50.0))
    stasis = low_shear * (1.0 + 0.5 * residence)
    lgrad_thr = max(float(loc.low_grad_thresh_si), 1e-6)
    T_gr = max(float(bio_cfg.soft_step_T_grad) * float(bio_cfg.soft_step_T_scale), 1e-6)
    low_grad_zone = torch.sigmoid(((lgrad_thr - grad_mag) / T_gr).clamp(-50.0, 50.0))

    w_ndx = max(float(loc.neg_dx_risk_weight), 0.0)
    w_sep = max(float(loc.sep_stream_risk_weight), 0.0)
    w_stag = max(float(loc.stasis_risk_weight), 0.0)
    w_lgrad = max(float(loc.low_grad_risk_weight), 0.0)
    shear_mode = (w_sep + w_stag + w_lgrad) > 1e-6 or bool(str(loc.aneurysm_size_mode).strip())

    sz_mode = str(loc.aneurysm_size_mode or "").strip().lower()
    if sz_mode == "auto":
        large = _vessel_span_nd(data) >= 0.018
        if large:
            w_stag *= 1.25
        else:
            w_ndx *= 1.25
    elif sz_mode == "small_neg_dx":
        w_ndx *= 1.35
    elif sz_mode == "large_stasis":
        w_stag *= 1.35

    def _norm(v: torch.Tensor) -> torch.Tensor:
        if not bool(pool.any().item()):
            return torch.zeros(n, device=device)
        return (v - v[pool].min()) / (v[pool].max() - v[pool].min() + 1e-12)

    if not shear_mode:
        w_rem = max(1.0 - w_ndx, 0.0)
        w_prior = 0.40 * w_rem / 0.75 if w_rem > 0 else 0.0
        w_stag_legacy = 0.35 * w_rem / 0.75 if w_rem > 0 else 0.0
        score = w_prior * _norm(prior) + w_stag_legacy * _norm(stag_legacy) + w_ndx * _norm(neg_dx)
    else:
        w_sum = w_ndx + w_sep + w_stag + w_lgrad
        if w_sum < 1e-6:
            w_ndx, w_sep, w_stag, w_lgrad = 0.35, 0.25, 0.25, 0.15
            w_sum = 1.0
        inv = 1.0 / w_sum
        score = (
            (w_ndx * inv) * _norm(neg_dx)
            + (w_sep * inv) * _norm(sep_stream)
            + (w_stag * inv) * _norm(stasis)
            + (w_lgrad * inv) * _norm(low_grad_zone)
        )

    rule = spatial_rule
    if rule and rule.rank_tie_break:
        dx = fields.flux_path_dx_raw.reshape(-1)
        wall = _wall_mask_from_data(data, device, n)
        hop = _hop_distance_from_seed(wall, data.edge_index.to(device)).float()
        score = score + 1e-6 * (_norm(dx) + _norm(-hop))
    score = score.clamp(0, 1) * pool.float()
    if loc.normalize_risk_per_half:
        score = normalize_risk_per_wall_half(score, data, device, pool, loc)
    return score


def _resolve_pool_risk(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    cfg: TemporalGrowthRuleConfig,
    t_out: int,
    t_in: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = cfg.spatial_rule or prior_rule_config_from_env()
    if cfg.localized is not None:
        pool = build_eligible_pool(data, device, ceiling, spatial, cfg.localized)
        risk = compute_localized_risk_score(
            data,
            device=device,
            bio_cfg=bio_cfg,
            t_in=cfg.risk_flow_time,
            ceiling=ceiling,
            pool=pool,
            spatial_rule=spatial,
            loc=cfg.localized,
        )
    else:
        pool = ceiling.reshape(-1).bool()
        risk = compute_spatial_risk_score(
            data,
            device=device,
            bio_cfg=bio_cfg,
            t_in=cfg.risk_flow_time,
            ceiling=ceiling,
            spatial_rule=spatial,
        )
    if cfg.localized is not None and cfg.localized.species_risk_weight > 0:
        ti = resolve_species_time_index(data, cfg.localized.species_gt_time, t_out, t_in)
        risk = blend_species_into_risk(risk, data, device, pool, cfg.localized, ti)
    elif cfg.localized is None:
        risk = risk * pool.float()
    return pool, risk


def _localized_static_support(
    data,
    *,
    device: torch.device,
    pool: torch.Tensor,
    risk: torch.Tensor,
    cfg: TemporalGrowthRuleConfig,
    t_out: int,
    t_in: int = 0,
) -> torch.Tensor:
    loc = cfg.localized
    if loc is None:
        raise ValueError("localized static support requires cfg.localized")
    return build_localized_static_support(
        risk, data, device, pool, loc, t_out=t_out, t_in=t_in
    )


def _progressive_frac_from_growth_u(
    cfg: TemporalGrowthRuleConfig,
    u_grow: float,
    *,
    extrap: bool = False,
    sim_end_scale: float = 1.0,
) -> float:
    from src.core_physics.clot_continuous_time import extrap_frac_headroom

    ug = max(0.0, float(u_grow))
    lo = float(cfg.start_frac)
    hi = min(0.95, float(cfg.end_frac) * max(float(cfg.promotion_boost), 1.0))
    power = max(float(cfg.power), 0.1)
    if extrap and ug > 1.0 + 1e-6:
        hi_ex = min(0.95, hi + extrap_frac_headroom())
        scale = max(float(sim_end_scale), 1.0 + 1e-6)
        t_extra = min((ug - 1.0) / max(scale - 1.0, 1e-6), 1.0)
        return hi + (hi_ex - hi) * t_extra
    ug = min(ug, 1.0)
    return lo + (hi - lo) * (ug ** power)


def _progressive_frac(cfg: TemporalGrowthRuleConfig, t_out: int, t_final: int) -> float:
    tf = max(int(t_final), 1)
    u = float(t_out) / tf
    return _progressive_frac_from_growth_u(cfg, u)


def predict_phi_temporal_at_time(
    data,
    t_out: int,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    cfg: TemporalGrowthRuleConfig,
    ceiling: torch.Tensor,
    risk: torch.Tensor,
    phi_prev: torch.Tensor | None,
    t_final: int,
    use_provided_risk: bool = False,
    onset_override: float | None = None,
    sim_end_scale: float | None = None,
) -> torch.Tensor:
    from src.core_physics.clot_continuous_time import (
        continuous_extrap_growth_enabled,
        feature_time_index,
        growth_time_frac,
        growth_u_from_t_frac,
        sim_end_scale_from_env,
    )

    n = int(data.num_nodes)
    t_virt = max(0, int(t_out))
    t_feat = feature_time_index(data, t_virt)
    scale = float(sim_end_scale if sim_end_scale is not None else sim_end_scale_from_env())
    extrap = continuous_extrap_growth_enabled()
    t_frac = growth_time_frac(data, t_virt, bio_cfg=bio_cfg)
    spatial = cfg.spatial_rule or prior_rule_config_from_env()
    onset_eff = (
        float(onset_override) if onset_override is not None else float(cfg.global_onset_frac)
    )

    if onset_eff > 0 and t_frac < onset_eff and t_virt <= t_final:
        return torch.zeros(n, device=device)

    pool, risk_hand = _resolve_pool_risk(
        data, device=device, bio_cfg=bio_cfg, ceiling=ceiling, cfg=cfg, t_out=t_feat
    )
    if use_provided_risk:
        risk_eff = risk.reshape(-1).to(device=device, dtype=risk_hand.dtype)
    else:
        risk_eff = risk_hand

    if cfg.kind == "static_spatial":
        if cfg.localized is not None:
            return _localized_static_support(
                data, device=device, pool=pool, risk=risk_eff, cfg=cfg, t_out=t_out
            ).float()
        phi, _ = predict_phi_prior_rule(
            data, device, bio_cfg, rule=spatial, t_in=cfg.risk_flow_time, ceiling_hops=2
        )
        return phi.reshape(-1).float()

    if cfg.kind == "progressive_topk":
        u_grow = growth_u_from_t_frac(
            t_frac,
            onset_eff,
            extrap=extrap,
            sim_end_scale=scale,
            tau_comsol_end=1.0,
        )
        frac = min(
            _progressive_frac_from_growth_u(
                cfg, u_grow, extrap=extrap, sim_end_scale=scale
            ),
            0.95,
        )
        if cfg.localized is not None:
            loc = cfg.localized
            loc_scale = frac / max(float(cfg.end_frac), 0.01)
            top_base = float(loc.segment_top_frac)
            if extrap and loc_scale > 1.0 + 1e-6:
                from src.core_physics.clot_continuous_time import extrap_frac_headroom

                top_cap = min(0.95, top_base + extrap_frac_headroom())
                eff_top = min(top_base * loc_scale, top_cap)
            else:
                eff_top = min(top_base * loc_scale, top_base)
            eff_top = max(eff_top, 0.01)
            flag = segment_topk_mask(risk_eff, data, device, pool, loc, top_frac_override=eff_top)
        else:
            flag = _top_frac_mask(risk_eff, pool, frac)
        out = flag.float()
        if phi_prev is not None:
            out = torch.maximum(out, phi_prev.reshape(-1).float())
        return out

    if cfg.kind == "ranked_onset":
        if cfg.localized is not None:
            static = _localized_static_support(
                data, device=device, pool=pool, risk=risk_eff, cfg=cfg, t_out=t_out
            )
        else:
            static, _ = predict_phi_prior_rule(
                data, device, bio_cfg, rule=spatial, t_in=cfg.risk_flow_time, ceiling_hops=2
            )
            static = static.reshape(-1).bool()
        static = static.reshape(-1).bool() & pool
        if not bool(static.any().item()):
            return torch.zeros(n, device=device)
        r_static = risk_eff[static]
        rmin = float(r_static.min())
        rmax = float(r_static.max()) + 1e-12
        rel = (r_static - rmin) / (rmax - rmin)
        onset = torch.full((n,), 1.0, device=device)
        idx = static.nonzero(as_tuple=False).reshape(-1)
        onset[idx] = float(cfg.min_onset_frac) + float(cfg.onset_spread) * (1.0 - rel)
        return static.float() * (t_frac >= onset).float()

    if cfg.kind == "hop_growth":
        seed = (
            segment_topk_mask(risk_eff, data, device, pool, cfg.localized)
            if cfg.localized is not None
            else _top_frac_mask(risk_eff, pool, max(float(cfg.seed_frac), 0.01))
        )
        committed = seed.clone()
        ei = data.edge_index.to(device=device)
        thr = torch.quantile(risk_eff[pool], float(cfg.risk_floor_quantile)) if bool(pool.any()) else risk_eff.median()
        extra_hops = max(0, t_virt - t_final) if extrap else 0
        for _ in range(max(int(t_feat), 0) + extra_hops):
            frontier = graph_dilate_hops(committed, ei, max(int(cfg.hop_per_step), 1))
            committed = committed | (frontier & pool & (risk_eff >= thr))
        return committed.float()

    if cfg.kind == "neighbor_ac":
        seed = (
            segment_topk_mask(risk_eff, data, device, pool, cfg.localized)
            if cfg.localized is not None
            else _top_frac_mask(risk_eff, pool, max(float(cfg.seed_frac), 0.01))
        )
        committed = seed.clone()
        ei = data.edge_index.to(device=device)
        src, dst = ei[0], ei[1]
        extra_hops = max(0, t_virt - t_final) if extrap else 0
        for step in range(max(int(t_feat), 0) + extra_hops):
            if bool(committed.any().item()):
                nb = torch.zeros(n, device=device)
                nb.scatter_add_(0, src, committed[dst].float())
                nb.scatter_add_(0, dst, committed[src].float())
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
                deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
                nb_frac = nb / deg.clamp(min=1.0)
            else:
                nb_frac = torch.zeros(n, device=device)
            rq = torch.quantile(risk_eff[pool], float(cfg.neighbor_risk_q)) if bool(pool.any()) else risk_eff.median()
            catalytic = (nb_frac >= 0.34) & pool & (risk_eff >= rq)
            prog_k = min(_progressive_frac(cfg, step + 1, t_final), 0.95)
            if cfg.localized is not None:
                prog = segment_topk_mask(risk_eff, data, device, pool, cfg.localized) & (
                    risk_eff >= torch.quantile(risk_eff[pool], 1.0 - prog_k)
                )
            else:
                prog = _top_frac_mask(risk_eff, pool, prog_k)
            committed = committed | catalytic | prog
        return committed.float()

    if cfg.kind == "threshold_accum":
        raise ValueError("threshold_accum is stateful; use rollout_temporal_phi")

    raise ValueError(f"unknown temporal rule kind {cfg.kind}")


def _rollout_threshold_accum(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
) -> dict[int, torch.Tensor]:
    n = int(data.num_nodes)
    n_times = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    ei = data.edge_index.to(device=device)
    src, dst = ei[0], ei[1]

    accum = torch.zeros(n, device=device)
    committed = torch.zeros(n, dtype=torch.bool, device=device)
    phi_by_t: dict[int, torch.Tensor] = {}

    gain = float(cfg.accum_gain)
    thr = max(float(cfg.accum_threshold), 1e-6)
    sw = float(cfg.accum_split_wall)
    sl = float(cfg.accum_split_lumen)
    onset = float(cfg.global_onset_frac)

    for t_out in range(0, n_times, max(int(time_stride), 1)):
        t_frac = _time_frac_at_index(data, t_out)
        if onset > 0.0 and t_frac < onset:
            phi_by_t[int(t_out)] = committed.float()
            continue

        pool, risk_eff = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg,
            t_out=t_out,
        )
        active = pool & (~committed)
        if bool(active.any().item()) and bool(pool.any().item()):
            rp = risk_eff[pool]
            rmin = float(rp.min())
            rmax = float(rp.max()) + 1e-12
            rnorm = torch.zeros_like(risk_eff)
            rnorm[pool] = (risk_eff[pool] - rmin) / (rmax - rmin)
            accum[active] = accum[active] + rnorm[active] * gain

        newly = active & (accum >= thr)
        if bool(newly.any().item()):
            committed = committed | newly
            accum[newly] = 0.0
            budget_w = thr * sw
            budget_l = thr * sl
            new_src = newly[src]
            wall_e = new_src & pool[dst]
            lumen_e = new_src & (~pool[dst])
            if bool(wall_e.any().item()):
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src[wall_e], torch.ones_like(src[wall_e], dtype=torch.float32))
                w_amt = budget_w / deg[src[wall_e]].clamp(min=1.0)
                accum.scatter_add_(0, dst[wall_e], w_amt)
            if bool(lumen_e.any().item()):
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src[lumen_e], torch.ones_like(src[lumen_e], dtype=torch.float32))
                l_amt = budget_l / deg[src[lumen_e]].clamp(min=1.0)
                accum.scatter_add_(0, dst[lumen_e], l_amt)

        phi_by_t[int(t_out)] = committed.float()
    return phi_by_t


def rollout_temporal_phi(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
    sim_end_scale: float | None = None,
) -> dict[int, torch.Tensor]:
    del phys_cfg
    if cfg.kind == "threshold_accum":
        return _rollout_threshold_accum(
            data, cfg, device=device, bio_cfg=bio_cfg, time_stride=time_stride
        )
    from src.core_physics.clot_continuous_time import feature_time_index, rollout_time_indices

    n_times = int(data.y.shape[0])
    t_indices = rollout_time_indices(data, time_stride=time_stride, sim_end_scale=sim_end_scale)
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    scale = float(sim_end_scale if sim_end_scale is not None else 1.0)
    for t_out in t_indices:
        pool, risk = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg,
            t_out=feature_time_index(data, int(t_out)),
        )
        phi = predict_phi_temporal_at_time(
            data,
            t_out,
            device=device,
            bio_cfg=bio_cfg,
            cfg=cfg,
            ceiling=ceiling,
            risk=risk,
            phi_prev=phi_prev,
            t_final=t_final,
            sim_end_scale=scale,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi
    return phi_by_t


