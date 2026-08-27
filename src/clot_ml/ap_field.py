"""v7: A learned residual on the time-varying wall AP field.

CONTEXT (docs/WOUND_PROGRESS.md §18.3). The upwind advective model (wall_ap_renewal.py)
gives a time-varying AP field, but diag_wall_ap_renewal.py proved it depletes AP in the
wrong spatial locations on complex wound vessels (AP_owner AUC drops to 0.22, far-field
is capped at 0.86).

This module trains a ClotGNN residual on top of that base field. At initialization (zero
residual), it recovers the upwind ODE exactly. Training targets COMSOL's AP_log1p_nd
on the solid boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Same sparse sampling as v6 mat_field (horizon crossing)
N_TIME_SAMPLES: int = 16

# We use 3 extra channels for the AP residual:
# t_frac: fraction of simulation time
# ap_ode: the base ND upwind renewal AP field
# gate: the static t=0 gate (crucial for consumption context)
EXTRA_CHANNELS: tuple[str, ...] = ("t_frac", "ap_ode", "gate")

# The AP_log1p_nd target is ~5e-8. We scale it by 1e7 during training to keep
# gradients in float32 range and avoid Adam drifting the residual.
AP_TARGET_SCALE: float = 1e7


@dataclass
class ApFieldConfig:
    """Configuration for the AP residual ClotGNN."""
    dim: int = 96
    layers: int = 6
    drop: float = 0.1
    lr: float = 2e-3
    wd: float = 1e-4
    epochs: int = 60
    # The AP field is purely a regression target on the solid boundary.
    reg_w: float = 1.0
    seed: int = 0
    train_stems: tuple[str, ...] = field(default_factory=tuple)


def sample_time_indices(T: int, k: int = N_TIME_SAMPLES) -> np.ndarray:
    T = int(T)
    if T <= k:
        return np.arange(T, dtype=np.int64)
    idx = np.unique(np.linspace(0, T - 1, int(k)).round().astype(np.int64))
    if idx[-1] != T - 1:
        idx = np.append(idx, T - 1)
    return idx


def build_ap_field_entry(data, bio_cfg, *, flow: str = "gt",
                         n_times: int = N_TIME_SAMPLES) -> dict:
    """Extract one vessel's AP training entry (t=0 upwind base + GT AP_log1p_nd target)."""
    from src.clot_ml.locked import build_sample
    from src.core_physics.physics_wall_model import PER_M3_TO_PER_CM3, deposition_gate, t0_flow_fields
    from src.core_physics.wall_ap_renewal import WallApRenewal, make_species_from_renewal

    # Base node features
    S = build_sample(data, bio_cfg, flow=flow, variant="v4")
    T = int(data.y.shape[0])
    ti = sample_time_indices(T, n_times)

    wall = np.asarray(S["wall"], dtype=bool)
    solid = np.asarray(S.get("solid", wall), dtype=bool)

    # 1. Base upwind-renewal AP field [T, N] (deploy-legal)
    f = t0_flow_fields(data, bio_cfg, hops=3, flow_source=flow)
    gate = deposition_gate(data, f, wall=wall, wound_source=True)
    renewal = WallApRenewal(renewal_scale=1.0)
    _, ap_traj_cgs = make_species_from_renewal(data, bio_cfg, f, renewal=renewal)
    
    # 2. Convert base CGS AP field to ND scale
    scales = bio_cfg.get_species_scales(device="cpu")
    ap_scale_cgs = float(scales[1]) * PER_M3_TO_PER_CM3
    # log1p(ap_cgs / ap_scale_cgs) matches the GT AP_log1p_nd transform exactly
    ode_nd = np.log1p(np.maximum(ap_traj_cgs, 0.0) / ap_scale_cgs) * AP_TARGET_SCALE
    
    # 3. Ground truth target
    ap_idx = data.y_channel_names.split(",").index("AP_log1p_nd")
    gt_nd = data.y[:, :, ap_idx].detach().cpu().numpy() * AP_TARGET_SCALE

    pos = np.asarray(S["pos"], dtype=np.float64)
    ei = np.asarray(S["edge_index"])

    return dict(
        X=np.asarray(S["X"], dtype=np.float32),
        edge_index=ei.astype(np.int64),
        pos=pos.astype(np.float32),
        u=np.asarray(S["u"], dtype=np.float32),
        v=np.asarray(S["v"], dtype=np.float32),
        wall=wall,
        solid=solid,
        gate=gate.astype(np.float32),
        t_idx=ti.astype(np.int64),
        T=np.int64(T),
        # per-time fields; [K, N]
        ode_t=ode_nd[ti].astype(np.float32),
        gt_t=gt_nd[ti].astype(np.float32),
    )


def extra_channels(entry: dict, k: int, dev) -> torch.Tensor:
    """``[N, 3]`` time-varying input for ``ClotGNN.extra`` at time sample ``k``."""
    ode = np.asarray(entry["ode_t"][k], dtype=np.float32)
    gate = np.asarray(entry["gate"], dtype=np.float32)
    T = int(entry["T"])
    tf = float(entry["t_idx"][k]) / max(T - 1, 1)
    
    cols = np.stack([
        np.full(ode.shape, tf, dtype=np.float32),
        ode,
        gate,
    ], axis=1)
    return torch.tensor(cols, dtype=torch.float32, device=dev)


def build_static_graph(entry: dict, mu: np.ndarray, sd: np.ndarray, dev) -> dict:
    from src.clot_ml.gnn import edge_features

    ei = np.asarray(entry["edge_index"])
    pos = np.asarray(entry["pos"], dtype=np.float64)
    u, v = np.asarray(entry["u"]), np.asarray(entry["v"])
    h_edge = float(np.median(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)))
    ea = edge_features(pos, ei, u, v, h_edge)
    cos_s = ea[:, 4:5]
    t = lambda a, d=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=d, device=dev)
    X = (np.asarray(entry["X"], dtype=np.float32) - mu) / sd
    return dict(
        x=t(X), ei=t(ei, torch.long), ea=t(ea),
        w_up=t(np.clip(cos_s, 0.0, None)), w_dn=t(np.clip(-cos_s, 0.0, None)),
        solid=t(np.asarray(entry["solid"], dtype=np.float32)),
        n=int(len(entry["solid"])),
    )


def make_model(in_dim: int, edim: int, cfg: ApFieldConfig):
    from src.clot_ml.gnn import ClotGNN

    return ClotGNN(in_dim, edim, dim=cfg.dim, layers=cfg.layers, drop=cfg.drop,
                   extra_dim=len(EXTRA_CHANNELS))


@torch.no_grad()
def predict_entry(model, entry: dict, mu, sd, dev, k: int) -> np.ndarray:
    """Returns the pure regression output at every node for time sample ``k``."""
    model.eval()
    g = build_static_graph(entry, mu, sd, dev)
    ex = extra_channels(entry, k, dev)
    base = torch.tensor(np.asarray(entry["ode_t"][k], dtype=np.float32), device=dev)
    # logit is discarded; we only need regression
    _, reg = model(g["x"], g["ei"], g["ea"], g["w_up"], g["w_dn"], base, extra=ex)
    return reg.cpu().numpy()


def correct_ap_cgs_trajectory(
    model,
    sample: dict,
    gate: np.ndarray,
    ap_cgs: np.ndarray,
    ap_scale_cgs: float,
    mu: np.ndarray,
    sd: np.ndarray,
    dev,
) -> np.ndarray:
    """Apply a ClotGNN residual to a live CGS AP trajectory.

    Untrained (zero-init residual) is the physics at the sampled times; values between
    samples are linearly interpolated.  Absent model is a caller concern -- this always
    runs the net.  Used by ``clot_ml_v0`` when a v7 checkpoint is on the artifact; without
    one the chemistry ODE stays the upwind-renewal field (docs/WOUND_PROGRESS.md 18.3).
    """
    T, _N = ap_cgs.shape
    t_idx = sample_time_indices(T)
    ode_nd = np.log1p(np.maximum(ap_cgs, 0.0) / float(ap_scale_cgs)) * AP_TARGET_SCALE
    solid = np.asarray(sample.get("solid", sample["wall"]), dtype=bool)
    entry = dict(
        X=np.asarray(sample["X"], dtype=np.float32),
        edge_index=np.asarray(sample["edge_index"]),
        pos=np.asarray(sample["pos"], dtype=np.float32),
        u=np.asarray(sample["u"], dtype=np.float32),
        v=np.asarray(sample["v"], dtype=np.float32),
        solid=solid,
        gate=np.asarray(gate, dtype=np.float32),
        t_idx=t_idx.astype(np.int64),
        T=np.int64(T),
        ode_t=ode_nd[t_idx].astype(np.float32),
        gt_t=np.zeros((len(t_idx), ode_nd.shape[1]), dtype=np.float32),
    )
    corrected = ode_nd.copy()
    for k, ti in enumerate(t_idx):
        corrected[int(ti)] = predict_entry(model, entry, mu, sd, dev, k)
    for i in range(len(t_idx) - 1):
        a, b = int(t_idx[i]), int(t_idx[i + 1])
        gap = b - a
        if gap <= 1:
            continue
        w = np.linspace(0.0, 1.0, gap + 1, dtype=np.float64)[1:-1]
        left, right = corrected[a], corrected[b]
        for j, ww in enumerate(w, start=1):
            corrected[a + j] = (1.0 - ww) * left + ww * right
    return np.expm1(corrected / AP_TARGET_SCALE) * float(ap_scale_cgs)
