"""Compatibility shim for legacy t0 mu physics helpers."""

from __future__ import annotations

import torch

from src.config import STATE_CHANNEL_MU_EFF_ND, BiochemConfig, PhysicsConfig


def gt_mu_anchor_cap_si(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return GT capped mu in SI units from anchor timeline."""
    del bio_cfg
    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=dev, dtype=torch.float32)
    mu_nd = y[:, STATE_CHANNEL_MU_EFF_ND]
    mu_si = phys_cfg.viscosity_nd_to_si(mu_nd)
    return torch.clamp(mu_si, min=1e-8)


def gt_clot_phi_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Binary GT clot target: growth-only ``relu(mu_eff(t) - mu_eff(t=0)) >= thresh``."""
    from src.core_physics.clot_phi_simple import (
        cap_mu_eff_si,
        clot_phi_thresh_si,
        gt_mu_anchor_cap_si,
    )

    dev = device or (data.y.device if torch.is_tensor(data.y) else torch.device("cpu"))
    ti = max(0, min(int(time_index), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=dev, dtype=torch.float32)
    mu = cap_mu_eff_si(phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, dev).reshape(-1)
    growth = (mu - anchor).clamp(min=0.0)
    return (growth >= clot_phi_thresh_si(phys_cfg)).to(dtype=torch.float32)

