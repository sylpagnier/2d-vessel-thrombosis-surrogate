"""Differentiable gelation readout for closed-loop species pushforward (Phase 3).

Maps accumulated FI/Mat log-ND on the ceiling band through soft Mat/FI gelation
sigmoids (deployable thresholds from ``BiochemConfig``), then optional mu_eff
coupling. Used as an auxiliary loss so species deltas feel downstream viscosity
threshold effects during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    clot_phi_physics_mu_ratio_max,
    gt_mu_anchor_cap_si,
    mat_si_for_gelation_from_log1p,
    species_log1p_nd_to_si,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.training.biochem_species_scope import (
    FI_CHANNEL,
    MAT_CHANNEL,
    pushforward_state_bulk_indices,
)
from src.utils.rheology import multiplicative_clot_mu_eff_nd, phi_clot_from_mat_fi


def gelation_temperature_scale() -> float:
    """Override sigmoid sharpness (1.0 = use biochem gnode temps)."""
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(float(rt.gelation.gelation_temp_scale), 0.1)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_GELATION_TEMP_SCALE") or "1.0").strip()
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return 1.0


def differentiable_mu_eff_from_species12(
    species_log12: torch.Tensor,
    mu_carreau_si: torch.Tensor,
    phi_clot: torch.Tensor,
    bio_cfg: BiochemConfig,
    *,
    gelation_beta: float | None = None,
) -> torch.Tensor:
    """mu_eff = mu_carreau * (1 + beta * (ratio_max - 1) * phi_clot).

    ``gelation_beta=None`` means beta 1, i.e. the historical readout. Passing beta here
    (not only at the graded readout) is what puts the gain *inside* the closed loop:
    viscosity -> occlusion -> stagnation -> more clot.
    """
    ratio = max(float(clot_phi_physics_mu_ratio_max(bio_cfg)), 1.0)
    if gelation_beta is not None:
        # Fold beta into the excess-over-baseline so beta=1 is exactly a no-op.
        ratio = 1.0 + max(float(gelation_beta), 0.0) * (ratio - 1.0)
    mu_c = mu_carreau_si.reshape(-1).to(device=species_log12.device, dtype=species_log12.dtype)
    return multiplicative_clot_mu_eff_nd(mu_c, phi_clot, ratio).reshape(-1).clamp(min=1e-8)


@dataclass(frozen=True)
class SpeciesPhysicsCtx:
    data: object
    phys_cfg: PhysicsConfig
    bio_cfg: BiochemConfig
    node_idx: torch.Tensor
    time_window: list[int]
    rest_band: torch.Tensor
    mu_anchor_si: torch.Tensor
    wall_band: torch.Tensor | None = None
    lumen_band: torch.Tensor | None = None


