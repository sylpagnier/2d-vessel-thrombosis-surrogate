"""Survey COMSOL clot nodes across biochem anchor graphs (where / when they form)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_kinematics_fields import (
    adjacent_band_mask,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
def _graph_props(data, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        u_ref = data.u_ref.to(device=device, dtype=torch.float32).reshape(-1)[:1]
        d_bar = data.d_bar.to(device=device, dtype=torch.float32).reshape(-1)[:1]
    else:
        u_ref = torch.as_tensor(data.u_ref, device=device, dtype=torch.float32).reshape(1)
        d_bar = torch.as_tensor(data.d_bar, device=device, dtype=torch.float32).reshape(1)
    return {"u_ref": u_ref, "d_bar": d_bar}


def _wall_mask(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        return data.mask_wall.view(-1).to(device=device).bool()
    return torch.zeros(n, dtype=torch.bool, device=device)


@dataclass
class AnchorClotSurvey:
    stem: str
    n_nodes: int
    n_times: int
    t_final_s: float
    mu_floor_si: float
    n_clot_strict_t0: int
    n_clot_strict_tfinal: int
    n_clot_p90_tfinal: int
    first_any_clot_time_s: float | None
    first_any_clot_frac: float | None
    pct_clot_on_wall_tfinal: float
    pct_clot_adjacent_tfinal: float
    pct_clot_off_wall_tfinal: float
    pct_clot_bulk_tfinal: float
    pct_clot_near_wall_sdf_tfinal: float
    median_sdf_nd_clot_tfinal: float
    median_sdf_nd_nonclot_adjacent: float
    dgamma_dx_clot_mean_tfinal: float
    dgamma_dx_non_mean_tfinal: float
    dshear_ds_clot_mean_tfinal: float
    gamma_clot_mean_tfinal: float
    prior_clot_mean_tfinal: float
    prior_non_mean_tfinal: float
    suggested_dx_thresh_p10: float
    inception_dgamma_dx_mean: float
    inception_n_nodes: int
    notes: list[str] = field(default_factory=list)


def discover_anchor_paths(anchor_dir: Path | None = None) -> list[Path]:
    root = anchor_dir or (Path(__file__).resolve().parents[2] / "data" / "processed" / "graphs_biochem_anchors")
    if not root.is_dir():
        return []
    return sorted(root.glob("comsol*.pt"))


