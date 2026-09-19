"""Step 5b/5c: per-macro-step mu -> GINO-DEQ feedback for temporal clot rollout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _resolve_pool_risk,
    _resolve_uv_for_temporal_risk,
    predict_phi_temporal_at_time,
    reset_temporal_kinematics_cache,
    temporal_vel_source,
)

if TYPE_CHECKING:
    pass

_coupled_uv: tuple[torch.Tensor, torch.Tensor] | None = None
_coupled_uv_key: tuple[int, int, int] | None = None


def _graph_key(data) -> tuple[int, int, int]:
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def reset_coupled_uv_cache() -> None:
    global _coupled_uv, _coupled_uv_key
    _coupled_uv = None
    _coupled_uv_key = None


def get_coupled_uv(data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _coupled_uv is None or _coupled_uv_key != _graph_key(data):
        return None
    u, v = _coupled_uv
    return u.to(device=device), v.to(device=device)


    


