"""Temporal clot-trigger rollout with deploy-faithful nucleation projection.

Forward contract (all T0+ stars at eval/deploy):
  phi_raw  = physics gelation and/or learned head (full mesh features)
  E(tau)   = wall @ tau=0 OR 1-hop from prior **predicted** commits
  phi(tau) = project_phi_with_nucleation(phi_raw, phi_prev, E(tau))

Loss / F1 support B_t is separate (``resolve_clot_loss_mask``); it may use GT
commits during T0-T2 training diagnostics but must not drive forward E(tau).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import growth_seed_mode, pred_clot_mask
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_phi_simple import build_clot_phi_step, clot_phi_model_uses_mpnn

GrowthSeed = Literal["gt", "pred"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def clot_trigger_nucleation_enabled() -> bool:
    """Apply ``project_phi_with_nucleation`` each macro step (default on)."""
    return _env_bool("CLOT_TRIGGER_NUCLEATION", True)


def clot_trigger_forward_seed_mode() -> GrowthSeed:
    """Seed for forward envelope E(tau): ``pred`` (deploy) or ``gt`` (oracle upper bound)."""
    raw = (os.environ.get("CLOT_TRIGGER_FORWARD_SEED") or "pred").strip().lower()
    if raw in ("gt", "oracle", "ground_truth"):
        return "gt"
    return "pred"


def clot_trigger_commit_thresh() -> float:
    try:
        return float(os.environ.get("CLOT_TRIGGER_COMMIT_THRESH", "0.5") or "0.5")
    except ValueError:
        return 0.5


def clot_trigger_use_dgamma_wall_seed() -> bool:
    """At tau=0, use dgamma wall band instead of geometry wall (debug only)."""
    return _env_bool("CLOT_TRIGGER_DGAMMA_WALL_SEED", False)


def clot_trigger_train_soft_commit() -> bool:
    """Training: avoid absorbing hard commits so BCE gradients reach the MLP head."""
    return _env_bool("CLOT_TRIGGER_TRAIN_SOFT_COMMIT", True)


def _project_step_phi(
    phi_raw: torch.Tensor,
    phi_prev: torch.Tensor | None,
    data,
    time_index: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    phi_pred_by_time: dict[int, torch.Tensor],
    growth_seed: GrowthSeed,
    hard_commit: bool | None = None,
) -> torch.Tensor:
    if not clot_trigger_nucleation_enabled():
        return phi_raw.reshape(-1).float()
    if hard_commit is None:
        hc = clot_trigger_commit_thresh() > 0 and not clot_trigger_train_soft_commit()
    else:
        hc = bool(hard_commit)
    elig = resolve_nucleation_eligibility(
        data,
        int(time_index),
        device,
        phys_cfg,
        bio_cfg,
        growth_seed=growth_seed,
        phi_pred_by_time=phi_pred_by_time if growth_seed == "pred" else None,
        commit_thresh=clot_trigger_commit_thresh(),
        use_dgamma_wall_seed=clot_trigger_use_dgamma_wall_seed(),
    )
    return project_phi_with_nucleation(
        phi_raw,
        phi_prev,
        elig,
        commit_thresh=clot_trigger_commit_thresh(),
        hard_commit=hc,
    )


def clot_phi_trigger_rollout_enabled() -> bool:
    return _env_bool("CLOT_PHI_TRIGGER_ROLLOUT", False)


def clot_phi_trigger_rollout_detach_prev() -> bool:
    raw = (os.environ.get("CLOT_PHI_TRIGGER_TBPTT_DETACH") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


