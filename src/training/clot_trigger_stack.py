"""Clot trigger stack (T0-T6): flow -> species -> phi commit (+ optional mu->kine loop).

Star 1 (T1): GT flow + GT species -> hybrid trigger (physics gelation + learned MLP).
Star 2: pred flow + GT species. Star 3: pred flow + pred species (biochem teacher).
Star 5: retrain deploy teacher (pred kine + FI/Mat). Star 6: T4/T5 + mu->kine coupling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_phi_simple import (
    ClotPhiStepBatch,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_forward_apply_region,
    clot_phi_hybrid_enabled,
    clot_phi_model_uses_mpnn,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    physics_mu_eff_si,
    physics_phi_from_mu,
)
from src.utils.paths import get_project_root


class ClotTriggerStar(str, Enum):
    T0_ORACLE = "t0"  # physics-only eval, no train
    T1_GT_INPUTS = "t1"  # GT flow + GT species, train hybrid trigger
    T2_PRED_FLOW = "t2"  # pred kine + GT species
    T3_DUMPED_SPECIES = "t3"  # pred kine + cached teacher species dump
    T4_LIVE_TEACHER = "t4"  # pred kine + live GraphSAGE species rollout (frozen global ckpt)
    T5_DEPLOY_TEACHER = "t5"  # retrain teacher (pred kine, FI/Mat) + pred-flow species dump
    T6_COUPLED = "t6"  # T4/T5 stack + phi/mu -> GINO-DEQ feedback each macro step


@dataclass
class ClotTriggerPaths:
    out_dir: Path
    ckpt_name: str = "clot_trigger_t1_best.pth"
    log_name: str = "clot_trigger_t1_train_log.jsonl"

    @property
    def ckpt_path(self) -> Path:
        return self.out_dir / self.ckpt_name


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def forward_physics_trigger_phi(
    step: ClotPhiStepBatch,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    species_log1p: torch.Tensor | None = None,
    use_soft: bool = True,
    apply_region: bool = True,
    time_index: int | None = None,
    mu_anchor_si: torch.Tensor | None = None,
    gelation_beta: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit Mat/FI gelation trigger (no learned head)."""
    sp = species_log1p if species_log1p is not None else step.species_log_gt
    mu_phys = cap_mu_eff_si(
        physics_mu_eff_si(
            step.mu_c_si,
            sp,
            bio_cfg,
            device=device,
            data=data,
            u_nd=step.u_flow_nd,
            v_nd=step.v_flow_nd,
            phys_cfg=phys_cfg,
            time_index=time_index,
            gelation_beta=gelation_beta,
        )
    )
    region = step.region if apply_region else None
    phi_phys = physics_phi_from_mu(
        mu_phys,
        step.mu_c_si,
        region,
        phys_cfg,
        soft=use_soft,
        mu_anchor_si=mu_anchor_si,
    )
    return phi_phys.reshape(-1), mu_phys.reshape(-1)


def apply_clot_trigger_deploy_env() -> None:
    """Deploy-faithful default: pred forward E(tau), fixed ceiling loss, growth-only GT labels."""
    apply_neighbor_band_trigger_env()
    os.environ.setdefault("CLOT_TRIGGER_IC_PHI_ZERO", "1")
    os.environ["CLOT_PHI_LOSS_SCOPE"] = "ceiling"
    os.environ["CLOT_PHI_GROWTH_SEED"] = "pred"
    os.environ["CLOT_PHI_CLOT_SEED_SOURCE"] = "wall"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "0"
    os.environ["CLOT_PHI_ORACLE_MU"] = "0"
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "3")
    apply_clot_trigger_nucleation_env()


def apply_clot_trigger_honest_env() -> None:
    """Alias for deploy-faithful env (legacy name kept for launchers/tests)."""
    apply_clot_trigger_deploy_env()


def clot_phi_trigger_rollout_enabled() -> bool:
    return _env_bool("CLOT_PHI_TRIGGER_ROLLOUT", False)


def clot_phi_trigger_rollout_detach_prev() -> bool:
    raw = (os.environ.get("CLOT_PHI_TRIGGER_TBPTT_DETACH") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def apply_clot_trigger_nucleation_env() -> None:
    """Deploy-faithful forward: wall + 1-hop from **predicted** commits each step."""
    os.environ["CLOT_TRIGGER_NUCLEATION"] = "1"
    os.environ["CLOT_TRIGGER_FORWARD_SEED"] = "pred"
    os.environ.setdefault("CLOT_V2_NUCLEATION_HOPS", "1")
    os.environ.setdefault("CLOT_TRIGGER_COMMIT_THRESH", "0.5")
    os.environ.setdefault("CLOT_TRIGGER_DGAMMA_WALL_SEED", "0")


def apply_neighbor_band_trigger_env() -> None:
    """Shared neighbor-band geometry defaults (seed/loss scope set by honest/oracle helpers)."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_WALL_HOPS": "1",
        "CLOT_PHI_CLOT_HOPS": "2",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_MU_CAP_SI": "0.10",
        "CLOT_PHI_THRESH_SI": "0.055",
        "CLOT_PHI_SHEAR_MIN_FRAC": "0",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def apply_star2_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
) -> None:
    """T2: frozen T1 trigger ckpt, pred GINO-DEQ flow + GT species."""
    apply_clot_trigger_honest_env()
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T2_PRED_FLOW.value
    os.environ["CLOT_PHI_VEL_SOURCE"] = "kinematics"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ["CLOT_PHI_KINE_CKPT"] = kine_ckpt
    os.environ["CLOT_PHI_KINE_TF"] = "0"
    os.environ["CLOT_PHI_ROLLOUT"] = "0"
    os.environ["CLOT_PHI_ORACLE_MU"] = "0"
    os.environ["CLOT_PHI_SPECIES_FEATURES"] = "1"
    os.environ["CLOT_PHI_JOINT_BIO"] = "0"
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ["CLOT_PHI_HYBRID"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND"] = "1"
    os.environ["CLOT_PHI_PHYSICS_BLEND_ALPHA"] = "0.55"
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "4"
    os.environ["CLOT_PHI_SOFT_LABELS"] = "1"


def reset_star2_kinematics_cache() -> None:
    """Clear cached steady GINO-DEQ uv between anchors."""
    from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache

    reset_temporal_kinematics_cache()


def default_teacher_checkpoint_path() -> Path:
    root = get_project_root()
    for rel in (
        "outputs/biochem/biochem_teacher_best_high_mu.pth",
        "outputs/biochem/biochem_teacher_last.pth",
        "outputs/biochem/sweep_mu_complexity_6h/FULL_step2/biochem_teacher_best_high_mu.pth",
    ):
        path = root / rel
        if path.is_file():
            return path
    return root / "outputs/biochem/biochem_teacher_best_high_mu.pth"


def default_dumped_species_anchor_dir() -> Path:
    return get_project_root() / "outputs" / "biochem" / "anchors_teacher_species"


@dataclass(frozen=True)
class ClotTriggerT5DeployPaths:
    """Artifacts for Star 5 deploy teacher retrain + pred-flow species cache."""

    out_root: Path

    @property
    def teacher_deploy(self) -> Path:
        return self.out_root / "biochem_teacher_deploy.pth"

    @property
    def manifest(self) -> Path:
        return self.out_root / "manifest.json"

    @property
    def eval_live_json(self) -> Path:
        return self.out_root / "t5_deploy_live.json"

    @property
    def eval_dumped_json(self) -> Path:
        return self.out_root / "t5_deploy_dumped.json"


def default_t5_deploy_paths() -> ClotTriggerT5DeployPaths:
    return ClotTriggerT5DeployPaths(
        out_root=get_project_root() / "outputs" / "biochem" / "clot_trigger" / "t5_deploy_teacher"
    )


def default_t5_predkine_species_dump_dir() -> Path:
    return get_project_root() / "outputs" / "biochem" / "anchors_teacher_species_predkine"


def default_t5_deploy_teacher_checkpoint_path() -> Path:
    """Promoted T5 deploy teacher, else global biochem best."""
    path = default_t5_deploy_paths().teacher_deploy
    if path.is_file():
        return path
    return default_teacher_checkpoint_path()


def apply_star3_dumped_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    dump_dir: str | None = None,
) -> None:
    """T3: pred kine + cached teacher species (``dump_teacher_species_to_anchors``)."""
    apply_star2_eval_env(kine_ckpt=kine_ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T3_DUMPED_SPECIES.value
    os.environ["CLOT_TRIGGER_SPECIES_SOURCE"] = "dumped"
    if dump_dir:
        os.environ["CLOT_TRIGGER_DUMPED_ANCHOR_DIR"] = dump_dir


def apply_star4_live_teacher_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
) -> None:
    """T4: pred kine + live GraphSAGE species rollout (slow; overnight path)."""
    apply_star2_eval_env(kine_ckpt=kine_ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T4_LIVE_TEACHER.value
    os.environ["CLOT_TRIGGER_SPECIES_SOURCE"] = "live"
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_GT_KINE_SKIP_DEQ"] = "0"
    os.environ["BIOCHEM_VAL_TIME_STRIDE"] = "1"
    os.environ.setdefault("BIOCHEM_DATA_BIO_SPECIES_SCOPE", "fi_mat")
    os.environ.setdefault("BIOCHEM_DATALOADER_WORKERS", "0")
    if teacher_ckpt:
        os.environ["CLOT_TRIGGER_TEACHER_CKPT"] = teacher_ckpt


def apply_star5_deploy_teacher_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
) -> None:
    """T5 eval: pred kine + live/deploy-retrained GraphSAGE species (``BIOCHEM_GT_KINE_VEL=0``)."""
    ckpt = teacher_ckpt or str(default_t5_deploy_teacher_checkpoint_path())
    apply_star4_live_teacher_env(kine_ckpt=kine_ckpt, teacher_ckpt=ckpt)
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T5_DEPLOY_TEACHER.value


def apply_star5_deploy_dumped_eval_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    dump_dir: str | None = None,
) -> None:
    """T5 fast eval: pred kine + pred-flow species dump from deploy teacher."""
    apply_star3_dumped_env(
        kine_ckpt=kine_ckpt,
        dump_dir=dump_dir or str(default_t5_predkine_species_dump_dir()),
    )
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T5_DEPLOY_TEACHER.value


def apply_star6_coupled_env(
    *,
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth",
    teacher_ckpt: str | None = None,
    species_live: bool = True,
    dump_dir: str | None = None,
) -> None:
    """T6: T4/T5 species + serial phi/mu -> GINO-DEQ MU_PRIOR feedback.

    Uses frozen T1 trigger (in_dim=5): no carry features in MLP; mu feedback via
    ``CLOT_PHI_FIXED_MU_FROM_PHI`` + ``KinematicsUvProvider`` (Step 5b pattern).
    Species remain offline/live teacher output (not re-rolled per macro step).
    """
    t5_teacher = str(default_t5_deploy_teacher_checkpoint_path())
    resolved_teacher = teacher_ckpt or t5_teacher
    if species_live:
        apply_star5_deploy_teacher_eval_env(kine_ckpt=kine_ckpt, teacher_ckpt=resolved_teacher)
    else:
        predkine = Path(dump_dir) if dump_dir else default_t5_predkine_species_dump_dir()
        if predkine.is_dir() and any(predkine.glob("*.pt")):
            apply_star5_deploy_dumped_eval_env(kine_ckpt=kine_ckpt, dump_dir=str(predkine))
        else:
            apply_star3_dumped_env(
                kine_ckpt=kine_ckpt,
                dump_dir=dump_dir or str(default_dumped_species_anchor_dir()),
            )
    os.environ["CLOT_TRIGGER_STAR"] = ClotTriggerStar.T6_COUPLED.value
    os.environ["CLOT_PHI_ROLLOUT"] = "1"
    os.environ["CLOT_PHI_ROLLOUT_DETACH"] = "1"
    os.environ["CLOT_PHI_CARRY_PHI"] = "0"
    os.environ["CLOT_PHI_CARRY_LOG_MU"] = "0"
    os.environ["CLOT_PHI_FIXED_MU_FROM_PHI"] = "1"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled"
    os.environ["CLOT_PHI_VEL_SOURCE"] = "kinematics"


def reset_star6_caches() -> None:
    """Clear frozen-kine, coupled-uv, and rollout GINO-DEQ caches."""
    from src.core_physics.clot_coupled_rollout import reset_coupled_uv_cache
    from src.core_physics.clot_phi_rollout import reset_rollout_kine_provider

    reset_star3_caches()
    reset_coupled_uv_cache()
    reset_rollout_kine_provider()


# Backward-compatible aliases (pre-T6 rename).
apply_star5_coupled_env = apply_star6_coupled_env
reset_star5_caches = reset_star6_caches


def reset_star3_caches() -> None:
    reset_star2_kinematics_cache()


