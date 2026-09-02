"""Typed defaults for Stage-A production / finetune launchers.

Process/IO env keys only (training recipe lives in train_kinematics_predictor).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.paths import get_project_root

PRODUCTION_OUTPUT_DIR = Path("outputs/kinematics/production_allfix")
CLINICAL_OUTPUT_DIR = Path("outputs/kinematics/clinical_anchor_finetune")
PROMOTED_BEST_PATH = Path("outputs/kinematics/kinematics_best.pth")
CLINICAL_ANCHOR_DIR = Path("data/processed/graphs_kinematics_anchors/carreau")

# Architecture toggles shared by production + finetune legs.
ALLFIX_ARCH_ENV: dict[str, str] = {
    "KINEMATICS_PHYS_GAT_PRIORS_MULTIPLY_BEFORE_ADDITIVE": "1",
    "KINEMATICS_BC_ENVELOPE": "1",
    "KINEMATICS_BC_LAMBDA": "10.0",
    "KINEMATICS_WSS_FUSE": "1",
    "KINEMATICS_FOURIER_LEARNABLE": "1",
    "KINEMATICS_VAL_EVERY": "1",
}

SKIP_LBFGS_FLAG = ".skip_lbfgs_after_crash"
STATE_LATEST = "kinematics_state_latest.pth"
CKPT_LATEST = "kinematics_ckpt_latest.pth"
BEST_CKPT = "kinematics_best.pth"


def _abs(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else get_project_root() / p


def has_clinical_anchor_packs() -> bool:
    root = _abs(CLINICAL_ANCHOR_DIR)
    if not root.is_dir():
        return False
    return any(root.glob("patient*.pt"))


def bind_env(updates: dict[str, str | None]) -> None:
    """Set env keys; ``None`` removes the variable."""
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[str(key)] = str(value)


def bind_quiet(*, quiet: bool) -> None:
    if quiet:
        bind_env(
            {
                "KINEMATICS_QUIET": "1",
                "KINEMATICS_VAL_PROGRESS": "0",
                "KINEMATICS_TQDM": "0",
            }
        )
    else:
        bind_env(
            {
                "KINEMATICS_QUIET": None,
                "KINEMATICS_VAL_PROGRESS": None,
                "KINEMATICS_TQDM": None,
            }
        )


def bind_allfix_arch(*, bc_lambda: str = "10.0") -> None:
    env = dict(ALLFIX_ARCH_ENV)
    env["KINEMATICS_BC_LAMBDA"] = str(bc_lambda)
    bind_env(env)


@dataclass
class FoundationConfig:
    """Phase 1: synthetic foundation on production_allfix."""

    output_dir: Path = field(default_factory=lambda: _abs(PRODUCTION_OUTPUT_DIR))
    epochs: int = 100
    adam_epochs: int = 85
    stage1_end: int = 40
    stage2_end: int = 60
    hard_mining_start: int = 16
    accum_steps: int = 2
    graph_cap: int = 0
    seed: int = 42
    fresh: bool = False
    quiet: bool = False
    max_attempts: int = 50
    retry_sleep_s: float = 5.0

    def bind_process_env(self) -> None:
        bind_allfix_arch()
        bind_env({"KINEMATICS_OUTPUT_DIR": str(self.output_dir.as_posix())})
        bind_quiet(quiet=self.quiet)
        if self.graph_cap > 0:
            bind_env({"KINEMATICS_GRAPH_CAP": str(self.graph_cap)})
        else:
            bind_env({"KINEMATICS_GRAPH_CAP": None})


@dataclass
class SyntheticPolishConfig:
    """Phase 2: ContinuityFocus Carreau finetune on production_allfix."""

    output_dir: Path = field(default_factory=lambda: _abs(PRODUCTION_OUTPUT_DIR))
    resume: str = "best"
    finetune_lr: float = 5e-6
    finetune_epochs: int = 40
    hard_mining_start: int = 20
    continuity_focus: bool = True
    try_lbfgs: bool = False
    quiet: bool = False

    def bind_process_env(self) -> None:
        bind_allfix_arch(bc_lambda="12.0" if self.continuity_focus else "10.0")
        bind_env(
            {
                "KINEMATICS_OUTPUT_DIR": str(self.output_dir.as_posix()),
                "KINEMATICS_GRAPH_CAP": None,
            }
        )
        bind_quiet(quiet=self.quiet)
        if self.try_lbfgs:
            bind_env({"KINEMATICS_SKIP_LBFGS": None})
        else:
            bind_env({"KINEMATICS_SKIP_LBFGS": "1"})


@dataclass
class ClinicalFinetuneConfig:
    """Phase 3: patient anchor finetune."""

    resume: Path = field(
        default_factory=lambda: _abs(PRODUCTION_OUTPUT_DIR / BEST_CKPT)
    )
    holdout: str = "patient007"
    finetune_epochs: int = 25
    finetune_lr: float = 5e-6
    synthetic_cap: int = 120
    clinical_boost: float = 10.0
    output_dir: Path = field(default_factory=lambda: _abs(CLINICAL_OUTPUT_DIR))

    def bind_process_env(self) -> None:
        bind_allfix_arch()
        bind_env(
            {
                "KINEMATICS_SKIP_LBFGS": "1",
                "KINEMATICS_INCLUDE_PATIENT_ANCHORS": "1",
                "KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS": self.holdout,
                "KINEMATICS_CLINICAL_ANCHOR_BOOST": str(self.clinical_boost),
                "KINEMATICS_OUTPUT_DIR": str(self.output_dir.as_posix()),
                "KINEMATICS_GRAPH_CAP": str(self.synthetic_cap),
                "KINEMATICS_SYNTHETIC_VAL_RATIO": "0.15",
                "KINEMATICS_SYNTHETIC_VAL_MIN": "20",
                "KINEMATICS_SYNTHETIC_VAL_MIN_L2": "6",
                "KINEMATICS_DUAL_PROMOTION_GATES": "1",
                "KINEMATICS_GATE_MAX_PATIENT_REL_L2": "0.25",
                "KINEMATICS_GATE_MAX_SYNTHETIC_REL_L2": "0.20",
                "KINEMATICS_GATE_MAX_SYNTHETIC_L2_REL_L2": "0.22",
                "KINEMATICS_QUIET": "1",
                "KINEMATICS_VAL_PROGRESS": "0",
            }
        )


@dataclass
class LadderConfig:
    """Full Stage-A ladder: foundation -> polish -> clinical -> promote."""

    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    polish: SyntheticPolishConfig = field(default_factory=SyntheticPolishConfig)
    clinical: ClinicalFinetuneConfig = field(default_factory=ClinicalFinetuneConfig)
    skip_foundation: bool = False
    skip_synthetic_polish: bool = False
    skip_clinical_anchors: bool = False
    skip_promote: bool = False
    require_clinical: bool = False
    resume_after_foundation: Path | None = None


@dataclass
class ProductionRunConfig:
    """Top-level entry used by scripts/run_kinematics_production.py."""

    ladder: LadderConfig = field(default_factory=LadderConfig)
    foundation_only: bool = False


__all__ = [
    "ALLFIX_ARCH_ENV",
    "BEST_CKPT",
    "CKPT_LATEST",
    "CLINICAL_ANCHOR_DIR",
    "CLINICAL_OUTPUT_DIR",
    "ClinicalFinetuneConfig",
    "FoundationConfig",
    "LadderConfig",
    "PRODUCTION_OUTPUT_DIR",
    "PROMOTED_BEST_PATH",
    "ProductionRunConfig",
    "SKIP_LBFGS_FLAG",
    "STATE_LATEST",
    "SyntheticPolishConfig",
    "bind_allfix_arch",
    "bind_env",
    "bind_quiet",
    "has_clinical_anchor_packs",
]
