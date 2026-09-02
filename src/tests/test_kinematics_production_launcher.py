"""Tests for Stage-A production launcher helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.kinematics_production_config import (
    ALLFIX_ARCH_ENV,
    FoundationConfig,
    bind_allfix_arch,
    bind_env,
)
from src.training.kinematics_production_runner import (
    checkpoint_next_epoch,
    clear_foundation_checkpoints,
)


def test_checkpoint_next_epoch_from_state(tmp_path: Path) -> None:
    ckpt = {"epoch": 79}
    torch.save(ckpt, tmp_path / "kinematics_state_latest.pth")
    assert checkpoint_next_epoch(tmp_path) == 80


def test_clear_foundation_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "kinematics_state_latest.pth").write_text("x", encoding="utf-8")
    (tmp_path / ".skip_lbfgs_after_crash").write_text("", encoding="utf-8")
    (tmp_path / "kinematics_ckpt_12.pth").write_text("x", encoding="utf-8")
    clear_foundation_checkpoints(tmp_path)
    assert not any(tmp_path.iterdir())


def test_foundation_config_binds_arch_env(monkeypatch) -> None:
    for key in ALLFIX_ARCH_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("KINEMATICS_OUTPUT_DIR", raising=False)
    cfg = FoundationConfig()
    cfg.bind_process_env()
    import os

    assert os.environ["KINEMATICS_BC_LAMBDA"] == "10.0"
    assert "production_allfix" in os.environ["KINEMATICS_OUTPUT_DIR"]


def test_bind_allfix_arch_overrides_bc_lambda(monkeypatch) -> None:
    bind_allfix_arch(bc_lambda="12.0")
    import os

    assert os.environ["KINEMATICS_BC_LAMBDA"] == "12.0"


def test_bind_env_can_unset(monkeypatch) -> None:
    monkeypatch.setenv("KINEMATICS_GRAPH_CAP", "99")
    bind_env({"KINEMATICS_GRAPH_CAP": None})
    import os

    assert "KINEMATICS_GRAPH_CAP" not in os.environ
