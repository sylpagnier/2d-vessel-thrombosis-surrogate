"""Tests for research sweep config loading and defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.research_sweep_config import (
    DEFAULT_RESEARCH_FLOW,
    DEFAULT_RESEARCH_MODEL,
    default_control,
    load_sweep_config,
    list_sweep_configs,
    normalize_sweep_config,
)


def test_default_control_matches_geometry_constants():
    ctrl = default_control()
    assert ctrl["re_target"] == pytest.approx(450.0)
    assert ctrl["t_final_s"] == pytest.approx(30000.0)
    assert ctrl["n_steps"] == 120
    assert ctrl["flow"] == DEFAULT_RESEARCH_FLOW
    assert ctrl["geometry"]["width"] == pytest.approx(0.012)


def test_active_sweeps_default_to_clot_ml_0():
    root = Path("configs/research_sweeps")
    for path in sorted(root.glob("*.json")):
        cfg = load_sweep_config(path)
        assert cfg["model"] == DEFAULT_RESEARCH_MODEL
        assert cfg["control"].get("flow") == DEFAULT_RESEARCH_FLOW


def test_legacy_sweep_model_is_no_longer_supported():
    """The mat-growth comparison arm is gone; a config asking for it must fail loudly."""
    with pytest.raises(Exception):
        normalize_sweep_config(
            {"id": "x", "arms": [{"name": "a"}], "model": "locked_canonical"},
            path=Path("x.json"),
        )


def test_normalize_fills_missing_flow():
    cfg = normalize_sweep_config(
        {"id": "x", "arms": [{"name": "a"}], "model": DEFAULT_RESEARCH_MODEL},
        path=Path("x.json"),
    )
    assert cfg["control"]["flow"] == DEFAULT_RESEARCH_FLOW


def test_list_sweep_configs_has_no_legacy_directory():
    """The legacy sweep directory was retired with the mat-growth stack."""
    cfgs = list_sweep_configs()
    assert cfgs, "no sweep configs found"
    assert not any("legacy" in p.parts for p in cfgs)
