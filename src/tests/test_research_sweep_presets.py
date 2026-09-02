"""Tests for research sweep presets and slim config normalization."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.evaluation.research_sweep_config import (
    DEFAULT_RESEARCH_MODEL,
    default_control,
    default_output_dir,
    load_sweep_config,
)
from src.evaluation.research_sweep_presets import (
    BEND_PRESETS,
    DEFAULT_WOUND_WIDTH_FRAC,
    expand_geometry_dict,
    resolve_wound_overlay,
    wall_roughness_amp_from_frac,
)


def test_bend_presets_use_named_angles():
    assert BEND_PRESETS["arc_mild"]["angle_span"] == pytest.approx(math.pi / 4.0)
    assert BEND_PRESETS["arc_strong"]["angle_span"] == pytest.approx(math.pi / 2.0)
    assert BEND_PRESETS["hook"]["angle_span"] == pytest.approx(2.0 * math.pi / 3.0)


def test_expand_geometry_dict_bend_preset():
    geom = expand_geometry_dict({"bend_preset": "arc_mild"})
    assert geom["curve_type"] == "arc"
    assert geom["angle_span"] == pytest.approx(math.pi / 4.0)


def test_expand_geometry_dict_roughness_frac():
    geom = expand_geometry_dict({"roughness_frac_width": 0.02, "width": 0.012})
    assert geom["wall_roughness_amp"] == pytest.approx(wall_roughness_amp_from_frac(0.02))


def test_resolve_wound_align_stenosis_uses_path_loc_frac():
    merged = {
        "wound_enabled": True,
        "wound_align_stenosis": True,
        "path_loc_frac": 0.35,
        "wound_width_frac": DEFAULT_WOUND_WIDTH_FRAC,
    }
    overlay = resolve_wound_overlay(merged)
    assert overlay is not None
    assert overlay["position_frac"] == pytest.approx(0.35)


def test_slim_configs_load_with_defaults():
    root = Path("configs/research_sweeps")
    for path in sorted(root.glob("*.json")):
        cfg = load_sweep_config(path)
        assert cfg["model"] == DEFAULT_RESEARCH_MODEL
        assert cfg["control"]["re_target"] == pytest.approx(450.0)
        assert cfg["control"]["n_steps"] == 120
        assert cfg["output_dir"] == default_output_dir(cfg["id"])


def test_bendiness_arms_expand_presets():
    cfg = load_sweep_config(Path("configs/research_sweeps/05_bendiness.json"))
    hook = next(a for a in cfg["arms"] if a["name"] == "hook")
    assert hook["geometry"]["curve_type"] == "hook"
    assert hook["geometry"]["angle_span"] == pytest.approx(2.0 * math.pi / 3.0)


def test_wound_control_preset_expands():
    cfg = load_sweep_config(Path("configs/research_sweeps/18_wound_x_stenosis.json"))
    geom = cfg["control"]["geometry"]
    assert geom["wound_enabled"] is True
    assert geom["wound_align_stenosis"] is True
    assert geom["wound_width_frac"] == pytest.approx(DEFAULT_WOUND_WIDTH_FRAC)
