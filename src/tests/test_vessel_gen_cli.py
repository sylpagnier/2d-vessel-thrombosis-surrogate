"""CLI resolution for biochem CFD vs synthetic vessel generation."""

from __future__ import annotations

import pytest

from src.config import VesselConfig
from src.data_gen.lib.vessel_generator import (
    _vessel_gen_arg_parser,
    default_unit_for_phase,
    resolve_vessel_gen_cli,
    resolve_wound_flags,
)
from src.data_gen.pipeline_biochem import _parse_batch_args, _wound_run_kwargs
from src.utils.paths import data_root


def _parse(*argv: str):
    return _vessel_gen_arg_parser().parse_args(list(argv))


def test_wound_at_pathology_implies_wound_sites():
    assert resolve_wound_flags(wound=False, wound_at_pathology=True) == (1.0, True)
    assert resolve_wound_flags(wound=True, wound_at_pathology=False) == (1.0, False)
    assert resolve_wound_flags(wound=False, wound_at_pathology=False) == (0.0, False)


def test_default_unit_cm_only_for_cfd_anchors():
    assert default_unit_for_phase("biochem_anchors") == "cm"
    assert default_unit_for_phase("biochem") == "m"
    assert default_unit_for_phase("kinematics") == "m"


def test_anchors_cli_matches_biochem_cfd_track():
    spec = resolve_vessel_gen_cli(
        _parse(
            "--anchors",
            "--level",
            "0",
            "-n",
            "10",
            "--pathology-mode",
            "straight_max",
            "--wound-at-pathology",
            "--seed",
            "42",
        )
    )
    assert spec.phase == "biochem_anchors"
    assert spec.unit == "cm"
    assert spec.wound_probability == 1.0
    assert spec.wound_at_pathology is True
    cfg = VesselConfig(phase=spec.phase)
    assert cfg.mesh_input_dir == data_root() / "raw" / "biochem_anchors"


def test_phase2_without_anchors_stays_synthetic_si():
    spec = resolve_vessel_gen_cli(
        _parse("--phase", "2", "--level", "0", "-n", "10", "--unit", "cm")
    )
    assert spec.phase == "biochem"
    assert spec.unit == "cm"
    cfg = VesselConfig(phase=spec.phase)
    assert cfg.mesh_input_dir == data_root() / "raw" / "biochem"


def test_phase2_default_unit_is_meters():
    spec = resolve_vessel_gen_cli(_parse("--phase", "2", "--level", "0", "-n", "1"))
    assert spec.unit == "m"
    assert spec.wound_probability == 0.0


def test_anchors_rejects_kinematics_phase():
    args = _parse("--phase", "1", "--level", "0", "-n", "1", "--anchors")
    with pytest.raises(ValueError, match="--anchors"):
        resolve_vessel_gen_cli(args)


def test_anchors_rejects_unit_meters():
    args = _parse("--anchors", "--level", "0", "-n", "1", "--unit", "m")
    with pytest.raises(ValueError, match="--unit cm"):
        resolve_vessel_gen_cli(args)


def test_pipeline_batch_anchor_wound_flags():
    args = _parse_batch_args(
        [
            "--batch",
            "--track",
            "anchor_meshes",
            "--level",
            "0",
            "-n",
            "10",
            "--pathology-mode",
            "straight_max",
            "--wound-at-pathology",
            "--seed",
            "42",
        ]
    )
    assert args is not None
    assert args.track == "anchor_meshes"
    assert args.level == 0
    assert args.seed == 42
    kw = _wound_run_kwargs(
        wound=bool(args.wound), wound_at_pathology=bool(args.wound_at_pathology)
    )
    assert kw == {"wound_probability": 1.0, "wound_at_pathology": True}
