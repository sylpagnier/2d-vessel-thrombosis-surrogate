"""Unit tests for research sweep geometry helpers (no Gmsh)."""

from __future__ import annotations

import numpy as np

from src.evaluation.research_sweep_geometry import (
    CONTROL_WIDTH_M,
    arm_geometry_cache_spec,
    build_research_vessel_params,
    geometry_spec_hash,
)
from src.evaluation.research_sweep_presets import resolve_wound_overlay


def test_build_research_params_straight_zero_noise():
    p = build_research_vessel_params(width=CONTROL_WIDTH_M, stenosis_occlusion=0.0)
    assert p["curve_type"] == "straight"
    assert p["v_type"] == "straight"
    assert all(abs(x) < 1e-15 for x in p["noise_top"])
    assert all(abs(x) < 1e-15 for x in p["offsets"])


def test_stenosis_occlusion_deterministic_and_negative_offsets():
    a = build_research_vessel_params(stenosis_occlusion=0.5, seed=42)
    b = build_research_vessel_params(stenosis_occlusion=0.5, seed=99)
    assert a["v_type"] == "stenosis"
    assert a["offsets"] == b["offsets"]  # Gaussian construction is seed-free
    assert min(a["offsets"]) < -1e-6
    assert a["stenosis_occlusion"] == 0.5


def test_aneurysm_factor_positive_offsets():
    p = build_research_vessel_params(aneurysm_factor=0.4)
    assert p["v_type"] == "aneurysm"
    assert max(p["offsets"]) > 1e-6


def test_path_loc_moves_peak():
    prox = build_research_vessel_params(stenosis_occlusion=0.5, path_loc_frac=0.2)
    dist = build_research_vessel_params(stenosis_occlusion=0.5, path_loc_frac=0.8)
    i_prox = int(np.argmin(prox["offsets"]))
    i_dist = int(np.argmin(dist["offsets"]))
    assert i_prox < i_dist


def test_path_loc_eccentricity_and_std_frac():
    both = build_research_vessel_params(stenosis_occlusion=0.5, path_loc=2)
    top = build_research_vessel_params(stenosis_occlusion=0.5, path_loc=0)
    assert both["path_loc"] == 2
    assert top["path_loc"] == 0
    narrow = build_research_vessel_params(
        stenosis_occlusion=0.5, pathology_std_frac=0.03
    )
    broad = build_research_vessel_params(
        stenosis_occlusion=0.5, pathology_std_frac=0.12
    )
    # Broader Gaussian has more non-near-zero samples.
    n_narrow = sum(1 for x in narrow["offsets"] if abs(x) > 1e-6)
    n_broad = sum(1 for x in broad["offsets"] if abs(x) > 1e-6)
    assert n_broad > n_narrow


def test_wall_roughness_deterministic():
    a = build_research_vessel_params(wall_roughness_amp=0.0006, seed=1)
    b = build_research_vessel_params(wall_roughness_amp=0.0006, seed=99)
    assert a["noise_top"] == b["noise_top"]
    assert max(abs(x) for x in a["noise_top"]) > 1e-6


def test_cache_spec_excludes_wound_so_wound_arms_share_a_mesh():
    """The wound is an overlay stamped after the graph loads, not baked into the mesh.

    `load_or_build_research_graph` calls `apply_research_wound_overlay` on the cached graph,
    so two arms that differ only in wound width or position must hash to the SAME mesh --
    otherwise every wound arm re-runs Gmsh for an identical vessel.
    """
    control = {
        "re_target": 450.0,
        "t_final_s": 30000.0,
        "n_steps": 120,
        "geometry": {"width": 0.012, "curve_type": "straight", "seed": 42},
    }
    base_geom = {"stenosis_occlusion": 0.5}
    arm_a = {"name": "w_narrow", "geometry": {**base_geom, "wound_enabled": True,
                                              "wound_width_frac": 0.2}}
    arm_b = {"name": "w_broad", "geometry": {**base_geom, "wound_enabled": True,
                                             "wound_width_frac": 0.6}}

    spec_a = arm_geometry_cache_spec(arm_a, control)
    spec_b = arm_geometry_cache_spec(arm_b, control)

    assert spec_a["stenosis_occlusion"] == 0.5
    assert not any(k.startswith("wound_") for k in spec_a)
    assert geometry_spec_hash(spec_a) == geometry_spec_hash(spec_b)

    # ...and the overlay is what actually separates the two arms.
    over_a = resolve_wound_overlay({**control["geometry"], **arm_a["geometry"]})
    over_b = resolve_wound_overlay({**control["geometry"], **arm_b["geometry"]})
    assert over_a is not None and over_b is not None
    assert over_a["width_frac"] != over_b["width_frac"]


def test_resolve_wound_align_stenosis():
    w = resolve_wound_overlay({
        "wound_enabled": True,
        "wound_align_stenosis": True,
        "path_loc_frac": 0.35,
        "wound_position_frac": 0.9,
    })
    assert w is not None
    assert abs(w["position_frac"] - 0.35) < 1e-9


def test_cache_spec_hash_stable():
    control = {
        "re_target": 450.0,
        "t_final_s": 30000.0,
        "n_steps": 120,
        "geometry": {"width": 0.012, "curve_type": "straight", "seed": 42},
    }
    arm = {"name": "x", "geometry": {"stenosis_occlusion": 0.5}, "re_target": 450.0}
    s1 = arm_geometry_cache_spec(arm, control)
    s2 = arm_geometry_cache_spec(arm, control)
    assert geometry_spec_hash(s1) == geometry_spec_hash(s2)
    assert s1["stenosis_occlusion"] == 0.5
    assert "path_loc" in s1
    assert "pathology_std_frac" in s1
    assert "wall_roughness_amp" in s1
