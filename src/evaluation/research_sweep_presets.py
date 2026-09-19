"""Named sweep axes and geometry presets for research configs.

Configs may reference preset keys instead of duplicating literals. Expansion happens
in ``normalize_sweep_config`` (control / arms) before mesh build and wound overlay.
"""

from __future__ import annotations

import math
from typing import Any

# Must match ``research_sweep_geometry.CONTROL_WIDTH_M``.
_DEFAULT_WIDTH_M = 0.012

# --- Reynolds / width / stenosis tiers (documented sweep grids) ---
RE_SWEEP = (150.0, 300.0, 450.0, 600.0, 900.0)
RE_TRIPLET = (300.0, 450.0, 900.0)
WIDTH_SWEEP_M = (0.008, 0.012, 0.016, 0.020)
WIDTH_TRIPLET_M = (0.008, 0.012, 0.020)
STENOSIS_SWEEP = (0.0, 0.25, 0.5, 0.75, 0.8)
STENOSIS_TRIPLET = (0.25, 0.5, 0.75)
STENOSIS_MID = 0.5
AXIAL_THIRDS = (0.2, 0.5, 0.8)
PATHOLOGY_STD_SWEEP = (0.03, 0.06, 0.12)
LENGTH_SWEEP_M = (0.05, 0.1, 0.15)

# --- Bend geometry (radians / meters) ---
BEND_ARC_MILD_ANGLE_RAD = math.pi / 4.0
BEND_ARC_STRONG_ANGLE_RAD = math.pi / 2.0
BEND_HOOK_ANGLE_RAD = 2.0 * math.pi / 3.0
BEND_S_CURVE_AMPLITUDE_M = 0.005

BEND_PRESETS: dict[str, dict[str, Any]] = {
    "straight": {
        "curve_type": "straight",
        "angle_span": 0.0,
        "amplitude": 0.0,
    },
    "arc_mild": {
        "curve_type": "arc",
        "angle_span": BEND_ARC_MILD_ANGLE_RAD,
        "amplitude": 0.0,
        "bend_sign": 1.0,
    },
    "arc_strong": {
        "curve_type": "arc",
        "angle_span": BEND_ARC_STRONG_ANGLE_RAD,
        "amplitude": 0.0,
        "bend_sign": 1.0,
    },
    "s_curve": {
        "curve_type": "s_curve",
        "angle_span": 0.0,
        "amplitude": BEND_S_CURVE_AMPLITUDE_M,
        "bend_sign": 1.0,
    },
    "hook": {
        "curve_type": "hook",
        "angle_span": BEND_HOOK_ANGLE_RAD,
        "amplitude": 0.0,
        "bend_sign": 1.0,
    },
}

# --- Wall roughness as fraction of channel width ---
WALL_ROUGHNESS_FRAC_WIDTH = (0.0, 0.02, 0.05, 0.08)

# --- Wound defaults (customer mirrored wound) ---
DEFAULT_WOUND_WIDTH_FRAC = 0.15
DEFAULT_WOUND_POSITION_FRAC = 0.5
WOUND_WIDTH_SWEEP = (0.08, 0.15, 0.25, 0.40)

# Control-level geometry bundles (merged into control.geometry).
CONTROL_GEOMETRY_PRESETS: dict[str, dict[str, Any]] = {
    "stenosis_mid": {"stenosis_occlusion": STENOSIS_MID},
    "wound_mid_enabled": {
        "wound_enabled": True,
        "wound_position_frac": DEFAULT_WOUND_POSITION_FRAC,
    },
    "wound_standard_width": {
        "wound_enabled": True,
        "wound_width_frac": DEFAULT_WOUND_WIDTH_FRAC,
    },
    "wound_geometry_mid": {
        "wound_position_frac": DEFAULT_WOUND_POSITION_FRAC,
        "wound_width_frac": DEFAULT_WOUND_WIDTH_FRAC,
    },
    "wound_in_stenosis": {
        "wound_enabled": True,
        "wound_width_frac": DEFAULT_WOUND_WIDTH_FRAC,
        "wound_align_stenosis": True,
    },
    "stenosis_mid_wound_standard": {
        "stenosis_occlusion": STENOSIS_MID,
        "wound_enabled": True,
        "wound_width_frac": DEFAULT_WOUND_WIDTH_FRAC,
    },
}


def wall_roughness_amp_from_frac(frac_width: float, *, width_m: float = _DEFAULT_WIDTH_M) -> float:
    """Absolute wall roughness amplitude [m] from fraction of channel width."""
    return float(frac_width) * float(width_m)


def expand_geometry_dict(
    geom: dict[str, Any] | None,
    *,
    width_m: float = _DEFAULT_WIDTH_M,
) -> dict[str, Any]:
    """Resolve bend_preset / roughness_frac_width into concrete geometry fields."""
    out = dict(geom or {})
    preset = out.pop("bend_preset", None)
    if preset is not None:
        key = str(preset).strip()
        if key not in BEND_PRESETS:
            raise ValueError(f"Unknown bend_preset={key!r}; known: {sorted(BEND_PRESETS)}")
        merged = dict(BEND_PRESETS[key])
        merged.update(out)
        out = merged
    rough_frac = out.pop("roughness_frac_width", None)
    if rough_frac is not None:
        out["wall_roughness_amp"] = wall_roughness_amp_from_frac(
            float(rough_frac),
            width_m=float(out.get("width", width_m)),
        )
    return out


def merge_geometry(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Control geometry + arm geometry with preset expansion."""
    g0 = expand_geometry_dict(control.get("geometry"))
    g1 = expand_geometry_dict(arm.get("geometry"), width_m=float(g0.get("width", _DEFAULT_WIDTH_M)))
    g0.update(g1)
    return g0


def resolve_wound_overlay(merged_geometry: dict[str, Any]) -> dict[str, Any] | None:
    """Return wound kwargs for ``apply_customer_mirrored_wound``, or None if disabled."""
    if not bool(merged_geometry.get("wound_enabled", False)):
        return None
    position = float(merged_geometry.get("wound_position_frac", DEFAULT_WOUND_POSITION_FRAC))
    if bool(merged_geometry.get("wound_align_stenosis", False)):
        position = float(merged_geometry.get("path_loc_frac", DEFAULT_WOUND_POSITION_FRAC))
    width = float(merged_geometry.get("wound_width_frac", DEFAULT_WOUND_WIDTH_FRAC))
    return {
        "enabled": True,
        "position_frac": position,
        "width_frac": width,
    }


__all__ = [
    "AXIAL_THIRDS",
    "BEND_ARC_MILD_ANGLE_RAD",
    "BEND_ARC_STRONG_ANGLE_RAD",
    "BEND_HOOK_ANGLE_RAD",
    "BEND_PRESETS",
    "BEND_S_CURVE_AMPLITUDE_M",
    "CONTROL_GEOMETRY_PRESETS",
    "DEFAULT_WOUND_POSITION_FRAC",
    "DEFAULT_WOUND_WIDTH_FRAC",
    "LENGTH_SWEEP_M",
    "PATHOLOGY_STD_SWEEP",
    "RE_SWEEP",
    "RE_TRIPLET",
    "STENOSIS_MID",
    "STENOSIS_SWEEP",
    "STENOSIS_TRIPLET",
    "WALL_ROUGHNESS_FRAC_WIDTH",
    "WIDTH_SWEEP_M",
    "WIDTH_TRIPLET_M",
    "WOUND_WIDTH_SWEEP",
    "expand_geometry_dict",
    "merge_geometry",
    "resolve_wound_overlay",
    "wall_roughness_amp_from_frac",
]
