"""Shared constants and config loading for geometry-sensitivity research sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.research_sweep_geometry import (
    CONTROL_BASE_LENGTH_M,
    CONTROL_RE,
    CONTROL_SEED,
    CONTROL_WIDTH_M,
    DEFAULT_HORIZON_N_STEPS,
    DEFAULT_T_FINAL_S,
)
from src.evaluation.research_sweep_presets import (
    CONTROL_GEOMETRY_PRESETS,
    expand_geometry_dict,
)
from src.utils.paths import get_project_root

SWEEPS_DIR = Path("configs/research_sweeps")
LEGACY_SWEEPS_DIR = Path("configs/research_sweeps/legacy")
OUTPUT_ROOT = Path("outputs/research_sweeps")

DEFAULT_RESEARCH_MODEL = "clot_ml_0"
DEFAULT_RESEARCH_FLOW = "fem"
DEFAULT_CLOT_MODEL = "clot_ml_0"
LEGACY_BIOCHEM_MODEL = "locked_canonical"

SUPPORTED_MODELS = frozenset(
    {
        DEFAULT_RESEARCH_MODEL,
        "clot_ml_v0",
        LEGACY_BIOCHEM_MODEL,
        "biochem",
        "legacy",
    }
)


def default_control() -> dict[str, Any]:
    """Canonical straight-channel control vessel (8 UI hours, Re=450)."""
    return {
        "re_target": float(CONTROL_RE),
        "t_final_s": float(DEFAULT_T_FINAL_S),
        "n_steps": int(DEFAULT_HORIZON_N_STEPS),
        "flow": DEFAULT_RESEARCH_FLOW,
        "include_velocity": False,
        "geometry": {
            "width": float(CONTROL_WIDTH_M),
            "curve_type": "straight",
            "angle_span": 0.0,
            "amplitude": 0.0,
            "base_length": float(CONTROL_BASE_LENGTH_M),
            "path_loc_frac": 0.5,
            "seed": int(CONTROL_SEED),
            "level": 0,
        },
    }


def default_output_dir(sweep_id: str) -> str:
    return f"{OUTPUT_ROOT.as_posix()}/{sweep_id}"


def _apply_control_geometry_preset(control: dict[str, Any]) -> None:
    preset = control.pop("geometry_preset", None)
    if preset is None:
        return
    key = str(preset).strip()
    if key not in CONTROL_GEOMETRY_PRESETS:
        raise ValueError(
            f"Unknown control geometry_preset={key!r}; "
            f"known: {sorted(CONTROL_GEOMETRY_PRESETS)}"
        )
    geom = dict(control.get("geometry") or {})
    geom.update(CONTROL_GEOMETRY_PRESETS[key])
    control["geometry"] = geom


def _normalize_arm_geometry(arm: dict[str, Any], control: dict[str, Any]) -> None:
    width = float((control.get("geometry") or {}).get("width", CONTROL_WIDTH_M))
    geom = expand_geometry_dict(arm.get("geometry"), width_m=width)
    if geom:
        arm["geometry"] = geom


def _abs(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def list_sweep_configs(*, include_legacy: bool = False) -> list[Path]:
    roots = [_abs(SWEEPS_DIR)]
    if include_legacy:
        roots.append(_abs(LEGACY_SWEEPS_DIR))
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.glob("*.json")))
    return out


def resolve_sweep_path(name: str, *, include_legacy: bool = False) -> Path:
    raw = str(name).strip()
    if raw.startswith("legacy/"):
        include_legacy = True
        raw = raw.split("/", 1)[1]
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    search_roots = [_abs(SWEEPS_DIR)]
    if include_legacy:
        search_roots.append(_abs(LEGACY_SWEEPS_DIR))
    for root in search_roots:
        cand = root / raw
        if cand.is_file():
            return cand
        if not raw.endswith(".json"):
            cand2 = root / f"{raw}.json"
            if cand2.is_file():
                return cand2
    raise FileNotFoundError(f"Sweep config not found: {name!r} (looked under {SWEEPS_DIR})")


def normalize_sweep_config(cfg: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Validate registry JSON and apply defaults for the active clot_ml_0 stack."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Sweep config must be a JSON object: {path}")
    if "arms" not in cfg or not cfg["arms"]:
        raise ValueError(f"Sweep config has no arms: {path}")
    if not cfg.get("id"):
        raise ValueError(f"Sweep config missing id: {path}")

    model = str(cfg.get("model") or DEFAULT_RESEARCH_MODEL).lower().strip()
    if model == "clot_ml_v0":
        model = DEFAULT_RESEARCH_MODEL
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model={model!r} in {path or cfg.get('id')}; "
            f"supported: {sorted(SUPPORTED_MODELS)}"
        )
    cfg["model"] = model

    control = dict(default_control())
    user_control = cfg.get("control")
    if isinstance(user_control, dict):
        geom = dict(control.get("geometry") or {})
        user_geom = user_control.get("geometry")
        if isinstance(user_geom, dict):
            geom.update(user_geom)
        control.update({k: v for k, v in user_control.items() if k != "geometry"})
        control["geometry"] = geom
    _apply_control_geometry_preset(control)
    control["geometry"] = expand_geometry_dict(control.get("geometry"))
    cfg["control"] = control

    cfg.setdefault("output_dir", default_output_dir(str(cfg["id"])))
    for arm in cfg["arms"]:
        if isinstance(arm, dict):
            _normalize_arm_geometry(arm, control)

    if model == DEFAULT_RESEARCH_MODEL:
        control.setdefault("flow", DEFAULT_RESEARCH_FLOW)
        cfg.setdefault("clot_model", DEFAULT_CLOT_MODEL)
    return cfg


def load_sweep_config(path: Path | str, *, include_legacy: bool = True) -> dict[str, Any]:
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return normalize_sweep_config(cfg, path=p)


__all__ = [
    "DEFAULT_CLOT_MODEL",
    "DEFAULT_RESEARCH_FLOW",
    "DEFAULT_RESEARCH_MODEL",
    "LEGACY_BIOCHEM_MODEL",
    "LEGACY_SWEEPS_DIR",
    "OUTPUT_ROOT",
    "SUPPORTED_MODELS",
    "SWEEPS_DIR",
    "default_control",
    "default_output_dir",
    "list_sweep_configs",
    "load_sweep_config",
    "normalize_sweep_config",
    "resolve_sweep_path",
]
