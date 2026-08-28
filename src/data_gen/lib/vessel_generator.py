"""
vessel_generator.py
-------------------
Generates 2D vessel meshes with parametric pathologies using Gmsh.

Performance design
~~~~~~~~~~~~~~~~~~
Gmsh holds global C++ state and is not thread-safe, so parallelism must be
achieved via *processes*, not threads.  The strategy is:

  1. The main process pre-samples ALL random parameters for every vessel and
     packages them as plain dicts (fully picklable, no Gmsh state).
  2. Worker processes each receive a *chunk* of those dicts.
  3. Every worker calls gmsh.initialize() once, iterates over its chunk, and
     calls gmsh.finalize() before exiting — keeping Gmsh init overhead at
     O(num_workers) rather than O(n).
  4. ProcessPoolExecutor dispatches chunks; tqdm tracks completion.

The result is near-linear scaling up to the physical core count.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from tqdm import tqdm

# Running ``python src/data_gen/lib/vessel_generator.py`` does not put the repo
# root on sys.path, so ``from src.config`` fails unless we add it first.
if __name__ == "__main__":
    import sys

    _proj = Path(__file__).resolve().parents[3]
    _ps = str(_proj)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from src.config import VesselConfig
from src.utils.paths import get_project_root, migrate_legacy_vessel_meshes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def summarize_vessel_mesh_inventory(output_dir: Path) -> Dict[str, Any]:
    """Scan ``output_dir`` for ``vessel_*.json`` (or ``.msh``) and report append state.

    New runs default to **append**: indices continue after ``max_idx`` so existing meshes
    are not overwritten. To regenerate from index 0, pass ``start_idx=0`` explicitly.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return {"count": 0, "max_idx": -1, "next_idx": 0}
    indices: List[int] = []
    for pat in ("vessel_*.json", "vessel_*.msh"):
        for p in output_dir.glob(pat):
            try:
                indices.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
    uniq = sorted(set(indices))
    if not uniq:
        return {"count": 0, "max_idx": -1, "next_idx": 0}
    mx = uniq[-1]
    return {
        "count": len(uniq),
        "max_idx": mx,
        "next_idx": mx + 1,
    }


def _next_vessel_index(output_dir: Path) -> int:
    return int(summarize_vessel_mesh_inventory(output_dir)["next_idx"])


def _params_by_idx(params_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(p["idx"]): p for p in params_list}


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centerline generators
# All accept pre-sampled scalar parameters so they are deterministic given
# a params dict — no random calls inside.
# ---------------------------------------------------------------------------

def _centerline_straight(
    n: int, length: float, jitter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Slightly jittered straight line along +X."""
    x = np.linspace(0.0, length, n)
    y = np.zeros(n)
    y[2 : n - 2] = jitter
    pts = np.column_stack([x, y])
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


def _centerline_arc(
    n: int,
    length: float,
    angle_span: float,
    *,
    bend_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular arc: starts at (0,0) pointing +X, sweeps by ``angle_span`` in the x–y plane.

    Default ``bend_sign=1`` matches the historical clockwise sweep (negative y at the tip for
    ``angle_span > 0``). Use ``bend_sign=-1`` to mirror across the x-axis (opposite vertical offset).
    ``radius = length / angle_span`` so arc length ~= ``length`` for small angles.
    """
    radius = length / max(angle_span, 1e-3)
    theta = np.linspace(0.0, angle_span, n)
    pts = np.column_stack([
        radius * np.sin(theta),
        radius * (np.cos(theta) - 1.0),
    ])
    bs = float(bend_sign)
    pts[:, 1] *= bs
    tangents = np.column_stack([np.cos(theta), -bs * np.sin(theta)])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


def _centerline_s_curve(
    n: int, length: float, amplitude: float
) -> Tuple[np.ndarray, np.ndarray]:
    """S-shaped: one full sine period transverse to flow."""
    t = np.linspace(0.0, 1.0, n)
    pts = np.column_stack([t * length, amplitude * np.sin(2.0 * np.pi * t)])
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Curve-type weight tables
# ---------------------------------------------------------------------------

_CURVE_WEIGHTS: Dict[int, Dict[str, float]] = {
    0: {"straight": 0.60, "arc": 0.30, "s_curve": 0.10, "hook": 0.00},
    1: {"straight": 0.15, "arc": 0.45, "s_curve": 0.20, "hook": 0.20},
}


def resolve_bend_sign_mode(explicit: Optional[str] = None) -> str:
    """
    ``down_only``: historical arcs (bend_sign=+1) and non-flipped S-curves (Apr-2026 style).
    ``bidirectional``: random mirror for L1+ arc/hook and signed S-curve (default since May 2026).
    """
    raw = (explicit or os.environ.get("KINEMATICS_BEND_SIGN_MODE", "bidirectional")).strip().lower()
    if raw in ("down_only", "down", "legacy", "historical", "apr26"):
        return "down_only"
    if raw in ("bidirectional", "both", "up_down", "default", "may26"):
        return "bidirectional"
    raise ValueError(
        f"Unknown bend sign mode {raw!r}; use 'down_only' or 'bidirectional' "
        "(or env KINEMATICS_BEND_SIGN_MODE)."
    )


def default_level_mix(n: int) -> Dict[int, int]:
    """Default kinematics cohort: mostly L0/L1, ~20% high-thrombus (L2)."""
    n = max(1, int(n))
    n2 = max(1, round(n * 0.2))
    rem = n - n2
    n0 = rem // 2
    n1 = rem - n0
    return {0: n0, 1: n1, 2: n2}


def parse_level_mix(spec: str, n: int) -> Dict[int, int]:
    """Parse ``n0,n1,n2`` counts; must sum to ``n``."""
    parts = [int(x.strip()) for x in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError("level_mix must have exactly three comma-separated integers (L0,L1,L2)")
    mix = {0: parts[0], 1: parts[1], 2: parts[2]}
    total = sum(mix.values())
    if total != n:
        raise ValueError(f"level_mix counts sum to {total}, expected n={n}")
    return mix


def cohort_levels(
    n: int,
    level: int,
    level_mix: Optional[Dict[int, int]],
    rng: np.random.Generator,
) -> List[int]:
    """Per-vessel geometry levels for one run (shuffled when ``level_mix`` is set)."""
    if level_mix is None:
        return [int(level)] * n
    total = sum(int(v) for v in level_mix.values())
    if total != n:
        raise ValueError(f"level_mix counts sum to {total}, expected n={n}")
    out: List[int] = []
    for lvl, cnt in sorted(level_mix.items()):
        out.extend([int(lvl)] * int(cnt))
    rng.shuffle(out)
    return out


_PATHOLOGY_MODE_CHOICES = ("random", "max_stenosis", "max_aneurysm", "straight_max")
_FORCED_MAX_PATHOLOGY_MODES = frozenset({"max_stenosis", "max_aneurysm", "straight_max"})
_ANEURYSM_WALL_MODE_CHOICES = ("mirrored", "one")


def parse_pathology_mix(spec, n, rng):
    """Per-vessel pathology modes from a mix spec, e.g. ``"random:0.75,max_stenosis:0.25"``.

    `--pathology-mode` is one mode for a whole run, so covering the severe-stenosis tail used to
    need a second command with a second seed and a second index range -- easy to get wrong, and
    the failure is silent (a cohort with no tail looks fine until the preflight reports 0%).

    Values are weights: integers summing to ``n`` are used as exact counts, anything else is
    allocated proportionally with the remainder going to the largest weight.  The assignment is
    shuffled with the caller's RNG so modes are not correlated with vessel index -- which would
    otherwise correlate them with the geometry-level schedule.
    """
    parts = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, val = chunk.partition(":")
        mode = normalize_pathology_mode(name.strip()) or "random"
        try:
            w = float(val) if val.strip() else 1.0
        except ValueError:
            raise ValueError(f"bad pathology-mix weight in {chunk!r}")
        if w < 0:
            raise ValueError(f"negative pathology-mix weight in {chunk!r}")
        parts.append((mode, w))
    if not parts:
        return ["random"] * n

    total = sum(w for _, w in parts)
    if total <= 0:
        return ["random"] * n
    exact = all(float(w).is_integer() for _, w in parts) and int(total) == int(n)
    counts = ([int(w) for _, w in parts] if exact
              else [int(n * w / total) for _, w in parts])
    short = int(n) - sum(counts)
    if short:
        counts[max(range(len(counts)), key=lambda i: parts[i][1])] += short

    out = []
    for (mode, _), c in zip(parts, counts):
        out.extend([mode] * max(0, c))
    out = out[: int(n)] + ["random"] * max(0, int(n) - len(out))
    rng.shuffle(out)
    return out


def normalize_pathology_mode(mode: str | None) -> str | None:
    """Return a canonical pathology mode or ``None`` for default random sampling."""
    if mode is None:
        return None
    raw = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "": "random",
        "default": "random",
        "none": "random",
        "max_stenosis": "max_stenosis",
        "maxstenosis": "max_stenosis",
        "stenosis_max": "max_stenosis",
        "max_aneurysm": "max_aneurysm",
        "maxaneurysm": "max_aneurysm",
        "aneurysm_max": "max_aneurysm",
        "straight_max": "straight_max",
        "straightmax": "straight_max",
        "max_straight": "straight_max",
        "straight_max_pathology": "straight_max",
    }
    resolved = aliases.get(raw, raw)
    if resolved == "random":
        return None
    if resolved not in _PATHOLOGY_MODE_CHOICES[1:]:
        raise ValueError(
            f"Unknown pathology_mode {mode!r}; use one of: {', '.join(_PATHOLOGY_MODE_CHOICES)}"
        )
    return resolved


def normalize_aneurysm_wall_mode(mode: str | None) -> str:
    """Return ``one`` (single wall max bulge, default) or ``mirrored`` (both walls)."""
    if mode is None:
        return "one"
    raw = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "": "one",
        "default": "one",
        "both": "mirrored",
        "both_walls": "mirrored",
        "symmetric": "mirrored",
        "mirror": "mirrored",
        "mirrored": "mirrored",
        "one": "one",
        "single": "one",
        "one_wall": "one",
        "single_wall": "one",
        "asymmetric": "one",
    }
    resolved = aliases.get(raw, raw)
    if resolved not in _ANEURYSM_WALL_MODE_CHOICES:
        raise ValueError(
            f"Unknown aneurysm_wall_mode {mode!r}; use one of: "
            f"{', '.join(_ANEURYSM_WALL_MODE_CHOICES)}"
        )
    return resolved


def prompt_pathology_mode() -> Optional[str]:
    """Interactive pathology-class picker shared by kinematics and biochem datagen."""
    print(
        "\nPathology class:\n"
        "  1 = random mix (default)\n"
        "  2 = max stenosis (~80% diameter occlusion at peak)\n"
        "  3 = max aneurysm (local width up to 3x inlet)\n"
        "  4 = straight max (straight x-vessel, no bend; max stenosis or aneurysm)\n"
    )
    while True:
        raw = input("Pathology class [1/2/3/4] [1]: ").strip()
        if raw in ("", "1"):
            return None
        if raw == "2":
            return "max_stenosis"
        if raw == "3":
            return "max_aneurysm"
        if raw == "4":
            return "straight_max"
        print("  Enter 1, 2, 3, or 4.")


def prompt_aneurysm_wall_mode(pathology_mode: str | None = None) -> str:
    """Ask one-wall vs mirrored when the cohort can include max aneurysms."""
    mode = normalize_pathology_mode(pathology_mode)
    if mode not in ("max_aneurysm", "straight_max"):
        return "one"
    print(
        "\nAneurysm wall placement (max-strength aneurysms):\n"
        "  1 = one wall only (default; max wall offset on top or bottom)\n"
        "  2 = mirrored / both walls (peak lumen up to 3x inlet)\n"
    )
    while True:
        raw = input("Aneurysm walls [1/2] [1]: ").strip()
        if raw in ("", "1"):
            return "one"
        if raw == "2":
            return "mirrored"
        print("  Enter 1 or 2.")


def stenosis_wall_offset_for_occlusion(
    width: float,
    cfg: VesselConfig,
    occlusion_frac: float | None = None,
) -> float:
    """Negative wall offset magnitude (Gaussian peak, both walls) for diameter occlusion."""
    if occlusion_frac is None:
        return cfg.max_stenosis_wall_offset(width)
    occlusion_frac = float(np.clip(occlusion_frac, 0.0, 0.95))
    lumen_frac = 1.0 - occlusion_frac
    return (lumen_frac - 1.0) * float(width) / 2.0


_WOUND_END_MARGIN_FRAC = 0.05  # keep Wound B-splines off inlet/outlet endpoints


def _clip_wound_center(center: float, half_width: float) -> float:
    """Clamp a wound center so the segment stays on the interior of the wall."""
    lo = float(half_width + _WOUND_END_MARGIN_FRAC)
    hi = float(1.0 - half_width - _WOUND_END_MARGIN_FRAC)
    if hi < lo:
        return 0.5
    return float(np.clip(center, lo, hi))


def _sample_wound_sites(
    rng: np.random.Generator,
    cfg: VesselConfig,
    *,
    wound_probability: float,
    wound_at_pathology: bool = False,
    pathology_peak_frac: float | None = None,
) -> list[dict[str, float]]:
    """Draw wound collars. Pathology placement is used only when a peak exists."""
    if float(rng.random()) >= float(wound_probability):
        return []
    n_wounds = int(rng.integers(cfg.wound_count_range[0], cfg.wound_count_range[1] + 1))
    sites: list[dict[str, float]] = []
    place_at_peak = bool(wound_at_pathology) and pathology_peak_frac is not None
    jitter = max(0.0, float(cfg.wound_pathology_jitter_frac))
    for _ in range(n_wounds):
        half_w = float(rng.uniform(*cfg.wound_half_width_frac_range))
        if place_at_peak:
            center = float(pathology_peak_frac) + float(rng.uniform(-jitter, jitter))
        else:
            center = float(rng.uniform(*cfg.wound_center_frac_range))
        sites.append({
            "center_frac": _clip_wound_center(center, half_w),
            "half_width_frac": half_w,
        })
    return sites


def _straighten_curve_weights(active: Dict[str, float], cfg: VesselConfig) -> Dict[str, float]:
    """Blend a level's curve weights toward the straight-vessel preference for severe pathology.

    Severe stenoses and aneurysms occur in straight vessels far more than in tortuous ones, and
    that is what the deploy cohort looks like.  Drawing severity independently of curvature
    produced shapes that are both clinically odd and hard to solve.

    A blend, not an override: ``w' = (1 - a) * w_level + a * w_severe``.  Each level keeps its
    character -- L2 is deliberately bendy and stays bendier than L0 -- while severe cases move
    toward straight within it.
    """
    a = float(np.clip(cfg.severe_pathology_straighten, 0.0, 1.0))
    if a <= 0.0:
        return dict(active)
    # A curve type the level has zeroed is an EXCLUSION, not a low weight.  L2 sets
    # `straight: 0.0` because its whole job is the bendy-and-pathological corner of the space;
    # L0 already supplies straight-and-severe in bulk (60% straight).  Blending straight back
    # into L2 would delete a deliberate contrast class to duplicate coverage we already have.
    # So the tilt only redistributes among the types the level actually allows -- which for L2
    # still means moving severe cases off hooks and onto arcs.
    # A zero entry counts as excluded even when the key is present: the pro-thrombotic map
    # spells out `{"straight": 0.0, ...}` rather than omitting it.
    active = {k: float(v) for k, v in active.items() if float(v) > 0.0}
    if not active:
        return {}
    pref = {k: float(cfg.severe_curve_weights.get(k, 0.0)) for k in active}
    tot = sum(pref.values())
    if tot <= 0.0:
        return dict(active)
    pref = {k: v / tot for k, v in pref.items()}
    out = {k: (1.0 - a) * float(active[k]) + a * pref[k] for k in active}
    out = {k: v for k, v in out.items() if v > 0}
    return out or dict(active)


def _sample_params(
        idx: int,
        level: int,
        cfg: VesselConfig,
        rng: np.random.Generator,
        pathology_mode: str | None = None,
        aneurysm_wall_mode: str | None = None,
        wound_probability: float | None = None,
        wound_at_pathology: bool = False,
        severity_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Draw ALL random numbers for one vessel and return a plain picklable dict.
    Level 2 triggers pro-thrombotic geometry (extreme expansions/stagnation zones).

    ``severity_scale`` < 1 softens the pathology for a repair re-draw: it scales the wall offset
    AND drops the max-magnitude shape presets, so the throat gets both shallower and gentler.
    Both halves are needed.  Scaling depth alone leaves `max_stenosis`'s sharp transition
    (``std_dev`` 0.02-0.05n against 0.04-0.10n for a normal draw), and sharpness is a large part
    of what makes these solves fail.  Without this the ladder is inert: `max_stenosis` pins the
    sampler at the class maximum, so every "easier" draw came back at the same severity 5.00
    (measured -- the 0.70x, 0.50x and 0.35x rungs were identical).
    """
    # Level-driven mode: 2 => pro-thrombotic cohort shaping.
    pro_thrombotic = (level == 2)
    pathology_mode = normalize_pathology_mode(pathology_mode)
    aneurysm_wall_mode = normalize_aneurysm_wall_mode(aneurysm_wall_mode)
    straight_max = pathology_mode == "straight_max"
    forced_max = pathology_mode in _FORCED_MAX_PATHOLOGY_MODES

    # Whether this vessel will carry a max-magnitude pathology is decided HERE, before the
    # curve is drawn, so the curve can be conditioned on it.  It used to be rolled inside the
    # offset block, which made severity and curvature independent.
    hit_max_roll = bool(rng.random() < float(cfg.pathology_max_hit_prob))

    if straight_max:
        # Dedicated class: straight along x, no bendiness, extreme pathology.
        curve_type = "straight"
        v_type = str(rng.choice(["stenosis", "aneurysm"]))
        magnitude_mode = "max_stenosis" if v_type == "stenosis" else "max_aneurysm"
    else:
        if pro_thrombotic:
            # Eliminate straight vessels; favor sharp turns and hooks
            active = {"straight": 0.0, "arc": 0.20, "s_curve": 0.40, "hook": 0.40}
        else:
            weights_map = _CURVE_WEIGHTS.get(min(level, 1), _CURVE_WEIGHTS)
            active = {k: v for k, v in weights_map.items() if v > 0}

        # A severe pathology belongs in a straighter vessel (`severe_pathology_straighten`).
        # `forced_max` is known now; for the random mode the roll above stands in for it.
        will_be_severe = forced_max or hit_max_roll
        if will_be_severe:
            active = _straighten_curve_weights(active, cfg)

        keys = list(active.keys())
        probs = np.array(list(active.values()), dtype=float)
        probs /= probs.sum()
        curve_type = str(rng.choice(keys, p=probs))

        if pathology_mode == "max_stenosis":
            v_type = "stenosis"
            magnitude_mode = "max_stenosis"
        elif pathology_mode == "max_aneurysm":
            v_type = "aneurysm"
            magnitude_mode = "max_aneurysm"
        elif pro_thrombotic:
            # Guarantee a pathology. Aneurysms (stagnation) and Stenosis (downstream deceleration)
            v_type = str(rng.choice(["stenosis", "aneurysm"], p=[0.3, 0.7]))
            magnitude_mode = None
        else:
            v_type = str(rng.choice(["straight", "stenosis", "aneurysm"]))
            magnitude_mode = None

    width = float(rng.uniform(cfg.width_min, cfg.width_max))
    n = cfg.num_ctrl_pts
    L = cfg.base_length
    t = np.linspace(0, 1, n)
    hit_configured_max = False

    # 1. Main Clinical Pathology
    offsets = np.zeros(n)
    pathology_peak_frac: float | None = None
    if v_type != "straight":
        if magnitude_mode == "max_stenosis":
            mag = cfg.max_stenosis_wall_offset(width)
            hit_configured_max = True
        elif magnitude_mode == "max_aneurysm":
            mag = cfg.max_aneurysm_wall_offset(width, pro_thrombotic=pro_thrombotic)
            hit_configured_max = True
        elif v_type in ("stenosis", "occlusion"):
            mult = cfg.stenosis_pro_thrombotic_mult if pro_thrombotic else 1.0
            stenosis_cap = 0.5 * float(cfg.max_stenosis_diameter_occlusion)
            if hit_max_roll:        # drawn before the curve, so the two are coupled
                mag = cfg.max_stenosis_wall_offset(width)
                hit_configured_max = True
            else:
                hi = min(float(cfg.stenosis_factor_max) * mult, stenosis_cap)
                lo = min(float(cfg.stenosis_factor_min), hi)
                mag = -float(rng.uniform(lo * width, hi * width))
        else:
            mult = cfg.aneurysm_pro_thrombotic_mult if pro_thrombotic else 1.0
            if float(rng.random()) < float(cfg.pathology_max_hit_prob):
                mag = cfg.max_aneurysm_wall_offset(width, pro_thrombotic=pro_thrombotic)
                hit_configured_max = True
            else:
                hi = min(float(cfg.aneurysm_factor_max) * mult, float(cfg.max_aneurysm_factor))
                lo = min(float(cfg.aneurysm_factor_min), hi)
                mag = float(rng.uniform(lo * width, hi * width))

        softened = False
        if float(severity_scale) != 1.0:
            mag = float(mag) * float(severity_scale)
            # No longer a max-magnitude draw: take the gentler transition and free peak/skew.
            hit_configured_max = False
            softened = True

        min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
        if (forced_max and not softened) or hit_configured_max:
            peak = int(min_idx + 0.5 * (max_idx - min_idx))
        else:
            peak = int(rng.integers(min_idx, max_idx))

        if (magnitude_mode == "max_stenosis" or pro_thrombotic) and not softened:
            # Sharper geometric transition to trigger sr_grad_flow < -750 (sgt)
            std_dev = float(rng.uniform(0.02 * n, 0.05 * n))
        else:
            std_dev = float(rng.uniform(0.04 * n, 0.10 * n))

        x_idx = np.arange(n)
        gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
        if (forced_max and not softened) or hit_configured_max:
            skew = np.ones(n, dtype=float)
        else:
            skew_factor = float(rng.uniform(-0.3, 0.3))
            skew = 1.0 + skew_factor * ((x_idx - peak) / n)
        offsets = mag * gauss * skew
        pathology_peak_frac = float(peak) / float(max(n - 1, 1))

    if v_type == "straight":
        path_loc = 2
    elif magnitude_mode in ("max_stenosis", "max_aneurysm") or hit_configured_max:
        # Max stenosis always uses both walls so diameter-occlusion targets stay exact.
        # Forced max aneurysm: one-wall (default) or mirrored (both walls -> up to 3x inlet).
        # Random pathology_max_hit snaps stay both-wall so the 3x lumen target remains defined.
        if (
            v_type == "aneurysm"
            and aneurysm_wall_mode == "one"
            and magnitude_mode == "max_aneurysm"
        ):
            path_loc = int(rng.choice([0, 1]))
        else:
            path_loc = 2  # both walls so occlusion / 3x-width targets are well-defined
    else:
        # Bias toward both-wall pathology so random draws reach the configured maxes more often.
        path_loc = int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))

    # 2. Universal Centerline Tortuosity (disabled for straight_max / clean forced maxes)
    if straight_max or forced_max:
        tortuosity: List[float] = [0.0] * max(0, n - 4)
    else:
        f1, f2 = rng.uniform(0.5, 1.5), rng.uniform(1.5, 2.5)
        meander = np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi)) + \
                  0.5 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi))

        # Higher tortuosity triggers separation on inner radii
        max_meander = (0.15 if pro_thrombotic else 0.10) * width
        meander = (meander / max(1e-9, float(np.max(np.abs(meander))))) * max_meander
        meander *= np.sin(np.pi * t)
        tortuosity = meander[2:n - 2].tolist()

    # 3. Independent Wall Roughness (zero at configured max so lumen targets stay exact)
    def get_wall_noise():
        if forced_max or hit_configured_max:
            return [0.0] * n
        if pro_thrombotic:
            # Higher frequency and amplitude to create local micro-cavities (sr < 25)
            f_h1, f_h2 = rng.uniform(2.0, 4.0), rng.uniform(4.0, 6.0)
            max_noise = 0.08 * width
        else:
            f_h1, f_h2 = rng.uniform(1.0, 2.5), rng.uniform(2.5, 4.0)
            max_noise = 0.05 * width

        noise = np.sin(2 * np.pi * f_h1 * t + rng.uniform(0, 2 * np.pi)) + \
                0.5 * np.sin(2 * np.pi * f_h2 * t + rng.uniform(0, 2 * np.pi))
        noise = (noise / max(1e-9, float(np.max(np.abs(noise))))) * max_noise
        noise *= np.sin(np.pi * t) ** 0.5
        return noise.tolist()

    noise_top = get_wall_noise()
    noise_bot = get_wall_noise()

    # 4. Safely Bound Parameters (Unchanged)
    max_half_width = (width / 2.0) + max(0, float(np.max(offsets))) + (0.08 * width)
    min_safe_radius = 1.6 * max_half_width
    max_safe_angle_span = L / min_safe_radius

    if curve_type == "straight":
        angle_span, amplitude = 0.0, 0.0
    elif curve_type in ("arc", "hook"):
        if curve_type == "arc":
            target_angle = float(rng.uniform(np.deg2rad(45), np.deg2rad(100)))
        else:
            target_angle = float(rng.uniform(np.deg2rad(100), np.deg2rad(125)))

        angle_span = min(target_angle, max_safe_angle_span)
        amplitude = 0.0
    else:  # s_curve
        angle_span = 0.0
        amp_mag = min(float(rng.uniform(0.003, 0.007)), L * 0.15)
        amplitude = amp_mag

    bend_mode = resolve_bend_sign_mode()
    bend_sign = 1.0
    if level >= 1 and not straight_max:
        if curve_type in ("arc", "hook"):
            bend_sign = (
                1.0
                if bend_mode == "down_only"
                else float(rng.choice([-1.0, 1.0]))
            )
        elif curve_type == "s_curve":
            if bend_mode == "bidirectional":
                amplitude *= float(rng.choice([-1.0, 1.0]))
            else:
                amplitude = abs(float(amplitude))

    # 5. Wound Sites
    wound_prob = wound_probability if wound_probability is not None else cfg.wound_probability
    wound_sites = _sample_wound_sites(
        rng,
        cfg,
        wound_probability=float(wound_prob),
        wound_at_pathology=wound_at_pathology,
        pathology_peak_frac=pathology_peak_frac,
    )

    return {
        "idx": idx,
        "level": level,
        "v_type": v_type,
        "curve_type": curve_type,
        # Carried into the written `.json` so a repair can re-draw the SAME class of vessel.
        "pathology_mode": pathology_mode,
        "magnitude_mode": magnitude_mode,
        "hit_configured_max": bool(hit_configured_max),
        "width": width,
        "angle_span": angle_span,
        "amplitude": amplitude,
        "bend_sign": bend_sign,
        "bend_sign_mode": bend_mode,
        "jitter": [],
        "tortuosity": tortuosity,
        "noise_top": noise_top,
        "noise_bot": noise_bot,
        "offsets": offsets.tolist(),
        "path_loc": path_loc,
        "pathology_mode": pathology_mode or "random",
        "aneurysm_wall_mode": aneurysm_wall_mode,
        "pathology_peak_frac": pathology_peak_frac,
        "wound_at_pathology": bool(wound_at_pathology),
        "wound_sites": wound_sites,
    }


def recompute_pathology_offsets(
    params: Dict[str, Any],
    cfg: VesselConfig,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    path_loc_frac: float | None = None,
) -> Dict[str, Any]:
    """Recompute stenosis/aneurysm offsets when pathology type or strength changes."""
    out = dict(params)
    v_type = str(out.get("v_type", "straight"))
    width = float(out.get("width", cfg.width_min))
    level = int(out.get("level", 0))
    pro_thrombotic = level == 2
    n = cfg.num_ctrl_pts
    t = np.linspace(0, 1, n)
    strength = float(np.clip(strength, 0.0, 1.0))

    offsets = np.zeros(n)
    at_max = False
    if v_type != "straight" and strength > 0.0:
        if v_type in ("stenosis", "occlusion"):
            mult = cfg.stenosis_pro_thrombotic_mult if pro_thrombotic else 1.0
            at_max = strength >= 1.0 - 1e-9
            if at_max:
                mag = cfg.max_stenosis_wall_offset(width)
            else:
                stenosis_cap = 0.5 * float(cfg.max_stenosis_diameter_occlusion)
                hi = min(float(cfg.stenosis_factor_max) * mult, stenosis_cap)
                lo = min(float(cfg.stenosis_factor_min), hi)
                mag = -float(rng.uniform(lo * width, hi * width))
                mag *= strength
        else:
            mult = cfg.aneurysm_pro_thrombotic_mult if pro_thrombotic else 1.0
            at_max = strength >= 1.0 - 1e-9
            if at_max:
                mag = cfg.max_aneurysm_wall_offset(width, pro_thrombotic=pro_thrombotic)
            else:
                hi = min(float(cfg.aneurysm_factor_max) * mult, float(cfg.max_aneurysm_factor))
                lo = min(float(cfg.aneurysm_factor_min), hi)
                mag = float(rng.uniform(lo * width, hi * width))
                mag *= strength

        min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
        if path_loc_frac is not None:
            peak = int(min_idx + float(np.clip(path_loc_frac, 0.0, 1.0)) * (max_idx - min_idx))
        else:
            peak = int(rng.integers(min_idx, max_idx))

        if pro_thrombotic:
            std_dev = float(rng.uniform(0.02 * n, 0.05 * n))
        else:
            std_dev = float(rng.uniform(0.04 * n, 0.10 * n))

        x_idx = np.arange(n)
        gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
        skew_factor = float(rng.uniform(-0.3, 0.3))
        skew = 1.0 + skew_factor * ((x_idx - peak) / n)
        offsets = mag * gauss * skew

    out["offsets"] = offsets.tolist()
    # Missing key => legacy both-wall max (research / recompute callers).
    if "aneurysm_wall_mode" in out:
        wall_mode = normalize_aneurysm_wall_mode(out.get("aneurysm_wall_mode"))
    else:
        wall_mode = "mirrored"
    out["aneurysm_wall_mode"] = wall_mode
    if v_type == "straight":
        out["path_loc"] = 2
    elif at_max:
        if v_type == "aneurysm" and wall_mode == "one":
            if int(out.get("path_loc", -1)) not in (0, 1):
                out["path_loc"] = int(rng.choice([0, 1]))
        else:
            out["path_loc"] = 2
    elif "path_loc" not in out:
        out["path_loc"] = int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))
    return out


def make_vessel_params(
    idx: int = 0,
    level: int = 0,
    cfg: VesselConfig | None = None,
    rng: np.random.Generator | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Sample one vessel parameter dict (same keys as ``_sample_params``)."""
    cfg = cfg or VesselConfig(phase="kinematics")
    rng = rng or np.random.default_rng(0)
    base = _sample_params(idx, level, cfg, rng)
    base.update(overrides)
    return base


def build_vessel_mesh(
    params: Dict[str, Any],
    cfg_dict: Dict[str, Any],
    output_dir: str | Path,
) -> Tuple[int, bool, str]:
    """Build and mesh one vessel via Gmsh; returns ``(idx, success, error_msg)``."""
    unit = cfg_dict.get("unit", "m")
    unit_scale = 100.0 if unit == "cm" else 1.0
    lc_min, mesh_lc = _gmsh_size_bounds(cfg_dict, unit_scale)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg_dict["mesh_size_factor"])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_lc)
        return _build_and_mesh(params, cfg_dict, str(output_dir))
    finally:
        gmsh.finalize()


def _gmsh_size_bounds(cfg_dict: Dict[str, Any], unit_scale: float,
                      d_bar: Optional[float] = None) -> Tuple[float, float]:
    """``(lc_min, lc_max)`` for Gmsh, in the mesh's own unit.

    With ``d_bar``, the open-lumen size is ``mesh_h_nd_target * d_bar`` -- resolution set in the
    non-dimensional units the model actually consumes, so every vessel lands on deployment's
    spacing regardless of its physical size.  ``Mesh.MeshSizeFactor`` still multiplies what Gmsh
    does with it, so the request is divided by it here; measured, achieved spacing then tracks
    the request to within 1%.  Without ``d_bar`` it falls back to the fixed ``mesh_lc``.

    ``CharacteristicLengthMin`` and ``Max`` used to BOTH be set to ``mesh_lc``, which clamps
    every element to one size and makes the per-point ``lc`` passed to ``addPoint`` inert -- the
    mesh was uniform no matter what the geometry asked for.  The floor is now
    ``mesh_lc * mesh_lc_min_ratio`` so lumen-aware point sizes actually take effect.

    ``mesh_refine`` (default 1.0) scales the whole request down for a repair pass on a vessel
    COMSOL could not solve.
    """
    h_nd = float(cfg_dict.get("mesh_h_nd_target", 0.0) or 0.0)
    if d_bar and h_nd > 0.0:
        size_factor = max(float(cfg_dict.get("mesh_size_factor", 1.0)), 1e-6)
        lc_max = (h_nd * float(d_bar) / size_factor) * float(cfg_dict.get("mesh_refine", 1.0))
    else:
        lc_max = float(cfg_dict["mesh_lc"]) * unit_scale * float(cfg_dict.get("mesh_refine", 1.0))
    ratio = float(cfg_dict.get("mesh_lc_min_ratio", 0.12))
    return max(lc_max * ratio, 1e-12), lc_max


def _install_lumen_size_callback(
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
    lc_min: float,
    lc_max: float,
    min_elems_across: int,
) -> None:
    """Size the mesh by the LOCAL LUMEN WIDTH, so a stenosis throat is resolved not spanned.

    A uniform ``mesh_lc`` puts about five elements across a severe throat -- the 2026-08-28
    cohort's tightest vessels close to 3.7 mm against a 0.75 mm element -- and COMSOL then fails
    to converge there.  Every one of the 39 unsolved vessels was a stenosis geometry, at a rate
    that climbed monotonically with stenosis ratio (RGP_DEQ_REPAIR_PLAN.md B27).

    This has to be a **size callback**, not per-point ``lc`` on ``addPoint``: the wall stations
    are B-spline control points, not model vertices, so ``Mesh.MeshSizeFromPoints`` ignores
    them.  Passing sizes there changed nothing at all -- the re-meshed node counts came back
    byte-identical to the shipped uniform meshes (3525, 3969, ...), which is how it was caught.

    ``top_coords[i]`` and ``bot_coords[i]`` are the two walls at one station, so their separation
    is the lumen width there.  A query point takes the width of its nearest station and asks for
    ``width / min_elems_across``, clamped to ``[lc_min, lc_max]``: the throat refines, the open
    lumen keeps ``lc_max`` and pays nothing.
    """
    top = np.asarray(top_coords, dtype=float)
    bot = np.asarray(bot_coords, dtype=float)
    mid = 0.5 * (top + bot)
    width = np.linalg.norm(top - bot, axis=1)
    want = np.clip(width / max(int(min_elems_across), 1), lc_min, lc_max)

    def _size(dim, tag, x, y, z, lc):
        k = int(np.argmin((mid[:, 0] - x) ** 2 + (mid[:, 1] - y) ** 2))
        return float(want[k])

    gmsh.model.mesh.setSizeCallback(_size)




def _build_and_mesh(
        params: Dict[str, Any],
        cfg_dict: Dict[str, Any],
        output_dir: str,
) -> Tuple[int, bool, str]:
    """Build, mesh, and save one vessel. Returns (idx, success, error_msg)."""
    from src.data_gen.lib.vessel_geometry import (
        GeometryValidationError,
        compute_geometry_from_params,
        compute_geometry_from_walls,
        validate_geometry,
    )

    idx = int(params["idx"])
    try:
        if str(params.get("geometry_mode", "parametric")) == "edited_walls":
            top = np.asarray(params["top_coords"], dtype=float)
            bot = np.asarray(params["bot_coords"], dtype=float)
            geom = compute_geometry_from_walls(
                top,
                bot,
                idx=idx,
                unit=str(cfg_dict.get("unit", "m")),
                params=params,
                base_length=float(cfg_dict["base_length"]),
            )
        else:
            geom = compute_geometry_from_params(params, cfg_dict)
        validate_geometry(geom, cfg_dict)
        return _mesh_geometry(geom, cfg_dict, output_dir)
    except GeometryValidationError as exc:
        return idx, False, str(exc)
    except Exception as exc:
        try:
            gmsh.model.remove()
        except Exception:
            pass
        return idx, False, str(exc)


def _split_wall_at_wounds(n: int, wound_sites: list) -> list[tuple[int, int, bool]]:
    """Split control-point array into segments: [(start, end, is_wound), ...]."""
    segments = []
    cursor = 0
    for ws in sorted(wound_sites, key=lambda w: w.center_frac):
        i_lo = max(1, int((ws.center_frac - ws.half_width_frac) * n))
        i_hi = min(n - 2, int((ws.center_frac + ws.half_width_frac) * n))
        # Enforce minimum size for wound segment (3 points) to prevent degenerate splines
        if i_hi - i_lo < 2:
            i_lo = max(1, i_lo - 1)
            i_hi = min(n - 2, i_hi + 1)
        if cursor < i_lo:
            segments.append((cursor, i_lo, False))
        segments.append((i_lo, i_hi, True))
        cursor = i_hi
    if cursor < n - 1:
        segments.append((cursor, n - 1, False))
    return segments


def _mesh_geometry(
    geom,
    cfg_dict: Dict[str, Any],
    output_dir: str,
) -> Tuple[int, bool, str]:
    """Gmsh meshing + file write from a ``VesselGeometry``."""
    idx = int(geom.idx)
    out = Path(output_dir)
    unit = str(cfg_dict.get("unit", "m"))
    unit_scale = 100.0 if unit == "cm" else 1.0
    # Per VESSEL, not per session: `d_bar` varies 3.9x across a cohort, and the session-level
    # bounds set in the worker cannot know it.
    lc_min, lc = _gmsh_size_bounds(cfg_dict, unit_scale, d_bar=float(getattr(geom, "d_bar", 0.0)))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    top_coords = geom.top_coords
    bot_coords = geom.bot_coords

    try:
        gmsh.model.add(f"vessel_{idx}")

        top_tags = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in top_coords]
        bot_tags = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in bot_coords]
        
        wound_sites = getattr(geom, "wound_sites", [])
        top_segments = _split_wall_at_wounds(len(top_tags), wound_sites)
        
        s_top_list = []
        healthy_top_curves = []
        wound_top_curves = []
        for (start, end, is_wound) in top_segments:
            pts = top_tags[start:end+1]
            curve = gmsh.model.geo.addBSpline(pts)
            s_top_list.append(curve)
            if is_wound:
                wound_top_curves.append(curve)
            else:
                healthy_top_curves.append(curve)
                
        # Bottom is reversed, so we create the curves in reverse order of the segments
        s_bot_list = []
        healthy_bot_curves = []
        wound_bot_curves = []
        for (start, end, is_wound) in reversed(top_segments):
            # reverse the points for the bottom curve
            pts = list(reversed(bot_tags[start:end+1]))
            curve = gmsh.model.geo.addBSpline(pts)
            s_bot_list.append(curve)
            if is_wound:
                wound_bot_curves.append(curve)
            else:
                healthy_bot_curves.append(curve)

        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        cl_components = s_top_list + [l_out] + s_bot_list + [l_in]
        cl = gmsh.model.geo.addCurveLoop(cl_components)
        s = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        tags = cfg_dict["TAGS"]
        gmsh.model.addPhysicalGroup(1, [l_in], tags["Inlet"], name="Inlet")
        gmsh.model.addPhysicalGroup(1, [l_out], tags["Outlet_1"], name="Outlet_1")
        gmsh.model.addPhysicalGroup(2, [s], tags["Fluid_Domain"], name="Fluid_Domain")

        # Boundary-layer patch cohort: split the lumped "Walls" group so COMSOL can apply
        # no-slip on the bottom (where the clot attaches) and slip/symmetry on the top
        # (unperturbed freestream). Standard vessels keep the single lumped "Walls" group.
        if "Wall_Bottom" in tags and "Slip_Boundary" in tags:
            gmsh.model.addPhysicalGroup(1, healthy_bot_curves, tags["Wall_Bottom"], name="Wall_Bottom")
            gmsh.model.addPhysicalGroup(1, healthy_top_curves, tags["Slip_Boundary"], name="Slip_Boundary")
        else:
            gmsh.model.addPhysicalGroup(1, healthy_top_curves + healthy_bot_curves, tags["Walls"], name="Walls")
            
        if "Wound" in tags and (wound_top_curves or wound_bot_curves):
            gmsh.model.addPhysicalGroup(1, wound_top_curves + wound_bot_curves, tags["Wound"], name="Wound")

        _install_lumen_size_callback(
            top_coords, bot_coords, lc_min, lc,
            int(cfg_dict.get("mesh_min_elems_across", 8)),
        )
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeSizeCallback()

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) < 50:
            raise RuntimeError(f"Too few nodes ({len(node_tags)})")

        gmsh.write(str(out / f"vessel_{idx}.msh"))
        gmsh.write(str(out / f"vessel_{idx}.nas"))
        gmsh.model.remove()

        with open(out / f"vessel_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(geom.meta, f, indent=4)

        return idx, True, ""
    except Exception as exc:
        try:
            gmsh.model.mesh.removeSizeCallback()
        except Exception:
            pass
        try:
            gmsh.model.remove()
        except Exception:
            pass
        return idx, False, str(exc)



def remesh_vessels_from_meta(
    stems: List[str],
    mesh_dir: Path | str,
    cfg_dict: Dict[str, Any],
    *,
    mesh_refine: float = 0.6,
    min_elems_across: Optional[int] = None,
) -> List[Tuple[str, bool, str]]:
    """Re-mesh existing vessels **at their own geometry** but finer.  Returns per-stem results.

    A vessel's ``.json`` carries ``top_wall_pts`` / ``bot_wall_pts`` -- the exact wall polylines
    its mesh was built from -- so a failed solve can be retried on the *same* vessel at higher
    resolution instead of being replaced by a different one.  That distinction matters: solve
    failure is not random, it rises monotonically with stenosis ratio (2.9% below 1.5, 40.6%
    above 3.0; RGP_DEQ_REPAIR_PLAN.md B27), so re-rolling geometry would quietly bias the cohort
    toward the shapes that solve easily -- exactly the tail the corpus exists to provide.
    """
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls

    mesh_dir = Path(mesh_dir)
    cfg = dict(cfg_dict)
    cfg["mesh_refine"] = float(mesh_refine)
    if min_elems_across is not None:
        cfg["mesh_min_elems_across"] = int(min_elems_across)

    unit_scale = 100.0 if str(cfg.get("unit", "m")) == "cm" else 1.0
    lc_min, lc_max = _gmsh_size_bounds(cfg, unit_scale)

    results: List[Tuple[str, bool, str]] = []
    # The repair runs inside a live pipeline process that may already hold a Gmsh session from
    # vessel generation.  Re-initialising is only a warning, but finalising someone else's
    # session is not -- so leave it alone if we did not open it.
    _owned = not gmsh.isInitialized()
    if _owned:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg["mesh_size_factor"])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

        for stem in stems:
            try:
                meta = json.loads((mesh_dir / f"{stem}.json").read_text(encoding="utf-8"))
                # `_build_meta` stores the wall polylines NON-DIMENSIONALLY (`coords / d_bar`),
                # while the mesher works in the mesh's own length unit.  `meta["d_bar"]` is
                # already in that unit, so it is the exact scale back.
                d_bar = float(meta["d_bar"])
                top = np.asarray(meta["top_wall_pts"], dtype=float) * d_bar
                bot = np.asarray(meta["bot_wall_pts"], dtype=float) * d_bar
                geom = compute_geometry_from_walls(
                    top, bot, idx=int(meta["id"]), unit=str(meta.get("unit", "m")), params=meta,
                )
                # Put the ORIGINAL meta back.  `compute_geometry_from_walls` stamps
                # `curve_type="edited"` defaults, and rewriting a vessel's `type` / `curve` /
                # `level` would corrupt the cohort's own record of what it contains -- those
                # fields are what preflight and the failure analysis read.
                repaired = dict(meta)
                repaired["mesh_repair"] = {
                    "mesh_refine": float(mesh_refine),
                    "min_elems_across": int(cfg.get("mesh_min_elems_across", 8)),
                    "rounds": int((meta.get("mesh_repair") or {}).get("rounds", 0)) + 1,
                }
                geom.meta = repaired
            except Exception as exc:
                results.append((stem, False, f"cannot rebuild geometry: {exc}"))
                continue
            _, ok, err = _mesh_geometry(geom, cfg, str(mesh_dir))
            results.append((stem, ok, err))
    finally:
        if _owned:
            gmsh.finalize()
    return results



def _wall_severity(top: np.ndarray, bot: np.ndarray, v_type: str) -> float:
    """Stenosis ratio ``median(w)/min(w)`` or aneurysm ratio ``max(w)/median(w)``."""
    w = np.linalg.norm(np.asarray(top) - np.asarray(bot), axis=1)
    med = float(np.median(w))
    if v_type == "aneurysm":
        return float(w.max() / max(med, 1e-12))
    return float(med / max(w.min(), 1e-12))


def _class_from_meta(meta: Dict[str, Any]) -> Tuple[str, str, float]:
    """``(v_type, pathology_mode, severity)`` of an existing vessel, from its own ``.json``.

    ``pathology_mode`` is recorded by the sampler since 2026-08-29; older packs are read off
    ``type`` instead.  The fallback must NOT be ``"random"`` for a pathological vessel -- that
    draws freely from ``[straight, stenosis, aneurysm]`` and turns a 5.0 stenosis into a healthy
    tube, which is how this was caught.
    """
    v_type = str(meta.get("type", "straight_straight")).split("_")[0]
    mode = meta.get("pathology_mode")
    if not mode or str(mode) == "random":
        mode = {"stenosis": "max_stenosis", "aneurysm": "max_aneurysm"}.get(v_type, "random")
    d_bar = float(meta.get("d_bar", 1.0))
    top = np.asarray(meta["top_wall_pts"], dtype=float) * d_bar
    bot = np.asarray(meta["bot_wall_pts"], dtype=float) * d_bar
    return v_type, str(mode), _wall_severity(top, bot, v_type)


def reshape_vessels_from_meta(
    stems: List[str],
    mesh_dir: Path | str,
    cfg: VesselConfig,
    cfg_dict: Dict[str, Any],
    *,
    attempt: int = 1,
    mesh_refine: float = 1.0,
    min_elems_across: Optional[int] = None,
    max_draws: int = 24,
    severity_target: float = 0.70,
) -> List[Tuple[str, bool, str]]:
    """Re-draw an unsolvable vessel as an EASIER sample of the same class.

    Refinement is the right first answer -- it preserves the exact geometry -- but some shapes
    are degenerate at the extreme of the sampler's range: an 80% occlusion whose throat walls
    very nearly touch, or a Gaussian bump landing on a tight bend.  No mesh saves those.  The
    2026-08-29 regeneration measured it: two rounds of global refinement (0.6x then 0.4x, the
    second reaching 25.7k nodes) recovered **2 of 39**, and the survivors were the extreme tail
    -- the worst fifteen all at stenosis ratio 4.4+ against a cohort median of 1.29.

    So the last stage replaces the *shape*: same ``level``, same pathology class, a different RNG
    draw -- at ``severity_target`` times the original's stenosis / aneurysm ratio.

    **The target descends across attempts, and that is the whole point.**  A first version held
    severity at >= 0.85x by rejection sampling, to stop the severe tail evaporating.  It did stop
    that, and it also made the substitution useless: an equally extreme vessel fails for the same
    reason the original did.  Measured on the 2026-08-29 regeneration -- 38 vessels re-drawn,
    **36 still unsolved**.  A replacement has to be easier than the thing it replaces or it is
    not a replacement.

    Candidates are built cheaply (no meshing) and scored by distance to ``severity_target * want``
    with a 3x penalty for overshoot, so the draw lands at or just under the goal rather than
    collapsing straight to a healthy tube.  ``reshaped_from`` records what was given up.
    """
    mesh_dir = Path(mesh_dir)
    cfg_d = dict(cfg_dict)
    cfg_d["mesh_refine"] = float(mesh_refine)
    if min_elems_across is not None:
        cfg_d["mesh_min_elems_across"] = int(min_elems_across)

    from src.data_gen.lib.vessel_geometry import (
        GeometryValidationError, compute_geometry_from_params, validate_geometry)

    unit_scale = 100.0 if str(cfg_d.get("unit", "m")) == "cm" else 1.0
    lc_min, lc_max = _gmsh_size_bounds(cfg_d, unit_scale)

    results: List[Tuple[str, bool, str]] = []
    _owned = not gmsh.isInitialized()
    if _owned:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg_d["mesh_size_factor"])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

        for stem in stems:
            try:
                meta = json.loads((mesh_dir / f"{stem}.json").read_text(encoding="utf-8"))
                idx = int(meta["id"])
                level = int(meta.get("level", 0))
                v_type, mode, want = _class_from_meta(meta)
            except Exception as exc:
                results.append((stem, False, f"cannot read class: {exc}"))
                continue

            # Solve for the target severity, do not just scale the wall offset by it.  The
            # stenosis ratio `median(w)/min(w)` is nonlinear in the offset -- scaling depth by
            # 0.70 measured 0.41x severity, so a blind scale overshoots and throws away more of
            # the severe tail than the rung asks for.  Severity is monotone in the scale, and
            # building a candidate is cheap (no meshing), so bisect.
            goal = max(float(severity_target) * want, 1.02)
            n_streams = max(1, int(max_draws) // 8)
            best = None       # (score, severity, params)
            for k in range(n_streams):
                lo, hi = 0.10, 1.0
                for _ in range(7):
                    mid = 0.5 * (lo + hi)
                    rng = np.random.default_rng(
                        abs(hash((idx, "reshape", int(attempt), k))) % (2**32))
                    params = _sample_params(idx, level, cfg, rng, pathology_mode=mode,
                                            severity_scale=mid)
                    if str(params.get("v_type")) != v_type:
                        break
                    try:
                        geom = compute_geometry_from_params(params, cfg_d)
                        validate_geometry(geom, cfg_d)
                    except (GeometryValidationError, Exception):
                        hi = mid            # invalid tends to mean too extreme
                        continue
                    sev = _wall_severity(geom.top_coords, geom.bot_coords, v_type)
                    # Overshooting the goal (still too hard) is penalised 3x: when the bisection
                    # cannot land exactly, err toward the easier side -- the whole point of the
                    # rung is that the vessel solves.
                    score = abs((sev - goal) if sev <= goal else 3.0 * (sev - goal))
                    if best is None or score < best[0]:
                        best = (score, sev, params)
                    if abs(sev - goal) <= 0.05 * goal:
                        break
                    if sev > goal:
                        hi = mid
                    else:
                        lo = mid

            if best is None:
                results.append((stem, False,
                                f"no valid {v_type} candidate for severity {goal:.2f}"))
                continue
            _, sev, params = best
            params["reshaped_from"] = {
                "attempt": int(attempt),
                "original_type": meta.get("type"),
                "pathology_mode": mode,
                "severity_was": round(float(want), 3),
                "severity_now": round(float(sev), 3),
                "severity_target": round(float(severity_target), 3),
            }
            _, ok, err = _build_and_mesh(params, cfg_d, str(mesh_dir))
            results.append((stem, ok, err))
    finally:
        if _owned:
            gmsh.finalize()
    return results


def _worker_run_chunk(
    chunk: List[Dict[str, Any]],
    cfg_dict: Dict[str, Any],
    output_dir: str,
) -> List[Tuple[int, bool, str]]:
    """
    Initialise Gmsh ONCE per process, process every sample in the chunk,
    then finalise.  Gmsh init/finalize is the heaviest fixed cost, so
    processing multiple samples per worker amortises it.
    """
    unit = cfg_dict.get("unit", "m")
    unit_scale = 100.0 if unit == "cm" else 1.0
    lc_min, mesh_lc = _gmsh_size_bounds(cfg_dict, unit_scale)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",          0)
    gmsh.option.setNumber("Mesh.Algorithm",            6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Smoothing",            5)
    gmsh.option.setNumber("Mesh.MshFileVersion",       2.2)
    gmsh.option.setNumber("Mesh.Binary",               0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",    1)
    gmsh.option.setNumber("Mesh.SaveAll",              0)
    gmsh.option.setNumber("Mesh.MeshSizeFactor",       cfg_dict["mesh_size_factor"])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_lc)

    results = [_build_and_mesh(p, cfg_dict, output_dir) for p in chunk]

    gmsh.finalize()
    return results


class VesselGenerator:
    """Generates 2D vessel meshes with parametric pathologies using Gmsh."""

    def __init__(self, phase: str = "kinematics", output_dir: Optional[str | Path] = None) -> None:
        self.cfg          = VesselConfig(phase=phase)
        self.project_root = get_project_root()
        self.output_dir   = Path(output_dir) if output_dir else self.project_root / self.cfg.mesh_input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            migrate_legacy_vessel_meshes(self.output_dir)

    def _sample_one(
        self,
        idx: int,
        level: int,
        rng: np.random.Generator,
        pathology_mode: str | None = None,
        aneurysm_wall_mode: str | None = None,
        wound_probability: float | None = None,
        wound_at_pathology: bool = False,
    ) -> Dict[str, Any]:
        """Sample one vessel parameter dict.

        Subclasses (e.g. ``BoundaryLayerPatchGenerator``) override this to swap in a
        different parameter sampler while reusing the shared ``run_pipeline`` driver.
        """
        return _sample_params(
            idx,
            level,
            self.cfg,
            rng,
            pathology_mode=pathology_mode,
            aneurysm_wall_mode=aneurysm_wall_mode,
            wound_probability=wound_probability,
            wound_at_pathology=wound_at_pathology,
        )

    def _cfg_dict(self) -> Dict[str, Any]:
        return {
            "num_ctrl_pts":       self.cfg.num_ctrl_pts,
            "base_length":        self.cfg.base_length,
            "mesh_lc":            self.cfg.mesh_lc,
            "mesh_size_factor":   self.cfg.mesh_size_factor,
            "mesh_min_elems_across": self.cfg.mesh_min_elems_across,
            "mesh_lc_min_ratio":  self.cfg.mesh_lc_min_ratio,
            "mesh_h_nd_target":   self.cfg.mesh_h_nd_target,
            "width_min":          self.cfg.width_min,
            "width_max":          self.cfg.width_max,
            "stenosis_factor_min": self.cfg.stenosis_factor_min,
            "stenosis_factor_max": self.cfg.stenosis_factor_max,
            "min_lumen_width_fraction": self.cfg.min_lumen_width_fraction,
            "aneurysm_factor_min": self.cfg.aneurysm_factor_min,
            "aneurysm_factor_max": self.cfg.aneurysm_factor_max,
            "max_stenosis_diameter_occlusion": self.cfg.max_stenosis_diameter_occlusion,
            "max_aneurysm_factor": self.cfg.max_aneurysm_factor,
            "pathology_max_hit_prob": self.cfg.pathology_max_hit_prob,
            "stenosis_pro_thrombotic_mult": self.cfg.stenosis_pro_thrombotic_mult,
            "aneurysm_pro_thrombotic_mult": self.cfg.aneurysm_pro_thrombotic_mult,
            "TAGS":               dict(self.cfg.TAGS),
        }

    # ------------------------------------------------------------------
    # Visualisation  (reads saved .msh files; main-process only)
    # ------------------------------------------------------------------

    def visualize_saved(self, indices: List[int], max_plots: int = 9) -> None:
        """Load and plot already-saved .msh files."""
        import meshio

        indices = indices[:max_plots]
        cols = min(3, len(indices))
        rows = math.ceil(len(indices) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for ax, idx in zip(axes, indices):
            path = self.output_dir / f"vessel_{idx}.msh"
            meta_path = self.output_dir / f"vessel_{idx}.json"
            if not path.exists():
                ax.set_visible(False)
                continue
            try:
                mesh  = meshio.read(str(path))
                nodes = mesh.points[:, :2]
                tris  = mesh.cells_dict.get("triangle")
                if tris is None:
                    ax.set_visible(False)
                    continue
                poly = PolyCollection(
                    nodes[tris], edgecolors="black", facecolors="lightblue", linewidths=0.1
                )
                ax.add_collection(poly)
                ax.autoscale_view()
                ax.set_aspect("equal")
                title = f"vessel_{idx}"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if "d_inlet" in meta:
                            inlet_diameter = float(meta["d_inlet"])
                            if str(meta.get("unit", "m")).lower() == "m":
                                inlet_diameter_cm = inlet_diameter * 100.0
                            else:
                                inlet_diameter_cm = inlet_diameter
                            title = f"{title} | d_inlet={inlet_diameter_cm:.2f} cm"
                    except Exception:
                        pass
                ax.set_title(title, fontsize=8)
            except Exception:
                ax.set_title(f"vessel_{idx} ERROR", fontsize=8)

        for ax in axes[len(indices):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        n: int = 50,
        level: int = 0,
        level_mix: Optional[Dict[int, int]] = None,
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
        unit: str = "m",
        pathology_mode: str | None = None,
        aneurysm_wall_mode: str | None = None,
        wound_probability: float | None = None,
        wound_at_pathology: bool = False,
    ) -> None:
        """
        Parallel batch vessel generation.

        Parameters
        ----------
        n           : number of vessels to produce in this run (file indices ``start_idx`` …)
        level       : geometry complexity when ``level_mix`` is None
                      (0 = mostly straight, 1 = curved, 2 = pro-thrombotic / high-clot)
        level_mix   : optional per-level counts ``{0: n0, 1: n1, 2: n2}`` (must sum to ``n``);
                      shuffled across indices for a mixed cohort in one run
        max_retries : retry attempts for failed samples (same ``idx`` as the failure; new parameters)
        num_workers : worker processes (default: cpu_count - 1, min 1)
        chunk_size  : samples per worker chunk (default: auto-balanced)
        seed        : integer seed for reproducibility (None = random)
        start_idx   : first vessel index ``vessel_{idx}.*``. If ``None``, appends after the
                      highest existing index in ``output_dir`` (no overwrite). Pass ``0`` to
                      fill from the beginning (may overwrite existing files).
        pathology_mode : ``None``/``random`` (default), ``max_stenosis`` (~80% diameter
                         occlusion at peak), ``max_aneurysm`` (local width up to 3x inlet),
                         or ``straight_max`` (straight vessel + max stenosis or aneurysm).
        aneurysm_wall_mode : ``one`` (default, max offset on a single wall) or
                             ``mirrored`` (both walls). Applies to max-strength aneurysms.
        wound_probability : chance each vessel gets a Wound physical group (``1`` via ``--wound``).
        wound_at_pathology : if True, center wounds on the stenosis/aneurysm peak when present;
                             straight vessels keep the random ``wound_center_frac_range`` draw.
        """
        # A mix spec (`a:0.7,b:0.3`) is expanded per vessel later; only a single mode is
        # normalised here, or the whole spec would be rejected as an unknown mode.
        _is_mix = bool(pathology_mode) and ("," in str(pathology_mode) or ":" in str(pathology_mode))
        if not _is_mix:
            pathology_mode = normalize_pathology_mode(pathology_mode)
        aneurysm_wall_mode = normalize_aneurysm_wall_mode(aneurysm_wall_mode)
        phys_cores  = os.cpu_count() or 1
        num_workers = max(1, phys_cores - 1) if num_workers is None else num_workers
        num_workers = min(num_workers, n)

        if start_idx is None:
            start_idx = _next_vessel_index(self.output_dir)

        if level_mix is not None:
            mix_msg = ", ".join(f"L{k}={v}" for k, v in sorted(level_mix.items()))
            logger.info(
                f"Generating {n} mixed-level vessels ({mix_msg}) → {self.output_dir} "
                f"[indices {start_idx}..{start_idx + n - 1}] "
                f"[{num_workers} workers / {phys_cores} logical cores]"
            )
        else:
            logger.info(
                f"Generating {n} Level-{level} vessels → {self.output_dir} "
                f"[indices {start_idx}..{start_idx + n - 1}] "
                f"[{num_workers} workers / {phys_cores} logical cores]"
            )
        if pathology_mode:
            logger.info("Pathology mode: %s", pathology_mode)
        if aneurysm_wall_mode != "one":
            logger.info("Aneurysm wall mode: %s", aneurysm_wall_mode)
        if wound_probability:
            extra = (
                "; placed at stenosis/aneurysm when present"
                if wound_at_pathology
                else ""
            )
            logger.info("Wound sites enabled (p=%.2f)%s", float(wound_probability), extra)

        cfg_d   = self._cfg_dict()
        cfg_d["unit"] = unit
        out_str = str(self.output_dir)
        rng     = np.random.default_rng(seed)

        # Pre-sample everything in the main process
        per_vessel_levels = cohort_levels(n, level, level_mix, rng)
        # `pathology_mode` may be a single mode (historical) or a mix spec -- see
        # `parse_pathology_mix`.  A mix lets one command cover the severe-stenosis tail that
        # random sampling under-represents, instead of a second run with a second seed.
        if pathology_mode and ("," in str(pathology_mode) or ":" in str(pathology_mode)):
            per_vessel_modes = parse_pathology_mix(pathology_mode, n, rng)
            from collections import Counter as _C
            logger.info("Pathology mix: %s", dict(_C(per_vessel_modes)))
        else:
            per_vessel_modes = [pathology_mode] * n
        all_params = [
            self._sample_one(
                start_idx + i,
                per_vessel_levels[i],
                rng,
                pathology_mode=per_vessel_modes[i],
                aneurysm_wall_mode=aneurysm_wall_mode,
                wound_probability=wound_probability,
                wound_at_pathology=wound_at_pathology,
            )
            for i in range(n)
        ]
        params_lookup = _params_by_idx(all_params)

        # Split into balanced chunks — larger chunks = less IPC overhead
        if chunk_size is None:
            chunk_size = max(1, math.ceil(n / num_workers))
        chunks = [all_params[i : i + chunk_size] for i in range(0, n, chunk_size)]

        # ---- Dispatch ----
        generated: int = 0
        failed_params: List[Dict[str, Any]] = []

        if num_workers == 1:
            logger.info("Executing sequentially in main process to avoid Windows spawn issues.")
            # We can just call the worker function directly
            results = _worker_run_chunk(all_params, cfg_d, out_str)

            # FIX: Initialize the progress bar (pbar) for the single-worker loop
            with tqdm(total=n, desc="Generating vessels", unit="vessel") as pbar:
                for idx, success, err in results:
                    pbar.update(1)
                    if success:
                        generated += 1
                    else:
                        logger.warning(f"[ {idx} ] failed: {err}")
                        if idx in params_lookup:
                            failed_params.append(params_lookup[idx])

        else:
            # Existing multiprocessing logic for num_workers > 1
            with mp.Pool(processes=num_workers) as pool:
                # Submit all chunks
                async_results = [
                    (pool.apply_async(_worker_run_chunk, (chunk, cfg_d, out_str)), chunk)
                    for chunk in chunks
                ]

                with tqdm(total=n, desc="Generating vessels", unit="vessel") as pbar:
                    for async_result, chunk in async_results:
                        try:
                            # Force a hard timeout (e.g., 60 seconds per chunk)
                            # Adjust time based on your average mesh generation speed
                            results = async_result.get(timeout=60)

                            for idx, success, err in results:
                                pbar.update(1)
                                if success:
                                    generated += 1
                                else:
                                    logger.warning(f"[ {idx} ] failed: {err}")
                                    if idx in params_lookup:
                                        failed_params.append(params_lookup[idx])

                        except mp.TimeoutError:
                            logger.error(f"Worker hung (Timeout) — {len(chunk)} samples queued for retry")
                            failed_params.extend(chunk)
                            pbar.update(len(chunk))
                            continue
                        except Exception as exc:
                            logger.error(f"Worker crash: {exc} — {len(chunk)} samples queued for retry")
                            failed_params.extend(chunk)
                            pbar.update(len(chunk))
                            continue

        # ---- Retry failed samples ----
        # Resample geometry parameters but keep the same vessel idx so outputs stay
        # vessel_{start_idx}..vessel_{start_idx+n-1} (replacement, no extra indices).
        for retry_round in range(1, max_retries + 1):
            if not failed_params:
                break
            logger.info(f"Retry {retry_round}/{max_retries}: {len(failed_params)} samples")

            # Reuse the mode this vessel was ORIGINALLY assigned, not the caller's argument.
            # With `--pathology-mix` the argument is a spec like "random:0.72,max_stenosis:0.18"
            # which `_sample_params` cannot parse -- it expects a single resolved mode, and the
            # main loop expands the spec per vessel before calling it.  Passing the raw spec
            # here crashed the run the moment any geometry was rejected and had to be resampled
            # (3 of 250: "outlet curled back past L/3").  Reusing the stored mode also keeps the
            # mix counts exact across retries instead of redrawing them.
            retry_batch = [
                self._sample_one(
                    int(failed_p["idx"]),
                    int(failed_p.get("level", level)),
                    rng,
                    pathology_mode=failed_p.get("pathology_mode") or "random",
                    aneurysm_wall_mode=aneurysm_wall_mode,
                    wound_probability=wound_probability,
                    wound_at_pathology=wound_at_pathology,
                )
                for failed_p in failed_params
            ]

            still_failed: List[Dict[str, Any]] = []
            retry_chunks = [retry_batch[i: i + chunk_size] for i in
                            range(0, len(retry_batch), chunk_size)]

            # [FIX 1: Indented inside the loop]
            # [FIX 2: Swapped to mp.Pool to prevent C++ deadlocks during retries]
            with mp.Pool(processes=min(num_workers, len(retry_batch))) as pool:
                async_results = [
                    (pool.apply_async(_worker_run_chunk, (chunk, cfg_d, out_str)), chunk)
                    for chunk in retry_chunks
                ]

                for async_result, chunk in async_results:
                    try:
                        # Apply the same 60-second timeout here
                        results = async_result.get(timeout=60)

                        for idx, success, err in results:
                            if success:
                                generated += 1
                            else:
                                # Find the original param dict to retry again
                                failed_p = next((p for p in chunk if p["idx"] == idx), None)
                                if failed_p:
                                    still_failed.append(failed_p)

                    except mp.TimeoutError:
                        logger.error(f"Retry worker hung (Timeout) — {len(chunk)} samples failed")
                        still_failed.extend(chunk)
                    except Exception as exc:
                        logger.error(f"Retry worker crash: {exc}")
                        still_failed.extend(chunk)

            failed_params = still_failed

        if failed_params:
            logger.warning(
                f"{len(failed_params)} samples could not be generated after {max_retries} retries."
            )

        logger.info(f"Done. {generated}/{n} vessels saved.")


class VesselGeneratorPhase3(VesselGenerator):
    """Synthetic Phase-3 vessel cohort (same geometry pipeline as ``VesselGenerator(phase='biochem')``)."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        super().__init__(phase="biochem", output_dir=output_dir)

    def run_pipeline(
        self,
        n: int = 50,
        level: int = 0,
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
        unit: str = "m",
        pathology_mode: str | None = None,
        aneurysm_wall_mode: str | None = None,
        wound_probability: float | None = None,
        wound_at_pathology: bool = False,
    ) -> None:
        if start_idx is None:
            start_idx = 0
        return super().run_pipeline(
            n=n,
            level=level,
            max_retries=max_retries,
            num_workers=num_workers,
            chunk_size=chunk_size,
            seed=seed,
            start_idx=start_idx,
            unit=unit,
            pathology_mode=pathology_mode,
            aneurysm_wall_mode=aneurysm_wall_mode,
            wound_probability=wound_probability,
            wound_at_pathology=wound_at_pathology,
        )


def _sample_patch_params(
    idx: int, cfg: VesselConfig, rng: np.random.Generator
) -> Dict[str, Any]:
    """Force a flat 2mm x 300-400um box and define the mu-clot metadata.

    The clot is NOT a hole in the mesh: the channel stays a perfectly flat fluid box.
    The clot morphology (center, height, width, shape, peak viscosity) and the inlet
    shear rate are emitted as metadata so the downstream COMSOL script can paint an
    analytical Carreau viscosity field (mu spikes to ``clot_mu_peak`` over the clot
    footprint on the no-slip bottom wall).

    Domain sizing (avoid the "nozzle"/Venturi artifact):
      * length 2mm gives ~700um unperturbed upstream development, a clot up to ~625um,
        and ~700um downstream for the wake/reattachment zone.
      * channel height 300-400um keeps the displaced flow far from the slip top wall so
        it diffuses upward instead of accelerating through a narrow gap.

    Morphology sweep (bias toward long, flat "smears", not just semicircles):
      * ``clot_width`` swept 100um (~20 nodes) -> 625um (~125 nodes), biased wide.
      * ``clot_shape`` in {plateau, gaussian, bbox}; plateau = sharp leading step,
        long flat top, sharp trailing step (teaches the GNN the front stagnation zone
        vs the parallel shear flow along the top).
    """
    length = float(cfg.base_length)  # 2mm (set by BoundaryLayerPatchGenerator)
    n = cfg.num_ctrl_pts
    lc = float(cfg.mesh_lc)

    # Channel height (wall-normal). Sweep 300-400um so the top slip wall behaves as a
    # near-infinite freestream relative to the boundary-layer-scale clot.
    width = float(rng.uniform(cfg.width_min, cfg.width_max))

    # Clot height: a boundary-layer-scale bump (~2-6 nodes). Keep the blockage ratio low
    # (height / channel height ~ 3-10%) to avoid the artificial nozzle effect.
    clot_height = float(rng.uniform(2.0 * lc, 6.0 * lc))

    # Clot width (streamwise): sweep 20 -> 125 nodes, heavily biased toward wide smears.
    # Beta(2,1) skews the fraction toward 1.0, favoring long flat morphologies.
    min_w_nodes, max_w_nodes = 20.0, 125.0
    wide_frac = float(rng.beta(2.0, 1.0))
    clot_width = (min_w_nodes + wide_frac * (max_w_nodes - min_w_nodes)) * lc

    # Morphology flag: occasionally a flat-topped "plateau", otherwise a smooth Gaussian
    # peak or a uniform bounding box.
    clot_shape = str(rng.choice(["plateau", "gaussian", "bbox"], p=[0.45, 0.35, 0.20]))
    # Transition length of the leading/trailing steps for the plateau (1-2 nodes => sharp).
    clot_edge_width = float(rng.uniform(lc, 2.0 * lc))

    # Sweep viscosity intensity from soft gel to near-solid.
    clot_mu_peak = float(rng.uniform(0.1, 10.0))

    # Sweep local shear rate for COMSOL inlet conditions (U = shear_rate * y).
    shear_rate = float(rng.uniform(50.0, 5000.0))

    # Center the clot so upstream development and downstream wake lengths are guaranteed
    # even for the widest (625um) smears.
    clot_x_center = length / 2.0

    return {
        "idx": idx,
        "level": 0,
        "v_type": "flat_patch",
        "curve_type": "straight",
        "path_loc": 2,
        "width": width,
        "base_length": length,
        "offsets": np.zeros(n).tolist(),
        "noise_top": np.zeros(n).tolist(),
        "noise_bot": np.zeros(n).tolist(),
        "tortuosity": np.zeros(max(n - 4, 0)).tolist(),
        # --- Metadata for COMSOL ---
        "clot_x_center": clot_x_center,
        "clot_height": clot_height,
        "clot_width": clot_width,
        "clot_shape": clot_shape,
        "clot_edge_width": clot_edge_width,
        "clot_mu_peak": clot_mu_peak,
        "inlet_shear_rate": shear_rate,
        "geometry_mode": "parametric",
    }


class BoundaryLayerPatchGenerator(VesselGenerator):
    """Generates pure fluid 2D boundary layer boxes for local Subgraph GNN training.

    NOTE (superseded for residual training): this Gmsh path emits *unstructured triangular*
    meshes. For the local patch baseline (pure linear shear ``u = shear_rate*y``), triangular
    faces bleed spurious ``v`` that would dominate the GNN residual label ``dU``. Prefer the
    structured-grid COMSOL-direct pipeline in ``patch_factory_comsol.py``
    (``PatchFactoryComsolGenerator``), which drives a mapped quad-mesh master ``.mph`` over
    clot parameters. This class is retained for non-residual / quick-mesh use cases.
    """

    # Flat-channel geometry: 2mm (streamwise) x 300-400um (wall-normal). The long domain
    # gives room for upstream development + a long clot + downstream wake; the tall channel
    # keeps the slip top wall a near-infinite freestream (no nozzle/Venturi artifact).
    PATCH_LENGTH: float = 2000e-6
    PATCH_WIDTH_MIN: float = 300e-6
    PATCH_WIDTH_MAX: float = 400e-6
    # Ultra-dense meshing for the local subgraph (~5um node spacing resolves the
    # boundary layer and the 20-125 node clot footprint).
    PATCH_MESH_LC: float = 5e-6

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        super().__init__(phase="patch_factory", output_dir=output_dir)
        # Keep the live VesselConfig consistent with the patch cfg_dict so the parameter
        # sampler (which reads cfg.mesh_lc / cfg.base_length / cfg.width_*) matches meshing.
        self.cfg.mesh_lc = self.PATCH_MESH_LC
        self.cfg.base_length = self.PATCH_LENGTH
        self.cfg.width_min = self.PATCH_WIDTH_MIN
        self.cfg.width_max = self.PATCH_WIDTH_MAX

    def _cfg_dict(self) -> Dict[str, Any]:
        cfg = super()._cfg_dict()
        # Ensure ultra-dense meshing and the flat-box base length for the local subgraph.
        cfg["mesh_lc"] = self.PATCH_MESH_LC
        cfg["base_length"] = self.PATCH_LENGTH
        # Custom tags for the top/bottom boundary-condition split.
        cfg["TAGS"].update({"Wall_Bottom": 4, "Slip_Boundary": 5})
        return cfg

    def _sample_one(
        self,
        idx: int,
        level: int,
        rng: np.random.Generator,
        pathology_mode: str | None = None,
        aneurysm_wall_mode: str | None = None,
        wound_probability: float | None = None,
        wound_at_pathology: bool = False,
        **_unused: Any,
    ) -> Dict[str, Any]:
        # Patch cohort ignores level / pathology / wound knobs: flat clot box.
        del pathology_mode, aneurysm_wall_mode, wound_probability, wound_at_pathology, _unused
        return _sample_patch_params(idx, self.cfg, rng)


def _prompt_int_choice(label: str, allowed: Tuple[int, ...]) -> int:
    """Read an integer from stdin until it is one of ``allowed``."""
    allowed_str = "/".join(str(x) for x in allowed)
    while True:
        raw = input(f"{label} ({allowed_str}): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Enter an integer: {allowed_str}")
            continue
        if v in allowed:
            return v
        print(f"  Must be one of: {allowed_str}")


def _prompt_positive_int(label: str, default: int = 500) -> int:
    """Read a positive integer from stdin; empty input returns ``default``."""
    while True:
        raw = input(f"{label} (>=1) [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  Enter a positive integer.")
            continue
        if v >= 1:
            return v
        print("  Must be at least 1.")


def _prompt_write_mode_vessel() -> bool:
    """Return True to overwrite from index 0, False to append with new indices."""
    while True:
        raw = input("Write mode [1=append new files / 2=overwrite from vessel_0] [1]: ").strip()
        if raw in ("", "1"):
            return False
        if raw == "2":
            return True
        print("  Enter 1 or 2.")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    """Read a yes/no answer; empty input returns ``default``."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{label} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y/yes or n/no.")


def resolve_wound_flags(*, wound: bool, wound_at_pathology: bool) -> tuple[float, bool]:
    """``--wound-at-pathology`` implies wound sites; returns ``(probability, at_peak)``."""
    at_peak = bool(wound_at_pathology)
    return (1.0 if (bool(wound) or at_peak) else 0.0), at_peak


def default_unit_for_phase(phase: str) -> str:
    """COMSOL biochem CFD anchors are CGS (cm); synthetic / kinematics meshes are SI (m)."""
    return "cm" if str(phase).strip().lower() == "biochem_anchors" else "m"


class VesselGenCliSpec(NamedTuple):
    """Resolved non-interactive vessel-generator destination (phase, units, wounds)."""

    phase: str
    unit: str
    wound_probability: float
    wound_at_pathology: bool


def resolve_vessel_gen_cli(args: argparse.Namespace) -> VesselGenCliSpec:
    """Map CLI flags onto the kinematics / synthetic-biochem / CFD-anchor tracks.

    ``--anchors`` is the biochem COMSOL CFD track: ``data/raw/biochem_anchors`` and
    ``unit=cm``. Plain ``--phase 2`` stays the synthetic SI track (``data/raw/biochem``).
    """
    anchors = bool(getattr(args, "anchors", False))
    phase_n = getattr(args, "phase", None)
    if anchors:
        if phase_n is not None and int(phase_n) != 2:
            raise ValueError("--anchors requires --phase 2 (biochem), not kinematics")
        phase = "biochem_anchors"
    else:
        if phase_n is None:
            raise ValueError("Provide --phase, or pass --anchors for biochem COMSOL CFD meshes")
        phase = {1: "kinematics", 2: "biochem"}[int(phase_n)]

    unit_arg = getattr(args, "unit", None)
    unit = str(unit_arg).lower() if unit_arg else default_unit_for_phase(phase)
    if phase == "biochem_anchors" and unit != "cm":
        raise ValueError("Biochem COMSOL anchor meshes must use --unit cm")

    wound_probability, wound_at_pathology = resolve_wound_flags(
        wound=bool(getattr(args, "wound", False)),
        wound_at_pathology=bool(getattr(args, "wound_at_pathology", False)),
    )
    return VesselGenCliSpec(
        phase=phase,
        unit=unit,
        wound_probability=wound_probability,
        wound_at_pathology=wound_at_pathology,
    )


def prompt_wound_options(
    *,
    wound_already: bool = False,
    wound_at_pathology_already: bool = False,
    default_at_pathology: bool = False,
) -> tuple[float, bool]:
    """Interactive wound prompts; CLI flags skip the questions."""
    if wound_already or wound_at_pathology_already:
        return resolve_wound_flags(
            wound=wound_already,
            wound_at_pathology=wound_at_pathology_already,
        )
    wound = _prompt_yes_no("Add wound sites to vessels?", default=False)
    at_peak = False
    if wound:
        at_peak = _prompt_yes_no(
            "Place wounds near stenosis/aneurysm when present?",
            default=default_at_pathology,
        )
    return resolve_wound_flags(wound=wound, wound_at_pathology=at_peak)


def _prompt_unit_choice(default: str = "m") -> str:
    """Read output unit from stdin; valid values are 'm' and 'cm'."""
    default = default.lower().strip()
    if default not in ("m", "cm"):
        default = "m"
    while True:
        raw = input(f"Mesh unit system [m/cm] [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("m", "cm"):
            return raw
        print("  Enter m or cm.")


def _vessel_gen_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate 2D vessel meshes (Gmsh). Use --anchors for biochem COMSOL CFD "
            "(cm meshes in data/raw/biochem_anchors)."
        )
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=None,
        help="Dataset (1=kinematics, 2=biochem; use with --level and -n). Implied by --anchors.",
    )
    p.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Geometry complexity (0=straight, 1=curved, 2=pro-clot)",
    )
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        default=None,
        metavar="N",
        help="How many vessels to generate (use with --phase/--anchors and --level)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility. Omit for a fresh random draw each run (default).",
    )
    p.add_argument("--num-workers", type=int, default=None, help="Worker processes (default: auto)")
    p.add_argument("--chunk-size", type=int, default=None, help="Samples per worker chunk (default: auto)")
    p.add_argument(
        "--unit",
        type=str,
        choices=("m", "cm"),
        default=None,
        help="Mesh unit system. Default: cm with --anchors, otherwise m.",
    )
    p.add_argument(
        "--anchors",
        action="store_true",
        help=(
            "Biochem COMSOL CFD track: write cm meshes to data/raw/biochem_anchors. "
            "Implies --phase 2. Do not use for synthetic SI graphs."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Start vessel indices at 0. Default is to append after existing meshes.",
    )
    p.add_argument(
        "--pathology-mode",
        choices=("random", "max_stenosis", "max_aneurysm", "straight_max"),
        default="random",
        help=(
            "Pathology sampling: random (default), max_stenosis (~80%% occlusion), "
            "max_aneurysm (up to 3x inlet width), or straight_max (straight + max pathology)."
        ),
    )
    p.add_argument(
        "--aneurysm-wall",
        choices=("mirrored", "one"),
        default="one",
        help=(
            "Max-strength aneurysm placement: one wall (default; max wall offset on top "
            "or bottom) or mirrored/both walls."
        ),
    )
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib preview of saved meshes (default: skip; avoids blocking on plot windows).",
    )
    p.add_argument(
        "--wound",
        action="store_true",
        help="Enable synthetic wound sites (adds Wound tag to boundary segments).",
    )
    p.add_argument(
        "--wound-at-pathology",
        action="store_true",
        help=(
            "Center wound sites on the stenosis or aneurysm peak when a pathology is present "
            "(small jitter; straight vessels keep random placement). Implies --wound."
        ),
    )
    p.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    parser = _vessel_gen_arg_parser()
    args = parser.parse_args()

    if args.anchors and args.phase is None:
        args.phase = 2
    trio = (args.phase is not None, args.level is not None, args.num_vessels is not None)
    if any(trio) and not all(trio):
        parser.error(
            "Provide --phase (or --anchors), --level, and -n/--num-vessels together "
            "for non-interactive mode."
        )

    if all(trio):
        try:
            spec = resolve_vessel_gen_cli(args)
        except ValueError as exc:
            parser.error(str(exc))
        phase = spec.phase
        level = args.level
        n_vessels = args.num_vessels
        start_idx = 0 if args.overwrite else None
        show_vessel_plot = bool(args.show_vessel_plot)
        unit_choice = spec.unit
        pathology_mode = normalize_pathology_mode(args.pathology_mode)
        aneurysm_wall_mode = normalize_aneurysm_wall_mode(args.aneurysm_wall)
        wound_probability = spec.wound_probability
        wound_at_pathology = spec.wound_at_pathology
        vg = VesselGenerator(phase=phase)
    else:
        phase_n = int(args.phase) if args.phase is not None else _prompt_int_choice(
            "Dataset (1=kinematics, 2=biochem)", (1, 2)
        )
        level = int(args.level) if args.level is not None else _prompt_int_choice(
            "Level (0=straight, 1=curved, 2=pro-clot)", (0, 1, 2)
        )
        anchors = bool(args.anchors)
        if phase_n == 1 and anchors:
            parser.error("--anchors requires biochem (--phase 2)")
        if phase_n == 2 and not anchors:
            anchors = _prompt_yes_no(
                "Biochem COMSOL CFD anchors (cm, data/raw/biochem_anchors)?",
                default=False,
            )
        phase = "biochem_anchors" if anchors else {1: "kinematics", 2: "biochem"}[phase_n]
        unit_choice = default_unit_for_phase(phase)
        if args.unit is not None:
            unit_choice = str(args.unit).lower()
            if phase == "biochem_anchors" and unit_choice != "cm":
                parser.error("Biochem COMSOL anchor meshes must use --unit cm")
        elif phase == "biochem" and level == 2:
            print("High-thrombus biochem generation detected.")
            print("Use 'cm' for thrombus CFD-compatible meshes, or keep 'm' for SI-scale meshes.")
            unit_choice = _prompt_unit_choice(default="cm")

        vg = VesselGenerator(phase=phase)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        unused_slots = index_span - n_on_disk if max_idx >= 0 else 0
        print("\n--- Vessel mesh inventory ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Unit: {unit_choice}")
        print(f"  Total number of phase vessels: {index_span}")
        print(f"  Number of vessel meshes already generated: {n_on_disk}")
        print(f"  Number of non-anchors remaining: {unused_slots}")
        print()
        if n_on_disk == 0:
            overwrite = True
            print("  No meshes on disk — starting indices at 0 (overwrite).\n")
        else:
            overwrite = True if args.overwrite else _prompt_write_mode_vessel()
        default_n = 50 if n_on_disk > 0 else 500
        n_vessels = (
            int(args.num_vessels)
            if args.num_vessels is not None
            else _prompt_positive_int("How many vessels to generate", default_n)
        )
        pathology_mode = prompt_pathology_mode()
        aneurysm_wall_mode = prompt_aneurysm_wall_mode(pathology_mode)
        start_idx = 0 if overwrite else None
        show_vessel_plot = bool(args.show_vessel_plot) or _prompt_yes_no(
            "Show matplotlib preview of generated meshes after this run?",
            default=False,
        )
        wound_probability, wound_at_pathology = prompt_wound_options(
            wound_already=bool(args.wound),
            wound_at_pathology_already=bool(args.wound_at_pathology),
            default_at_pathology=anchors,
        )

    if args.seed is not None:
        logger.info("Using fixed RNG seed=%s", args.seed)
    else:
        logger.info("Using random RNG seed (each run draws a new cohort)")

    vg.run_pipeline(
        n=n_vessels,
        level=level,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        start_idx=start_idx,
        unit=unit_choice,
        pathology_mode=pathology_mode,
        aneurysm_wall_mode=aneurysm_wall_mode,
        wound_probability=wound_probability,
        wound_at_pathology=wound_at_pathology,
    )

    if show_vessel_plot:
        saved_indices = sorted(
            int(p.stem.split("_")[-1])
            for p in vg.output_dir.glob("vessel_*.msh")
        )[:9]
        if saved_indices:
            vg.visualize_saved(saved_indices)