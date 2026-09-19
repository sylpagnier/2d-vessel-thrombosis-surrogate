"""Customer geometry inbox + load helpers for the Predict app.

Supports:
  - existing 2D Thrombosis Surrogate ``.pt`` graphs
  - ``.msh`` / ``.nas`` 2D triangle meshes (linear or quadratic, m / cm / mm). Wall, inlet and
    outlet are read off the mesh outline (``mesh_and_meta_from_outline``); a same-stem sidecar
    ``.json`` from our own generator is still used when present
  - parametric vessel build (caller supplies mesh+meta via ``graph_from_mesh_meta``)

STL is not supported (it has no 2D triangle mesh to solve on).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import meshio
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.mesh_to_graph_biochem import default_biochem_bio_inlet_bc
from src.utils.safe_load import load_untrusted_graph
from src.utils.channel_schema import (
    BIO_Y_SCHEMA,
    Y_SCHEMAS,
    attach_channel_metadata,
    infer_missing_schema,
)
from src.utils.paths import get_project_root
from src.utils.units import MESH_UNIT_M

INBOX_DIRNAME = "customer_geometries"
SUPPORTED_SUFFIXES = (".pt", ".msh", ".nas")
DEFAULT_N_STEPS = 60
DEFAULT_RE = 450.0

# Element sizing for a parametric customer build, calibrated to the COMSOL anchor meshes
# `clot_ml_0` was trained on -- not to `VesselConfig.mesh_h_nd_target`, which is the KINEMATICS
# corpus's density.  The anchors are meshed at one ABSOLUTE size whatever the vessel width
# (median nearest-node spacing 0.35 mm on comsol040 at 8.3 mm wide and comsol044 at 18.4 mm),
# with a mild throat refinement (0.24 mm interior / 0.32 mm wall on comsol041/044).  The model
# counts depth in graph hops, so the physical size of a hop is part of what it learned.
#
# Measured 2026-09-16 against comsol041 / comsol040 / comsol017 at matched width:
#   open spacing 0.352 / 0.344 / 0.358 mm vs 0.350 / 0.347 / 0.352;
#   node count 13,917 / 9,607 / 14,487 vs 14,295 / 9,443 / 14,501; throat 0.26 mm.
# The previous d_bar-relative size made an aneurysm 20% coarser than its anchor, and a throat
# twice as fine (0.16 mm): `Mesh.MeshSizeFactor` also scales the lumen-width size callback,
# which `_gmsh_size_bounds` only compensates for `lc_max`.  Size factor 1.0 removes that.
# The former 2x "draft" mesh inflated 77% stenosis wall clot from 0.23 to 0.36.
ANCHOR_MESH_LC_M = 0.72e-3
ANCHOR_MESH_MIN_ELEMS_ACROSS = 7

# --- What the deployed model was trained on ------------------------------------------------
# Measured 2026-09-14 on the 36 COMSOL vessels in clot_ml_final's `training_pool` plus the 6
# wound vessels the wound branch was fit on (raw meshes in data/raw/biochem_anchors, packs in
# data/processed/graphs_biochem_anchors). The Predict app keeps its controls inside these
# ranges and flags uploads outside them; src/tests/test_customer_training_envelope.py re-measures
# them from that data when it is present, so they cannot silently drift from the cohort.
TRAINED_INLET_WIDTH_M = (0.0083, 0.0200)     # comsol040 .. comsol020; median ~15 mm
TRAINED_LENGTH_M = (0.099, 0.106)            # every vessel is ~10 cm along its centerline
TRAINED_MAX_BEND_DEG = 115.0                 # comsol032 / comsol035 / comsol037
TRAINED_MAX_S_AMPLITUDE_M = 0.005            # largest two-sided centerline wiggle, +/-4.7 mm
TRAINED_STENOSIS_NARROWING = 0.77            # comsol041/042/044: 77% diameter reduction, all at max
TRAINED_ANEURYSM_WIDENING = 0.99             # comsol040/047: peak width 1.99x inlet
TRAINED_PATHOLOGY_CENTER_FRAC = 0.51         # every trained stenosis/aneurysm sits at mid-length
TRAINED_STENOSIS_FWHM_FRAC = 0.12            # half-depth width of the narrowing, 0.09-0.13 of length
TRAINED_ANEURYSM_FWHM_FRAC = 0.20            # 0.195 / 0.207
TRAINED_WOUND_CENTER_FRAC = (0.32, 0.69)     # wound_comsol001..006
TRAINED_WOUND_WIDTH_FRAC = (0.06, 0.15)
TRAINED_WOUND_HORIZON_S = 12000.0            # 6,136-11,975 s; wound_comsol003 (19,118 s) is the outlier
# `apply_customer_max_pathology` settings that reproduce the trained shapes. Calibrated by building
# the generator geometry and measuring it exactly as the cohort was measured: location 0.5 puts the
# peak at 0.51 of length; sharpness 0.77 gives a 0.12 FWHM narrowing, 0.49 a 0.20 FWHM widening.
TRAINED_PATHOLOGY_LOCATION = 0.5
TRAINED_STENOSIS_SHARPNESS = 0.77
TRAINED_ANEURYSM_SHARPNESS = 0.49
# Built depth / requested depth at those settings.  The Gaussian's peak falls between wall control
# points, so the spline never reaches the requested offset: measured 2026-09-16 on the throat of
# the built wall (4,000-station interpolation), stenosis 0.967 / 0.969 / 0.969 at 25 / 50 / 77%,
# aneurysm 0.987 at 99%.  Uncorrected, "77%" built 74.6% against comsol041/042/044's 77.1%.
STENOSIS_DEPTH_GAIN = 0.969
ANEURYSM_DEPTH_GAIN = 0.987


def default_customer_mesh_cache_dir(root: Path | None = None) -> Path:
    """Persistent Gmsh meshes for parametric customer vessels (FEM t=0 flow)."""
    return (root or get_project_root()) / "outputs" / "customer" / "_meshes"


def _parametric_cache_key(
    params: dict[str, Any],
    *,
    re_target: float,
    t_final_s: float,
    n_steps: int,
) -> str:
    payload = {
        "params": params,
        "re_target": float(re_target),
        "t_final_s": float(t_final_s),
        "n_steps": int(n_steps),
        "mesh": ["anchor", ANCHOR_MESH_LC_M, ANCHOR_MESH_MIN_ELEMS_ACROSS],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]

CustomerMaxPathology = Literal["max_stenosis", "max_aneurysm"]


class CustomerGeometryError(ValueError):
    """User-facing geometry / tag / sidecar failure."""


def apply_customer_max_pathology(
    params: dict[str, Any],
    cfg: VesselConfig | None = None,
    kind: str = "max_stenosis",
    strength: float = 1.0,
    location: float = 0.5,
    sharpness: float = 1.0,
    one_wall: bool = False,
) -> dict[str, Any]:
    """Overlay a mid-vessel max stenosis or max aneurysm on parametric vessel params.

    Uses both-wall peak offsets so stenosis hits ``max_stenosis_diameter_occlusion``
    and aneurysm hits ``max_aneurysm_width_scale`` (3x inlet). Wall noise is cleared
    so the preview matches the configured strength. ``strength``/``location``/
    ``sharpness`` scale the magnitude, axial position, and taper of the overlay
    around those defaults (1.0 / 0.5 / 1.0 reproduce the fixed max preset).

    ``one_wall`` puts the whole offset on the bottom wall instead of splitting it across both,
    so ``strength`` then moves the width by ``mag / width`` rather than ``2 * mag / width``.
    """
    cfg = cfg or VesselConfig(phase="kinematics")
    if kind not in ("max_stenosis", "max_aneurysm", "stenosis", "aneurysm"):
        raise ValueError(f"Unknown max pathology kind {kind!r}")

    out = dict(params)
    width = float(out.get("width", cfg.width_min))
    n = int(cfg.num_ctrl_pts)
    if "stenosis" in kind:
        mag = float(cfg.max_stenosis_wall_offset(width)) * strength
        out["v_type"] = "stenosis"
    else:
        mag = float(cfg.max_aneurysm_wall_offset(width)) * strength
        out["v_type"] = "aneurysm"

    min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
    peak = min_idx + location * (max_idx - min_idx)
    std_dev = max(1.0, 0.035 * n) / max(0.01, sharpness)
    x_idx = np.arange(n, dtype=float)
    gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
    out["offsets"] = (mag * gauss).tolist()
    out["path_loc"] = 1 if one_wall else 2
    out["noise_top"] = [0.0] * n
    out["noise_bot"] = [0.0] * n
    out["pathology_mode"] = kind
    return out


def inbox_dir(root: Path | None = None) -> Path:
    return (root or get_project_root()) / INBOX_DIRNAME


def ensure_inbox(root: Path | None = None) -> Path:
    d = inbox_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_inbox(root: Path | None = None) -> list[Path]:
    d = ensure_inbox(root)
    files = [
        p
        for p in sorted(d.iterdir(), key=lambda q: q.name.lower())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def copy_into_inbox(src: Path | str, root: Path | None = None) -> Path:
    """Copy a customer file into the inbox (same name). Returns destination path."""
    src_p = Path(src).resolve()
    if not src_p.is_file():
        raise CustomerGeometryError(f"File not found: {src_p}")
    if src_p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CustomerGeometryError(
            f"Unsupported type '{src_p.suffix}'. Use .pt, .msh, or .nas."
        )
    dest_dir = ensure_inbox(root)
    dest = dest_dir / src_p.name
    if src_p.resolve() != dest.resolve():
        shutil.copy2(src_p, dest)
        sidecar = src_p.with_suffix(".json")
        if src_p.suffix.lower() in (".msh", ".nas") and sidecar.is_file():
            shutil.copy2(sidecar, dest_dir / sidecar.name)
    return dest


def _physics(re_target: float) -> PhysicsConfig:
    # Mesh build uses kinematics feature layout; Re only affects u_ref / priors.
    return PhysicsConfig(phase="kinematics", re_target=float(re_target))


def _validate_masks(data: Data, *, stem: str) -> None:
    for name in ("mask_inlet", "mask_outlet", "mask_wall"):
        if not hasattr(data, name) or getattr(data, name) is None:
            raise CustomerGeometryError(
                f"{stem}: missing {name}. Meshes need Gmsh tags "
                f"Inlet={VesselConfig().TAGS['Inlet']}, "
                f"Outlet_1={VesselConfig().TAGS['Outlet_1']}, "
                f"Walls={VesselConfig().TAGS['Walls']}. "
                f"See {INBOX_DIRNAME}/README.txt."
            )
    n_in = int(data.mask_inlet.reshape(-1).bool().sum().item())
    n_out = int(data.mask_outlet.reshape(-1).bool().sum().item())
    n_wall = int(data.mask_wall.reshape(-1).bool().sum().item())
    if n_in < 1 or n_out < 1 or n_wall < 1:
        raise CustomerGeometryError(
            f"{stem}: bad boundary masks (inlet={n_in}, outlet={n_out}, wall={n_wall}). "
            f"Check physical line tags in {INBOX_DIRNAME}/README.txt."
        )


def apply_re_target(data: Data, re_target: float) -> Data:
    """Rescale ``u_ref`` / inlet velocity BCs for a new Reynolds number."""
    out = data.clone() if hasattr(data, "clone") else data
    phys = _physics(re_target)
    d_bar = float(out.d_bar.reshape(-1)[0].item()) if hasattr(out, "d_bar") else 0.0
    if d_bar <= 0:
        raise CustomerGeometryError("Graph is missing a valid d_bar length scale.")
    u_new = float(phys.get_u_ref(d_bar))
    u_old = float(out.u_ref.reshape(-1)[0].item()) if hasattr(out, "u_ref") else u_new
    out.u_ref = torch.tensor([u_new], dtype=torch.float32)
    if hasattr(out, "re_actual"):
        out.re_actual = torch.tensor([float(re_target)], dtype=torch.float32)
    scale = (u_new / u_old) if abs(u_old) > 1e-12 else 1.0
    if hasattr(out, "u_inlet_bc") and out.u_inlet_bc is not None and abs(scale - 1.0) > 1e-8:
        out.u_inlet_bc = out.u_inlet_bc * scale
    # ND prior columns in x (u_prior / v_prior) stay ND; absolute Re is carried by u_ref.
    return out


def apply_customer_mirrored_wound(
    data: Data,
    *,
    enabled: bool,
    position_frac: float = 0.50,
    width_frac: float = 0.15,
) -> Data:
    """Assign a symmetric wound patch on both wall sides of a customer graph.

    ``position_frac`` and ``width_frac`` are measured along the inlet-to-outlet axis,
    not in global x.  Every solid-boundary node inside that axial interval becomes a
    wound node; therefore the patch is mirrored across the lumen by construction.
    The healthy-wall mask is made disjoint from it, matching the wound complement's
    domain convention.  With ``enabled=False`` the existing solid boundary is restored
    as healthy wall and the result is a no-wound graph.
    """
    out = data.clone() if hasattr(data, "clone") else data
    if not hasattr(out, "mask_wall") or out.mask_wall is None:
        raise CustomerGeometryError("Graph is missing mask_wall; cannot place a wound.")
    wall = out.mask_wall.reshape(-1).bool()
    existing = getattr(out, "mask_wound", None)
    if existing is not None and torch.is_tensor(existing) and existing.numel():
        solid = wall | existing.reshape(-1).bool().to(device=wall.device)
    else:
        solid = wall.clone()
    n = int(solid.numel())
    wound = torch.zeros(n, dtype=torch.bool, device=wall.device)
    if enabled and bool(solid.any()):
        pos = out.x[:, :2].detach().to(device="cpu", dtype=torch.float64).numpy()
        inlet = getattr(out, "mask_inlet", None)
        outlet = getattr(out, "mask_outlet", None)
        axis: np.ndarray | None = None
        if inlet is not None and outlet is not None:
            inn = inlet.reshape(-1).bool().detach().cpu().numpy()
            outn = outlet.reshape(-1).bool().detach().cpu().numpy()
            if inn.any() and outn.any():
                axis = pos[outn].mean(axis=0) - pos[inn].mean(axis=0)
        if axis is None or float(np.linalg.norm(axis)) < 1e-12:
            centered = pos - pos.mean(axis=0, keepdims=True)
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
        axis = np.asarray(axis, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        q = pos @ axis
        solid_np = solid.detach().cpu().numpy()
        lo, hi = float(q[solid_np].min()), float(q[solid_np].max())
        along = (q - lo) / max(hi - lo, 1e-12)
        center = float(np.clip(position_frac, 0.0, 1.0))
        width = float(np.clip(width_frac, 0.01, 0.80))
        wound_np = solid_np & (np.abs(along - center) <= 0.5 * width)
        wound = torch.as_tensor(wound_np, dtype=torch.bool, device=wall.device)
    out.mask_wound = wound
    out.mask_wall = solid & ~wound
    out.customer_wound_enabled = bool(enabled)
    out.customer_wound_position_frac = float(np.clip(position_frac, 0.0, 1.0))
    out.customer_wound_width_frac = float(np.clip(width_frac, 0.01, 0.80))
    return out


def _bio_y_width() -> int:
    return int(Y_SCHEMAS[BIO_Y_SCHEMA].width)


def _seed_bio_frame_from_data(data: Data, n_nodes: int) -> torch.Tensor:
    """One (N, 16) frame: copy kinematics u/v/p/mu when present; rest resting / zero."""
    c = _bio_y_width()
    frame = torch.zeros(n_nodes, c, dtype=torch.float32)
    y = getattr(data, "y", None)
    if y is not None and torch.is_tensor(y):
        if y.dim() == 3:
            src = y[0]
        elif y.dim() == 2:
            src = y
        else:
            src = None
        if src is not None:
            n_copy = min(int(src.shape[-1]), 4, c)
            frame[:, :n_copy] = src[:, :n_copy].detach().cpu().float()
    if abs(float(frame[:, 3].mean().item())) < 1e-12:
        # mu_eff_nd ~ 1 when unknown
        frame[:, 3] = 1.0
    if abs(float(frame[:, 4].mean().item())) < 1e-12:
        # RP_log1p_nd / AP_log1p_nd (resting + activated platelets) left at 0 here
        # silently starve `wall_platelet_constants`'s rp/ap read of data.y[0]: with both
        # zero, the wound module's chemistry ODE (src.clot_ml.wound.mat_trajectory_torch)
        # has no source term at all (dep = sat*(k_rs*rp + k_as*ap_i)), so every
        # customer/parametric pack (no COMSOL RP/AP field to copy) predicts zero wound
        # clot regardless of geometry or wound placement. Seed the resting bulk
        # concentrations (c_RP0, c_AP0) that measured COMSOL packs carry at t=0
        # (src/tests/test_ap_closure.py: RP_log1p_nd ~1e-6, AP_log1p_nd ~5e-8 -- these
        # ND encodings are `concentration / (c_RP0 * bulk_scale)`, not `/ c_RP0` alone).
        bio_cfg = BiochemConfig(phase="biochem")
        scale0 = float(bio_cfg.get_species_scales(device="cpu")[0])
        frame[:, 4] = float(np.log1p(bio_cfg.c_RP0 / scale0))
        frame[:, 5] = float(np.log1p(bio_cfg.c_AP0 / scale0))
    # Prothrombin, antithrombin and fibrinogen start at their bulk plasma level (ND 1.0, i.e.
    # log1p(1)) on every node of every COMSOL training pack -- all 42 fit vessels carry exactly
    # that t=0 value. They were left at 0 here, so every customer vessel started with no clotting
    # substrate at all (docs/DEPLOY_ALIGNMENT.md measures what matching it recovers).
    names = list(Y_SCHEMAS[BIO_Y_SCHEMA].channels)
    for species in ("PT_log1p_nd", "AT_log1p_nd", "FG_log1p_nd"):
        i = names.index(species)
        if abs(float(frame[:, i].mean().item())) < 1e-12:
            frame[:, i] = float(np.log1p(1.0))
    return frame


def synthesize_deploy_timeline(
    data: Data,
    *,
    t_final_s: float,
    n_steps: int | None = None,
) -> Data:
    """Attach a macro time axis and biochem ``y`` scaffold for kinematics-driven deploy."""
    out = data.clone() if hasattr(data, "clone") else data
    n_nodes = int(out.x.shape[0])
    steps = int(n_steps) if n_steps is not None else DEFAULT_N_STEPS
    steps = max(2, steps)
    t_end = max(float(t_final_s), 1.0)

    frame0 = _seed_bio_frame_from_data(out, n_nodes)
    y_series = frame0.unsqueeze(0).expand(steps, -1, -1).contiguous().clone()
    # Keep existing species IC from a biochem graph when present
    y_old = getattr(data, "y", None)
    has_real_biochem_y = (
        y_old is not None
        and torch.is_tensor(y_old)
        and y_old.dim() == 3
        and int(y_old.shape[-1]) == _bio_y_width()
    )
    if has_real_biochem_y:
        src = y_old.detach().cpu().float()
        t_src = int(src.shape[0])
        for i in range(steps):
            j = min(int(round(i * (t_src - 1) / max(steps - 1, 1))), t_src - 1)
            y_series[i] = src[j]

    out.y = y_series
    # No real (COMSOL-backed) species/velocity history to draw on -- the local FEM t=0 solve
    # must not read `y[0, :, 0:2]` as if it were ground-truth inlet velocity (it is at best a
    # broadcast kinematics snapshot, at worst exactly zero), or it Dirichlet-clamps the inlet
    # to that placeholder and solves a trivial zero-flow field.  See
    # `src.clot_ml.v0.solve_fem_into_pack`.
    if not has_real_biochem_y:
        out.research_synthetic = True
    out.t = torch.linspace(0.0, t_end, steps=steps, dtype=torch.float32)
    if not hasattr(out, "bio_inlet_bc") or out.bio_inlet_bc is None:
        out.bio_inlet_bc = default_biochem_bio_inlet_bc(n_nodes)
    infer_missing_schema(out, phase_hint="biochem")
    if getattr(out, "y_schema", None) != BIO_Y_SCHEMA:
        attach_channel_metadata(
            out,
            x_schema=getattr(out, "x_schema", None) or "kine_x_v1_18ch",
            y_schema=BIO_Y_SCHEMA,
            mask_wall=getattr(out, "mask_wall", None),
        )
    return out


def _ensure_p2_topology(data: Data) -> Data:
    """Elevate a Gmsh-built P1 (linear ``triangle``) graph to the P2 (``triangle6``) topology
    every deploy checkpoint was trained on.

    ``clot_ml_0``'s base GNN and wound complement are fit exclusively on COMSOL biochem
    anchor packs (see the ``fit_anchors`` list in every locked manifest), and COMSOL always
    exports ``triangle6``: no deploy checkpoint has ever seen a mesh without mid-side nodes.
    Gmsh-built customer meshes (parametric or uploaded ``.msh``/``.nas``) are linear
    ``triangle`` by default, so the off-wall/wound shell logic -- which walks mesh HOPS, not
    physical distance (``first_corner_shell`` in ``physics_lumen_model.py``, the wound
    module's own shell-1 owner map) -- lands its first shell one full corner-hop further from
    the wall than it does on any vessel the model was tuned against, and the near-wall layer
    never fires at all. Measured end to end on a customer parametric+wound vessel: off-wall
    clot sat at 3.1-5.5x the local mesh pitch from the nearest wall/wound node with this
    elevation skipped, against 1.6x / 3.2x two-layer bands on the equivalent COMSOL pack run
    through the identical ``flow="fem"`` path.

    Reuses the exact elevation recipe already used to close this same P1/P2 mismatch for the
    kinematics predictor (``src/data_gen/lib/p2_elevation.py``, RGP_DEQ_REPAIR_PLAN.md Sec 8) --
    a mid-side node is inserted on every edge, matching COMSOL's own corner-midside-only
    convention, with interpolated features/labels. ``src/core_physics/local_fem_solver.py``
    registers the FEM solve's own facet DOFs against these same positions, so the local t=0
    flow at an elevated mid-side node is a genuine solved P2 value, not a placeholder.
    """
    from src.data_gen.lib.p1_corner_graph import identify_midside_nodes
    from src.data_gen.lib.p2_elevation import elevate_to_p2

    already, _ = identify_midside_nodes(data)
    if bool(already.any()):
        return data
    return elevate_to_p2(data, keep_wls=False)


def _inlet_width_length_scale(meta: dict[str, Any]) -> dict[str, Any]:
    """Use the INLET width as ``d_bar``, the convention of every vessel the deployed model saw.

    ``d_bar`` is the length scale of the whole graph: node positions, wall distance and the
    inlet velocity (``u_ref = Re * mu / (rho * d_bar)``) are all expressed in it. Every COMSOL
    training pack has ``d_bar`` equal to its inlet width (checked on all 36 fit vessels and the 6
    wound vessels), while the geometry generator's sidecar uses the MEAN width. The two agree on a
    plain vessel and part as soon as a stenosis or aneurysm changes the mean -- a 77% stenosis
    moves ``d_bar`` ~10%, and with it the inlet velocity and every non-dimensional feature.
    """
    d_inlet = meta.get("d_inlet")
    d_bar = meta.get("d_bar")
    if not d_inlet or not d_bar or float(d_inlet) <= 0 or abs(float(d_inlet) - float(d_bar)) < 1e-12:
        return meta
    out = dict(meta)
    ratio = float(d_bar) / float(d_inlet)
    for key in ("centerline_pts", "top_wall_pts", "bot_wall_pts"):   # stored in units of d_bar
        if out.get(key) is not None:
            out[key] = (np.asarray(out[key], dtype=np.float64) * ratio).tolist()
    out["d_bar_mean_width"] = float(d_bar)
    out["d_bar"] = float(d_inlet)
    return out


def _training_recipe_node_x(data: Data) -> None:
    """Rebuild ``data.x`` exactly the way every COMSOL training pack's node channels were built.

    Training packs come from `ComsolAnchorDataExtractor` -> `build_kinematics_graph_from_comsol_steady`
    on the final P2 node set: graph-derived wall normals oriented at the centerline, KD-tree wall
    distance, hydraulic width, and ``width_d1`` / ``width_d2`` through the WLS sparse operators.
    `MeshToGraph` computes the same channels differently (line-segment normals on the P1 mesh,
    interpolated through P2 elevation), and the model reads four of them (``sdf_nd``,
    ``wall_normal``, ``width_nd``, ``width_d1/d2``). On comsol032 the two builders' wall normals
    differed by 12-60% and the width derivatives by 100-180%; this recipe matches the pack to
    <=5% and <=17%. The centerline comes from the graph (no COMSOL sidecar exists for a customer
    vessel) -- on the training packs that gives identical channels to the sidecar centerline.
    """
    from scipy.spatial import cKDTree

    from src.data_gen.lib.centerline_utils import resolve_centerline_nd
    from src.data_gen.lib.kinematics_graph_builder import wall_normals_and_sdf_mesh_to_graph_style
    from src.data_gen.lib.mesh_wls import (
        precompute_wls_operators,
        solid_boundary_mask,
        wls_sparse_operators,
    )
    from src.data_gen.lib.node_feature_assembly import (
        build_kinematics_node_x_tensor,
        resolve_anchor_kine_phys_cfg,
    )

    d_bar = float(data.d_bar.reshape(-1)[0])
    u_ref = float(data.u_ref.reshape(-1)[0])
    n = int(data.x.shape[0])
    nodes_nd = data.x[:, :2].detach().cpu().float()
    nodes_si = nodes_nd.numpy().astype(np.float64) * d_bar
    edge_index = data.edge_index.detach().cpu()
    m_in = data.mask_inlet.reshape(-1).bool().cpu()
    m_out = data.mask_outlet.reshape(-1).bool().cpu()
    m_wall = data.mask_wall.reshape(-1).bool().cpu()
    wound = getattr(data, "mask_wound", None)
    m_wound = wound.reshape(-1).bool().cpu() if torch.is_tensor(wound) and wound.numel() else None
    solid = solid_boundary_mask(m_wall, m_wound)

    V, W, M_inv = precompute_wls_operators(edge_index, n, nodes_nd)
    G_x, G_y, _ = wls_sparse_operators(edge_index, n, M_inv, V, W)
    cl_pts, cl_tan, _src = resolve_centerline_nd(
        nodes_nd, m_in, m_out, edge_index=edge_index, mask_wall=solid, stem="", raw_sidecar_dir=None
    )
    no_line_cells = meshio.Mesh(np.column_stack([nodes_si, np.zeros(n)]), [])
    sdf, normal = wall_normals_and_sdf_mesh_to_graph_style(
        no_line_cells, nodes_si, mask_wall=m_wall, mask_inlet=m_in, mask_outlet=m_out,
        mask_wound=m_wound, d_bar_si=d_bar, centerline_pts_si=np.asarray(cl_pts) * d_bar,
        edge_index=edge_index,
    )
    phys = resolve_anchor_kine_phys_cfg()
    x, _, _ = build_kinematics_node_x_tensor(
        pos_nd=nodes_nd, sdf_nd=sdf, wall_normal=normal, mask_inlet=m_in, mask_outlet=m_out,
        mask_wall=solid, d_bar_si=d_bar, u_ref=u_ref, phys_cfg=phys,
        wall_tree=cKDTree(nodes_si[solid.numpy()]), edge_index=edge_index, G_x=G_x, G_y=G_y,
        centerline_pts_nd=cl_pts, centerline_tangents_nd=cl_tan, inlet_uv_nd=None,
        mu_nd_scale=phys.mu_viscosity_nd_scale,
    )
    data.x = x.to(dtype=data.x.dtype)


def _keep_opening_corners(data: Data, mesh: meshio.Mesh) -> None:
    """Put the two corner nodes of each opening back into ``mask_inlet`` / ``mask_outlet``.

    `gmsh_line_boundary_masks` carves every wall node out of the inlet and outlet, so the corner
    where a cut meets the wall ends up wall-only -- and after P2 elevation the mid-side nodes next
    to it lose the tag too (41 inlet nodes on comsol041 instead of 45). Every COMSOL training pack
    tags ALL nodes on the cut, both corners shared with the wall, and the t=0 FEM takes its inlet
    and outlet facets from these masks. On comsol041, giving the mesh-built graph the training
    pack's masks and nothing else moved the deploy wall metric 0.60 -> 0.85
    (docs/DEPLOY_ALIGNMENT.md).
    """
    tags = VesselConfig().TAGS
    try:
        lines = np.asarray(mesh.cells_dict["line"])
        line_tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["line"])
    except (KeyError, AttributeError):
        return
    n = int(data.mask_inlet.numel())
    for attr, tag in (("mask_inlet", tags["Inlet"]), ("mask_outlet", tags["Outlet_1"])):
        on_cut = np.unique(lines[line_tags == tag])
        on_cut = on_cut[on_cut < n]
        mask = getattr(data, attr).reshape(-1).bool().clone()
        mask[torch.as_tensor(on_cut, dtype=torch.long)] = True
        setattr(data, attr, mask.to(getattr(data, attr).dtype).reshape(getattr(data, attr).shape))


def graph_from_mesh_meta(
    mesh: meshio.Mesh,
    meta: dict[str, Any],
    *,
    re_target: float = DEFAULT_RE,
    stem: str = "customer",
) -> Data:
    """Build a kinematics graph then upgrade to a deploy-ready biochem timeline scaffold."""
    meta = _inlet_width_length_scale(meta)
    phys = _physics(re_target)
    # MeshToGraph uses phys_cfg from constructor; override re via PhysicsConfig
    builder = MeshToGraph(phase="kinematics", rheology="carreau")
    builder.phys_cfg = phys
    data = builder.process_mesh(mesh, meta, stem=stem)
    if data is None:
        raise CustomerGeometryError(
            f"{stem}: mesh has no triangles or wall nodes."
        )
    _validate_masks(data, stem=stem)
    _keep_opening_corners(data, mesh)
    data = apply_re_target(data, re_target)
    data = _ensure_p2_topology(data)
    _training_recipe_node_x(data)
    # Default scaffold; callers usually re-synthesize with the UI horizon.
    return synthesize_deploy_timeline(data, t_final_s=8000.0, n_steps=DEFAULT_N_STEPS)


# --- Boundary inference for plain meshes ---------------------------------------------------
# An uploaded vessel mesh usually carries no Gmsh physical tags (COMSOL and Nastran exports never
# do) and no sidecar JSON. Everything the graph builder needs from those can be read off the mesh
# itself: its boundary is one closed outline, and a 2D vessel outline has four corners -- the two
# short, straight caps between them are the inlet and outlet, the two long runs are the walls,
# and the centerline/d_bar follow from the walls. When the corners are ambiguous the caller
# supplies a point near each opening instead (the app asks the user to click them).

MESH_UNIT_SCALE: dict[str, float] = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_CORNER_TURN_DEG = 50.0          # a cap/wall corner; smooth wall curvature turns far less per node
_HINT_CORNER_TURN_DEG = 30.0     # looser when the user has already said where the opening is
_CAP_MAX_WALL_FRACTION = 0.8     # each cap must be shorter than this fraction of the shorter wall
_CAP_MIN_STRAIGHTNESS = 0.95     # chord / arc length of a cap
_CENTERLINE_STATIONS = 200


class BoundaryNeedsInput(CustomerGeometryError):
    """The inlet and outlet could not be found automatically; the user has to point at them.

    Carries the scaled (metre) corner-node positions and the outline mask so the app can still
    draw the vessel to click on.
    """

    def __init__(self, message: str, *, pos: np.ndarray, boundary_mask: np.ndarray, unit: str) -> None:
        super().__init__(message)
        self.pos = pos
        self.boundary_mask = boundary_mask
        self.unit = unit


def _corner_triangles(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Linear triangles (quadratic ``triangle6`` reduced to its corners), compacted to used nodes."""
    blocks = []
    for block in mesh.cells:
        if block.type in ("triangle", "triangle6"):
            blocks.append(np.asarray(block.data)[:, :3])
    if not blocks:
        raise CustomerGeometryError("The mesh has no triangles. Upload a 2D triangle mesh of the vessel.")
    tris = np.vstack(blocks).astype(np.int64)
    used, inverse = np.unique(tris.reshape(-1), return_inverse=True)
    return np.asarray(mesh.points, dtype=np.float64)[used, :2], inverse.reshape(-1, 3)


def _boundary_loop(tris: np.ndarray, n_nodes: int) -> np.ndarray:
    """Ordered node indices of the single closed outline (edges used by exactly one triangle)."""
    edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    neighbours: dict[int, list[int]] = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    if not neighbours or any(len(v) != 2 for v in neighbours.values()):
        raise CustomerGeometryError(
            "The mesh outline is not a single clean loop. Upload one vessel with no holes or separate pieces."
        )
    start = min(neighbours)
    loop, prev, cur = [start], -1, start
    while True:
        a, b = neighbours[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    if len(loop) != len(neighbours):
        raise CustomerGeometryError(
            "The mesh has more than one outline (a hole or a separate piece). Upload a single vessel."
        )
    return np.asarray(loop, dtype=np.int64)


def _turn_angles_deg(P: np.ndarray, k: int = 2) -> np.ndarray:
    d_in = P - np.roll(P, k, axis=0)
    d_out = np.roll(P, -k, axis=0) - P
    cos = np.sum(d_in * d_out, axis=1) / np.maximum(
        np.linalg.norm(d_in, axis=1) * np.linalg.norm(d_out, axis=1), 1e-30
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _corner_positions(turn: np.ndarray, min_deg: float, window: int = 3) -> list[int]:
    m = len(turn)
    out = []
    for i in range(m):
        if turn[i] < min_deg:
            continue
        neighbourhood = turn[[(i + d) % m for d in range(-window, window + 1)]]
        if turn[i] >= neighbourhood.max() and not any((i - d) % m in out for d in range(1, window + 1)):
            out.append(i)
    return out


def _cyclic_range(a: int, b: int, m: int) -> np.ndarray:
    """Loop positions a..b inclusive, walking forward (wrapping)."""
    return np.arange(a, a + ((b - a) % m) + 1) % m


def _arc_length(P: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _segment_containing(j: int, corners: list[int], m: int) -> tuple[int, int]:
    """The corner-to-corner stretch of the loop that contains position ``j``."""
    cs = sorted(corners)
    before = [c for c in cs if c < j] or [cs[-1]]
    after = [c for c in cs if c > j] or [cs[0]]
    if j in cs:  # a click right on a corner: take the shorter neighbouring stretch (caps are short)
        i = cs.index(j)
        prev_c, next_c = cs[i - 1], cs[(i + 1) % len(cs)]
        return (prev_c, j) if (j - prev_c) % m <= (next_c - j) % m else (j, next_c)
    return before[-1], after[0]


def _extend_straight(P: np.ndarray, a: int, b: int, m: int, tol: float = 1e-3) -> tuple[int, int]:
    """Grow the loop stretch a..b outward while the next node stays on the stretch's line."""
    seg = P[_cyclic_range(a, b, m)]
    c = seg.mean(axis=0)
    direction = np.linalg.svd(seg - c, full_matrices=False)[2][0]
    normal = np.array([-direction[1], direction[0]])
    limit = tol * max(float(np.linalg.norm(seg[-1] - seg[0])), 1e-30)
    on_line = lambda k: abs(float((P[k % m] - c) @ normal)) <= limit
    for _ in range(m):
        if not on_line(a - 1) or (a - 1) % m == b:
            break
        a = (a - 1) % m
    for _ in range(m):
        if not on_line(b + 1) or (b + 1) % m == a:
            break
        b = (b + 1) % m
    return a, b


def split_outline(
    points: np.ndarray,
    loop: np.ndarray,
    *,
    inlet_hint: Any = None,
    outlet_hint: Any = None,
) -> dict[str, Any]:
    """Split a vessel outline into inlet cap, outlet cap and two walls (loop positions).

    Returns ``inlet``/``outlet`` (loop-position arrays, corners included), ``walls`` (two
    loop-position arrays, each ordered inlet end -> outlet end) and ``source`` ("auto" or "user").
    Raises ``ValueError`` with a user-facing reason when the split is ambiguous.
    """
    P = points[loop]
    m = len(loop)
    turn = _turn_angles_deg(P)
    if inlet_hint is not None and outlet_hint is not None:
        corners = _corner_positions(turn, _HINT_CORNER_TURN_DEG)
        if len(corners) < 4:
            raise ValueError("Couldn't find the corners of those openings. Click closer to the middle of each open end.")
        spans = []
        for hint in (inlet_hint, outlet_hint):
            j = int(np.argmin(np.linalg.norm(P - np.asarray(hint, dtype=np.float64)[:2], axis=1)))
            spans.append(_segment_containing(j, corners, m))
        (ia, ib), (oa, ob) = spans
        if (ia, ib) == (oa, ob) or ib == oa or ob == ia:
            raise ValueError("The inlet and outlet you picked are on the same or touching stretch of the outline. Pick the two open ends.")
        source = "user"
    else:
        corners = _corner_positions(turn, _CORNER_TURN_DEG)
        if len(corners) != 4:
            raise ValueError(f"Found {len(corners)} corners on the outline instead of 4, so the open ends are ambiguous.")
        c = sorted(corners)
        segs = [(c[i], c[(i + 1) % 4]) for i in range(4)]
        lengths = [_arc_length(P[_cyclic_range(a, b, m)]) for a, b in segs]
        pair = min(((0, 2), (1, 3)), key=lambda pr: max(lengths[pr[0]], lengths[pr[1]]))
        walls_idx = [i for i in range(4) if i not in pair]
        if max(lengths[i] for i in pair) >= _CAP_MAX_WALL_FRACTION * min(lengths[i] for i in walls_idx):
            raise ValueError("The open ends aren't clearly shorter than the walls.")
        for i in pair:
            a, b = segs[i]
            chord = float(np.linalg.norm(P[b] - P[a]))
            if chord < _CAP_MIN_STRAIGHTNESS * lengths[i]:
                raise ValueError("The open ends aren't straight cuts across the vessel.")
        # Inlet = the cap further along -x of the vessel's long axis (left, in the usual layout).
        centred = points - points.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        if axis[0] < 0:
            axis = -axis
        caps = sorted(pair, key=lambda i: float(P[_cyclic_range(*segs[i], m)].mean(axis=0) @ axis))
        (ia, ib), (oa, ob) = segs[caps[0]], segs[caps[1]]
        source = "auto"
    # Corner detection looks a couple of nodes either side, so on a finely graded corner it can
    # land one edge inside the cut. An opening is a straight cut: grow each cap along its own line
    # to the real corners, so every node ON the cut is inlet/outlet -- exactly how the COMSOL
    # training packs tag it (all nodes of the cut, both corners shared with the wall).
    ia, ib = _extend_straight(P, ia, ib, m)
    oa, ob = _extend_straight(P, oa, ob, m)
    wall_a = _cyclic_range(ib, oa, m)            # inlet end -> outlet end, walking forward
    wall_b = _cyclic_range(ob, ia, m)[::-1]      # outlet end -> inlet end forward, so reverse
    return {
        "inlet": _cyclic_range(ia, ib, m),
        "outlet": _cyclic_range(oa, ob, m),
        "walls": (wall_a, wall_b),
        "source": source,
    }


def _resample_polyline(P: np.ndarray, n: int) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def detect_mesh_unit(points: np.ndarray) -> str:
    """Pick m / cm / mm so the vessel's longest extent is closest (log scale) to the trained ~0.1 m."""
    extent = float(np.max(np.ptp(points[:, :2], axis=0)))
    if extent <= 0:
        return "m"
    target = float(VesselConfig().base_length)
    return min(MESH_UNIT_SCALE, key=lambda u: abs(np.log10(extent * MESH_UNIT_SCALE[u] / target)))


def _tag_hint_points(mesh: meshio.Mesh, points_m: np.ndarray, scale: float) -> tuple[Any, Any]:
    """Centroids of Gmsh-tagged inlet/outlet lines, as click-style hints (None if untagged)."""
    tags = VesselConfig().TAGS
    try:
        lines = mesh.cells_dict["line"]
        line_tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["line"])
    except (KeyError, AttributeError):
        return None, None
    pts = np.asarray(mesh.points, dtype=np.float64)[:, :2] * scale
    out = []
    for key in ("Inlet", "Outlet_1"):
        sel = lines[line_tags == tags[key]]
        out.append(pts[np.unique(sel)].mean(axis=0) if len(sel) else None)
    return (out[0], out[1]) if out[0] is not None and out[1] is not None else (None, None)


def mesh_and_meta_from_outline(
    mesh: meshio.Mesh,
    *,
    unit: str = "auto",
    inlet_hint: Any = None,
    outlet_hint: Any = None,
    stem: str = "customer",
) -> tuple[meshio.Mesh, dict[str, Any], dict[str, Any]]:
    """Tag an untagged (or sidecar-less) vessel mesh from its own outline.

    ``inlet_hint`` / ``outlet_hint`` are points in METRES (the coordinates the app previews in).
    Returns a linear-triangle mesh with synthesised Inlet/Outlet_1/Walls line tags, the sidecar
    meta the graph builder needs, and a small ``info`` dict describing what was inferred.
    """
    raw_points, tris = _corner_triangles(mesh)
    unit = detect_mesh_unit(raw_points) if unit == "auto" else unit
    if unit not in MESH_UNIT_SCALE:
        raise CustomerGeometryError(f"Unknown mesh unit {unit!r}; use m, cm or mm.")
    scale = MESH_UNIT_SCALE[unit]
    points = raw_points * scale
    loop = _boundary_loop(tris, len(points))
    if inlet_hint is None or outlet_hint is None:
        inlet_hint, outlet_hint = _tag_hint_points(mesh, points, scale)
    try:
        split = split_outline(points, loop, inlet_hint=inlet_hint, outlet_hint=outlet_hint)
    except ValueError as exc:
        boundary_mask = np.zeros(len(points), dtype=bool)
        boundary_mask[loop] = True
        raise BoundaryNeedsInput(
            f"{exc} Click the inlet opening on the preview, then the outlet.",
            pos=points, boundary_mask=boundary_mask, unit=unit,
        ) from exc
    m = len(loop)
    top = _resample_polyline(points[loop[split["walls"][0]]], _CENTERLINE_STATIONS)
    bot = _resample_polyline(points[loop[split["walls"][1]]], _CENTERLINE_STATIONS)
    # "top" is the wall on the left of the inlet -> outlet direction, matching the generator.
    travel = top[-1] + bot[-1] - top[0] - bot[0]
    left = np.array([-travel[1], travel[0]])
    if float((top - bot).mean(axis=0) @ left) < 0:
        top, bot = bot, top
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls

    meta = dict(compute_geometry_from_walls(top, bot, unit=MESH_UNIT_M).meta)
    meta["boundary_source"] = split["source"]

    inlet_edges = set(split["inlet"][:-1].tolist())
    outlet_edges = set(split["outlet"][:-1].tolist())
    tags = VesselConfig().TAGS
    lines = np.column_stack([loop, np.roll(loop, -1)])
    line_tags = np.array([
        tags["Inlet"] if j in inlet_edges else tags["Outlet_1"] if j in outlet_edges else tags["Walls"]
        for j in range(m)
    ], dtype=np.int64)
    tagged = meshio.Mesh(
        np.column_stack([points, np.zeros(len(points))]),
        [("triangle", tris), ("line", lines)],
        cell_data={
            "gmsh:physical": [np.full(len(tris), tags["Fluid_Domain"], dtype=np.int64), line_tags],
            "gmsh:geometrical": [np.ones(len(tris), dtype=np.int64), np.ones(m, dtype=np.int64)],
        },
    )
    info = {
        "boundary_source": split["source"],
        "unit": unit,
        "length_m": float(_arc_length(0.5 * (top + bot))),
        "inlet_width_m": float(meta["d_inlet"]),
        "warnings": training_envelope_warnings(top, bot),
    }
    return tagged, meta, info


def training_envelope_warnings(top: np.ndarray, bot: np.ndarray) -> list[str]:
    """Plain-language notes where a vessel (walls ordered inlet -> outlet, metres) leaves the
    geometry the model was trained on. Empty when it is inside every measured range."""
    widths = np.linalg.norm(top - bot, axis=1)
    inlet = float(widths[0])
    centre = 0.5 * (top + bot)
    length = _arc_length(centre)
    tangent = np.diff(centre, axis=0)
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    bend = float(np.degrees(heading.max() - heading.min()))
    out = []
    lo, hi = TRAINED_INLET_WIDTH_M
    if not lo * 0.95 <= inlet <= hi * 1.05:
        out.append(f"Inlet width {inlet * 1e3:.1f} mm; trained vessels are {lo * 1e3:.0f}-{hi * 1e3:.0f} mm.")
    lo, hi = TRAINED_LENGTH_M
    if not lo * 0.9 <= length <= hi * 1.1:
        out.append(f"Length {length * 1e2:.1f} cm; trained vessels are all about 10 cm long.")
    if bend > TRAINED_MAX_BEND_DEG * 1.1:
        out.append(f"Bends through {bend:.0f} degrees; trained vessels bend at most {TRAINED_MAX_BEND_DEG:.0f}.")
    narrowing = 1.0 - float(widths.min()) / inlet
    widening = float(widths.max()) / inlet - 1.0
    if narrowing > TRAINED_STENOSIS_NARROWING + 0.05:
        out.append(f"Narrows by {narrowing:.0%}; the tightest trained stenosis narrows by {TRAINED_STENOSIS_NARROWING:.0%}.")
    if widening > TRAINED_ANEURYSM_WIDENING + 0.1:
        out.append(f"Widens to {1 + widening:.1f}x the inlet; trained aneurysms reach {1 + TRAINED_ANEURYSM_WIDENING:.1f}x.")
    return out


def _load_mesh_and_meta(
    path: Path, *, unit: str = "auto", inlet_hint: Any = None, outlet_hint: Any = None,
) -> tuple[meshio.Mesh, dict[str, Any], dict[str, Any]]:
    mesh = meshio.read(path)
    json_path = path.with_suffix(".json")
    tagged = _tag_hint_points(mesh, np.asarray(mesh.points), 1.0) != (None, None)
    if json_path.is_file() and tagged and inlet_hint is None:
        # Meshes from our own generator ship tags plus an exact sidecar; keep using both. A
        # sidecar next to an UNTAGGED mesh (e.g. a COMSOL export) cannot supply the boundary.
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise CustomerGeometryError(f"{json_path.name}: expected a JSON object.")
        return mesh, meta, {"boundary_source": "tags", "unit": str(meta.get("unit", MESH_UNIT_M))}
    return mesh_and_meta_from_outline(mesh, unit=unit, inlet_hint=inlet_hint, outlet_hint=outlet_hint, stem=path.stem)


def load_customer_geometry(
    path: Path | str,
    *,
    re_target: float = DEFAULT_RE,
    t_final_s: float | None = None,
    n_steps: int | None = None,
    mesh_unit: str = "auto",
    inlet_hint: Any = None,
    outlet_hint: Any = None,
) -> Data:
    """Load ``.pt`` / ``.msh`` / ``.nas`` into a deploy-ready ``Data`` graph.

    For meshes, ``mesh_unit`` ("auto", "m", "cm", "mm") and the optional ``inlet_hint`` /
    ``outlet_hint`` (points in metres) steer boundary inference; see ``mesh_and_meta_from_outline``.
    The inferred boundary summary is attached as ``data.customer_boundary``.
    """
    p = Path(path)
    if not p.is_file():
        raise CustomerGeometryError(f"File not found: {p}")
    suffix = p.suffix.lower()
    stem = p.stem

    if suffix == ".pt":
        data = load_untrusted_graph(p)
        if not isinstance(data, Data):
            raise CustomerGeometryError(f"{p.name}: expected a PyG Data graph.")
        _validate_masks(data, stem=stem)
        data = apply_re_target(data, re_target)
        t_end = float(t_final_s) if t_final_s is not None else float(
            getattr(data, "t", torch.tensor([8000.0])).reshape(-1)[-1].item()
        )
        steps = n_steps
        if steps is None and hasattr(data, "y") and torch.is_tensor(data.y) and data.y.dim() == 3:
            steps = int(data.y.shape[0])
        return synthesize_deploy_timeline(data, t_final_s=t_end, n_steps=steps)

    if suffix in (".msh", ".nas"):
        mesh, meta, boundary = _load_mesh_and_meta(
            p, unit=mesh_unit, inlet_hint=inlet_hint, outlet_hint=outlet_hint
        )
        data = graph_from_mesh_meta(mesh, meta, re_target=re_target, stem=stem)
        data.mesh_path = str(p.resolve())
        if boundary["boundary_source"] != "tags":
            # The t=0 FEM solve re-reads the mesh from `mesh_path` and registers its nodes onto
            # the graph exactly, so it must see the linear, metre-scaled mesh the graph was built
            # from -- not the original (a quadratic export's mid-side nodes are not the graph's).
            key = hashlib.sha1(
                p.read_bytes() + json.dumps([mesh_unit, inlet_hint, outlet_hint], default=str).encode()
            ).hexdigest()[:16]
            cache = default_customer_mesh_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            tagged_path = cache / f"upload_{key}.msh"
            meshio.write(tagged_path, mesh, file_format="gmsh22", binary=False)
            data.mesh_path = str(tagged_path)
        data.customer_boundary = boundary
        if t_final_s is not None or n_steps is not None:
            data = synthesize_deploy_timeline(
                data,
                t_final_s=float(t_final_s if t_final_s is not None else 8000.0),
                n_steps=n_steps,
            )
        return data

    raise CustomerGeometryError(
        f"Unsupported type '{suffix}'. Use .pt, .msh, or .nas (see {INBOX_DIRNAME}/README.txt)."
    )


def build_parametric_customer_graph(
    *,
    re_target: float = DEFAULT_RE,
    t_final_s: float = 8000.0,
    n_steps: int = DEFAULT_N_STEPS,
    width: float | None = None,
    angle_span: float | None = None,
    amplitude: float | None = None,
    level: int = 0,
    params_override: dict[str, Any] | None = None,
) -> Data:
    """Build a synthetic vessel mesh and return a deploy-ready graph.

    ``params_override`` may be an edited-walls params dict from
    ``geometry_to_params_override`` (skips width/angle sampling).

    The mesh is sized to the COMSOL anchors ``clot_ml_0`` trained on (``ANCHOR_MESH_LC_M``).

    Meshes are cached under ``outputs/customer/_meshes/`` so local FEM can resolve
    ``mesh_path`` after the build returns.
    """
    from src.data_gen.lib.vessel_generator import (
        VesselGenerator,
        build_vessel_mesh,
        make_vessel_params,
    )

    # `mesh_h_nd_target=0` selects the absolute `mesh_lc` in `_gmsh_size_bounds`.
    cfg = VesselConfig(
        phase="kinematics", mesh_h_nd_target=0.0, mesh_lc=ANCHOR_MESH_LC_M,
        mesh_min_elems_across=ANCHOR_MESH_MIN_ELEMS_ACROSS,
    )
    cfg.mesh_size_factor = 1.0  # set after init: the phase default would override a kwarg
    gen = VesselGenerator(phase="kinematics")
    gen.cfg = cfg
    cfg_dict = dict(gen._cfg_dict())
    cfg_dict["unit"] = "m"

    if params_override is not None:
        params = dict(params_override)
        params.setdefault("idx", 0)
    else:
        overrides: dict[str, Any] = {}
        if width is not None:
            overrides["width"] = float(width)
        if angle_span is not None:
            overrides["angle_span"] = float(angle_span)
            overrides["curve_type"] = "arc" if abs(float(angle_span)) > 1e-6 else "straight"
        if amplitude is not None:
            overrides["amplitude"] = float(amplitude)
            if float(amplitude) > 0:
                overrides["curve_type"] = "sine"
        params = make_vessel_params(idx=0, level=int(level), cfg=cfg, **overrides)

    cache_key = _parametric_cache_key(
        params,
        re_target=float(re_target),
        t_final_s=float(t_final_s),
        n_steps=int(n_steps),
    )
    cache_dir = default_customer_mesh_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    msh_path = cache_dir / f"customer_{cache_key}.msh"
    json_path = cache_dir / f"customer_{cache_key}.json"

    if not msh_path.is_file() or not json_path.is_file():
        idx, ok, err = build_vessel_mesh(params, cfg_dict, cache_dir)
        if not ok:
            raise CustomerGeometryError(err or "parametric mesh build failed")
        built_msh = cache_dir / f"vessel_{idx}.msh"
        built_json = cache_dir / f"vessel_{idx}.json"
        if built_msh.resolve() != msh_path.resolve():
            if msh_path.is_file():
                msh_path.unlink()
            built_msh.replace(msh_path)
        if built_json.resolve() != json_path.resolve():
            if json_path.is_file():
                json_path.unlink()
            built_json.replace(json_path)
        for leftover in cache_dir.glob("vessel_*.*"):
            leftover.unlink(missing_ok=True)

    mesh = meshio.read(msh_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    data = graph_from_mesh_meta(
        mesh,
        meta,
        re_target=re_target,
        stem=f"customer_{cache_key}",
    )
    data.mesh_path = str(msh_path.resolve())
    return synthesize_deploy_timeline(data, t_final_s=t_final_s, n_steps=n_steps)


def preview_points_from_graph(data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return SI positions and inlet/outlet/wall masks for a lightweight preview."""
    d_bar = float(data.d_bar.reshape(-1)[0].item()) if hasattr(data, "d_bar") else 1.0
    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64) * d_bar
    inlet = data.mask_inlet.reshape(-1).bool().cpu().numpy()
    outlet = data.mask_outlet.reshape(-1).bool().cpu().numpy()
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    return pos, inlet, outlet, wall
