"""Rebuild a pack's node features in place, faithfully to the extractor.

The COMSOL exports behind `data/processed/graphs_biochem_anchors` are multi-GB and several
are gone (all three `wound_patient*` runs), so a pack whose builder had a bug cannot simply
be re-extracted.  It has to be repaired on disk, and a repair is only trustworthy if it goes
through **the same call the extractor uses** -- `build_kinematics_node_x_tensor` with
`resolve_anchor_kine_phys_cfg`'s Carreau config -- changing exactly one input.

This module is the shared engine for the two repair scripts:

* `scripts/repair_wound_pack_geometry.py` -- the solid-boundary fix (WOUND_PROGRESS 6):
  wall-derived geometry measured against `mask_wall` alone put wound nodes in the lumen.
* `scripts/repair_pack_wall_normals.py` -- the dead-channel fix
  (MODEL_REVIEW_2026-08-22 6.5): `wall_normal` zero at every boundary node, `node_type_*`
  all-zero everywhere.

It lives in `src/` rather than in one of the scripts because both need it and because a
script importing another script depends on `scripts/` being on `sys.path`.

TWO THINGS EVERY CALLER MUST KNOW.

**A full rebuild does NOT reproduce the stored packs**, and that is not this code's fault:
the cohort was written by more than one extractor revision and they disagree about the
*prior* channels.  On `patient020` a fresh build puts `wss_prior_nd` at ~45 at the wall where
the pack has 0, and moves `u_prior` by 0.55 in the interior (WOUND_PROGRESS 8, last
paragraph -- still unresolved).  So a repair must never write `rebuild_x`'s output wholesale;
it writes either the affected *rows* or the *delta* against a pre-fix rebuild.

**`data.x` is not the only copy.**  `data.x_biochem` carries its own `wall_normal`
(`BIO_X_SCHEMA` channels 3:5) and `assert_anchor_dual_x_aligned` requires the two to agree,
so use :func:`write_x` rather than assigning `data.x` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.config import BiochemNodeFeat, NodeFeat
from src.data_gen.lib.centerline_utils import resolve_centerline_nd
from src.data_gen.lib.mesh_wls import (
    boundary_normals_from_graph,
    solid_boundary_mask,
    solid_boundary_nodes,
)
from src.data_gen.lib.node_feature_assembly import (
    build_kinematics_node_x_tensor,
    resolve_anchor_kine_phys_cfg,
)

SIDECAR_DIR = Path("data/raw/biochem_anchors")

__all__ = ["sdf_and_normals", "rebuild_x", "write_x", "solid_of"]


def solid_of(data) -> np.ndarray:
    """Every no-slip boundary node as a numpy mask: healthy wall union wound."""
    return solid_boundary_nodes(data)


def sdf_and_normals(
    pos_nd: torch.Tensor,
    mask_solid: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    *,
    graph_normals: bool = True,
):
    """KD-tree SDF and wall normals measured against the solid boundary.

    Mirrors ``wall_normals_and_sdf_mesh_to_graph_style``.  Positions here are already
    non-dimensional, so the query distance *is* the SDF.

    Off the boundary the normal is the KD-tree offset direction, as it always was.  **On**
    the boundary a node is its own nearest boundary point, so that offset is the zero vector
    -- which was the bug: every pack carried ``wall_normal == 0`` at every wall node, because
    the COMSOL ``.msh`` files export no line cells and the exact segment-normal branch never
    ran (WOUND_PROGRESS 8).  With ``graph_normals=True`` (default) the boundary normal comes
    from :func:`boundary_normals_from_graph` instead.

    ``graph_normals=False`` reproduces the old zero convention exactly, which is what a
    pre-fix pack contains -- pass it to build the baseline a delta is measured against.
    """
    p = pos_nd.detach().cpu().numpy().astype(np.float64)
    solid_np = mask_solid.detach().cpu().numpy()
    solid_idx = np.where(solid_np)[0]
    if solid_idx.size == 0:
        raise ValueError("empty solid boundary mask")
    tree = cKDTree(p[solid_idx])
    dist, nearest = tree.query(p)
    diff = p - p[solid_idx][nearest]
    if graph_normals and edge_index is not None:
        gn = boundary_normals_from_graph(p, solid_np, edge_index.detach().cpu().numpy())
        got = np.linalg.norm(gn, axis=1) > 0.5
        diff[got] = gn[got]
    normal = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
    sdf = torch.clamp(torch.tensor(dist, dtype=torch.float32).view(-1, 1), min=1e-6)
    return sdf, torch.tensor(normal, dtype=torch.float32)


def rebuild_x(data, *, use_wound: bool = True, graph_normals: bool = True):
    """Return a freshly built 18ch ``x`` (+ prior vectors), extractor-faithful.

    ``use_wound=False`` measures geometry against ``mask_wall`` alone (the pre-WOUND_PROGRESS-6
    convention); ``graph_normals=False`` restores the zero boundary normal.  Both exist so a
    repair can be diffed against exactly what it replaces.
    """
    wound = getattr(data, "mask_wound", None) if use_wound else None
    mask_solid = solid_boundary_mask(data.mask_wall, wound)

    pos_nd = data.x[:, NodeFeat.XY]
    d_bar_si = float(data.d_bar.reshape(-1)[0])
    u_ref = float(data.u_ref.reshape(-1)[0])
    phys_cfg = resolve_anchor_kine_phys_cfg()

    sdf_nd, wall_normal = sdf_and_normals(
        pos_nd, mask_solid, data.edge_index, graph_normals=graph_normals)

    # compute_hydraulic_width_nd probes with SI coordinates, so the tree is SI.
    wall_tree = cKDTree(pos_nd[mask_solid].detach().cpu().numpy() * d_bar_si)

    cl_pts, cl_tan, _ = resolve_centerline_nd(
        pos_nd,
        data.mask_inlet,
        data.mask_outlet,
        edge_index=data.edge_index,
        mask_wall=mask_solid,
        stem=getattr(data, "graph_stem", None),
        raw_sidecar_dir=SIDECAR_DIR,
    )

    # The extractor passes inlet-only Dirichlet values, zero elsewhere.
    n = int(pos_nd.shape[0])
    u_bc = torch.zeros(n, dtype=torch.float32)
    v_bc = torch.zeros(n, dtype=torch.float32)
    if getattr(data, "u_inlet_bc", None) is not None and data.u_inlet_bc.shape[1] >= 2:
        mi = data.mask_inlet
        u_bc[mi] = data.u_inlet_bc[mi, 0]
        v_bc[mi] = data.u_inlet_bc[mi, 1]

    return build_kinematics_node_x_tensor(
        pos_nd=pos_nd,
        sdf_nd=sdf_nd,
        wall_normal=wall_normal,
        mask_inlet=data.mask_inlet,
        mask_outlet=data.mask_outlet,
        mask_wall=mask_solid,
        d_bar_si=d_bar_si,
        u_ref=u_ref,
        phys_cfg=phys_cfg,
        wall_tree=wall_tree,
        edge_index=data.edge_index,
        G_x=getattr(data, "G_x", None),
        G_y=getattr(data, "G_y", None),
        centerline_pts_nd=cl_pts,
        centerline_tangents_nd=cl_tan,
        inlet_uv_nd=(u_bc, v_bc),
        mu_nd_scale=phys_cfg.mu_viscosity_nd_scale,
    )


def write_x(data, x_new: torch.Tensor) -> None:
    """Install a repaired ``x`` and keep ``x_biochem``'s duplicate normal in step.

    ``BIO_X_SCHEMA`` repeats ``wall_normal_x``/``wall_normal_y`` at channels 3:5 and
    ``assert_anchor_dual_x_aligned`` fails if the two tensors disagree, so every in-place
    repair of ``data.x`` must touch both.  A fresh extract does not need this -- the builder
    hands the same ``wall_normal`` tensor to both layouts.
    """
    names = data.x_channel_names.split(",")
    data.x = x_new
    xb = getattr(data, "x_biochem", None)
    if xb is None:
        return
    xb = xb.clone()
    xb[:, BiochemNodeFeat.WALL_NORMAL] = x_new[
        :, [names.index("wall_normal_x"), names.index("wall_normal_y")]]
    data.x_biochem = xb
