"""Shared COMSOL-boundary to mesh-vertex snap tolerance.

Geometry samples on a slanted/curved inlet sit ~0.3-0.5 mesh edges off vertices
(comsol048: 90-180 um on a 354 um edge). A fixed 10 um remap dropped those
points and shipped a 3-node Dirichlet mask. First-snap (volume nodes onto
datasets) and remap (csv onto mesh) share this helper.

``BIOCHEM_BOUNDARY_SNAP_CM`` (cm) overrides both paths when set.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

# Former environment overrides that nothing in the tree ever set and no doc
# named, so each always resolved to the value below.  Kept as named constants
# rather than inlined literals so the value stays greppable and explainable.
BIOCHEM_BOUNDARY_SNAP_CM = ""


BOUNDARY_SNAP_EDGE_FRAC = 0.55
BOUNDARY_SNAP_FLOOR_M = 2.0e-5  # 20 um
BOUNDARY_SNAP_SMALL_CLOUD_M = 1.0e-4  # 0.01 cm; <4-point dataset fallback


def boundary_snap_tol_m(*, mesh_edge_scale_m: float | None = None) -> float:
    """SI snap distance: env override, else ``max(20 um, 0.55 * mesh_edge_scale_m)``."""
    raw = BIOCHEM_BOUNDARY_SNAP_CM.strip()
    if raw:
        return float(raw) * 0.01
    if mesh_edge_scale_m is None or not np.isfinite(mesh_edge_scale_m) or mesh_edge_scale_m <= 0.0:
        return BOUNDARY_SNAP_FLOOR_M
    return max(BOUNDARY_SNAP_FLOOR_M, BOUNDARY_SNAP_EDGE_FRAC * float(mesh_edge_scale_m))


def boundary_snap_tol_cm(coords_cm: np.ndarray) -> float:
    """Distance (cm) to snap volume mesh nodes onto a COMSOL boundary dataset."""
    raw = BIOCHEM_BOUNDARY_SNAP_CM.strip()
    if raw:
        return float(raw)
    pts = np.asarray(coords_cm[:, :2], dtype=np.float64)
    if pts.shape[0] < 4:
        return BOUNDARY_SNAP_SMALL_CLOUD_M * 100.0
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nn_cm = float(np.median(np.asarray(dist[:, 1], dtype=np.float64)))
    return boundary_snap_tol_m(mesh_edge_scale_m=nn_cm * 0.01) * 100.0
