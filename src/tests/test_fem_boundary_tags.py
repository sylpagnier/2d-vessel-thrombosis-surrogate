"""The local FEM solver's inlet/outlet facet tagging, and the geometric completion of it.

WHY THIS EXISTS.  Inlet and outlet facets used to be tagged by requiring BOTH corner vertices
of a boundary facet to carry COMSOL's node selection.  That selection is not always complete
on a quadratic mesh, and two of the 2026-09-02 packs prove it:

    comsol038   no two adjacent inlet corners tagged at all -- 0 facets, and the solve was
                 refused outright ("an untagged inlet or outlet leaves ... a singular solve")
    comsol048   4 of 21 outlet facets tagged with one corner each, so those four silently
                 took the no-slip WALL condition instead of the outlet one and the solve was
                 wrong without saying so

An inlet or outlet is a straight cut through the lumen, so the tagged nodes determine it
exactly: fit the line through them, take every boundary facet whose midpoint lies on it
inside the tagged nodes' own along-line extent.  The completion is accepted only when it
CONTAINS what the corner rule already agreed on.

These tests pin both halves: that the completion recovers the two broken packs, and that it
is a bit-exact no-op wherever the corner rule was already complete.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.paths import anchor_meshes_dir, anchor_packs_dir, get_project_root

REPO = get_project_root()
PACKS = anchor_packs_dir()
MESHES = anchor_meshes_dir()

skfem = pytest.importorskip("skfem")

#: packs where the corner rule was already complete -- the completion must not move a facet
INTACT = ("comsol012", "comsol020", "comsol044", "wound_comsol005")
#: (pack, expected inlet facets, expected outlet facets) after completion
REPAIRED = (("comsol038", 20, 20), ("comsol048", 21, 21))


def _register(stem: str):
    """Load the pack and its mesh, at the unit scale that makes them coincide."""
    from scipy.spatial import KDTree

    logging.disable(logging.INFO)
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    mesh = skfem.Mesh.load(str(MESHES / f"{stem}.nas"))
    pos = d.x[:, 0:2].numpy() * float(d.d_bar.reshape(-1)[0])
    kd = KDTree(pos)
    best = None
    for scale in (1.0, 0.01, 0.001):
        cand = mesh if scale == 1.0 else mesh.scaled(scale)
        dist, idx = kd.query(cand.p.T)
        key = (int(np.unique(idx).size), -float(np.median(dist)))
        if best is None or key > best[0]:
            best = (key, cand, idx)
    return d, best[1], best[2]


def _corner_and_plane(stem: str):
    """(corner-rule, completed) facet index sets for inlet and outlet."""
    d, mesh, idx = _register(stem)
    bf = mesh.boundary_facets()
    P = mesh.p.T
    a, b = P[mesh.facets[0, bf]], P[mesh.facets[1, bf]]
    mid, h = 0.5 * (a + b), np.linalg.norm(b - a, axis=1)
    out = {}
    for name, mask in (("inlet", d.mask_inlet), ("outlet", d.mask_outlet)):
        m = mask.reshape(-1).bool().numpy()[idx]
        corner = bf[np.all(m[mesh.facets], axis=0)[bf]]
        pts = P[m]
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        along, normal = vt[0], vt[1]
        extent = float(np.abs((pts - c) @ along).max())
        sel = (np.abs((mid - c) @ normal) < 0.05 * h) & (np.abs((mid - c) @ along) <= extent)
        out[name] = (set(corner.tolist()), set(bf[sel].tolist()))
    return out


@pytest.mark.parametrize("stem", INTACT)
def test_completion_is_a_no_op_where_the_corner_rule_was_complete(stem):
    if not (PACKS / f"{stem}.pt").exists() or not (MESHES / f"{stem}.nas").exists():
        pytest.skip(f"{stem} not on disk")
    for name, (corner, plane) in _corner_and_plane(stem).items():
        assert corner, f"{stem} {name}: corner rule found nothing to compare against"
        assert corner == plane, (
            f"{stem} {name}: the planar completion moved a facet where the corner rule was "
            f"already complete (corner {len(corner)}, plane {len(plane)})")


@pytest.mark.parametrize("stem,n_inlet,n_outlet", REPAIRED)
def test_completion_recovers_the_two_incompletely_tagged_packs(stem, n_inlet, n_outlet):
    if not (PACKS / f"{stem}.pt").exists() or not (MESHES / f"{stem}.nas").exists():
        pytest.skip(f"{stem} not on disk")
    got = _corner_and_plane(stem)
    for name, want in (("inlet", n_inlet), ("outlet", n_outlet)):
        corner, plane = got[name]
        assert len(plane) == want, f"{stem} {name}: completed to {len(plane)}, expected {want}"
        assert corner <= plane, (
            f"{stem} {name}: the completion dropped a facet the corner rule had tagged; it "
            f"is only ever allowed to ADD")
    assert not (got["inlet"][1] & got["outlet"][1]), f"{stem}: inlet and outlet overlap"
    # the point of the fix: at least one of the two ends was genuinely broken before it
    assert any(len(c) != len(p) for c, p in got.values()), (
        f"{stem} is listed as repaired but the corner rule already agreed -- move it to INTACT")


def test_the_solver_uses_the_completion_and_solves_the_pack_that_used_to_refuse():
    """End-to-end: `comsol038` raised on `inlet=0 outlet=0` before this fix."""
    stem = "comsol038"
    if not (PACKS / f"{stem}.pt").exists() or not (MESHES / f"{stem}.nas").exists():
        pytest.skip(f"{stem} not on disk")
    from src.clot_ml.v0 import solve_fem_into_pack

    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    solve_fem_into_pack(d)
    u = d.u0_pred.numpy()
    v = d.v0_pred.numpy()
    assert np.isfinite(u).all() and np.isfinite(v).all()
    wall = d.mask_wall.reshape(-1).bool().numpy()
    speed = np.hypot(u, v)
    assert speed[~wall].max() > 0.0, "solved field is identically zero off the wall"
    assert float(np.median(speed[wall])) < 0.05 * float(np.median(speed[~wall])), (
        "no-slip is not being imposed: wall speed is not small against the lumen")
