"""The Predict app's training-envelope constants must match the cohort the deployed model was fit on.

Re-measures the COMSOL training vessels (raw meshes + packs) with the same outline code the app
uses for uploads. Skipped when that local data is absent (it is not in the public repository).
"""
from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pytest

from src.data_gen.lib import customer_geometry_import as cgi

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "outputs" / "clot_ml" / "locked" / "clot_ml_final" / "manifest.json"
RAW = ROOT / "data" / "raw" / "biochem_anchors"
PACKS = ROOT / "data" / "processed" / "graphs_biochem_anchors"

pytestmark = pytest.mark.skipif(
    not (MANIFEST.is_file() and RAW.is_dir() and PACKS.is_dir()), reason="training cohort not present locally"
)


def _cohort():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return manifest["training_pool"], manifest["geometry_classes"]


def _walls(stem):
    pts, tris = cgi._corner_triangles(meshio.read(RAW / f"{stem}.nas"))
    pts = pts * cgi.MESH_UNIT_SCALE["cm"]
    loop = cgi._boundary_loop(tris, len(pts))
    split = cgi.split_outline(pts, loop)
    P = pts[loop]
    return (cgi._resample_polyline(P[split["walls"][0]], 400), cgi._resample_polyline(P[split["walls"][1]], 400))


def test_inlet_width_is_the_length_scale_and_ranges_match():
    from src.utils.safe_load import load_untrusted_graph

    pool, classes = _cohort()
    inlet, length, narrowing, widening = [], [], [], []
    for stem in pool:
        top, bot = _walls(stem)
        w = np.linalg.norm(top - bot, axis=1)
        pack = load_untrusted_graph(PACKS / f"{stem}.pt")
        assert float(pack.d_bar.reshape(-1)[0]) == pytest.approx(w[0], rel=2e-3), stem   # d_bar == inlet width
        inlet.append(w[0])
        length.append(cgi._arc_length(0.5 * (top + bot)))
        if classes[stem] == "stenosis":
            narrowing.append(1 - w.min() / w[0])
        if classes[stem] == "aneurysm":
            widening.append(w.max() / w[0] - 1)
    lo, hi = cgi.TRAINED_INLET_WIDTH_M
    assert min(inlet) == pytest.approx(lo, abs=2e-4) and max(inlet) == pytest.approx(hi, abs=2e-4)
    assert cgi.TRAINED_LENGTH_M[0] <= min(length) and max(length) <= cgi.TRAINED_LENGTH_M[1]
    assert max(narrowing) == pytest.approx(cgi.TRAINED_STENOSIS_NARROWING, abs=0.01)
    assert max(widening) == pytest.approx(cgi.TRAINED_ANEURYSM_WIDENING, abs=0.01)


def test_app_built_graph_matches_its_training_pack():
    """A training vessel uploaded as a raw mesh must give the model the inputs its pack does.

    Guards the train/deploy fixes in docs/DEPLOY_ALIGNMENT.md: inlet/outlet masks keep their
    corners, d_bar is the inlet width, and the node channels the model reads are built with the
    training extractor's recipe. comsol032 has no COMSOL tagging gaps, so masks match exactly.
    """
    from scipy.spatial import cKDTree

    from src.utils.safe_load import load_untrusted_graph

    stem = "comsol032"
    pack = load_untrusted_graph(PACKS / f"{stem}.pt")
    app = cgi.load_customer_geometry(RAW / f"{stem}.nas", t_final_s=3600.0, n_steps=20)
    assert app.num_nodes == pack.num_nodes
    assert float(app.d_bar.reshape(-1)[0]) == pytest.approx(float(pack.d_bar.reshape(-1)[0]), rel=1e-4)
    dist, nn = cKDTree(app.x[:, :2].numpy() * float(app.d_bar)).query(
        pack.x[:, :2].detach().numpy() * float(pack.d_bar))
    assert dist.max() < 2e-5 and len(np.unique(nn)) == pack.num_nodes
    for k in ("mask_inlet", "mask_outlet", "mask_wall"):
        assert (getattr(pack, k).reshape(-1).bool().numpy() == getattr(app, k).reshape(-1).bool().numpy()[nn]).all(), k
    X, Y = pack.x.detach().numpy(), app.x.numpy()[nn]
    rel = lambda j: float(np.abs(X[:, j] - Y[:, j]).mean() / (np.abs(X[:, j]).mean() + 1e-9))
    assert rel(2) < 0.01 and rel(15) < 0.01          # sdf_nd, width_nd
    assert rel(4) < 0.10 and rel(5) < 0.10           # wall normals
    y0 = app.y[0].numpy()
    assert np.allclose(y0[:, [8, 10, 11]], np.log1p(1.0))   # PT / AT / FG at bulk, as every pack


def test_customer_t0_pressure_comes_from_the_fem_solve():
    """`p_nd` is a model feature; a customer vessel gets it from the FEM, in the pack's own units."""
    from scipy.spatial import cKDTree

    from src.clot_ml.v0 import solve_fem_into_pack
    from src.utils.safe_load import load_untrusted_graph

    stem = "comsol032"
    pack = load_untrusted_graph(PACKS / f"{stem}.pt")
    app = cgi.load_customer_geometry(RAW / f"{stem}.nas", t_final_s=3600.0, n_steps=20)
    assert float(np.abs(app.y[0, :, 2].numpy()).max()) == 0.0      # nothing before the solve
    solve_fem_into_pack(app)
    _, nn = cKDTree(app.x[:, :2].numpy() * float(app.d_bar)).query(
        pack.x[:, :2].detach().numpy() * float(pack.d_bar))
    p_pack, p_app = pack.y[0, :, 2].detach().numpy(), app.y[0, :, 2].numpy()[nn]
    assert np.linalg.norm(p_app - p_pack) / np.linalg.norm(p_pack) < 0.1
    assert (app.y[:, :, 2] == app.y[0:1, :, 2]).all()              # every frame carries it


def test_wound_ranges_match_the_wound_cohort():
    from src.utils.safe_load import load_untrusted_graph

    centres, widths, horizons = [], [], []
    for i in range(1, 7):
        g = load_untrusted_graph(PACKS / f"wound_comsol00{i}.pt")
        xy = g.x[:, :2].numpy()
        inl, outl, wound = (getattr(g, k).bool().numpy() for k in ("mask_inlet", "mask_outlet", "mask_wound"))
        axis = xy[outl].mean(0) - xy[inl].mean(0)
        s = (xy - xy[inl].mean(0)) @ axis / float(axis @ axis)
        centres.append((s[wound].min() + s[wound].max()) / 2)
        widths.append(s[wound].max() - s[wound].min())
        horizons.append(float(g.t.reshape(-1)[-1]))
    assert min(centres) == pytest.approx(cgi.TRAINED_WOUND_CENTER_FRAC[0], abs=0.015)
    assert max(centres) == pytest.approx(cgi.TRAINED_WOUND_CENTER_FRAC[1], abs=0.015)
    assert min(widths) == pytest.approx(cgi.TRAINED_WOUND_WIDTH_FRAC[0], abs=0.015)
    assert max(widths) == pytest.approx(cgi.TRAINED_WOUND_WIDTH_FRAC[1], abs=0.015)
    assert sorted(horizons)[-2] <= cgi.TRAINED_WOUND_HORIZON_S   # all but the wound_comsol003 outlier
