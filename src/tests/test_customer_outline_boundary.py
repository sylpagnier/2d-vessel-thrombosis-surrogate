"""Uploaded meshes need no tags or sidecar: wall / inlet / outlet come from the mesh outline."""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest

from src.data_gen.lib.customer_geometry_import import (
    BoundaryNeedsInput,
    detect_mesh_unit,
    mesh_and_meta_from_outline,
)
from src.config import VesselConfig

TAGS = VesselConfig().TAGS


def _channel_mesh(length=100.0, width=8.0, nx=60, ny=8, stenosis=0.0, step=False):
    """Structured triangle mesh of a straight channel (mm), untagged, optionally narrowed."""
    xs = np.linspace(0.0, length, nx)
    pts, index = [], {}
    for i, x in enumerate(xs):
        half = 0.5 * width * (1.0 - stenosis * np.exp(-0.5 * ((x - length / 2) / (length / 12)) ** 2))
        if step and 0.4 * length <= x <= 0.6 * length:  # abrupt widening: extra sharp corners
            half *= 2.5
        for j, t in enumerate(np.linspace(-1.0, 1.0, ny)):
            index[i, j] = len(pts)
            pts.append([x, half * t, 0.0])
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
            tris += [[a, b, c], [a, c, d]]
    return meshio.Mesh(np.asarray(pts), [("triangle", np.asarray(tris))])


def _cap_x(mesh: meshio.Mesh, tag: int) -> float:
    lines = mesh.cells_dict["line"][mesh.cell_data_dict["gmsh:physical"]["line"] == tag]
    return float(mesh.points[np.unique(lines), 0].mean())


def test_straight_channel_boundary_is_found_automatically():
    tagged, meta, info = mesh_and_meta_from_outline(_channel_mesh())
    assert info["boundary_source"] == "auto" and info["unit"] == "mm"
    assert _cap_x(tagged, TAGS["Inlet"]) == pytest.approx(0.0, abs=1e-9)
    assert _cap_x(tagged, TAGS["Outlet_1"]) == pytest.approx(0.1, abs=1e-9)
    assert meta["d_inlet"] == pytest.approx(0.008, rel=1e-6)
    assert info["warnings"] == []
    assert info["length_m"] == pytest.approx(0.1, rel=1e-3)
    n_wall_edges = int(np.sum(tagged.cell_data_dict["gmsh:physical"]["line"] == TAGS["Walls"]))
    assert n_wall_edges == 2 * (60 - 1)


def test_stenosis_keeps_its_ends_and_mean_width():
    tagged, meta, info = mesh_and_meta_from_outline(_channel_mesh(stenosis=0.6), unit="mm")
    assert info["boundary_source"] == "auto"
    assert _cap_x(tagged, TAGS["Inlet"]) < _cap_x(tagged, TAGS["Outlet_1"])
    assert 0.004 < meta["d_bar"] < 0.008          # the generator's mean width...
    from src.data_gen.lib.customer_geometry_import import graph_from_mesh_meta
    data = graph_from_mesh_meta(tagged, meta)     # ...but the graph uses the INLET width, like training
    assert float(data.d_bar.reshape(-1)[0]) == pytest.approx(0.008, rel=1e-6)


def test_user_picked_ends_override_the_automatic_direction():
    mesh = _channel_mesh()
    tagged, _, info = mesh_and_meta_from_outline(mesh, inlet_hint=[0.1, 0.0], outlet_hint=[0.0, 0.001])
    assert info["boundary_source"] == "user"
    assert _cap_x(tagged, TAGS["Inlet"]) == pytest.approx(0.1, abs=1e-9)
    assert _cap_x(tagged, TAGS["Outlet_1"]) == pytest.approx(0.0, abs=1e-9)


def test_ambiguous_outline_asks_for_the_ends():
    with pytest.raises(BoundaryNeedsInput) as exc:
        mesh_and_meta_from_outline(_channel_mesh(step=True), unit="mm")
    assert exc.value.boundary_mask.sum() > 0 and exc.value.pos.shape[1] == 2
    assert "Click the inlet" in str(exc.value)
    # ...and the same outline loads once the user has pointed at the two open ends.
    tagged, _, info = mesh_and_meta_from_outline(
        _channel_mesh(step=True), unit="mm", inlet_hint=[0.0, 0.0], outlet_hint=[0.1, 0.0]
    )
    assert info["boundary_source"] == "user"
    assert _cap_x(tagged, TAGS["Inlet"]) == pytest.approx(0.0, abs=1e-9)


def test_unit_detection_targets_the_trained_vessel_length():
    pts = np.array([[0.0, 0.0], [100.0, 8.0]])
    assert detect_mesh_unit(pts) == "mm"
    assert detect_mesh_unit(pts / 10) == "cm"
    assert detect_mesh_unit(pts / 1000) == "m"


ANCHOR_MESH = Path(__file__).resolve().parents[2] / "data" / "raw" / "biochem_anchors" / "comsol041.msh"


@pytest.mark.skipif(not ANCHOR_MESH.is_file(), reason="local COMSOL anchor mesh not present")
def test_untagged_quadratic_comsol_export_loads_without_a_sidecar():
    """comsol041.msh: triangle6, centimetres, no line tags -- the demo vessel's own mesh."""
    from src.data_gen.lib.customer_geometry_import import load_customer_geometry

    data = load_customer_geometry(ANCHOR_MESH, t_final_s=3600.0, n_steps=20)
    assert data.customer_boundary["boundary_source"] == "auto"
    assert data.customer_boundary["unit"] == "cm"
    x = data.x[:, 0]
    assert float(x[data.mask_inlet.bool()].mean()) < float(x[data.mask_outlet.bool()].mean())
    assert Path(data.mesh_path).suffix == ".msh" and Path(data.mesh_path).is_file()


def test_upload_ids_and_points_are_validated():
    from src.tools.customer_predict_web import _point, _store_upload

    for bad in ("../../secret.msh", "abc.msh", "0" * 32 + ".exe"):
        with pytest.raises(ValueError):
            _store_upload({"upload_id": bad})
    assert _point({}, "inlet_hint") is None
    assert _point({"inlet_hint": [0.01, -0.002]}, "inlet_hint") == [0.01, -0.002]
    for bad in ([1.0], ["x", 1], [float("nan"), 0.0], "0,0"):
        with pytest.raises(ValueError):
            _point({"inlet_hint": bad}, "inlet_hint")


def test_uploads_outside_the_training_envelope_are_flagged():
    _, _, info = mesh_and_meta_from_outline(_channel_mesh(width=3.0), unit="mm")
    assert any("Inlet width 3.0 mm" in w for w in info["warnings"])
