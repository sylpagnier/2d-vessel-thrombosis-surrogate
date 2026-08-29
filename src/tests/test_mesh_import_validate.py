"""Gmsh NAS import validation (no COMSOL required)."""

import numpy as np
import pytest

from src.data_gen.lib.comsol_t0_fluid import validate_mesh_import


def test_validate_mesh_import_identical_passes():
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [0.05, 0.01]], dtype=float)
    stats = validate_mesh_import(xy, xy)
    assert stats["gmsh_vertices"] == 3.0
    assert stats["nn_max_m"] == 0.0


def test_validate_mesh_import_count_mismatch_raises():
    a = np.zeros((3, 2))
    b = np.zeros((4, 2))
    with pytest.raises(ValueError, match="vertex count"):
        validate_mesh_import(a, b)


def test_validate_mesh_import_offset_raises():
    xy = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
    shifted = xy + np.array([2e-4, 0.0])
    with pytest.raises(ValueError, match="offset too large"):
        validate_mesh_import(xy, shifted, max_nn_m=2e-5)
