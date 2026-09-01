"""Helpers for scripts/exp_template_ab.py -- unit detection and npz scalars."""

from __future__ import annotations

import numpy as np

from scripts.archive.tier1_retired.exp_template_ab import detect_xy_scale, expr_to_key, npz_scalar, rel_l2


def test_expr_to_key_matches_export():
    assert expr_to_key("d(spf.sr,x)") == "d_spf_sr_x"
    assert expr_to_key("spf.mu") == "spf_mu"


def test_detect_xy_scale_metres_already_match():
    ab = np.array([[0.0, 0.0], [0.1, 0.0]])
    pk = np.array([[0.0, 0.0], [0.1, 0.0]])
    assert abs(detect_xy_scale(ab, pk) - 1.0) < 1e-12


def test_detect_xy_scale_cm_labelled_as_metres():
    # Pack in metres (0.1 m span), A/B dumped in cm (10 span) -- the old matching failure.
    ab = np.array([[0.0, 0.0], [10.0, 0.0]])
    pk = np.array([[0.0, 0.0], [0.1, 0.0]])
    s = detect_xy_scale(ab, pk)
    assert abs(s - 0.01) < 1e-12


def test_npz_scalar_zero_dim_and_missing(tmp_path):
    path = tmp_path / "t.npz"
    np.savez(path, mesh_unit=np.array("cm"), coords_supplied=False, d_bar=0.015)
    z = np.load(path, allow_pickle=True)
    assert npz_scalar(z, "mesh_unit") == "cm"
    assert npz_scalar(z, "coords_supplied") is False
    assert abs(float(npz_scalar(z, "d_bar")) - 0.015) < 1e-12
    assert npz_scalar(z, "nope", default="x") == "x"


def test_rel_l2_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert rel_l2(a, a) == 0.0
    assert rel_l2(a, a, mask=np.array([True, False, True])) == 0.0
