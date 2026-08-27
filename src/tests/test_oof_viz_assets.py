"""Pins for honest out-of-fold visualization assets."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch


def test_cv_fold_checkpoint_records_the_exact_exclusion(tmp_path):
    from scripts.run_phase9_cv import _save_fold_member
    from src.clot_ml.gnn import ClotGNN

    model = ClotGNN(3, 7, dim=4, layers=1, drop=0.0)
    predict = SimpleNamespace(model=model, norm=(np.zeros(3), np.ones(3)))
    _save_fold_member(
        tmp_path, tag="c0shape", fold=2, held=["patient040", "patient041"],
        train=["patient001", "patient005"],
        pool=["patient001", "patient005", "patient040", "patient041"],
        cols=["a", "b", "phys_mask"], cfg={"shape_w": 2.0, "rounds": 3},
        seed=0, predict=predict,
    )

    root = tmp_path / "c0shape" / "fold_02"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["purpose"].startswith("outer-fold checkpoint")
    assert manifest["held_out"] == ["patient040", "patient041"]
    assert not set(manifest["held_out"]) & set(manifest["train_anchors"])
    assert manifest["members"][0]["cfg"]["shape_w"] == 2.0
    saved = torch.load(root / "member_s0.pth", map_location="cpu", weights_only=False)
    assert saved["held_out"] == manifest["held_out"]
    assert saved["train_anchors"] == manifest["train_anchors"]


def test_strict_oof_masks_keep_domains_separate(monkeypatch):
    import eval_strict_temporal as temporal

    wall = np.array([True, True, False, False])
    V = {"v": {"S": {"wall": wall, "solid": wall, "owner": np.array([0, 1, 0, 0])},
               "times": [0, 1, 2]}}
    gm = np.array([True, False, True, True])
    P = np.array([[0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.2, 0.8],
                  [1.0, 0.0, 1.0, 1.0]])
    monkeypatch.setattr(temporal, "candidate_mask", lambda *_args, **_kwargs: gm)
    masks = temporal.predict_masks(
        V, "v", P, {"v4": {"v": np.zeros(4)}}, {}, ((0.5, True), (0.5, True)))

    assert masks.dtype == bool
    assert np.array_equal(masks[-1], gm)
    assert not masks[:, 1].any(), "an uncommitted wall node must not leak in"
    assert np.all(masks[1:] >= masks[:-1]), "OOF visualization must preserve monotonicity"


def test_oof_viz_generator_and_series_export_refuse_final_half():
    import inspect

    import scripts.eval_strict_temporal as temporal
    import scripts.gen_clot_ml_v0_oof_viz_data as generator

    exporter = inspect.getsource(temporal.main)
    assert "--save-oof-series" in exporter
    assert "set(oof_series) & set(SEALED)" in exporter
    source = inspect.getsource(generator.main)
    assert "set(vessels) & set(SEALED)" in source
    assert "base-model exclusion" in source


def test_oof_viz_template_prioritizes_viewer_and_distributions():
    from scripts import build_v4_temporal_artifact as builder

    # The legacy Phase-10 template remains in the source for historical reference, but the
    # active template must not regress to its long model-review narrative.
    assert "Generalization, vessel by vessel." in builder.TEMPLATE
    assert "Final deploy score distribution" in builder.TEMPLATE
    assert "Mean-over-time distribution" in builder.TEMPLATE
    assert "vessel-select" in builder.TEMPLATE
    assert "The stricter protocol, and what survives it." not in builder.TEMPLATE


def test_wound_viz_mode_is_explicit_and_uses_lovo_dispatcher():
    import inspect

    import scripts.gen_clot_ml_v0_oof_viz_data as generator

    source = inspect.getsource(generator.main)
    payload = inspect.getsource(generator.build_wound_payload)
    assert "--wound" in source
    assert "predict_clot_ml_v0" in payload
    assert "wound_rate_train" in payload
    assert "wound region" in payload
