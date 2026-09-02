from __future__ import annotations

import numpy as np


def test_customer_web_payload_contains_scrubbable_fields_and_metrics():
    from src.inference.customer_pipeline import CustomerTrajectory
    from src.tools.customer_predict_web import _trajectory_payload

    n = 12
    times = np.array([0.0, 3750.0, 7500.0])
    pos = np.c_[np.linspace(0.0, 1.0, n), np.zeros(n)]
    wall = np.zeros(n, dtype=bool)
    wall[[0, -1]] = True
    phi = {i: np.linspace(0.0, min(1.0, i / 2), n) for i in range(3)}
    vel = {i: np.full(n, 1.0 + i) for i in range(3)}
    traj = CustomerTrajectory(
        t_sec=times,
        pos=pos,
        vel_mag=vel,
        mu_eff_si={i: np.ones(n) for i in range(3)},
        phi=phi,
        n_steps=3,
        mask_wall=wall,
        mask_inlet=np.zeros(n, dtype=bool),
        mask_outlet=np.zeros(n, dtype=bool),
        hop_from_wall=np.arange(n, dtype=np.int32),
        meta={"include_velocity": True},
    )

    payload = _trajectory_payload(traj, run_mode="scientific", csv_url="/api/job/x/csv")
    assert len(payload["pos"]) == n
    assert len(payload["phi"]) == 3
    assert len(payload["velocity"]) == 3
    assert len(payload["metrics"]) == 3
    assert payload["meta"]["run_mode"] == "scientific"
    assert payload["csv_url"].endswith("/csv")
    import json

    # The browser contract must be strict JSON; NaN/Infinity are converted to null.
    json.dumps(payload, allow_nan=False)


def test_customer_web_page_has_existing_customer_modes():
    from src.tools.customer_predict_web import PAGE

    for token in (
        "Inbox geometry", "Parametric vessel", "Clot + Velocity", "Scientific", "field-canvas", "/api/run",
        "Estimate runtime", "renderPreview", "startJob('estimate')", "Add mirrored wound",
        "markGeometryDirty", "clot_ml_0",
    ):
        assert token in PAGE


def test_runtime_estimate_scales_with_mesh_and_rollout_length():
    from types import SimpleNamespace

    from src.tools.customer_predict_web import _runtime_estimate

    data = SimpleNamespace(
        num_nodes=4_000,
        x=np.zeros((4_000, 2), dtype=np.float32),
        edge_index=np.zeros((2, 20_000), dtype=np.int64),
    )
    estimate = _runtime_estimate(data, n_steps=120, hours=8.0)

    assert estimate["n_nodes"] == 4_000
    assert estimate["n_edges"] == 20_000
    assert estimate["estimate_low_s"] < estimate["estimate_mid_s"] < estimate["estimate_high_s"]


def test_customer_baseline_uses_the_clot_ml_0_alias_without_loading_weights():
    import torch

    from src.inference.customer_pipeline import CustomerDeployPipeline, DEFAULT_CUSTOMER_CLOT_MODEL

    pipeline = CustomerDeployPipeline(device=torch.device("cpu"), require_cuda=False)
    assert DEFAULT_CUSTOMER_CLOT_MODEL == "clot_ml_0"
    assert pipeline.model_name == "clot_ml_0"
    assert pipeline.locked_model_name == "clot_ml_0"


def test_parametric_vessel_is_clean_and_defined_only_by_visible_controls():
    from src.tools.customer_predict_web import _parametric_params

    straight = _parametric_params({"width": 0.004, "bend": 0.0, "amp": 0.0, "pathology": "none"})
    curved = _parametric_params({"width": 0.012, "bend": 70.0, "amp": 0.0, "pathology": "none"})
    s_curve = _parametric_params({"width": 0.008, "bend": 0.0, "amp": 0.010, "pathology": "none"})

    assert straight["curve_type"] == "straight"
    assert curved["curve_type"] == "arc"
    assert s_curve["curve_type"] == "s_curve"
    assert straight["width"] == 0.004
    assert curved["width"] == 0.012
    assert not any(straight["offsets"])
    assert not any(straight["noise_top"])
    assert not any(straight["noise_bot"])
    assert not any(straight["tortuosity"])
