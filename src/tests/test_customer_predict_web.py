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

    # The mode DROPDOWN ("Clot + Velocity" as one option) was replaced by a
    # `mode-switch` button group with `data-mode` values, and the twin velocity panels by a
    # single panel with a field picker.  These assert the CAPABILITIES still reachable, not
    # the old control's label -- a UI rewrite should be free to move a control without
    # breaking a test whose subject is "the customer flows still exist".
    for token in (
        "Upload geometry", "Parametric vessel", "field-canvas", "/api/run",
        "Estimate runtime", "renderPreview", "startJob('estimate')", "Add mirrored wound",
        "markGeometryDirty",
    ):
        assert token in PAGE
    for mode in ('data-mode="clot"', 'data-mode="flow"', 'data-mode="retrain"'):
        assert mode in PAGE, f"customer mode {mode} is gone from the UI"


def test_customer_web_page_geometry_and_clot_display():
    import re

    from src.tools.customer_predict_web import PAGE

    # Parametric is the default source and the Inbox picker is gone from the UI.
    options = re.search(r'<select id="source"[^>]*>(.*?)</select>', PAGE, re.S).group(1)
    assert re.findall(r'value="(\w+)"', options) == ["parametric", "upload"]
    assert 'value="parametric" selected' in options
    assert "Inbox geometry" not in PAGE
    # Uploads need no sidecar: the boundary is inferred, or picked on the preview.
    assert "sidecar" not in PAGE.lower()
    for token in ('id="mesh-unit"', 'id="pick-ends"', 'id="swap-ends"', "upload_id"):
        assert token in PAGE
    # Clot is one uniform class: no off-wall/on-wall split and no colour ramp.
    assert "off-wall clot" not in PAGE and "--clot-off" not in PAGE and "lerpRgb" not in PAGE
    for token in ('id="source-info"', "bindHover(", 'id="run-progress"', "progressUpdate("):
        assert token in PAGE


def test_progress_stage_follows_pipeline_messages():
    from src.tools.customer_predict_web import _progress_stage

    assert _progress_stage("Building parametric geometry…") == 0
    assert _progress_stage("[i] Solving local FEM flow at t=0...") == 1
    assert _progress_stage("[i] Rolling out clot_ml_0...") == 2
    assert _progress_stage("[OK] clot_ml_0 rollout done (60 steps)") == 3
    assert _progress_stage("[WARN] shear rate unavailable: x") is None


def test_local_launcher_runs_the_release_ui():
    """The local Predict launcher and both release zips must start the same app module."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    module = "src.tools.customer_predict_web"
    launcher = (root / "scripts" / "_launcher_common.ps1").read_text(encoding="utf-8")
    body = launcher.split("function Invoke-GoCustomerPredict {", 1)[1].split("\n}", 1)[0]
    assert "Invoke-GoCustomerPredictWeb" in body
    assert f'-Module "{module}"' in launcher
    assert f"-m {module}" in (root / "scripts" / "build_customer_bundle.ps1").read_text(encoding="utf-8")
    assert f"-m {module}" in (root / "scripts" / "bundle" / "run_unix.sh").read_text(encoding="utf-8")
    assert not (root / "src" / "tools" / "customer_predict_app.py").exists()

    # The artifact binding, asserted where it actually LIVES.  This used to look for the
    # literal "clot_ml_0" inside PAGE, where it appeared only in a topbar caption -- so the
    # 2026-09-04 rebrand to "ClotML" broke it, while a genuine repointing of the UI to the
    # wrong model would have passed. Check the constant the pipeline resolves instead.
    from src.clot_ml.v0 import resolve_clot_ml_name
    from src.inference.customer_pipeline import _LOCKED_CUSTOMER_CLOT_MODEL

    assert _LOCKED_CUSTOMER_CLOT_MODEL == "clot_ml_0"
    assert resolve_clot_ml_name(_LOCKED_CUSTOMER_CLOT_MODEL)


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

    straight = _parametric_params({"width": 0.010, "bend": 0.0, "amp": 0.0, "pathology": "none"})
    curved = _parametric_params({"width": 0.018, "bend": 70.0, "amp": 0.0, "pathology": "none"})
    s_curve = _parametric_params({"width": 0.015, "bend": 0.0, "amp": 0.004, "pathology": "none"})

    assert straight["curve_type"] == "straight"
    assert curved["curve_type"] == "arc"
    assert s_curve["curve_type"] == "s_curve"
    assert straight["width"] == 0.010
    assert curved["width"] == 0.018
    assert not any(straight["offsets"])
    assert not any(straight["noise_top"])
    assert not any(straight["noise_bot"])
    assert not any(straight["tortuosity"])


def test_server_refuses_cross_site_and_rebound_requests():
    """A page on another site, or a rebound hostname, must not reach the job API."""
    import http.client
    import threading

    from src.tools.customer_predict_web import Handler, Server, allowed_hosts

    server = Server(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    server.allowed_hosts = allowed_hosts("127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def status(method: str, path: str, headers: dict[str, str], body: str | None = None) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{port}", **headers})
        code = conn.getresponse().status
        conn.close()
        return code

    try:
        assert status("GET", "/", {}) == 200
        assert status("GET", "/", {"Host": f"localhost:{port}"}) == 200
        assert status("GET", "/", {"Host": f"evil.example:{port}"}) == 403
        # cross-site "simple" POST: text/plain, needs no CORS preflight
        assert status("POST", "/api/run", {"Content-Type": "text/plain"}, "{}") == 415
        assert status("POST", "/api/run", {"Content-Type": "application/json",
                                           "Origin": "http://evil.example"}, "{}") == 403
        assert status("POST", "/api/open-folder", {"Content-Type": "text/plain"}, "{}") == 415
        # same-origin JSON passes the guard; no worker is running, so it stops at 503
        assert status("POST", "/api/run", {"Content-Type": "application/json",
                                           "Origin": f"http://127.0.0.1:{port}"}, "{}") == 503
    finally:
        server.shutdown()
        server.server_close()


def test_numeric_request_fields_are_bounded_and_finite():
    """A raw API call bypasses the sliders; the server must clamp and reject non-finite values."""
    import json
    import math

    import pytest

    from src.tools.customer_predict_web import (
        _parametric_params, _reject_json_constant, _request_settings,
    )

    wide = _parametric_params({"width": 5.0, "bend": 900.0, "amp": -1.0, "pathology": "none"})
    assert wide["width"] == 0.02
    assert wide["amplitude"] == 0.0
    assert wide["curve_type"] == "arc"
    _, hours, t_final_s, n_steps, _ = _request_settings({"hours": 1e9})
    # The timeline is the model's own 150 s pack axis at the capped horizon, not a knob.
    assert t_final_s <= 30000.0 and n_steps == 201 and math.isfinite(hours)

    for bad in ({"hours": float("nan")}, {"re": float("inf")}, {"hours": "x"}):
        with pytest.raises(ValueError):
            _request_settings(bad)
    with pytest.raises(ValueError):
        _parametric_params({"width": float("nan")})
    with pytest.raises(ValueError):
        json.loads('{"hours": NaN}', parse_constant=_reject_json_constant)


def test_finished_jobs_are_pruned_without_evicting_recent_csv_jobs(tmp_path, monkeypatch):
    import src.tools.customer_predict_web as web

    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "MAX_FINISHED_CSV_JOBS", 2)
    monkeypatch.setattr(web, "MAX_FINISHED_OTHER_JOBS", 3)
    monkeypatch.setattr(web, "JOBS", {})
    (tmp_path / "outputs" / "customer_predict").mkdir(parents=True)

    # Real job ids are uuid4 and `_metrics_csv_path` now insists on that, so the fixture uses
    # UUIDs whose last field counts, keeping the assertions readable.
    csv_ids = [f"00000000-0000-4000-8000-00000000000{i}" for i in range(4)]
    preview_ids = [f"00000000-0000-4000-8000-0000000001{i:02d}" for i in range(10)]
    running_id = "00000000-0000-4000-8000-0000000000ff"

    for i, job_id in enumerate(csv_ids):
        web._metrics_csv_path(job_id).write_text("t\n")
        web._set_job(job_id, status="done", finished_at=float(i), result={"csv_url": "x"})
    web._set_job(running_id, status="running")
    for i, job_id in enumerate(preview_ids):
        web._set_job(job_id, status="done", finished_at=10.0 + i, result={"kind": "preview"})

    assert {j for j in web.JOBS if j in csv_ids} == set(csv_ids[2:])
    assert {j for j in web.JOBS if j in preview_ids} == set(preview_ids[7:])
    assert running_id in web.JOBS
    assert not web._metrics_csv_path(csv_ids[0]).exists()
    assert web._metrics_csv_path(csv_ids[3]).exists()


def test_controls_stay_inside_the_training_envelope():
    """Every geometry control is bounded by what clot_ml_0 was trained on, in the page and the server."""
    import re

    import numpy as np

    from src.config import VesselConfig
    from src.data_gen.lib.customer_geometry_import import (
        ANEURYSM_DEPTH_GAIN, STENOSIS_DEPTH_GAIN,
        TRAINED_ANEURYSM_FWHM_FRAC, TRAINED_ANEURYSM_WIDENING, TRAINED_MAX_S_AMPLITUDE_M,
        TRAINED_STENOSIS_FWHM_FRAC, TRAINED_STENOSIS_NARROWING, TRAINED_WOUND_HORIZON_S,
        _resample_polyline,
    )
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_params
    from src.tools.customer_predict_web import (
        PAGE, PARAM_WIDTH_DEFAULT_M, PARAM_WIDTH_M, WOUND_POSITION_PCT, WOUND_WIDTH_PCT,
        _parametric_params, _request_settings,
    )

    def slider(el_id):
        tag = re.search(r'<input id="' + el_id + r'"[^>]*>', PAGE).group(0)
        return tuple(float(re.search(k + r'="([^"]+)"', tag).group(1)) for k in ("min", "max", "value"))

    assert slider("width") == (*PARAM_WIDTH_M, PARAM_WIDTH_DEFAULT_M)
    assert slider("amp")[1] == TRAINED_MAX_S_AMPLITUDE_M
    assert slider("wound-position")[:2] == WOUND_POSITION_PCT
    assert slider("wound-width")[:2] == WOUND_WIDTH_PCT
    assert 'id="p-location"' not in PAGE and 'id="p-sharpness"' not in PAGE

    assert slider("p-strength") == (0.0, 100 * TRAINED_STENOSIS_NARROWING, 100 * TRAINED_STENOSIS_NARROWING)

    # The default strength builds the trained shape: depth, mid-length position and taper width.
    # These are the control-point walls, which deliberately overshoot by 1/gain: gmsh's B-spline
    # does not pass through its control points, so the MESHED wall lands on the trained depth
    # (`test_parametric_mesh_matches_the_trained_throat_and_spacing`).
    cfg = VesselConfig(phase="kinematics").__dict__
    for kind, depth, fwhm in (("stenosis", -TRAINED_STENOSIS_NARROWING / STENOSIS_DEPTH_GAIN, TRAINED_STENOSIS_FWHM_FRAC),
                              ("aneurysm", TRAINED_ANEURYSM_WIDENING / ANEURYSM_DEPTH_GAIN, TRAINED_ANEURYSM_FWHM_FRAC)):
        g = compute_geometry_from_params(_parametric_params({"width": 0.015, "bend": 0, "pathology": kind}), cfg)
        top = _resample_polyline(np.asarray(g.top_coords), 400)
        bot = _resample_polyline(np.asarray(g.bot_coords), 400)
        dev = np.linalg.norm(top - bot, axis=1)
        dev = dev / dev[0] - 1.0
        k = int(np.argmax(np.abs(dev)))
        assert abs(dev[k] - depth) < 0.02, (kind, dev[k])
        assert abs(k / 399 - 0.51) < 0.02
        assert abs(float((np.abs(dev) >= 0.5 * abs(dev[k])).mean()) - fwhm) < 0.02, kind

    # Strength is the percent width change at the centre: 50% -> 0.5x stenosis, 1.5x aneurysm,
    # on the MESHED wall, so the control polygon carries the 1/gain overshoot here too.
    # Requests past the trained maximum clamp to it.
    for kind, pct, ratio in (("stenosis", 50, 1 - 0.5 / STENOSIS_DEPTH_GAIN),
                             ("aneurysm", 50, 1 + 0.5 / ANEURYSM_DEPTH_GAIN),
                             ("stenosis", 95, 1 - TRAINED_STENOSIS_NARROWING / STENOSIS_DEPTH_GAIN)):
        g = compute_geometry_from_params(
            _parametric_params({"width": 0.015, "bend": 0, "pathology": kind, "pathology_strength": pct}), cfg)
        top = _resample_polyline(np.asarray(g.top_coords), 400)
        bot = _resample_polyline(np.asarray(g.bot_coords), 400)
        w = np.linalg.norm(top - bot, axis=1) / np.linalg.norm(top[0] - bot[0])
        extreme = w.min() if kind == "stenosis" else w.max()
        assert abs(extreme - ratio) < 0.01, (kind, pct, extreme)

    # A wounded vessel stops at the longest wound horizon in training.
    _, _, t_wound, _, _ = _request_settings({"hours": 8.0, "wound_enabled": True})
    _, _, t_plain, _, _ = _request_settings({"hours": 8.0})
    assert t_wound == TRAINED_WOUND_HORIZON_S and t_plain == 8.0 * 3600


def test_app_timeline_is_the_trained_grid():
    """The app must query clot_ml_0 on the grid its temporal head was fitted on."""
    from src.clot_ml.locked import trained_query_grid, trained_time_axis

    full = trained_time_axis(30000.0)
    assert len(full) == 201 and full[1] == 150.0 and full[-1] == 30000.0
    early = trained_time_axis(4193.8)                  # comsol003: truncated final step
    assert len(early) == 29 and abs(early[-1] - 4193.8) < 1e-6
    assert trained_query_grid({"n_times": 11}, 201) == list(range(0, 201, 20))


def test_parametric_mesh_matches_the_trained_throat_and_spacing():
    """The graph the model sees -- not the control polygon -- must match the COMSOL anchors.

    Targets are comsol041/042/044 (throat 0.227-0.232 of inlet width) and comsol040/047 (peak
    1.987x), and the anchors' uniform ~0.35 mm open-lumen node spacing.  A d_bar-relative mesh
    was 20% coarser on an aneurysm and 2x finer at a stenosis throat; a 2x draft mesh inflated
    stenosis wall clot from 0.23 to 0.36.
    """
    import pytest

    pytest.importorskip("gmsh")
    from scipy.spatial import cKDTree

    from src.tools.customer_predict_web import _parametric_data

    def built(pathology):
        req = {"width": 0.01594, "bend": 0, "amp": 0, "pathology": pathology}
        data = _parametric_data(req, re_target=450.0, t_final_s=3600.0, n_steps=25)
        pos = data.x[:, :2].numpy().astype(float) * float(data.d_bar.reshape(-1)[0])
        wall = data.mask_wall.reshape(-1).bool().numpy()
        yc = 0.5 * (pos[wall, 1].min() + pos[wall, 1].max())
        top = pos[wall & (pos[:, 1] > yc)]
        bot = pos[wall & (pos[:, 1] < yc)]
        top, bot = top[np.argsort(top[:, 0])], bot[np.argsort(bot[:, 0])]
        xs = np.linspace(max(top[0, 0], bot[0, 0]), min(top[-1, 0], bot[-1, 0]), 4000)
        w = np.interp(xs, top[:, 0], top[:, 1]) - np.interp(xs, bot[:, 0], bot[:, 1])
        nn = cKDTree(pos).query(pos, k=2)[0][:, 1]
        frac = (pos[:, 0] - xs[0]) / (xs[-1] - xs[0])
        open_spacing = float(np.median(nn[(frac < 0.25) | (frac > 0.8)]))
        return w / np.median(w[:200]), open_spacing

    sten, sten_h = built("stenosis")
    aneu, aneu_h = built("aneurysm")
    assert abs(sten.min() - 0.229) < 0.01
    assert abs(aneu.max() - 1.987) < 0.02
    for h in (sten_h, aneu_h):
        assert abs(h - 0.35e-3) < 0.02e-3


def test_metrics_csv_path_refuses_a_non_uuid_job_id(tmp_path, monkeypatch):
    """The CSV route's filename comes off the request line, so it may only be a UUID.

    CodeQL reads `/api/job/<id>/csv` as a path expression fed by user input (py/path-injection).
    The route's regex and the `JOBS` membership check already make traversal unreachable; this
    pins the third guard, which does not depend on either of them staying that way.
    """
    import uuid

    import pytest

    import src.tools.customer_predict_web as web

    monkeypatch.setattr(web, "ROOT", tmp_path)
    good = str(uuid.uuid4())
    assert web._metrics_csv_path(good).name == f"web_scientific_metrics_{good}.csv"

    for bad in ("../../etc/passwd", "a/b", "..", "csv-0", "", "*"):
        with pytest.raises(ValueError):
            web._metrics_csv_path(bad)
