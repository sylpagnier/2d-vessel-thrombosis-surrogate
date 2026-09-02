"""Release invariants for the C0-tail ``clot_ml_0`` baseline alias."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


LOCKED = Path("outputs/clot_ml/locked")
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def _alias_manifest_path() -> Path:
    for name in ("clot_ml_0", "clot_ml_v0"):
        path = LOCKED / name / "manifest.json"
        if path.exists():
            return path
    return LOCKED / "clot_ml_0" / "manifest.json"


def test_clot_ml_0_binds_the_reviewed_c0_and_wound_artifacts():
    path = _alias_manifest_path()
    if not path.exists():
        return
    alias = json.loads(path.read_text())
    source = alias["source"]
    base_path = LOCKED / source["base_artifact"] / "manifest.json"
    wound_path = LOCKED / source["wound_artifact"] / "manifest.json"
    base = json.loads(base_path.read_text())
    wound = json.loads(wound_path.read_text())

    assert alias["kind"] == "temporal_v4_wound"
    assert alias["base_model"] == source["base_artifact"] == "clot_gnn_v6"
    assert alias["alias_of"] == source["wound_artifact"] == "clot_gnn_v6w"
    assert sha256(base_path.read_bytes()).hexdigest().upper() == source["base_manifest_sha256"]
    assert base["fingerprint"] == source["base_fingerprint"]
    assert alias["wound"] == wound["wound"]


def test_clot_ml_0_records_generalization_and_deploy_gates():
    path = _alias_manifest_path()
    if not path.exists():
        return
    alias = json.loads(path.read_text())
    val = alias["validation"]
    assert val["training_pool"]["clot_carrying"] == 23
    assert val["training_pool"]["clot_free_false_positive_only"] == 8
    assert set(val["training_pool"]["final_half_excluded"]) == FINAL_HALF
    assert val["strict_cv"]["final"]["wall"] == 0.9203
    assert val["strict_cv"]["final"]["off"] == 0.7078
    assert val["c0_ablation"]["shape_w"] == 0.0
    assert val["flow"]["training_and_strict_cv"] == "GT t=0 flow only"
    assert alias["release_status"]["research_baseline"] == "validated"
    assert alias["release_status"]["cold_deploy"] == "blocked"
