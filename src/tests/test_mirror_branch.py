"""The mirror-branch evaluation rule (`src/clot_ml/mirror_branch.py`) and its guard."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.clot_ml import mirror_branch as mb

ROOT = Path(__file__).resolve().parents[2]


def _symmetric_cloud(seed=0):
    """A jittered point cloud reflected exactly across y = 0, rotated off the axes."""
    rng = np.random.default_rng(seed)
    half = np.column_stack([rng.uniform(0, 10, 400), rng.uniform(0.2, 1.0, 400)])
    pts = np.vstack([half, half * [1, -1]])
    th = 0.4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return pts @ R.T


def test_symmetric_mesh_maps_onto_itself_as_an_involution():
    m = mb.mirror_permutation(_symmetric_cloud())
    assert m.symmetric
    assert np.array_equal(m.perm[m.perm], np.arange(len(m.perm)))


def test_asymmetric_mesh_is_not_symmetric():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0, 10, 800), rng.uniform(0, 1, 800) ** 3])
    assert not mb.mirror_permutation(pts).symmetric


def test_branch_is_decided_from_flow_not_clot():
    pts = _symmetric_cloud()
    m = mb.mirror_permutation(pts)
    rng = np.random.default_rng(2)
    comsol = rng.normal(size=(len(pts), 2))
    on_mirror = mb.reflect_vectors(comsol, m)
    assert mb.flow_branch(on_mirror, comsol, m)["mirror_branch"]
    assert not mb.flow_branch(comsol, comsol, m)["mirror_branch"]


def test_eval_gt_is_identity_for_unregistered_vessels():
    gt = np.array([0.0, 1.0, 0.7, 0.2])
    out = mb.eval_gt("not_a_registered_vessel", gt, np.zeros((4, 2)))
    assert out.tolist() == [False, True, True, False]


def test_registered_stem_refuses_positions_that_are_not_symmetric(monkeypatch):
    monkeypatch.setattr(mb, "mirror_branch_stems", lambda: ("fake_vessel",))
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0, 10, 500), rng.uniform(0, 1, 500) ** 3])
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        mb.eval_gt("fake_vessel", np.ones(500, bool), pts)


def test_registered_stem_is_mirrored_for_every_frame(monkeypatch):
    monkeypatch.setattr(mb, "mirror_branch_stems", lambda: ("fake_vessel",))
    mb._PERMS.clear()
    pts = _symmetric_cloud()
    perm = mb.mirror_permutation(pts).perm
    gts = np.random.default_rng(4).random((3, len(pts))) > 0.5
    assert np.array_equal(mb.eval_gt("fake_vessel", gts, pts), gts[:, perm])
    mb._PERMS.clear()


def test_labels_and_positions_in_different_node_orders_are_refused(monkeypatch):
    monkeypatch.setattr(mb, "mirror_branch_stems", lambda: ("fake_vessel",))
    mb._PERMS.clear()
    pts = _symmetric_cloud()
    with pytest.raises(ValueError, match="same node order"):
        mb.eval_gt("fake_vessel", np.ones(len(pts) - 1, bool), pts)
    mb._PERMS.clear()


def test_comsol_labels_switch_never_mirrors(monkeypatch):
    monkeypatch.setattr(mb, "mirror_branch_stems", lambda: ("fake_vessel",))
    pts = _symmetric_cloud()
    gt = np.random.default_rng(5).random(len(pts)) > 0.5
    assert np.array_equal(mb.eval_gt("fake_vessel", gt, pts, mirror=False), gt)


def test_registry_obeys_its_own_rule():
    """Every flagged vessel is symmetric and its flow is on the mirror side by the margin."""
    rows = json.loads(mb.REGISTRY.read_text(encoding="utf-8"))["vessels"]
    for stem, r in rows.items():
        expect = r["symmetric"] and r["rel_l2_mirrored"] * mb.BRANCH_MARGIN < r["rel_l2_direct"]
        assert r["mirror_branch"] == expect, stem


#: Files that still read a raw label (`["y"] ... > 0.5` or `gt_clot_phi_at_time(`) outside the
#: evaluation-label helper.  Each is a TRAINING target, a label builder, or a diagnostic outside
#: the published results.  This set may only SHRINK: an evaluation path that scores against
#: ground truth must use `mirror_branch.eval_gt` or `evaluate.gt_series`.
RAW_LABEL_ALLOWED = frozenset({
    "scripts/build_clot_ml_cache.py", "scripts/diag_batc_sweep_wall.py",
    "scripts/diag_coupled_flow_delta.py", "scripts/diag_fp_geography.py",
    "scripts/diag_offwall_ranking.py", "scripts/diag_offwall_readout_sensitivity.py",
    "scripts/diag_offwall_score_geography.py", "scripts/diag_sealed_offwall_gap.py",
    "scripts/diagnose_crack_001_root.py", "scripts/diagnose_lumen_001_vs_007.py",
    "scripts/eda_clot_physics.py", "scripts/eval_by_class.py",
    "scripts/eval_expected_score_readout.py", "scripts/eval_wound_ab_pair.py",
    "scripts/gen_clot_ml_0_oof_viz_data.py", "scripts/gen_offwall_temporal_data.py",
    "scripts/predict_wall_clot.py", "scripts/probe_pocket_ranking.py",
    "scripts/promote_clot_gnn_v4_temporal.py", "scripts/publication/generate_wound_ab_data.py",
    "src/archive/differentiable_wall_model/evaluation.py",
    "src/archive/differentiable_wall_model/improved_heads.py",
    "src/clot_ml/features.py", "src/clot_ml/protocol.py",
    "src/core_physics/physics_wall_model.py", "src/core_physics/t0_mu_physics.py",
    "src/tools/diagnostics/field_calibration.py", "src/tools/diagnostics/wound_composition.py",
    "src/tools/diagnostics/wound_p003_causes.py",
    # these contain a raw read on purpose, next to the evaluation label:
    "src/clot_ml/evaluate.py",                           # gt_series wraps it in eval_gt
    "scripts/eval_strict_temporal.py",                   # `gt` trains the temporal head
    "scripts/publication/generate_operating_point_data.py",  # docstring mention only
    # a LEAKAGE gate, not a model score: its oracle feature is built from COMSOL's own clot, so
    # the label it is checked against must stay COMSOL's too
    "scripts/publication/diag_coupling_leakage.py",
})
RAW_LABEL = re.compile(r'\["y"\][^\n]{0,30}>\s*0\.5|gt_clot_phi_at_time\(')


def test_no_new_evaluation_path_reads_a_raw_label():
    found = set()
    for base in ("src", "scripts"):
        for p in (ROOT / base).rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            if "/tests/" in rel:
                continue
            if RAW_LABEL.search(p.read_text(encoding="utf-8", errors="ignore")):
                found.add(rel)
    new = sorted(found - RAW_LABEL_ALLOWED)
    assert not new, ("these files read a raw ground-truth label; score against "
                     "`mirror_branch.eval_gt` (or `evaluate.gt_series`) instead: %s" % new)
    gone = sorted(RAW_LABEL_ALLOWED - found)
    assert not gone, "no longer read raw labels -- remove from RAW_LABEL_ALLOWED: %s" % gone
