"""An artifact must record which t=0 flow field it was measured under.

`pack.u0_pred` is one slot every reconstructed flow source writes, so its contents are a MODE
the packs are in, not a property of them.  Two legitimate consumers need different fields in
it at the same time -- the split family wants the full-pool checkpoint, the leak-free flow
diagnostics want each vessel's held-out cross-fit fold -- so the slot gets rewritten whenever
the other measurement is wanted.

On 2026-09-06 a full-pool precache silently invalidated `outputs/runs/flow_diagnostics.json`,
which was a bare list of per-vessel rows with no checkpoint field anywhere.  The file still
looked current, and nothing could have told you otherwise.

These pin the two properties that prevent a repeat: the mode is derivable from what the packs
carry, and old artifacts written before stamping still load.
"""

from __future__ import annotations

import json

import pytest

from src.core_physics.flow_provenance import (
    ABSENT, CROSS_FIT, FULL_POOL, MIXED, SOLVED,
    classify, is_fold_run, pack_stamp, read_diagnostics,
)


class _Pack:
    """Minimal stand-in: `pack_stamp` reads one attribute and nothing else."""

    def __init__(self, path=None):
        if path is not None:
            self.u0_pred_provenance = json.dumps({"checkpoint": {"path": path}})


def test_fold_runs_are_distinguished_from_the_full_pool_arm():
    assert is_fold_run("E8_xf5_f0")
    assert is_fold_run("E8_xf5_f12")
    assert not is_fold_run("E5_band_gateup")
    assert not is_fold_run(SOLVED)


def test_pack_stamp_keeps_the_run_directory_not_just_the_basename():
    """Every arm writes the same filename, so the basename cannot identify the arm."""
    s = pack_stamp(_Pack(r"outputs\runs\E8_xf5_f3\kinematics_best_calibrated.pth"))
    assert s == {"run": "E8_xf5_f3", "checkpoint": "kinematics_best_calibrated.pth"}
    other = pack_stamp(_Pack("outputs/runs/E5_band_gateup/kinematics_best_calibrated.pth"))
    assert other["checkpoint"] == s["checkpoint"], "the ambiguity this exists for"
    assert other["run"] != s["run"], "the run directory is what disambiguates them"


def test_absent_and_unreadable_stamps_do_not_raise():
    assert pack_stamp(_Pack())["run"] == ABSENT
    bad = _Pack()
    bad.u0_pred_provenance = "{not json"
    assert pack_stamp(bad)["run"] == "(unreadable)"


@pytest.mark.parametrize("runs,expected", [
    (["E8_xf5_f0", "E8_xf5_f1", "E5_band_gateup"], CROSS_FIT),
    (["E5_band_gateup"] * 5, FULL_POOL),
    ([SOLVED] * 5, SOLVED),
    (["E5_band_gateup", "E7_other"], MIXED),
    ([ABSENT, ABSENT], ABSENT),
    ([], ABSENT),
])
def test_classify_names_the_cohort_mode(runs, expected):
    assert classify(runs) == expected


def test_a_solved_cohort_is_not_called_a_pool():
    """`fem` solves its field; naming that a training pool would credit absent weights."""
    assert classify([SOLVED] * 37) == SOLVED
    assert classify([SOLVED] * 37) != FULL_POOL


def test_read_diagnostics_accepts_both_payload_shapes(tmp_path):
    rows = [{"stem": "comsol020", "rel_l2": 0.01}]

    old = tmp_path / "old.json"
    old.write_text(json.dumps(rows), encoding="utf-8")
    got, prov = read_diagnostics(old)
    assert got == rows and prov["mode"] == ABSENT, "pre-stamping files must still load"

    new = tmp_path / "new.json"
    new.write_text(json.dumps(
        {"flow": "pred", "provenance": {"mode": CROSS_FIT, "runs": {"E8_xf5_f0": 1}},
         "vessels": rows}), encoding="utf-8")
    got, prov = read_diagnostics(new)
    assert got == rows and prov["mode"] == CROSS_FIT


def test_metric_identity_names_what_is_actually_computed():
    """The figures resolve BATC_0's settings; the report had been calling them BATC."""
    from scripts.publication.utils import metric_identity, metric_label

    mid = metric_identity()
    assert mid["metric"] in ("BATC", "BATC_0", "custom")
    s = mid["settings"]
    if mid["metric"] == "BATC_0":
        assert (s["relax_hops"], s["f_beta"], s["iou_weight"]) == (2, 0.5, 0.5)
    elif mid["metric"] == "BATC":
        assert (s["relax_hops"], s["f_beta"], s["iou_weight"]) == (4, 1.0, 0.2)
    assert mid["metric"] in metric_label()


def test_metric_identity_refuses_to_keep_a_published_name_for_other_settings():
    """An unrecognised combination must degrade to `custom`, not to the nearest label."""
    import scripts.publication.utils as u

    assert u._PUBLISHED_SETTINGS.get((3, 0.7, 0.9)) is None
    assert (2, 0.5, 0.5) in u._PUBLISHED_SETTINGS
    assert (4, 1.0, 0.2) in u._PUBLISHED_SETTINGS
