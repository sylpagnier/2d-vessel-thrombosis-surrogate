"""A cohort archive must hold every file its manifest describes.

`archive_model_cohort.py` copied checkpoints into a flat `checkpoints/` directory keyed by
BASENAME.  Every training run writes its weights as `kinematics_best_calibrated.pth`, so the
six checkpoints behind the split cohort -- `E5_band_gateup` plus the five `E8_xf5_f*` folds --
all resolved to one destination and overwrote each other.  Each was hashed immediately after
its own copy, so the manifest recorded six distinct digests for a path that ended up holding
one file: five of the six entries described bytes the archive did not contain.

That is worse than not archiving the checkpoints, because the manifest reads as proof.  It was
found on 2026-09-06 only because a restore was verified against it and five entries
"mismatched"; the originals were still under `outputs/runs/`, so nothing was actually lost.

Two guards, tested here: destinations for name-ambiguous sources are scoped by their parent
directory, and a duplicate path anywhere in the record list is refused outright.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.archive_model_cohort import _assert_unique, _copy_in, _run_scoped


def test_same_basename_from_different_runs_gets_distinct_destinations():
    a = Path("outputs/runs/E5_band_gateup/kinematics_best_calibrated.pth")
    b = Path("outputs/runs/E8_xf5_f0/kinematics_best_calibrated.pth")
    ra, rb = _run_scoped(a, "checkpoints"), _run_scoped(b, "checkpoints")
    assert ra != rb, "checkpoints from different runs must not share a destination"
    assert ra == "checkpoints/E5_band_gateup/kinematics_best_calibrated.pth"
    assert rb == "checkpoints/E8_xf5_f0/kinematics_best_calibrated.pth"


def test_duplicate_paths_are_refused():
    dup = [
        {"path": "checkpoints/kinematics_best_calibrated.pth", "bytes": 1, "sha256": "aa"},
        {"path": "checkpoints/kinematics_best_calibrated.pth", "bytes": 2, "sha256": "bb"},
        {"path": "scores/dc_xf5.npz", "bytes": 3, "sha256": "cc"},
    ]
    with pytest.raises(SystemExit) as exc:
        _assert_unique(dup)
    assert "kinematics_best_calibrated.pth" in str(exc.value)
    assert "dc_xf5" not in str(exc.value), "only the clashing path should be reported"


def test_unique_paths_pass():
    _assert_unique([
        {"path": "checkpoints/E5_band_gateup/w.pth", "bytes": 1, "sha256": "aa"},
        {"path": "checkpoints/E8_xf5_f0/w.pth", "bytes": 2, "sha256": "bb"},
    ])


def test_archive_round_trip_keeps_every_file(tmp_path):
    """The end-to-end property: N distinct sources -> N distinct files, hashes that verify."""
    import hashlib

    root = tmp_path / "snapshot"
    files: list[dict] = []
    for i, run in enumerate(("E5_band_gateup", "E8_xf5_f0", "E8_xf5_f1")):
        src_dir = tmp_path / "runs" / run
        src_dir.mkdir(parents=True)
        src = src_dir / "kinematics_best_calibrated.pth"
        src.write_bytes(f"weights-{i}".encode())
        files += _copy_in(src, root, _run_scoped(src, "checkpoints"))

    _assert_unique(files)
    assert len(files) == 3
    for rec in files:
        on_disk = root / rec["path"]
        assert on_disk.is_file(), f"manifest names {rec['path']} but the archive lacks it"
        assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == rec["sha256"]
    assert len({r["sha256"] for r in files}) == 3, "each run's distinct bytes must survive"
