"""Pins for the cache feature fingerprint (:mod:`src.clot_ml.feature_fingerprint`).

The thing being protected: a stale feature cache does not announce itself.  The arrays load,
the shapes are right, and every downstream number is quietly computed on superseded features.
`load_cache` used to detect exactly one historical change (the missing `solid` key); this
detects any change to the feature builders, structurally.
"""
from __future__ import annotations

import numpy as np

from src.clot_ml.feature_fingerprint import (
    FEATURE_SOURCES, FINGERPRINT_KEY, check, feature_fingerprint, stamp,
)


def test_fingerprint_is_stable_and_short():
    a, b = feature_fingerprint(), feature_fingerprint()
    assert a == b, "the hash must not depend on call order or time"
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_fingerprint_changes_when_a_feature_source_changes():
    """The whole point: edit a builder, every prior cache becomes detectably stale."""
    base = feature_fingerprint()
    moved = feature_fingerprint(FEATURE_SOURCES + ("src/clot_ml/severity_metric.py",))
    assert moved != base


def test_fingerprint_is_newline_normalised():
    """A CRLF checkout on Windows and an LF one on CI must agree, or every cache reads as
    stale on the other platform and the warning gets ignored into uselessness."""
    import hashlib
    from pathlib import Path

    rel = FEATURE_SOURCES[0]
    p = Path(__file__).resolve().parents[2] / rel
    txt = p.read_text(encoding="utf-8")
    h_lf = hashlib.sha256(txt.replace("\r\n", "\n").encode()).hexdigest()
    h_crlf = hashlib.sha256(txt.replace("\r\n", "\n").replace("\n", "\r\n")
                            .replace("\r\n", "\n").encode()).hexdigest()
    assert h_lf == h_crlf


def test_stamp_then_check_round_trips():
    payload = stamp({"X": np.zeros(3)})
    assert FINGERPRINT_KEY in payload
    assert check({"vessel": payload}) is None


def test_check_distinguishes_absent_from_mismatched():
    """Absent means 'built before this existed, unknown'; mismatched means 'known wrong'.
    Collapsing the two would make the mechanism useless on exactly the caches it protects."""
    absent = check({"v": {}}, "c")
    assert absent is not None and "no feature fingerprint" in absent

    wrong = check({"v": {FINGERPRINT_KEY: np.str_("0" * 16)}}, "c")
    assert wrong is not None and "MISMATCH" in wrong
    assert "Rebuild it" in wrong


def test_check_is_quiet_on_an_empty_cache():
    assert check({}, "c") is None
