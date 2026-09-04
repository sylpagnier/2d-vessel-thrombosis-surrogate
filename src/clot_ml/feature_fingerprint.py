"""A content hash of the feature builders, written into every cache at build time.

WHY.  `data.load_cache` currently detects exactly ONE historical change: it warns when a
cache has no `solid` key, because `build_clot_ml_cache.py` has written one since the
2026-08-22 geometry union.  That is a hand-maintained heuristic for a single past event.  It
cannot see the next change, and a stale cache is the expensive failure here precisely because
it does not announce itself -- the arrays load, the shapes are right, and every downstream
number is quietly computed on superseded features.

This replaces the heuristic with something structural: a SHA-256 over the source of the two
modules that actually define the feature columns.  If either changes, every cache built
before the change is detectably stale, whatever the change was and whether or not anyone
remembered to add a marker for it.

WHAT IS HASHED, and why only these two.  `features.py` builds the v3 sample and
`features_v4.py` appends the v4 transport channels; together they determine the column set
and every value in it.  Hashing more (config, physics) would fire on edits that cannot move a
cached feature, and a fingerprint that cries wolf is one people learn to override.

The fingerprint is ADVISORY at load time, not fatal.  Refusing to load would strand every
cache built before this landed, including the ones the shipped artifact was trained on.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]

#: The modules whose contents define a cached feature vector.
FEATURE_SOURCES: tuple[str, ...] = (
    "src/clot_ml/features.py",
    "src/clot_ml/features_v4.py",
)

#: npz key the fingerprint is stored under.  Absent on any cache built before 2026-09-03.
FINGERPRINT_KEY = "feature_fingerprint"


def feature_fingerprint(sources: tuple[str, ...] = FEATURE_SOURCES) -> str:
    """SHA-256 over the feature-builder sources, newline-normalised.

    Normalising line endings keeps the hash stable across a CRLF checkout on Windows and an
    LF one on CI, which would otherwise report every cache as stale on the other platform.
    """
    h = hashlib.sha256()
    for rel in sources:
        p = _REPO / rel
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        if p.exists():
            h.update(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
        else:  # a missing source is itself a state worth distinguishing
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()[:16]


def stamp(payload: dict) -> dict:
    """Add the fingerprint to a dict destined for ``np.savez_compressed``."""
    payload[FINGERPRINT_KEY] = np.str_(feature_fingerprint())
    return payload


def check(cache: dict, name: str = "cache") -> str | None:
    """Compare a loaded cache against the current sources.  Returns a warning, or ``None``.

    ``cache`` is the ``{stem: {key: array}}`` mapping `load_cache` returns.  A cache with no
    fingerprint predates this mechanism and is reported as unknown rather than stale -- it may
    be fine, and it is exactly the case the `solid`-key heuristic was written for.
    """
    if not cache:
        return None
    want = feature_fingerprint()
    seen: dict[str, int] = {}
    for S in cache.values():
        got = S.get(FINGERPRINT_KEY)
        key = "<none>" if got is None else str(np.asarray(got).item()
                                               if np.ndim(got) == 0 else got)
        seen[key] = seen.get(key, 0) + 1
    if list(seen) == [want]:
        return None
    if list(seen) == ["<none>"]:
        return (f"{name}: no feature fingerprint (built before 2026-09-03). Cannot verify it "
                f"matches the current {', '.join(FEATURE_SOURCES)}; rebuild to make this "
                f"checkable.")
    parts = ", ".join(f"{k}x{v}" for k, v in sorted(seen.items()))
    return (f"{name}: feature fingerprint MISMATCH -- current sources hash {want}, cache "
            f"holds {parts}. The feature builders changed since this cache was written, so "
            f"every number computed from it is on superseded features. Rebuild it.")
