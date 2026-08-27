"""The adaptive-cut clamp must be an exact no-op inside its fitted support.

That property is the whole licence for shipping it: it cannot change any result the
19-vessel cohort is able to measure (`scripts/eval_adapt_clamp.py` verifies this on the real
vessels, bit-identical, under two different partitions), so it carries no selection risk.
Outside the support it must hold the statistic at the boundary rather than extrapolate.

These tests use synthetic samples so they run without the cache or any checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import apply_adapt, vessel_stat  # noqa: E402


def _sample(n=40, seed=0):
    """A minimal cache-entry shape: wall/phys_mask/edge_index are all `readout_resid` uses."""
    rng = np.random.default_rng(seed)
    wall = np.zeros(n, bool)
    wall[: n // 2] = True
    return dict(wall=wall, phys_mask=rng.random(n) > 0.5,
                edge_index=np.stack([np.arange(n - 1), np.arange(1, n)]))


def wall_of(S):
    return S["wall"]


TH = (0.86, 0.47, 0.95, 0.83)
B, MED = -0.6, 0.22


def test_clamp_is_exact_noop_inside_support():
    """A vessel whose statistic lies inside [lo, hi] must get a byte-identical mask."""
    S = _sample()
    rng = np.random.default_rng(1)
    for _ in range(25):
        sc = rng.random(len(S["wall"]))
        stat = vessel_stat(S, sc, wall_of(S))
        lo, hi = stat - 0.05, stat + 0.05          # support that brackets this vessel
        plain = apply_adapt(S, sc, "resid", TH, wall_of, B, MED)
        clamped = apply_adapt(S, sc, "resid", TH, wall_of, B, MED, lo, hi)
        assert np.array_equal(plain, clamped)


def test_clamp_binds_only_outside_support():
    """Above `hi` the offset must freeze at the boundary's value, not keep growing."""
    S = _sample(seed=3)
    sc = np.random.default_rng(2).random(len(S["wall"]))
    stat = vessel_stat(S, sc, wall_of(S))
    hi = stat - 0.20                                # force this vessel to be exterior
    lo = hi - 0.10

    at_boundary = apply_adapt(S, sc, "resid", TH, wall_of, B, MED, lo, hi)
    # the same vessel scored as if its statistic WERE the boundary, with no clamp active
    shifted = apply_adapt(S, sc, "resid", tuple(np.clip(np.array(TH) + B * (hi - MED),
                                                        0.02, 0.98)), wall_of, 0.0, 0.0)
    assert np.array_equal(at_boundary, shifted)


def test_unbounded_extrapolation_can_saturate_the_cut():
    """Why the guardrail exists: unclamped, a far-out statistic drives cuts to the floor."""
    S = _sample(seed=5)
    sc = np.full(len(S["wall"]), 0.30)
    sc[:5] = 1.0
    far = apply_adapt(S, sc, "resid", TH, wall_of, B, med=-5.0)   # stat - med is huge
    near = apply_adapt(S, sc, "resid", TH, wall_of, B, MED, -5.0, -4.9)
    assert far.sum() > near.sum(), "unbounded offset should commit strictly more"


def test_default_signature_unchanged():
    """Callers that pass no bounds must get the historical behaviour exactly."""
    S = _sample(seed=7)
    sc = np.random.default_rng(4).random(len(S["wall"]))
    a = apply_adapt(S, sc, "resid", TH, wall_of, B, MED)
    b = apply_adapt(S, sc, "resid", TH, wall_of, B, MED, None, None)
    assert np.array_equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
