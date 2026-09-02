"""The pre-flight gate check: verdicts, and the thresholds it is calibrated against.

The check exists to refuse a *vacuous* prediction -- an empty wall gate zeroes the readout's
seed and every downstream channel with it -- so the FAIL branch is the one that must never
regress.  These tests exercise the decision logic directly, without solving any flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.clot_ml.preflight import (
    FAIL, FIRE_FRAC_MAX, FIRE_FRAC_MIN, PASS, WARN, PreflightResult, preflight_check,
)


class _FakeData:
    """Minimal stand-in exposing only what `preflight_check` reads via `wall_gate_firing`."""

    def __init__(self, gate: np.ndarray, wall: np.ndarray):
        self._gate, self._wall = gate, wall

    @property
    def mask_wall(self):
        import torch

        return torch.tensor(self._wall)


@pytest.fixture
def patched_gate(monkeypatch):
    """Drive `preflight_check` from a chosen (gate, wall) pair."""

    def _apply(gate, wall):
        import src.clot_ml.preflight as pf

        monkeypatch.setattr(pf, "wall_gate_firing",
                            lambda data, flow, bio_cfg=None: (np.asarray(gate, float),
                                                              np.asarray(wall, bool)))
        return preflight_check(object(), "pred")

    return _apply


def test_empty_wall_gate_fails(patched_gate):
    """The failure the whole module exists for: gate fires on no wall node."""
    res = patched_gate(gate=np.zeros(10), wall=np.ones(10, bool))
    assert res.verdict == FAIL
    assert res.ok is False
    assert res.n_fire == 0
    assert any("no node" in r.lower() or "empty" in r.lower() for r in res.reasons)


def test_gate_firing_off_the_wall_still_fails(patched_gate):
    """Firing in the lumen does not save it -- the seed is `(gate > 0) & wall`."""
    wall = np.zeros(10, bool)
    wall[:3] = True
    gate = np.zeros(10)
    gate[5:] = 1.0          # fires, but nowhere on the wall
    res = patched_gate(gate=gate, wall=wall)
    assert res.verdict == FAIL
    assert res.n_fire == 0


def test_healthy_firing_fraction_passes(patched_gate):
    """A firing fraction inside the reference envelope is a clean pass."""
    n = 200
    wall = np.ones(n, bool)
    gate = np.zeros(n)
    gate[: int(0.13 * n)] = 1.0      # ~0.13, the cohort median under GT and FEM
    res = patched_gate(gate=gate, wall=wall)
    assert res.verdict == PASS
    assert res.ok is True
    assert res.reasons == []
    assert FIRE_FRAC_MIN < res.fire_frac < FIRE_FRAC_MAX


@pytest.mark.parametrize("frac,word", [(0.01, "under-firing"), (0.95, "over-firing")])
def test_out_of_envelope_warns_but_does_not_refuse(patched_gate, frac, word):
    """Outside the envelope is a vessel to inspect, not a vacuous prediction."""
    n = 400
    wall = np.ones(n, bool)
    gate = np.zeros(n)
    gate[: max(int(frac * n), 1)] = 1.0
    res = patched_gate(gate=gate, wall=wall)
    assert res.verdict == WARN
    assert res.ok is True                      # WARN must never block a rollout
    assert any(word in r for r in res.reasons)


def test_no_wall_nodes_fails(patched_gate):
    """A pack with no wall selection cannot be scored at all."""
    res = patched_gate(gate=np.ones(5), wall=np.zeros(5, bool))
    assert res.verdict == FAIL
    assert res.n_wall == 0


def test_thresholds_bracket_the_measured_cohort():
    """The envelope must sit outside every GT/FEM vessel measured, or it fires on good flow.

    Cohort minima/maxima (33 vessels, docs/PUBLICATION_NOTES.md s2):
        GT   0.0465 .. 0.4286
        FEM  0.0428 .. 0.4416
    """
    assert FIRE_FRAC_MIN < 0.0428, "would false-alarm on the lowest-firing FEM vessel"
    assert FIRE_FRAC_MAX > 0.4416, "would false-alarm on the highest-firing FEM vessel"


def test_result_is_reportable():
    """`str()` is what a deploy log shows; it must name the counts and the reasons."""
    res = PreflightResult(WARN, 100, 2, 0.02, ["under-firing"])
    text = str(res)
    assert "WARN" in text and "2/100" in text and "under-firing" in text
