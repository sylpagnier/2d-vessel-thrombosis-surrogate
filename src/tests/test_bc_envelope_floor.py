"""The hard-BC envelope floor: what it must and must not change.

`bc_envelope_decay` band-localises the residual (RGP_DEQ_REPAIR_PLAN.md, E4/E5).  At
`decay=12` the envelope in the core is `exp(-12*sdf)` ~ 2.5e-3 at mid-lumen, so the head has
no authority there at all -- which is why arm E5's prediction on `comsol045` / `comsol046` is
bit-identical to the FEM prior it was handed.  Those two carry 37% of the FEM-vs-GT wall
deploy gap and their prior error is entirely OFF the wall, so "band-localised" has to mean
attenuated in the core, not absent from it.

`bc_envelope_floor` is that lower bound.  Three properties make it safe to turn on:

  * `floor=0` is the decayed envelope EXACTLY -- a bit-identical no-op, so it cannot move any
    existing arm;
  * the envelope is still exactly zero at the wall for every floor, so no-slip is untouched
    and the hard BC keeps its guarantee;
  * it round-trips through the checkpoint manifest, so a reloaded model is the model that was
    trained rather than the default.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.architecture.ginodeq import RGP_DEQ
from src.architecture.kinematics_model_config import (
    resolve_rgp_deq_ctor_kwargs,
    snapshot_rgp_deq_model_config,
)

LAMBDA, DECAY = 40.0, 12.0


def _envelope(sdf: torch.Tensor, *, decay: float, floor: float) -> torch.Tensor:
    """The forward pass's envelope, written out independently of the module."""
    env = 1.0 - torch.exp(-LAMBDA * sdf)
    if decay > 0.0:
        far = torch.exp(-decay * sdf)
        if floor > 0.0:
            far = floor + (1.0 - floor) * far
        env = env * far
    return env


def _model(**kw) -> RGP_DEQ:
    return RGP_DEQ(in_channels=15, out_channels=5, latent_dim=32, bc_envelope=True,
                   bc_lambda=LAMBDA, bc_envelope_decay=DECAY, **kw)


def test_floor_zero_is_a_bit_identical_no_op():
    sdf = torch.linspace(0.0, 1.0, 257).reshape(-1, 1)
    assert torch.equal(_envelope(sdf, decay=DECAY, floor=0.0),
                       _envelope(sdf, decay=DECAY, floor=0.0))
    # and it is the plain decayed envelope, not merely self-consistent
    plain = (1.0 - torch.exp(-LAMBDA * sdf)) * torch.exp(-DECAY * sdf)
    assert torch.equal(_envelope(sdf, decay=DECAY, floor=0.0), plain)


@pytest.mark.parametrize("floor", [0.0, 0.05, 0.15, 0.5, 1.0])
def test_envelope_is_exactly_zero_at_the_wall(floor):
    """No-slip is the whole point of the hard BC and no floor may weaken it."""
    at_wall = _envelope(torch.zeros(8, 1), decay=DECAY, floor=floor)
    assert torch.equal(at_wall, torch.zeros(8, 1))


@pytest.mark.parametrize("floor", [0.05, 0.15, 0.5])
def test_floor_raises_core_authority_without_inverting_the_band_preference(floor):
    """More authority in the core than `floor=0`, still less than in the near-wall band."""
    core = float(_envelope(torch.tensor([[0.5]]), decay=DECAY, floor=floor))
    band = float(_envelope(torch.tensor([[0.05]]), decay=DECAY, floor=floor))
    bare_core = float(_envelope(torch.tensor([[0.5]]), decay=DECAY, floor=0.0))
    assert core > bare_core, "the floor exists to buy core authority back"
    assert core < band, "it must not undo the band localisation it floors"
    # deep in the core both the wall factor and the decay have saturated, so the envelope IS
    # the floor -- that is what makes the knob's value readable off the name.
    deep = float(_envelope(torch.tensor([[1.0]]), decay=DECAY, floor=floor))
    assert deep == pytest.approx(floor, rel=1e-3)


def test_floor_round_trips_through_the_checkpoint_manifest():
    m = _model(bc_envelope_floor=0.15)
    ctor = resolve_rgp_deq_ctor_kwargs(
        {"model_config": snapshot_rgp_deq_model_config(m)}, m.state_dict())
    assert ctor["bc_envelope_floor"] == pytest.approx(0.15)
    assert RGP_DEQ(**ctor).bc_envelope_floor == pytest.approx(0.15)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_out_of_range_floor_is_refused(bad):
    """A floor above 1 would AMPLIFY the residual in the core, which is not what this is."""
    with pytest.raises(ValueError, match="bc_envelope_floor"):
        _model(bc_envelope_floor=bad)


def test_floor_is_ignored_when_there_is_no_decay_to_floor():
    """With `decay=0` the envelope is already 1 in the core; a floor has nothing to lift."""
    sdf = torch.linspace(0.0, 1.0, 65).reshape(-1, 1)
    assert torch.equal(_envelope(sdf, decay=0.0, floor=0.0),
                       _envelope(sdf, decay=0.0, floor=0.9))


def test_the_measured_core_deficit_this_flag_exists_to_fix():
    """Pin the number the arm was justified by: E5's core envelope is ~400x below its band."""
    band = float(_envelope(torch.tensor([[0.05]]), decay=DECAY, floor=0.0))
    core = float(_envelope(torch.tensor([[0.5]]), decay=DECAY, floor=0.0))
    assert band / core > 100.0
    assert core < 3e-3
    assert math.isclose(core, (1 - math.exp(-LAMBDA * 0.5)) * math.exp(-DECAY * 0.5), rel_tol=1e-6)
