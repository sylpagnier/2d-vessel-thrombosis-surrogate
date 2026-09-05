"""What makes `scripts/calibrate_residual_scale.py` a checkpoint transform and not a new model.

The calibration fits two numbers that the training objective leaves unconstrained -- the
residual's amplitude (`residual_scale`) and how tightly it is confined to the wall band
(`bc_envelope_decay`, RGP_DEQ_REPAIR_PLAN.md s18.5).  It fits them from ONE forward pass per
vessel, by claiming that the model's contribution obeys

    delta(alpha, decay) = alpha * exp(-(decay - decay_trained) * sdf) * delta_trained

and the calibrator therefore sweeps both knobs as arithmetic on a single prediction.  If that
identity does not hold, the sweep is scoring fields the deployed checkpoint will never produce
and every number it picks is wrong -- so these tests check it against the real forward pass
rather than against the formula's own restatement.

It is exact in exact arithmetic but not bit-exact in practice, for a reason worth knowing:
`RGP_DEQ` runs `outer_iters=3`, so the decoded field is fed back into the next equilibrium
solve and `residual_scale` is therefore inside the loop, not merely applied after it.  Measured
here, the deviation from linearity is 3e-5 of `max|delta|` at alpha=5 -- float32 round-off
through 3 x 4 Picard iterations, four decades below anything that could move a choice of
alpha.  The tolerances below are set at that scale deliberately; if they ever have to be
loosened, the feedback has stopped being negligible and the one-pass sweep needs re-deriving.
"""

from __future__ import annotations

import pytest
import torch

from src.config import NodeFeat
from src.tests.test_rgp_deq_repair import _s17_graph, _s17_model

LAMBDA, DECAY = 40.0, 12.0


def _delta(model, graph):
    """The model's whole additive contribution: `pred - prior`."""
    with torch.no_grad():
        out = model(graph, solver="picard")
    pred = (out[0] if isinstance(out, tuple) else out)[:, :2].double()
    return pred - graph.x[:, NodeFeat.UV_PRIOR].double()


def _agrees(got, want):
    """Equal in exact arithmetic; the forward pass is float32, so compare at its resolution.

    `delta` entries near the wall are ~1e-7 against a prior of ~1e-3, four decades below the
    quantity they are computed from, so an elementwise `rtol` there tests float32 rounding and
    nothing else.  The scale-relative `atol` is the honest comparison, set an order of
    magnitude above the 3e-5 deviation the outer-iteration feedback actually produces.
    """
    scale = float(want.abs().max())
    return torch.allclose(got, want, rtol=1e-2, atol=1e-4 * max(scale, 1e-30))


def _calibrated(model, *, alpha=1.0, decay=DECAY):
    """A copy of `model` with the two calibrated numbers changed and nothing else."""
    import copy

    out = copy.deepcopy(model)
    out.bc_envelope_decay = float(decay)
    with torch.no_grad():
        out.residual_scale.mul_(float(alpha))
    return out.eval()


@pytest.fixture(scope="module")
def base():
    m = _s17_model(bc_envelope=True, bc_lambda=LAMBDA, bc_envelope_decay=DECAY,
                   residual_rezero=True, decoder_skip=True, residual_gain=True)
    # ReZero starts the scale at 0, which would make every delta identically zero and every
    # assertion below vacuously true.  Give it the order of magnitude a trained arm reaches.
    with torch.no_grad():
        m.residual_scale.fill_(0.0067)
    return m, _s17_graph()


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 5.0])
def test_delta_is_exactly_linear_in_residual_scale(base, alpha):
    """`alpha` is a multiplier on the field, not merely on a parameter."""
    model, graph = base
    got = _delta(_calibrated(model, alpha=alpha), graph)
    want = alpha * _delta(model, graph)
    assert _agrees(got, want)


@pytest.mark.parametrize("decay", [24.0, 36.0, 72.0])
def test_redecaying_is_an_exponential_reweighting_of_the_same_delta(base, decay):
    """Changing the decay re-weights `delta` by `exp(-(decay - trained) * sdf)`, node by node."""
    model, graph = base
    got = _delta(_calibrated(model, decay=decay), graph)
    sdf = graph.x[:, NodeFeat.SDF].reshape(-1, 1).double()
    want = torch.exp(-(decay - DECAY) * sdf) * _delta(model, graph)
    assert _agrees(got, want)


def test_the_two_knobs_compose(base):
    """The sweep varies both at once, so the joint identity is the one that has to hold."""
    model, graph = base
    got = _delta(_calibrated(model, alpha=5.0, decay=24.0), graph)
    sdf = graph.x[:, NodeFeat.SDF].reshape(-1, 1).double()
    want = 5.0 * torch.exp(-(24.0 - DECAY) * sdf) * _delta(model, graph)
    assert _agrees(got, want)


@pytest.mark.parametrize("alpha", [1.0, 5.0, 50.0])
def test_no_slip_survives_any_calibration(alpha):
    """The hard BC is the one thing calibration must never be able to break."""
    model = _s17_model(bc_envelope=True, bc_lambda=LAMBDA, bc_envelope_decay=DECAY,
                       residual_rezero=True, decoder_skip=True, residual_gain=True)
    with torch.no_grad():
        model.residual_scale.fill_(0.0067)
    graph = _s17_graph()
    graph.x[:, NodeFeat.SDF] = 0.0                      # every node on the wall
    delta = _delta(_calibrated(model, alpha=alpha, decay=48.0), graph)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_a_large_decay_makes_the_core_contribution_negligible():
    """The property the core guard relies on, stated as a bound rather than a measurement.

    At `decay=24` against a trained `decay=12`, the residual at mid-lumen is down by `exp(-6)`,
    which is why raising the amplitude 5x can leave the off-wall features untouched.
    """
    band, core = 0.05, 0.5
    ratio_band = torch.exp(torch.tensor(-(24.0 - DECAY) * band))
    ratio_core = torch.exp(torch.tensor(-(24.0 - DECAY) * core))
    assert float(ratio_band) > 0.5, "the band must keep most of its authority"
    assert float(ratio_core) < 1e-2, "the core must lose essentially all of it"
    assert float(ratio_band) / float(ratio_core) > 100.0
