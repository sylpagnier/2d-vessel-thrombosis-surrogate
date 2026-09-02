"""Property guards for the physics-informed wall-shear operator.

The whole argument for this architecture is that its failure modes are IMPOSSIBLE rather than
unlikely (see the module docstring).  Each claim below is one of those, as a test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.core_physics.wall_shear_attenuation import DELTA_MU_HALF_SI, shear_attenuation
from src.core_physics.pi_wall_shear import (
    FEATURE_NAMES, P_FLUX_INIT, PIWallShear, assemble_features, hydraulic_h,
    physics_log_ratio)


def _corpus(n=400, p_true=2.0, seed=0, offset=0.0, hydraulic=False):
    """Synthetic wall rows obeying the model exactly, so a fit has a known right answer."""
    rng = np.random.default_rng(seed)
    dmu = np.exp(rng.uniform(np.log(0.05), np.log(3.0), n))
    h = rng.uniform(0.25, 1.0, n)
    sr0 = np.exp(rng.uniform(np.log(2.0), np.log(200.0), n))
    h_use = hydraulic_h(dmu, h) if hydraulic else h
    log_ratio = -np.log1p(dmu / DELTA_MU_HALF_SI) - p_true * np.log(h_use) + offset
    return dict(sr0=sr0, sr_fem=sr0 * np.exp(log_ratio), delta_mu=dmu, h_over_h0=h,
                s_signed=rng.normal(size=n), width_nd=rng.uniform(0.5, 2.0, n),
                width_d1=rng.normal(size=n) * 0.1, width_d2=rng.normal(size=n) * 0.1,
                sdf_nd=np.zeros(n), in_clot=(dmu > 0).astype(float))


# --- the closed-form baseline -----------------------------------------------------------

def test_no_clot_is_exactly_the_identity():
    """dmu = 0 and h = h0 must return sr0 UNCHANGED -- a clot-free vessel is untouched."""
    lr = physics_log_ratio(np.zeros(5), np.ones(5))
    assert np.allclose(lr, 0.0, atol=0.0)


def test_baseline_reduces_to_the_tier1_prior_when_the_lumen_is_open():
    """With h = h0 the operator must BE `shear_attenuation`, or the two laws have forked."""
    dmu = np.array([0.0, 0.068, 0.68, 6.8])
    got = np.exp(physics_log_ratio(dmu, np.ones_like(dmu)))
    assert np.allclose(got, shear_attenuation(dmu), rtol=1e-12)


def test_flux_term_can_raise_shear_above_the_base_field():
    """The measured behaviour a dmu-only operator CANNOT produce: sr/sr0 > 1 at high occlusion
    (patient008 reads 1.565 in `outputs/diag_corrector_severe_occlusion.json`)."""
    ratio = float(np.exp(physics_log_ratio(np.array([0.002]), np.array([0.2]), p=2.0))[0])
    assert ratio > 1.0


def test_shear_is_monotone_decreasing_in_delta_mu_at_fixed_geometry():
    dmu = np.geomspace(1e-4, 1e3, 200)
    r = np.exp(physics_log_ratio(dmu, np.full_like(dmu, 0.7)))
    assert np.all(np.diff(r) < 0.0)


def test_shear_is_monotone_increasing_as_the_lumen_closes():
    h = np.linspace(0.05, 1.0, 200)
    r = np.exp(physics_log_ratio(np.full_like(h, 0.68), h))
    assert np.all(np.diff(r) < 0.0)      # h increasing -> ratio decreasing


def test_torch_and_numpy_baselines_agree():
    dmu, h = np.array([0.0, 0.3, 2.0]), np.array([1.0, 0.6, 0.3])
    a = physics_log_ratio(dmu, h)
    b = physics_log_ratio(torch.tensor(dmu), torch.tensor(h)).numpy()
    assert np.allclose(a, b, atol=1e-12)


# --- the learned residual ----------------------------------------------------------------

def test_untrained_operator_is_exactly_the_physics():
    """eps is init'd to zero, so an untrained model IS the closed form -- never worse."""
    d = _corpus(n=64)
    m = PIWallShear()
    X = torch.tensor(assemble_features(d), dtype=torch.float32)
    m.set_normalizer(assemble_features(d))
    with torch.no_grad():
        got = m(X, torch.tensor(d["delta_mu"], dtype=torch.float32),
                torch.tensor(d["h_over_h0"], dtype=torch.float32)).numpy()
    want = physics_log_ratio(d["delta_mu"], d["h_over_h0"])
    assert np.allclose(got, want, atol=1e-5)


def test_residual_is_bounded_even_on_absurd_input():
    """OOD safety: far from data the worst case is a BOUNDED multiple of the physics, never
    the `max|du|_nd = 2.96` blow-up the unconstrained GAT corrector produced."""
    m = PIWallShear(eps_max=0.4)
    torch.manual_seed(0)
    for lin in m.net:
        if isinstance(lin, torch.nn.Linear):
            torch.nn.init.normal_(lin.weight, std=50.0)
            torch.nn.init.normal_(lin.bias, std=50.0)
    X = torch.randn(256, len(FEATURE_NAMES)) * 1e4
    with torch.no_grad():
        e = m.eps(X)
    assert float(e.abs().max()) <= 0.4 + 1e-6


def test_physics_parameters_are_initialised_at_their_physical_values():
    m = PIWallShear()
    assert m.p == pytest.approx(P_FLUX_INIT, rel=1e-6)
    assert m.delta_mu_half == pytest.approx(DELTA_MU_HALF_SI, rel=1e-6)


@pytest.mark.parametrize("hyd", [False, True])
def test_closed_form_exponent_fit_recovers_a_known_p(hyd):
    """`fit_p` is STEP 3 and carries the headline claim; pin BOTH backbones against truth."""
    from scripts.train_pi_wall_shear import fit_p

    for p_true in (1.5, 2.0, 2.75):
        data = {f"v{i}": _corpus(n=500, p_true=p_true, seed=i, hydraulic=hyd)
                for i in range(3)}
        assert fit_p(data, hydraulic=hyd) == pytest.approx(p_true, abs=1e-6)


def test_hydraulic_lumen_has_both_limits_exact():
    """B(0) = 0 -> an infinitely soft clot blocks nothing; B(inf) = 1 -> a solid one blocks
    its full geometric extent.  Neither limit may be approximate: they are what let the
    backbone keep the identity at dmu = 0 while still reaching Poiseuille at high dmu."""
    h = np.array([0.2, 0.5, 0.9])
    assert np.allclose(hydraulic_h(np.zeros(3), h), 1.0, atol=0.0)
    assert np.allclose(hydraulic_h(np.full(3, 1e9), h), h, rtol=1e-6)
    # monotone in dmu: a stiffer clot blocks strictly more
    dmu = np.geomspace(1e-4, 1e4, 300)
    he = hydraulic_h(dmu, np.full_like(dmu, 0.4))
    assert np.all(np.diff(he) < 0.0)


def test_hydraulic_backbone_still_reduces_to_the_prior_with_an_open_lumen():
    """h = h0 must give A(dmu) exactly, hydraulic or not -- otherwise the two laws have forked."""
    dmu = np.array([0.0, 0.068, 0.68, 6.8])
    ones = np.ones_like(dmu)
    assert np.allclose(np.exp(physics_log_ratio(dmu, ones, hydraulic=True)),
                       shear_attenuation(dmu), rtol=1e-12)


def test_training_absorbs_a_per_vessel_offset():
    """What the residual is FOR: the corpus fit showed the leftover is mostly a per-vessel
    offset (log-residual means -0.75 to +0.50).  Training must remove one."""
    from scripts.train_pi_wall_shear import arm_scores, train_one

    train = {f"v{i}": _corpus(n=600, seed=i, offset=0.35) for i in range(3)}
    m = train_one(train, epochs=250, lr=1e-2, eps_max=0.6, hidden=16, seed=0)
    d = train["v0"]
    X = torch.tensor(assemble_features(d), dtype=torch.float32)
    with torch.no_grad():
        full = m(X, torch.tensor(d["delta_mu"], dtype=torch.float32),
                 torch.tensor(d["h_over_h0"], dtype=torch.float32)).numpy()
    phys = physics_log_ratio(d["delta_mu"], d["h_over_h0"])
    assert arm_scores(d, full.astype(np.float64))["mae_log"] < \
        0.5 * arm_scores(d, phys)["mae_log"]
