r"""Physics-informed wall-shear operator: a BOUNDED learned correction on a closed-form baseline.

    sr_pred / sr0  =  A(dmu) * (h0/h)^p * exp(eps_theta(x))
                      \_____/  \_______/  \________________/
                      anchored   flux       bounded residual
                      Tier 1    p init 2    |eps| <= eps_max

WHY THIS SHAPE, and not a network that predicts the ratio directly.  Every failure this module
has ever had is a failure of extrapolation, and this structure makes those failures impossible
rather than unlikely:

* **The Delta-mu response cannot vanish.**  The shipped GAT corrector moved wall `sr` by 2-4%
  over a 100x viscosity sweep, non-monotonically, because the response was absent from its
  training distribution (`corrector-is-delta-mu-blind`).  Here `A(dmu)` is analytic and
  anchored on a measured constant, so no amount of bad data can flatten it.
* **The flux term is the input every existing operator lacks.**  FEM shows `sr/sr0` RISING
  with occlusion in 5 of 5 vessels and exceeding 1.0 on comsol008 -- viscous shielding at low
  occlusion, flux-redistribution acceleration at high (`clot-shear-map-is-non-monotone`).  A
  function of `dmu` alone is ANTI-correlated with truth (-0.374); adding `(h0/h)^p` at the
  textbook exponent flips that to +0.554 (+0.677 in log).  That measurement is what fixes the
  architecture; `p` is initialised at the Poiseuille value 2 and refined, not invented.
* **Out of distribution it degrades to physics.**  `eps` is a `tanh` bounded at `eps_max`, so
  the worst case far from data is a bounded multiple of the analytic baseline -- never the
  `max|du|_nd = 2.96` blow-up the unconstrained corrector produced at high `dmu`.
* **The identity is exact.**  At `dmu = 0`, `h = h0`, `eps = 0` the operator returns `sr0`
  unchanged, so a clot-free vessel is provably untouched.

The learned term therefore only has to absorb what the backbone leaves behind, which the corpus
fit shows is mostly a PER-VESSEL offset (log-residual means -0.75 to +0.50, within-vessel
scatter as low as 0.13).  That is a small job, which is the point at this data scale.

Scope: this predicts wall shear MAGNITUDE on the wall manifold (~500-1000 nodes/vessel), which
is what the deposition gate consumes.  It says nothing about the interior velocity field.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.core_physics.wall_shear_attenuation import DELTA_MU_HALF_SI

#: Poiseuille: wall shear ~ Q/h^2 at fixed flux, so the flux term starts at the textbook value.
P_FLUX_INIT = 2.0

#: Columns the operator consumes, in order.  All computable at deploy from the model's own clot
#: state -- no GT velocity, no GT `Mat`.
FEATURE_NAMES = (
    "log_h",        # log(h/h0) -- the flux term's own argument, so eps can bend it locally
    "log_dmu",      # log1p(delta_mu) -- lets eps correct A() without being able to erase it
    "in_clot",
    "s_signed",     # signed arclength along the base flow: upstream < 0 < downstream
    "abs_s",
    "width_nd",
    "width_d1",
    "width_d2",
    "log_sr0",      # the base shear regime (a 2 1/s vessel behaves unlike a 200 1/s one)
)


def assemble_features(d: dict) -> np.ndarray:
    """Build the residual's input matrix from a corpus dict (or a live deploy dict)."""
    h = np.clip(np.asarray(d["h_over_h0"], dtype=np.float64), 1e-3, 1.0)
    s = np.asarray(d["s_signed"], dtype=np.float64)
    cols = [
        np.log(h),
        np.log1p(np.clip(np.asarray(d["delta_mu"], dtype=np.float64), 0.0, None)),
        np.asarray(d["in_clot"], dtype=np.float64),
        s,
        np.abs(s),
        np.asarray(d["width_nd"], dtype=np.float64),
        np.asarray(d["width_d1"], dtype=np.float64),
        np.asarray(d["width_d2"], dtype=np.float64),
        np.log1p(np.clip(np.asarray(d["sr0"], dtype=np.float64), 0.0, None)),
    ]
    return np.stack(cols, axis=1)


def hydraulic_h(delta_mu, h_over_h0, *, delta_mu_half: float = DELTA_MU_HALF_SI):
    """Effective residual lumen: a soft clot does not block flux the way a solid one does.

    MEASURED, not assumed.  Regressing `log(sr/sr0)` on `log(h)` inside `dmu` terciles of the
    FEM corpus gives slopes **-0.218 / -0.588 / -2.073** from low to high `dmu`
    (`python -m src.tools.diagnostics pi-flux-interaction`).  A CONSTANT flux exponent is therefore wrong: the
    Poiseuille value -2 is recovered only where the occlusion is stiff, and a soft gel barely
    redirects flux at all.  That is physical -- flux is diverted in proportion to how solid the
    obstruction is -- so it belongs in the geometry, not in a fitted exponent:

        B(dmu) = 1 - A(dmu)             how solid the clot is;  B(0) = 0,  B(inf) = 1
        h_eff  = 1 - (1 - h) * B(dmu)   only the SOLID part of the clot blocks the lumen

    `B` reuses `A`'s single anchored constant, so this adds no free parameter, and it keeps
    both limits exact: at `dmu = 0` the lumen is unblocked (`h_eff = 1`) and at `dmu -> inf`
    the geometric occlusion applies in full (`h_eff = h`).
    """
    if isinstance(delta_mu, torch.Tensor):
        A = 1.0 / (1.0 + delta_mu.clamp(min=0.0) / delta_mu_half)
        h = h_over_h0.clamp(min=1e-3, max=1.0)
        return (1.0 - (1.0 - h) * (1.0 - A)).clamp(min=1e-3, max=1.0)
    A = 1.0 / (1.0 + np.clip(np.asarray(delta_mu, dtype=np.float64), 0.0, None) / delta_mu_half)
    h = np.clip(np.asarray(h_over_h0, dtype=np.float64), 1e-3, 1.0)
    return np.clip(1.0 - (1.0 - h) * (1.0 - A), 1e-3, 1.0)


def physics_log_ratio(delta_mu, h_over_h0, *, p: float = P_FLUX_INIT,
                      delta_mu_half: float = DELTA_MU_HALF_SI, hydraulic: bool = True):
    """``log(A(dmu) * (h0/h_eff)^p)`` -- the closed-form baseline, no learning involved.

    ``hydraulic=False`` uses the raw geometric ``h``, i.e. a constant flux exponent.  Both are
    scored side by side in `scripts/train_pi_wall_shear.py` so the choice is reported rather
    than assumed.
    """
    h_use = hydraulic_h(delta_mu, h_over_h0, delta_mu_half=delta_mu_half) if hydraulic else (
        h_over_h0.clamp(min=1e-3, max=1.0) if isinstance(h_over_h0, torch.Tensor)
        else np.clip(np.asarray(h_over_h0, dtype=np.float64), 1e-3, 1.0))
    if isinstance(delta_mu, torch.Tensor):
        return -torch.log1p(delta_mu.clamp(min=0.0) / delta_mu_half) - p * torch.log(h_use)
    dmu = np.clip(np.asarray(delta_mu, dtype=np.float64), 0.0, None)
    return -np.log1p(dmu / delta_mu_half) - p * np.log(h_use)


class PIWallShear(nn.Module):
    """The operator.  ``forward`` returns ``log(sr_pred / sr0)``.

    ``p`` and ``delta_mu_half`` are LEARNABLE but physically initialised, so a fit that moves
    them far from 2.0 / 0.095 Pa.s is a signal the mechanism is wrong -- not a free parameter
    to be quietly absorbed.  Both are kept positive by construction.
    """

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 32,
                 *, eps_max: float = 0.4, learn_physics: bool = True,
                 hydraulic: bool = True):
        super().__init__()
        self.eps_max = float(eps_max)
        self.hydraulic = bool(hydraulic)
        self.log_p = nn.Parameter(torch.tensor(float(np.log(P_FLUX_INIT))),
                                  requires_grad=learn_physics)
        self.log_half = nn.Parameter(torch.tensor(float(np.log(DELTA_MU_HALF_SI))),
                                     requires_grad=learn_physics)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Start at eps == 0 so an untrained operator IS the physics baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std", torch.ones(n_features))

    @property
    def p(self) -> float:
        return float(self.log_p.detach().exp())

    @property
    def delta_mu_half(self) -> float:
        return float(self.log_half.detach().exp())

    def set_normalizer(self, X: np.ndarray) -> None:
        self.x_mean = torch.tensor(X.mean(axis=0), dtype=torch.float32)
        self.x_std = torch.tensor(np.maximum(X.std(axis=0), 1e-6), dtype=torch.float32)

    def eps(self, X: torch.Tensor) -> torch.Tensor:
        z = (X - self.x_mean) / self.x_std
        return self.eps_max * torch.tanh(self.net(z).reshape(-1))

    def forward(self, X: torch.Tensor, delta_mu: torch.Tensor,
                h_over_h0: torch.Tensor, *, physics_only: bool = False) -> torch.Tensor:
        base = physics_log_ratio(delta_mu, h_over_h0, p=self.log_p.exp(),
                                 delta_mu_half=self.log_half.exp(),
                                 hydraulic=self.hydraulic)
        return base if physics_only else base + self.eps(X)

    def predict_sr(self, X, delta_mu, h_over_h0, sr0, *, physics_only: bool = False):
        """Wall shear in 1/s, the units the deposition gate reads."""
        with torch.no_grad():
            lr = self.forward(_t(X), _t(delta_mu), _t(h_over_h0), physics_only=physics_only)
        return np.asarray(sr0, dtype=np.float64) * np.exp(lr.numpy())


def _t(a) -> torch.Tensor:
    return a if isinstance(a, torch.Tensor) else torch.tensor(np.asarray(a), dtype=torch.float32)
