"""Pre-flight check on the t=0 flow, run BEFORE a clot rollout is trusted.

WHY THIS EXISTS.  The clot readout seeds its physics mask from ``(gate > 0) & wall``.  When the
supplied t=0 flow makes that gate fire on **no** wall node, the seed is empty and thirteen
downstream physics/advection/ownership channels become identically zero rather than degraded --
the prediction is not merely worse, it is vacuous.  On ``comsol010`` this is mask 131 -> 0 nodes
and wall F1 0.969 -> 0.000.

Measured over 33 vessels (``docs/PUBLICATION_NOTES.md`` s2), the empty-gate indicator is the
**strongest single predictor** of the score drop when flow is swapped (r = +0.745 against the
drop), while velocity rel-L2 is uninformative (r = +0.029).

THE POINT OF THIS MODULE: the predictive statistics are exactly the ones that need **no ground
truth**, so the diagnosis is deployable.  Gate Jaccard, ``dsrx`` correlation and rel-L2 all
require a reference field and are unavailable on a new vessel; the firing set of the gate is
self-contained.  We can therefore refuse a vacuous prediction *before* paying for it.

REFERENCE RANGE, and where it comes from.  Wall-node firing fraction over the 33-vessel cohort:

    GT  flow   min 0.0465   p5 0.0563   median 0.1322   max 0.4286   empty on 0
    FEM flow   min 0.0428   p5 0.0569   median 0.1245   max 0.4416   empty on 0
    RGP-DEQ    min 0.0000   p5 0.0000   median 0.0917   max 0.4544   empty on 5

FEM tracks ground truth closely; the learned surrogate empties the gate on 5 of 33.  The bounds
below are set just outside the GT/FEM envelope, so a vessel is flagged when its gate behaves
unlike anything the model was fitted or validated against -- not merely when it is unusual.

Reproduce the calibration: ``python scripts/publication/generate_flow_diagnostics.py --flow
{gt,fem,pred}`` then ``python scripts/validate_preflight.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

M_TO_CM = 100.0

# Firing fraction outside this band is unlike any GT or FEM vessel in the 33-vessel cohort.
FIRE_FRAC_MIN = 0.040   # below the GT min (0.0465) and the FEM min (0.0428)
FIRE_FRAC_MAX = 0.460   # above the GT max (0.4286) and the FEM max (0.4416)

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class PreflightResult:
    """Verdict on a t=0 flow field, from statistics that need no ground truth."""

    verdict: str
    n_wall: int
    n_fire: int
    fire_frac: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True unless the flow would produce a vacuous prediction."""
        return self.verdict != FAIL

    def __str__(self) -> str:
        head = (f"[preflight] {self.verdict.upper()}  "
                f"wall gate fires on {self.n_fire}/{self.n_wall} nodes "
                f"({self.fire_frac:.4f})")
        return head + ("".join("\n  - " + r for r in self.reasons) if self.reasons else "")


def wall_gate_firing(data, flow: str, bio_cfg=None) -> tuple[np.ndarray, np.ndarray]:
    """``(gate, wall)`` at t=0 for one flow source.

    Built from the same primitives as ``src/clot_ml/features.py`` -- the same ``lss``/``sgt``
    constants, the same ``_flow_hops`` stencil per flow source and the same ``dsrx_gain``
    amplitude correction -- so this is the consumer's own gate, not a re-derivation that might
    drift from it.
    """
    from src.clot_ml.temporal import _flow_hops
    from src.config import BiochemConfig
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d
    from src.core_physics.physics_wall_model import dsrx_gain

    bio = bio_cfg if bio_cfg is not None else BiochemConfig(phase="biochem")
    ei = data.edge_index.detach().cpu().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])

    if flow == "gt":
        u = data.y[0, :, 0].reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.y[0, :, 1].reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)

    Dx, Dy = build_mls_gradient(node_positions(data), ei, hops=_flow_hops(flow))
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = ((Dx @ sr) / (d_bar * M_TO_CM)) * dsrx_gain(flow)

    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    gate = (dsrx < sgt).astype(np.float64) * coef * np.abs(dsrx) + (sr < lss).astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    return gate, wall


def preflight_check(data, flow: str, bio_cfg=None) -> PreflightResult:
    """Is this t=0 flow fit to drive a clot rollout?

    ``FAIL`` means the wall gate fires nowhere: the readout's seed is empty and every downstream
    channel will be identically zero.  Do not spend a rollout on it -- supply a converged flow
    field instead (a local FEM solve empties the gate on 0 of 33 cohort vessels).

    ``WARN`` means the gate fires, but on a fraction of the wall unlike any vessel in the
    reference cohort.  The prediction is not vacuous and may well be fine; treat it as a vessel
    to inspect rather than to discard.
    """
    gate, wall = wall_gate_firing(data, flow, bio_cfg)
    n_wall = int(wall.sum())
    if n_wall == 0:
        return PreflightResult(FAIL, 0, 0, float("nan"),
                               ["no wall nodes: the pack carries no `mask_wall` selection"])

    fire = (gate > 0) & wall
    n_fire = int(fire.sum())
    frac = n_fire / n_wall

    if n_fire == 0:
        return PreflightResult(
            FAIL, n_wall, 0, 0.0,
            ["wall gate fires on NO node: the readout seed `(gate > 0) & wall` is empty, so "
             "the rollout would return identically-zero channels, not a degraded prediction",
             "supply a converged t=0 flow (local FEM) rather than a learned surrogate field"])

    reasons = []
    if frac < FIRE_FRAC_MIN:
        reasons.append(
            f"gate fires on {frac:.4f} of wall nodes, below anything seen under GT or FEM "
            f"flow in the reference cohort (min {FIRE_FRAC_MIN:.3f}) -- under-firing, expect "
            "recall loss")
    elif frac > FIRE_FRAC_MAX:
        reasons.append(
            f"gate fires on {frac:.4f} of wall nodes, above the reference cohort "
            f"(max {FIRE_FRAC_MAX:.3f}) -- over-firing, expect false positives")

    return PreflightResult(WARN if reasons else PASS, n_wall, n_fire, frac, reasons)
