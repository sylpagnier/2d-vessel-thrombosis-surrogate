"""Analytic wall-shear attenuation under clot viscosity elevation.

Used by ``pi_wall_shear`` (physics-informed wall shear operator).  The learned local
kinematic corrector that previously lived in ``coupled_shear_gnn`` was removed after
measurement showed it did not improve deploy clot scores; see ``docs/LOCAL_KINEMATIC_CORRECTOR.md``.
"""
from __future__ import annotations

import numpy as np
import torch

#: Clot viscosity elevation at which wall shear is halved [Pa.s].  Anchored on
#: (dmu = 0.68 Pa.s -> sr/sr0 = 0.1226); see LOCAL_KINEMATIC_CORRECTOR.md.
DELTA_MU_HALF_SI = 0.0950

#: The anchor itself, kept next to the constant it fixes so a test can pin the pair.
SHEAR_ANCHOR_DELTA_MU_SI = 0.68
SHEAR_ANCHOR_RATIO = 0.1226


def shear_attenuation(delta_mu_si, *, delta_mu_half_si: float = DELTA_MU_HALF_SI):
    """Wall-shear attenuation factor ``A(dmu)`` in ``(0, 1]``.

    ``sr_clot = sr_base * A(delta_mu)``.  Monotone decreasing, ``A(0) = 1``, ``A -> 0`` as the
    occlusion becomes solid.
    """
    half = max(float(delta_mu_half_si), 1e-12)
    if isinstance(delta_mu_si, torch.Tensor):
        return 1.0 / (1.0 + delta_mu_si.clamp(min=0.0) / half)
    dmu = np.clip(np.asarray(delta_mu_si, dtype=np.float64), 0.0, None)
    return 1.0 / (1.0 + dmu / half)
