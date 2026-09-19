"""Project a learned velocity residual onto the divergence-free fields that vanish at the wall.

**Why.**  Under the hard BC the model's contribution is an additive field
``delta = pred - prior``, and nothing in the objective asks it to be solenoidal.  Measured over
14 deploy vessels, median ``|div u| / |u|``:

    plain FEM                        0.00011
    + residual, band-confined        0.00090     (8x)
    + residual, as trained           0.00150     (14x)
    + residual, 4x amplitude         0.00569     (52x)

and the biochem off-wall deploy score against plain FEM runs −0.025 / −0.051 / −0.162 over the
same three arms — monotone in the divergence and in nothing else that was measured.  That is a
mechanism, not a coincidence: the off-wall feature block is built by transporting material
along the flow (``dMat/dt + u.grad(Mat) = 0``, PHASE7 1.1), and a transport solve on a field
that is not solenoidal creates and destroys material that the equation says is conserved.

The information is there — every transport CHANNEL improves under the residual
(``log_mat_adv`` 0.5099 -> 0.5311).  What this removes is the part of the correction that
cannot be a velocity perturbation of an incompressible flow.

**How.**  Write the corrected residual as the curl of a stream function, which is solenoidal by
construction, and fit that stream function to the residual the model actually produced:

    delta' = (D_y psi, -D_x psi),     psi = argmin |D_y psi - delta_u|^2 + |D_x psi + delta_v|^2

with ``psi`` pinned to zero on the wall so the correction cannot move the no-slip boundary.
``D_x``/``D_y`` are the same MLS gradient operators every downstream consumer differentiates
with, so "divergence-free" here means *in the discretisation the features are computed in*,
which is the only sense that can matter to them.

This is a post-hoc projection of an existing checkpoint's output: it needs no retraining, and
it is the cheap test of whether a stream-function decoder is worth building.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def project_solenoidal(delta: np.ndarray, Dx, Dy, *, wall: np.ndarray | None = None,
                       ridge: float = 1e-6, rtol: float = 1e-8, maxiter: int = 2000):
    """Least-norm divergence-free correction of ``delta`` (n, 2), in the MLS discretisation.

    With ``B = [Dx, Dy]`` the discrete divergence, the nearest field to ``delta`` with
    ``B d' = 0`` is the classical projection

        d' = delta - B^T (B B^T)^-1 B delta

    and ``B B^T = Dx Dx^T + Dy Dy^T`` is symmetric positive semi-definite, so it takes a CG
    solve.  Every operator is the one the feature builders differentiate with, so the result is
    divergence-free in the sense that matters to them.

    **Two formulations that do NOT work, measured, so they are not retried.**

    * A stream function, ``delta = curl(psi)``: solenoidal in the continuum, but discretely
      ``div(curl psi) = (Dx Dy - Dy Dx) psi`` and these MLS operators do not commute.  Fitting
      one RAISED median divergence from 0.0049 to 0.0097 on five deploy vessels -- a second
      derivative applied to a noisy field amplifies what it was meant to remove.
    * The Poisson form ``L phi = div(delta)`` with ``L = Dx Dx + Dy Dy``: that composition is
      not symmetric and is near-singular, and a direct solve returned corrections 1e9 times the
      field.

    ``ridge`` is relative to the operator's diagonal scale, so it is dimensionless.

    **No-slip is enforced, not hoped for.**  The projection uses natural conditions, so
    ``B^T y`` is not zero on the wall and the raw projected residual moves it -- measured at
    3-7e-3 of the field's peak, which is small but not nothing, and
    ``test_deployable_flow_fixes.py`` rightly fails on it: a wall node with velocity makes
    every edge pointing into it carry a direction the clot ensemble never saw at training
    time.  The hard BC's guarantee outranks the divergence one, so the correction is zeroed on
    the wall afterwards and ``info["div_after"]`` reports what that costs.
    """
    from scipy.sparse.linalg import cg

    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError(f"delta must be (n, 2), got {delta.shape}")

    BBt = (Dx @ Dx.T + Dy @ Dy.T).tocsr()
    scale = float(np.abs(BBt.diagonal()).mean()) or 1.0
    A = BBt + (ridge * scale) * sp.eye(BBt.shape[0], format="csr")
    rhs = Dx @ delta[:, 0] + Dy @ delta[:, 1]
    y, _ = cg(A, rhs, rtol=rtol, maxiter=maxiter)
    out = delta - np.stack([Dx.T @ y, Dy.T @ y], axis=1)

    info = {"div_before": float(np.abs(rhs).mean()),
            "moved": float(np.linalg.norm(out - delta) / (np.linalg.norm(delta) + 1e-30))}
    if wall is not None:
        w = np.asarray(wall, dtype=bool)
        if w.any():
            info["wall_leak_raw"] = float(np.abs(out[w]).max())
            out[w] = 0.0
    info["div_after"] = float(np.abs(Dx @ out[:, 0] + Dy @ out[:, 1]).mean())
    info["wall_leak"] = (0.0 if wall is None
                         else float(np.abs(out[np.asarray(wall, dtype=bool)]).max())
                         if np.any(wall) else 0.0)
    return out, info


def divergence(uv: np.ndarray, Dx, Dy) -> np.ndarray:
    """``div u`` in the MLS discretisation the feature builders use."""
    uv = np.asarray(uv, dtype=np.float64)
    return Dx @ uv[:, 0] + Dy @ uv[:, 1]


__all__ = ["project_solenoidal", "divergence"]
