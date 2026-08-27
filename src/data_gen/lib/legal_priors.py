"""Deploy-legal prior fields for the RGP-DEQ input block (WALL_MODEL_PLAN.md s16.1, s17 Z1/Z2).

**Why this module exists.** `data.x[:, UV_PRIOR|MU_PRIOR]` as stored in the anchor packs are
bit-identical to the converged clot-free CFD solution `y[0]` -- they contain backflow, which the
clamped parabolic magnitude in `build_poiseuille_priors` cannot produce, and `wss_prior_nd` is
identically zero because it has no `y[0]` counterpart to be overwritten with (s16.1).

The RGP-DEQ consumes those columns as *inputs* (`ginodeq.py:438-440`), so the flow surrogate is
handed the field it exists to predict.

**The deployment contract (s17 Z2, decided):** at deploy we are given **geometry + initial and
boundary conditions only**. No clot-free CFD solve is available. The stored priors are therefore
not legal inputs, and anything trained on them is trained on information that will not exist.

This module supplies legal replacements computable from `(sdf_nd, width_nd, mask_inlet,
mask_outlet, edge_index)` alone.
"""
from __future__ import annotations

import torch

from src.config import PhysicsConfig
from src.data_gen.lib.node_feature_assembly import (
    mass_conserving_umax_nd,
    width_nd_to_radius_nd,
)

# data.x column layout (kine_x_v1_18ch)
COL_XY = slice(0, 2)
COL_SDF = 2
COL_WALL_NORMAL = slice(4, 6)
COL_U_PRIOR = 11
COL_V_PRIOR = 12
COL_MU_PRIOR = 13
COL_WSS_PRIOR = 14
COL_WIDTH = 15

PRIOR_SOURCES = ("stored", "analytic", "zero")


def potential_flow_direction(
    data, *, iters: int = 3000, tol: float = 1e-8, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit flow direction from a graph Laplace solve with inlet/outlet Dirichlet BCs.

    Potential-flow approximation: solve ``div(grad phi) = 0`` with ``phi=1`` on the inlet and
    ``phi=0`` on the outlet, then take the normalised ``-grad phi`` as the streamwise direction.
    Uses only geometry and the boundary masks, so it is legal under the s17 Z2 contract.

    Solved by conjugate gradient on the free nodes. **Jacobi is not viable here**: these meshes
    run ~274 hops inlet-to-outlet, so Jacobi needs O(diameter^2) ~ 75k sweeps to converge, and
    an under-converged potential yields a direction field uncorrelated with the flow.

    Returns ``(dir_x, dir_y)``, each ``[N]``, unit-norm where the gradient is resolvable.
    """
    dev = device or data.x.device
    n = int(data.num_nodes)
    row, col = data.edge_index.to(dev)
    pos = data.x[:, COL_XY].to(dev, torch.float32)

    inlet = _mask(data, "mask_inlet", n, dev)
    outlet = _mask(data, "mask_outlet", n, dev)
    fixed = inlet | outlet
    free = ~fixed

    deg = torch.zeros(n, device=dev)
    deg.index_add_(0, row, torch.ones(row.shape[0], device=dev))

    def lap(v: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(n, device=dev)
        acc.index_add_(0, row, v[col])
        return deg * v - acc

    phi_fixed = torch.zeros(n, device=dev)
    phi_fixed[inlet] = 1.0
    # Solve L[free,free] x = -L[free,fixed] phi_fixed  by CG on the free block.
    b = (-lap(phi_fixed))[free]
    x = torch.zeros(int(free.sum()), device=dev)

    def A(v: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(n, device=dev)
        full[free] = v
        return lap(full)[free]

    r = b - A(x)
    p = r.clone()
    rs = torch.dot(r, r)
    for _ in range(int(iters)):
        if rs.sqrt() < tol:
            break
        Ap = A(p)
        alpha = rs / torch.dot(p, Ap).clamp(min=1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        p = r + (rs_new / rs.clamp(min=1e-30)) * p
        rs = rs_new

    phi = phi_fixed.clone()
    phi[free] = x

    gx, gy = _lsq_gradient(phi, pos, row, col, n, dev)
    # Flow runs down the potential gradient.
    dx, dy = -gx, -gy
    mag = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-9)
    return dx / mag, dy / mag


def _shallow_view(data):
    """A new ``Data`` object with its own attribute store but the caller's tensors.

    ``Data.clone()`` deep-copies every tensor, which on these packs means duplicating a ~335 MB
    ``y``.  All this module ever rewrites is ``x``, so a fresh store over shared tensors is
    both cheaper and sufficient -- and, unlike returning ``data`` itself, a later ``.to(dev)``
    on the result cannot reach back and mutate the caller's pack.

    Anything that is not a PyG ``Data`` (duck-typed stand-ins in the tests, plain namespaces)
    is returned as-is: there is nothing to alias-protect and no store to rebuild.
    """
    to_dict = getattr(data, "to_dict", None)
    if not callable(to_dict):
        return data
    try:
        out = data.__class__()
        for key, value in to_dict().items():
            out[key] = value
    except Exception:
        return data
    return out


def _mask(data, name: str, n: int, dev) -> torch.Tensor:
    m = getattr(data, name, None)
    if m is None:
        return torch.zeros(n, dtype=torch.bool, device=dev)
    return m.reshape(-1).to(dev).bool()


#: Relative eigenvalue floor below which a node's 2x2 LSQ normal matrix is treated as rank-1.
#: COMSOL exports ``triangle6``: 74.5% of biochem mesh nodes are P2 mid-side nodes of degree 2,
#: and their two edge vectors are EXACTLY antiparallel (measured ``cos = -1.0000`` on 100% of
#: them), so ``A`` is rank-1 by construction, not by accident.
GRAD_RCOND = 1e-6

#: Passes of neighbour-fill for rank-deficient rows.  A P2 mid-side node is the exact midpoint
#: of its two corner neighbours, so one pass is 2nd-order exact; the extra passes only cover
#: the rare deficient node with no well-conditioned neighbour.
GRAD_FILL_PASSES = 3


def _lsq_gradient(f, pos, row, col, n, dev):
    """Per-node least-squares gradient of a scalar field over graph edges.

    **Rank-aware** (RGP_DEQ_REPAIR_PLAN.md B3).  The previous implementation added a
    scale-relative ridge ``A + 1e-6 * |A|max * I`` and inverted.  On a collinear stencil that
    lifts the null direction *just above* the solver's tolerance, so the transverse gradient
    component is inverted rather than truncated and comes back as amplified noise.  Measured
    consequence on :func:`potential_flow_direction`: ``cos`` vs COMSOL read **+1.00 on P1
    corner nodes and +0.65 on the 74.5% of nodes that are P2 mid-side** -- a ~50 degree error
    on three quarters of every biochem mesh.

    The fix is two steps, both cheap:

    1. **Truncate, do not ridge.**  Eigen-decompose the symmetric 2x2 and solve only in the
       directions whose eigenvalue clears :data:`GRAD_RCOND` relative to the largest.  That is
       the minimum-norm solution: along a collinear stencil the gradient is resolved, and the
       transverse component is honestly 0 instead of noise.
    2. **Fill the deficient rows from well-conditioned neighbours.**  A truncated mid-side
       gradient is unbiased but incomplete.  Since a P2 mid-side node lies exactly on the
       segment between its two corner neighbours, averaging their (well-conditioned) gradients
       is 2nd-order exact and recovers the transverse component the stencil cannot see.
    """
    dv = pos[col] - pos[row]
    df = f[col] - f[row]
    w = 1.0 / dv.norm(dim=1).clamp(min=1e-9) ** 2
    # Follow the inputs' dtype: hard-coding float32 here raises `index_add_(): self (Float) and
    # source (Double) must have the same scalar type` the moment a caller passes float64
    # positions, which is the natural thing to do when checking this operator's accuracy.
    dt = torch.promote_types(dv.dtype, df.dtype)
    dv, df, w = dv.to(dt), df.to(dt), w.to(dt)
    A = torch.zeros(n, 2, 2, device=dev, dtype=dt)
    b = torch.zeros(n, 2, device=dev, dtype=dt)
    for k in range(2):
        for j in range(2):
            A[:, k, j].index_add_(0, row, w * dv[:, k] * dv[:, j])
        b[:, k].index_add_(0, row, w * dv[:, k] * df)

    # Symmetric by construction; eigh is stable on the 2x2 and gives the null direction.
    evals, evecs = torch.linalg.eigh(A.double())
    lam_max = evals.abs().amax(dim=1, keepdim=True).clamp(min=1e-300)
    keep = (evals.abs() / lam_max) > GRAD_RCOND
    inv = torch.where(keep, 1.0 / torch.where(keep, evals, torch.ones_like(evals)),
                      torch.zeros_like(evals))
    # g = V diag(inv) V^T b, computed without materialising the pseudo-inverse.
    bt = torch.einsum("nij,nj->ni", evecs.transpose(1, 2), b.double())
    g = torch.einsum("nij,nj->ni", evecs, inv * bt).to(b.dtype)

    deficient = ~keep.all(dim=1)
    if bool(deficient.any()):
        good = (~deficient).to(g.dtype)
        for _ in range(GRAD_FILL_PASSES):
            if not bool(deficient.any()):
                break
            acc = torch.zeros_like(g)
            cnt = torch.zeros(n, device=dev, dtype=g.dtype)
            acc.index_add_(0, row, g[col] * good[col].unsqueeze(1))
            cnt.index_add_(0, row, good[col])
            fillable = deficient & (cnt > 0)
            if not bool(fillable.any()):
                break
            g = torch.where(
                fillable.unsqueeze(1), acc / cnt.clamp(min=1.0).unsqueeze(1), g
            )
            good = good.clone()
            good[fillable] = 1.0
            deficient = deficient & ~fillable
    return g[:, 0], g[:, 1]


#: Peak ND velocity cap, expressed as a MULTIPLE of the vessel's own inlet peak rather than as
#: the absolute 2.0 in ``graph_velocity_priors``.  The absolute cap was set because uncapped
#: ``1/R`` reached ~8 ND on tight stenoses; but the vessels it clips are precisely the stenoses
#: and wound packs the clot model exists to score.  Measured inlet-to-throat radius ratios:
#: patient020 1.10, patient001 1.51, wound_patient001 3.50, patient041 4.55 -- and COMSOL's own
#: peak on patient041 is 5.32 ND against the 2.0 clip, a 2.7x truncation of a real physical
#: acceleration.  A relative cap keeps the blow-up guard without deleting the signal.
UMAX_CAP_X_INLET = 6.0


def inlet_anchored_umax_nd(data, r_nd: torch.Tensor, *, device=None) -> torch.Tensor:
    """Poiseuille peak from 2D mass conservation, anchored on the vessel's OWN inlet BC.

    ``mass_conserving_umax_nd`` uses fixed module constants (``U_MAX_BASE_ND = 1.5``,
    ``R_REF_ND = 0.5``) and never reads the pack.  The inlet Dirichlet BC *is* available under
    the s17 Z2 deploy contract -- ``data.u_inlet_bc`` on the inlet mask is exactly the COMSOL
    inlet profile -- so the reference peak and reference radius should both come from it.

    ``u_max(x) = u_peak_inlet * (R_inlet / R(x))`` is exact for 2D planar mass conservation.
    Falls back to the module constants when the pack carries no usable inlet BC.
    """
    dev = device or r_nd.device
    n = int(data.num_nodes)
    inlet = _mask(data, "mask_inlet", n, dev)
    bc = getattr(data, "u_inlet_bc", None)
    u_peak = None
    if bc is not None and torch.is_tensor(bc) and bool(inlet.any()):
        b = bc.to(dev).reshape(n, -1).float()
        speed = b.norm(dim=1) if b.shape[1] >= 2 else b[:, 0].abs()
        cand = speed[inlet]
        if cand.numel() > 0 and float(cand.max()) > 0.0:
            u_peak = cand.max()
    if u_peak is None:
        return mass_conserving_umax_nd(r_nd).reshape(-1)

    r_in = r_nd[inlet]
    r_ref = r_in.median() if r_in.numel() > 0 else r_nd.median()
    u_max = u_peak * (r_ref.clamp(min=1e-6) / r_nd.clamp(min=1e-6))
    return u_max.clamp(max=float(u_peak) * UMAX_CAP_X_INLET).reshape(-1)


def build_analytic_priors(
    data, *, phys_cfg: PhysicsConfig | None = None, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical Poiseuille priors from geometry + BCs only. Returns (u, v, mu, wss).

    The magnitude, shear rate, Carreau viscosity and wall shear stress are pure functions of
    ``(sdf_nd, width_nd)`` and the inlet BC -- no flow direction needed. Direction is supplied
    by :func:`potential_flow_direction` and only sets the sign/orientation of ``u``/``v``.
    """
    dev = device or data.x.device
    ph = phys_cfg or PhysicsConfig()
    x = data.x.to(dev)
    sdf = x[:, COL_SDF].reshape(-1).clamp_min(0.0)
    width = x[:, COL_WIDTH].reshape(-1)

    r_nd = width_nd_to_radius_nd(width).reshape(-1)
    u_max = inlet_anchored_umax_nd(data, r_nd, device=dev)
    r_lane = (r_nd - torch.minimum(sdf, r_nd)).clamp_min(0.0)

    mag = torch.clamp(u_max * (1.0 - (r_lane**2 / (r_nd**2 + 1e-12))), min=0.0)
    gamma = torch.abs(-2.0 * u_max * r_lane / (r_nd**2 + 1e-12))

    if getattr(ph, "viscosity_model", "carreau") == "newtonian":
        mu = torch.ones_like(mag)
    else:
        ref = float(ph.mu_viscosity_nd_scale)
        u_ref = float(getattr(data, "u_ref", torch.tensor(1.0)).reshape(-1)[0])
        d_bar = float(getattr(data, "d_bar", torch.tensor(1.0)).reshape(-1)[0])
        lam_nd = ph.lam * (u_ref / max(d_bar, 1e-12))
        mu = (ph.mu_inf / ref) + ((ph.mu_0 / ref) - (ph.mu_inf / ref)) * (
            1.0 + (lam_nd * gamma) ** ph.a
        ) ** ((ph.n - 1.0) / ph.a)

    wall = _mask(data, "mask_wall", int(data.num_nodes), dev)
    wss = mu * gamma * wall.to(mu.dtype)

    dx, dy = potential_flow_direction(data, device=dev)
    return mag * dx, mag * dy, mu, wss


def resolve_prior_source(default: str = "stored") -> str:
    """Active prior source from the runtime config, else ``SPECIES_PRIOR_SOURCE``, else default."""
    import os

    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return str(rt.rollout.prior_source or default).strip().lower()
    except Exception:
        pass
    return (os.environ.get("SPECIES_PRIOR_SOURCE") or default).strip().lower()


def assert_train_deploy_prior_parity(train_source: str, deploy_source: str) -> None:
    """Fail loudly when training uses a prior block deploy will not have (s17 Z3).

    v1-v10 trained with the leaked CFD priors and deployed against a predicted field. That is a
    distribution shift sitting under every result in sections 9-13, and it was never checked.
    """
    t, d = (train_source or "").strip().lower(), (deploy_source or "").strip().lower()
    if t == d:
        return
    raise ValueError(
        f"prior_source mismatch: training uses {t!r} but deploy uses {d!r}. "
        "Under the s17 Z2 contract the model must never train on a prior block it will not "
        "have at deploy. Set both to 'analytic', or pass them equal deliberately."
    )


def apply_prior_source(data, source: str = "analytic", *, phys_cfg: PhysicsConfig | None = None):
    """Return ``data`` with the four prior columns rewritten according to ``source``.

    * ``stored``   -- leave as-is. **Illegal under the s17 Z2 contract** (these are GT CFD).
    * ``analytic`` -- Poiseuille magnitude + potential-flow direction. Legal.
    * ``zero``     -- all four columns zeroed. The ablation control for Z1.

    Always returns a NEW ``Data`` object whose tensors are shared with ``data`` except for the
    rewritten ``x`` (RGP_DEQ_REPAIR_PLAN.md B12).  The old code returned the caller's own
    object for ``stored`` and a full ``.clone()`` otherwise, so a loop of the form
    ``apply_prior_source(pack, src).to(device)`` moved the caller's pack to the GPU on the
    ``stored`` pass and then silently compared the next source against a mutated original.
    Sharing rather than deep-cloning also avoids duplicating ``y`` -- these packs run ~335 MB
    and ``y`` is nearly all of it.
    """
    src = (source or "stored").strip().lower()
    if src not in PRIOR_SOURCES:
        raise ValueError(f"prior source must be one of {PRIOR_SOURCES}, got {source!r}")

    out = _shallow_view(data)
    if src == "stored":
        return out

    x = out.x.clone()
    if src == "zero":
        x[:, COL_U_PRIOR] = 0.0
        x[:, COL_V_PRIOR] = 0.0
        x[:, COL_MU_PRIOR] = 0.0
        x[:, COL_WSS_PRIOR] = 0.0
    else:
        u, v, mu, wss = build_analytic_priors(out, phys_cfg=phys_cfg)
        x[:, COL_U_PRIOR] = u.to(x.dtype)
        x[:, COL_V_PRIOR] = v.to(x.dtype)
        x[:, COL_MU_PRIOR] = mu.to(x.dtype)
        x[:, COL_WSS_PRIOR] = wss.to(x.dtype)
    out.x = x
    return out
