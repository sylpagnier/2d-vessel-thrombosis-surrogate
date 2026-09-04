"""Phase-3 physics wall-clot model: the COMSOL deposition law integrated on t=0 gates.

PHASE3_HANDOFF 1.2-1.5.  Nothing here is learned except a single global scalar
(``da_scale``) that sets the deposition rate level.

Inputs are deploy-legal under the Phase-3 bandaid: node positions, mesh connectivity,
``u_ref``/``d_bar``, the boundary/initial conditions, and the **GT velocity field at
t=0 only**.

The one substantive change from the previous stack: shear rate and its x-gradient are
computed with :mod:`src.core_physics.mls_gradient` instead of the packs' ``G_x``/``G_y``.
Audited against COMSOL's own ``spf.sr`` / ``d(spf.sr,x)`` on comsol007:

    operator                  spearman(spf.sr)   spearman(d(spf.sr,x))
    packs' G_x / G_y                0.19                0.00
    MLS, 3 graph hops               0.998               0.990

Everything downstream -- both deposition gates -- was previously being evaluated on
noise.  See ``scripts/step0_mls_validate.py``.

Unit system here is COMSOL-native CGS, matching
``src/core_physics/comsol_surface_deposition.py`` and ``viscosity_mat_crit`` (2e7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.core_physics.mls_gradient import (
    build_mls_gradient,
    node_positions,
    shear_rate_2d,
)

M_TO_CM = 100.0
PER_M3_TO_PER_CM3 = 1.0e-6
PER_M2_TO_PER_CM2 = 1.0e-4


@dataclass
class T0Fields:
    sr: np.ndarray        # spf.sr [1/s]
    dsrx: np.ndarray      # d(spf.sr,x) [1/(s*cm)]
    gate_low: np.ndarray  # spf.sr < lss
    gate_sep: np.ndarray  # d(spf.sr,x) < sgt
    gate: np.ndarray      # (L/gamma_m)*|dsrx|*[sep] + [low]   -- the law's bracket prefactor
    # The velocity these were differentiated FROM, carried so downstream consumers cannot
    # silently re-read `data.y[0]` and reintroduce GT flow under `flow_source='pred'`.
    # `None` on the hand-built T0Fields in shear_redistribution / the diag scripts, which
    # synthesise sr/dsrx directly; consumers that need u/v must check rather than fall back.
    u: np.ndarray | None = None
    v: np.ndarray | None = None


def gate_from_shear(
    sr: np.ndarray, dsrx: np.ndarray, bio_cfg, *, wall: np.ndarray | None = None
) -> np.ndarray:
    """The COMSOL deposition law's bracket prefactor from a shear field.

    ``gate = [dsrx < sgt] * (L_char/gamma_m) * |dsrx|  +  [sr < lss]``

    Single source of truth: every arm that re-evaluates the gate on a *different* flow
    field -- the GT-flow oracle, the corrector rollout, the frozen t=0 fields -- must use
    the same expression, or their comparison measures the transcription and not the flow.
    ``sr`` in 1/s, ``dsrx`` in 1/(s*cm), both COMSOL-native CGS.
    """
    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM
    coef = float(bio_cfg.L_char) * M_TO_CM / float(bio_cfg.gamma_m)
    g = (dsrx < sgt_cgs) * coef * np.abs(dsrx) + (sr < float(bio_cfg.lss))
    return g if wall is None else g * wall


#: The injured wall's deposition prefactor.  COMSOL's `srf2` is `srf1` with both shear gates
#: deleted, so the bracket multiplier there is a hard 1 (docs/WOUND_PROGRESS.md 1).  Not a fit.
WOUND_UNGATED_PREFACTOR: float = 1.0


def deposition_gate(data, fields=None, *, gate: np.ndarray | None = None,
                    wall: np.ndarray | None = None, wound_source: bool = True,
                    prefactor: float = WOUND_UNGATED_PREFACTOR) -> np.ndarray:
    """The full surface-deposition prefactor: gated `srf1` on healthy wall, ungated `srf2`
    on the wound.

    WHY THIS EXISTS.  Every consumer used to write ``fields.gate * mask_wall``, which is the
    right transcription of `srf1` and a wrong description of the VESSEL: on a wound pack the
    injured segment then deposits nothing at all, while carrying **50-88% of the vessel's
    total surface ``Mat``**.  Everything derived from that trajectory inherits the hole --
    ``mat_phys``, the advective source, and ``mat_owner`` on the 5.6-16.4% of the mesh whose
    nearest solid node is a wound node (docs/WOUND_PROGRESS.md 14.6).

    This is the same defect WOUND_PROGRESS 6/6b found in the geometry layer, one level up.
    The split those sections preserved still holds and is the reason this is not simply
    ``fields.gate * solid``: ``mask_wall`` selects the **gated law**, and the wound gets the
    **ungated one** rather than `srf1`'s value.

    Pass either ``fields`` (uses ``fields.gate``) or an explicit ``gate`` array -- the v4
    indicator backbone builds its own gate and needs the same union applied to it.

    ``wound_source=False`` reproduces the healthy-wall-only prefactor bit-for-bit, and on a
    pack with no wound mask both branches are identical.
    """
    if (fields is None) == (gate is None):
        raise ValueError("pass exactly one of `fields` or `gate`")
    wall = (data.mask_wall.reshape(-1).bool().cpu().numpy() if wall is None
            else np.asarray(wall, dtype=bool))
    base = fields.gate if gate is None else np.asarray(gate, dtype=np.float64)
    gate = base * wall
    if not wound_source:
        return gate
    wnd = getattr(data, "mask_wound", None)
    if wnd is None or not torch.is_tensor(wnd) or wnd.numel() == 0:
        return gate
    wnd = wnd.reshape(-1).bool().cpu().numpy()
    if not wnd.any():
        return gate
    return np.where(wnd, float(prefactor), gate)


def gt_flow_gate_series(
    data, bio_cfg, *, hops: int = 3, wall: np.ndarray | None = None
) -> np.ndarray:
    """``[T, N]`` gate recomputed from the GT velocity at EVERY timestep -- an ORACLE.

    Upper bound on any evolving-flow model, learned or not: zero flow error. Illegal as a
    model, decisive as a ceiling. Prefer ``outputs/wall_species_cache/<v>.npz``'s
    ``sr_t``/``dsrx_t`` when only wall nodes are needed -- this recomputes MLS gradients at
    all T timesteps and is minutes per vessel.
    """
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    nt = int(data.y.shape[0])
    out = np.zeros((nt, int(data.num_nodes)))
    for ti in range(nt):
        u = data.y[ti, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[ti, :, 1].detach().cpu().numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        out[ti] = gate_from_shear(sr, (Dx @ sr) / (d_bar * M_TO_CM), bio_cfg, wall=wall)
    return out


#: h3->h6 stencil attenuation: a property of the MLS operator measured on the GT field alone
#: (2026-08-23, wall nodes, 10 vessels).  A wide stencil smooths the second derivative of any
#: field, regardless of which solver produced it.
DSRX_STENCIL_GAIN = 2.18

#: Surrogate-specific amplitude deficit at hops=6, like-for-like (corr 0.95, 2026-08-23).
#: The RGP-DEQ surrogate's near-wall velocity is further 1.38x underestimated relative to GT
#: at the same stencil width.  FEM and GT both carry zero surrogate deficit.
DSRX_SURROGATE_GAIN = 1.38   # = PRED_DSRX_GAIN / DSRX_STENCIL_GAIN

#: Combined amplitude correction for the RGP-DEQ surrogate at ``hops=6``.
#:
#: ``sgt`` is a PHYSICAL constant read off COMSOL's deposition law, but the discrete ``dsrx``
#: it is compared against depends on the stencil that produced it, and the two flow sources do
#: not use the same one: GT is differentiated at ``hops=3`` and the surrogate needs ``hops=6``
#: to escape a sign flip (``src/clot_ml/features.py``).  Two effects stack, both measured
#: 2026-08-23 at the wall: a 6-hop stencil attenuates ``dsrx`` **2.18x** relative to a 3-hop
#: one on the GT field alone (DSRX_STENCIL_GAIN), and the surrogate is a further **1.38x** low
#: like-for-like at hops=6 (DSRX_SURROGATE_GAIN, corr 0.95).  Uncorrected, the ``sgt`` gate
#: branch fires on 0.56x the nodes it should and agrees with the GT gate on almost none of them.
#:
#: **This is a fitted constant, not a derivation, and it is fitted on FIT only.**  The obvious
#: deploy-legal alternative was tested and fails: the same operator chain applied to a smooth
#: SYNTHETIC field reads a ratio of 1.00 on every vessel, because a wide stencil is exact on a
#: resolved field -- the attenuation is a statement about the near-wall shear field's own
#: spectral content, not a property of the operator, so it cannot be estimated per vessel
#: without ground truth.  Least squares on FIT (n=25) gives 3.00; DEV (n=5, held out) would
#: have chosen 2.56, which is the honest measure of how well it transfers.  Two significant
#: figures: anything finer is inside that spread.
#:
#: Wall gate union Jaccard against the GT gate, median:  FIT **0.20 -> 0.52**, DEV (held out)
#: **0.47 -> 0.54**.  Fire rate relative to GT: FIT x0.61 -> x0.89, DEV x0.67 -> x1.24.
#:
#: Kept as a DERIVED ALIAS (= DSRX_STENCIL_GAIN * DSRX_SURROGATE_GAIN) so the RGP-DEQ arm's
#: numbers are bit-identical and the split can be verified: 2.18 * 1.38 = 3.01, rounded to 3.00.
#: Applied here and in ``src/clot_ml/features.py`` -- the two places that gate on `sgt`.
#: Deliberately NOT applied in ``src/differentiable_wall_model``: its gates are soft and its
#: thresholds are *learned* per artifact (``compute_soft_gates`` / ``ParameterMap``), so they
#: absorb the scale themselves and rescaling the input would invalidate its trained weights.
PRED_DSRX_GAIN = 3.00   # = DSRX_STENCIL_GAIN * DSRX_SURROGATE_GAIN; keep for DEQ arm


def dsrx_gain(flow_source: str) -> float:
    """Wall-shear x-derivative amplitude correction, per flow source.

    `pred` carries the surrogate's fitted deficit; `fem` and `gt` are converged fields on
    COMSOL's own scale and take 1.0.  `CLOT_PRED_DSRX_GAIN` overrides the reconstructed
    sources so the constant can be swept apart from the field.

    Centralised because the override used to be read inline here only, while `features.py`
    applied a hardcoded `PRED_DSRX_GAIN` to its own `dsrx` -- so an ablation moved one of the
    two derivatives and not the other.
    """
    import os as _os

    src = str(flow_source)
    if src in ("pred", "fem"):
        raw = _os.environ.get("CLOT_PRED_DSRX_GAIN", "").strip()
        if raw:
            return float(raw)
    return {"pred": PRED_DSRX_GAIN}.get(src, 1.0)


def t0_flow_fields(
    data, bio_cfg, *, hops: int = 3, time_index: int = 0, flow_source: str = "gt"
) -> T0Fields:
    """Shear rate and shear-gradient at ``time_index`` from the velocity field.

    ``flow_source='gt'`` is the Phase-3 bandaid (GT flow at t=0).  ``'pred'`` uses the
    deployable kinematic model's ``u0_pred``/``v0_pred`` and is the Phase-5 arm -- the
    delta between the two IS the deployability gap (PHASE3_HANDOFF 4a).
    """
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])           # m/s
    d_bar = float(data.d_bar.reshape(-1)[0])           # m
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    # `fem` reads the same slot: the local solver writes its field into `u0_pred` and the
    # rest of the pipeline treats it as a reconstructed field, which is what it is.  An
    # UNRECOGNISED source raises -- it used to fall through to the ground-truth branch, so a
    # `flow="fem"` run silently scored GT and looked like a perfect solver.
    if flow_source not in ("gt", "pred", "fem"):
        raise ValueError(f"unknown flow_source {flow_source!r}; expected gt, pred or fem")
    if flow_source in ("pred", "fem"):
        if getattr(data, "u0_pred", None) is None:
            raise ValueError(f"pack has no u0_pred (flow_source={flow_source!r})")
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        # MLS on predicted velocity, not the kinematics shear head.  On comsol005 the
        # cached head has wall corr 0.17 vs GT MLS (median 54 1/s against 193) and
        # dsrx from that field never trips sgt, so the wall gate is empty.  MLS-on-u0
        # keeps wall corr 0.82.  ``sr0_pred`` stays on the pack for a later head.
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    else:
        u = data.y[time_index, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[time_index, :, 1].detach().cpu().numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)                                   # 1/(s*cm)
    dsrx = dsrx * dsrx_gain(flow_source)

    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM             # 1/(s*m) -> 1/(s*cm)
    return T0Fields(
        sr=sr, dsrx=dsrx,
        gate_low=(sr < float(bio_cfg.lss)).astype(np.float64),
        gate_sep=(dsrx < sgt_cgs).astype(np.float64),
        gate=gate_from_shear(sr, dsrx, bio_cfg),
        u=u, v=v,
    )


#: Washout coefficient, DIMENSIONLESS (it multiplies ``sr`` [1/s] to make a rate).  Fit as a
#: single global scalar on WALL_COHORT_V2_TRAIN by ``scripts/diag_mat_washout.py``; 16 of 19
#: vessels pick this value under leave-one-vessel-out, and the LOO gain is 0.310 -> 0.442
#: against the in-sample 0.464, so almost none of it is the fit reading its own answer.
WASHOUT_LAMBDA = 1.54e-6


def washout_step(mat: np.ndarray, source: np.ndarray, h: float, decay: np.ndarray):
    """One step of ``dMat/dt = source - decay*Mat``, unconditionally stable and positive.

    Backward-Euler in the removal term only:  ``mat <- (mat + h*source) / (1 + h*decay)``.

    NOT explicit Euler, and the difference is not cosmetic.  The stored timestep is 150 s and
    ``decay = lambda*sr`` reaches ~1e-2 1/s where the separation branch fires on fast flow, so
    ``h*decay`` passes 2 and an explicit update oscillates and then diverges on exactly the
    high-shear nodes this term exists to suppress.  Shared by the model and by
    ``scripts/diag_mat_washout.py`` so the fitted ``lambda`` transfers between them instead of
    silently meaning two different things.
    """
    return (mat + h * source) / (1.0 + h * np.asarray(decay, dtype=np.float64))


def wall_platelet_constants(data, bio_cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(rp, ap)`` at the wall in CGS [plt/cm^3], read from the t=0 initial condition.

    PHASE3_HANDOFF 1.3 / 26.16: both are spatially flat (CV 0.3% / 10%) and vary 0.2%
    across the cohort, so this is an initial condition, not a learned field.
    """
    names = data.y_channel_names.split(",")
    scales = bio_cfg.get_species_scales(device="cpu")
    rp_nd = torch.expm1(data.y[0, :, names.index("RP_log1p_nd")].clamp(-10, 8)).numpy()
    ap_nd = torch.expm1(data.y[0, :, names.index("AP_log1p_nd")].clamp(-10, 8)).numpy()
    rp = rp_nd * float(scales[0]) * PER_M3_TO_PER_CM3
    ap = ap_nd * float(scales[1]) * PER_M3_TO_PER_CM3
    return rp, ap


def graded_gate(
    fields: T0Fields,
    bio_cfg,
    *,
    mode: str = "hard",
    tau_low: float = 0.25,
    tau_sep: float = 0.25,
) -> np.ndarray:
    """The law's bracket prefactor, either COMSOL's hard step or a graded surrogate.

    WHY GRADE A LAW WHOSE GATE IS PROVABLY A HARD STEP.  COMSOL's gate is evaluated on
    the *current* shear at every step; this model freezes it at t=0.  A node sitting just
    below ``lss`` at t=0 is the one most likely to leave the gate as the clot narrows the
    lumen and accelerates the flow, while a node deep inside a stagnation zone stays
    gated for the whole run.  So the correct t=0 surrogate for the *time-averaged* gate is
    not the t=0 indicator, it is a decreasing function of the margin.  ``tau_*`` are the
    margins in units of the thresholds themselves (``temp = tau * lss``), i.e.
    dimensionless, so they transfer across vessels.

    ``mode='hard'`` reproduces ``t0_flow_fields``'s gate exactly.
    """
    lss = float(bio_cfg.lss)
    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM
    L_cm = float(bio_cfg.L_char) * M_TO_CM
    coef = L_cm / float(bio_cfg.gamma_m)
    def soft(x, thresh, tau, scale):
        t = max(tau * scale, 1e-12)
        return 1.0 / (1.0 + np.exp(np.clip((x - thresh) / t, -50, 50)))

    if mode == "hard":
        g_low, g_sep = fields.gate_low, fields.gate_sep
    elif mode == "sigmoid":
        g_low = soft(fields.sr, lss, tau_low, lss)
        g_sep = soft(fields.dsrx, sgt_cgs, tau_sep, abs(sgt_cgs))
    elif mode == "sigmoid_low":
        # Grade only the stagnation branch: the separation branch already carries a
        # magnitude through |dsrx|, so it is not the one that flashes.
        g_low = soft(fields.sr, lss, tau_low, lss)
        g_sep = fields.gate_sep
    else:
        raise ValueError(f"unknown gate mode {mode!r}")
    return g_sep * coef * np.abs(fields.dsrx) + g_low


def integrate_mat_trajectory(
    data,
    bio_cfg,
    gate: np.ndarray,
    *,
    da_scale: float = 40.0,
    da_scale_auto: float | None = None,
    blockage=None,
    species=None,
    ap_boost=None,
    ap_closure=None,
    washout: float = 0.0,
    washout_sr: np.ndarray | None = None,
    wall_ap_renewal=None,
    wall_ap_fields=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the surface ODEs, returning ``Mat`` at EVERY timestep.

    ``wall_ap_renewal`` -- optional :class:`src.core_physics.wall_ap_renewal.WallApRenewal`
        instance.  When set, the time-varying wall-AP ODE (:mod:`wall_ap_renewal`) is
        resolved first and the result is used as ``species``, overriding any explicit
        ``species`` argument and clearing ``ap_closure`` (the dynamic field supersedes the
        static Damkohler correction).  ``wall_ap_fields`` must be provided alongside it
        (a :class:`T0Fields` instance carrying ``u``/``v`` for the upwind transport solve).

        Both parameters default to ``None`` — when absent the function is bit-identical to
        the previous signature and every existing caller is unaffected.


    ``da_scale`` defaults to 40, not the 100 the final-mask sweep settled on.  Every value
    above ~50 gives a bit-identical committed set (docs/PHASE3_RESULTS.md 3), so the mask
    metric could not distinguish them -- but the growth CURVE can: 100 -> 40 takes
    ``curve_l1`` from 0.2998 to 0.1018 at an unchanged deploy score (13.1).

    ``da_scale_auto`` -- separate rate scalar for the AUTOCATALYTIC term.  ``None`` means
        "same as ``da_scale``", which is bit-identical to the one-scalar model.
        WHY THIS EXISTS: COMSOL's own export never supported one scalar.  Refitting the two
        terms independently across 19 TRAIN vessels (``scripts/diag_damkohler_cohort.py``)
        gives ``A_s/Da`` median 20.7 and ``A_a/Da`` median 67.6 -- a ratio of **3.07**,
        positive in 15/19 vessels, and the same signature COMSOL's own numbers carry
        (``d(Mas,t)/J0_Mas`` = 25.8 against ``d(Mat,t)/J0_Mat`` = 145.6).  The single
        ``da_scale`` was absorbing the smaller of the two.  This matters for TIMING and not
        for the mask: the autocatalytic term is what decides how long a node idles below
        ``crit`` before it runs away, so the ratio sets the delay between the first
        deposition and the commitment that the score sees.

    ``species`` -- ``(rp, ap)`` in CGS, each ``[N]`` (frozen) or ``[T, N]`` (time-varying).
        Defaults to the t=0 constants.  Passing GT trajectories makes this a CHEMISTRY
        oracle, the analogue of the time-varying-gate flow oracle in 13.4.
    ``ap_boost`` -- optional ``(mat, step) -> multiplier [N]``, the thrombin coupling:
        committed nodes generate thrombin, thrombin activates platelets, and ``k_as`` is
        12x ``k_rs``.  This is the mechanism the ad-hoc graph-growth term stands in for.
    ``ap_closure`` -- optional ``(gate, sat, mas, mat) -> multiplier [N]``, the wall-AP
        Damkohler balance (:mod:`src.core_physics.ap_closure`).  Applied AFTER ``ap_boost``
        and evaluated on the rollout's OWN surface state, so it is self-consistent.  This
        is what breaks the flash: with ``ap`` frozen and uniform, every ``gate == 1`` node
        integrates the identical ODE and they all cross ``crit`` in the same step.
        Leaving it ``None`` reproduces the frozen-``ap`` trajectory bit-for-bit.

    ``blockage`` -- optional callable ``(mat, gate0) -> gate`` applied every step, used by
    the shear-redistribution arm to let the growing clot close its own gates.

    ``washout`` -- dimensionless coefficient on the REMOVAL term ``- washout*sr*Mat``, with
        ``washout_sr`` the per-node shear rate [1/s] (normally ``T0Fields.sr``).  ``0.0``
        reproduces the accumulate-only trajectory bit-for-bit.  ``washout_sr`` may be
        static ``[N]``, time-varying ``[T, N]``, or a callable ``(mat, step) -> [N]`` so a
        wake/blockage can update the sink on the same committed state the source sees.

        THIS IS THE ONE STRUCTURAL TERM THE LAW WAS MISSING, and it is missing because the
        repo treats ``Mat`` as a surface coverage.  It is not: in the ``.mph`` it is a
        *Transport of Diluted Species* DOMAIN concentration on ``tds2``, with convection
        enabled (nonconservative form, Do Carmo and Galeao crosswind stabilisation), sourced
        at the wall by the ``J0_Mat`` flux.  Material deposited at the wall therefore sits in
        the near-wall fluid and is carried off by the flow; accumulating it forever, as this
        function did, has no removal channel and no steady state.

        WHY IT MATTERS MORE THAN ANY RATE SCALAR.  Handed a perfect oracle -- GT ``RP``,
        ``AP``, ``M``, ``Mas``, ``sr`` and ``d(sr,x)`` at every timestep -- the accumulate-only
        ODE still ranks GT ``Mat`` at only 0.31 on live wall nodes, and is ANTI-correlated on
        5 of 19 train vessels (``scripts/diag_local_ode_closure.py``).  No input model and no
        choice of ``da_scale`` can fix that, because the deficit is in the equation.  The
        removal term takes the same oracle to 0.464 in-sample and 0.442 leave-one-vessel-out.

        WHY IT IS PROPORTIONAL TO ``sr`` AND NOT A BARE LIFETIME.  The gate has two branches.
        The stagnation branch fires where ``sr < lss``, so those nodes deposit AND retain.
        The separation branch fires on ``d(sr,x) < sgt``, which happens at reattachment points
        where ``sr`` itself is large -- those nodes deposit and are immediately scoured.  With
        no removal the model ranks the second group far too high, and since that branch is the
        one carrying a magnitude (``(L/gamma_m)*|dsrx|`` reaches ~1.5) it dominates the
        predicted ordering.  Measured against the two cheaper stories: a bare lifetime
        ``-lam*Mat`` reaches 0.431 and pure saturation ``J0*(1-Mat/Msat)`` reaches 0.310,
        i.e. exactly nothing.  Saturation is dead; flow-proportional removal beats a bare
        lifetime by 0.033, which is a thin margin on 19 vessels -- the strong claim here is
        that removal exists, and the ``sr`` scaling is the better of two live options and the
        only one with a mechanism in the model tree.

    Returns ``(traj [T, N], t [T])`` in COMSOL model units.
    """
    # ---- wall_ap_renewal convenience entry point ---------------------------
    # Lazy import avoids the circular dependency: wall_ap_renewal imports from
    # physics_wall_model (T0Fields, wall_platelet_constants), so physics_wall_model
    # must not import wall_ap_renewal at module level.
    if wall_ap_renewal is not None:
        from src.core_physics.wall_ap_renewal import make_species_from_renewal  # noqa: PLC0415
        if species is not None:
            raise ValueError(
                "pass either `species` or `wall_ap_renewal`, not both; "
                "`wall_ap_renewal` resolves to a species array internally"
            )
        if wall_ap_fields is None:
            raise ValueError(
                "`wall_ap_renewal` requires `wall_ap_fields` (a T0Fields instance "
                "carrying u/v for the upwind transport solve)"
            )
        species = make_species_from_renewal(data, bio_cfg, wall_ap_fields,
                                            renewal=wall_ap_renewal)
        ap_closure = None   # dynamic field supersedes the static Damkohler correction

    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)
    da_a = da if da_scale_auto is None else float(bio_cfg.surface_damkohler) * float(da_scale_auto)
    if species is None:
        rp, ap = wall_platelet_constants(data, bio_cfg)
    else:
        rp, ap = species
    rp = np.asarray(rp, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)
    t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gate0 = gate
    n = gate0.shape[0]
    mas = np.zeros(n)
    mat = np.zeros(n)
    traj = np.zeros((len(t), n))
    gate_s = float(bio_cfg.surface_time_gate_s)
    slope = float(bio_cfg.surface_time_gate_slope)
    lam = float(washout)
    if lam != 0.0 and washout_sr is None:
        raise ValueError("washout != 0 needs washout_sr (the per-node shear rate)")
    wsr_static = None
    if lam != 0.0 and not callable(washout_sr):
        wsr_arr = np.asarray(washout_sr, dtype=np.float64)
        if wsr_arr.ndim not in (1, 2):
            raise ValueError("washout_sr must be [N], [T, N], or a callable")
        wsr_static = wsr_arr
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        g = gate0 if blockage is None else blockage(mat, gate0, i)
        rp_i = rp[i] if rp.ndim == 2 else rp
        ap_i = ap[i] if ap.ndim == 2 else ap
        if ap_boost is not None:
            ap_i = ap_i * ap_boost(mat, i)
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        if ap_closure is not None:
            ap_i = ap_i * ap_closure(g, sat, mas, mat)
        dep = sat * (k_rs * rp_i + k_as * ap_i)
        auto = (mas / minf) * k_aa * ap_i
        mas = mas + h * da * g * dep * step2t
        src = g * (da * dep + da_a * auto) * step2t
        if lam == 0.0:
            mat = mat + h * src
        else:
            if callable(washout_sr):
                sr_i = np.asarray(washout_sr(mat, i), dtype=np.float64).reshape(-1)
            elif wsr_static.ndim == 2:
                sr_i = wsr_static[min(i, wsr_static.shape[0] - 1)].reshape(-1)
            else:
                sr_i = wsr_static.reshape(-1)
            mat = washout_step(mat, src, h, lam * np.abs(sr_i))
        traj[i + 1] = mat
    return traj, t


def first_crossing(traj: np.ndarray, thresh: float) -> np.ndarray:
    """[T,N] -> per-node index of the first crossing of ``thresh``, or -1 if never."""
    hot = traj >= thresh
    any_hot = hot.any(axis=0)
    idx = np.where(any_hot, hot.argmax(axis=0), -1)
    return idx


def integrate_mat(
    data,
    bio_cfg,
    fields: T0Fields,
    *,
    da_scale: float = 1.0,
    wall_only: bool = True,
) -> np.ndarray:
    """Integrate the COMSOL surface ODEs with the t=0 gates held fixed.

    ``dMas/dt = Da*gate*Sat*(k_rs*rp + k_as*ap)*step2t``
    ``dMat/dt = Da*gate*(Sat*(k_rs*rp + k_as*ap) + (Mas/Minf)*k_aa*ap)*step2t``
    ``Sat = 1 - Mas/Minf``   (verified against the exported ``Sat(M)`` column, rel 1.8e-12)

    Returns ``Mat`` at the final time in COMSOL model units (compare to
    ``viscosity_mat_crit`` = 2e7).
    """
    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)

    rp, ap = wall_platelet_constants(data, bio_cfg)
    t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gate = fields.gate.copy()
    if wall_only:
        gate = gate * data.mask_wall.reshape(-1).bool().cpu().numpy()

    n = gate.shape[0]
    mas = np.zeros(n)
    mat = np.zeros(n)
    gate_s = float(bio_cfg.surface_time_gate_s)
    slope = float(bio_cfg.surface_time_gate_slope)
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        dep = sat * (k_rs * rp + k_as * ap)
        auto = (mas / minf) * k_aa * ap
        mas = mas + h * da * gate * dep * step2t
        mat = mat + h * da * gate * (dep + auto) * step2t
    return mat




def _wall_adjacency(data):
    import scipy.sparse as sp

    ei = data.edge_index.detach().cpu().numpy()
    n = int(data.num_nodes)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def predicted_seed_mask(data, bio_cfg, fields, *, relax=2.0, grow_hops=6, adj=None):
    """The shipped t=0 prediction: both gates, then shear-admitted graph growth."""
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    A = _wall_adjacency(data) if adj is None else adj
    cur = (fields.gate > 0) & wall
    adm = (fields.sr < float(bio_cfg.lss) * float(relax)) & wall
    for _ in range(int(grow_hops)):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur, adm, A


# ---------------------------------------------------------------------------
# The gelation shear collapse: one measured constant, and its measured limits
# ---------------------------------------------------------------------------
#
# WHAT IT IS.  `sr/sr0` is 1.000 before gelation and **0.1226** after: p25-p75 0.113-0.136,
# 584 observations across all three wound vessels (MODEL_REVIEW_2026-08-22 9e.4, memory
# `gelation-steps-shear-to-one-eighth`).  Tight, and genuinely measured.
#
# WHAT IT IS NOT.  It is NOT a general clot->flow blockage law, and the temptation to use it as
# one has been tested and refused.  On a 96-case FEM sweep with injected clots
# (`outputs/pi_corpus`, occlusion to 75%, mu 0.1-3.0 Pa.s) the case-median ratio spans
# **0.004 to 19.7** -- it exceeds 1.0 outright where flux redistribution accelerates the
# residual lumen (`clot-shear-map-is-non-monotone`).
#
# AND AN OCCLUSION-DEPENDENT REPLACEMENT WAS FITTED AND REJECTED.  Against the obvious
# candidate `A(dmu) * (1/h_eff)^p`, at case level, 93 cases, 12 vessels:
#
#     const (this value)                 median|log err| 0.524
#     A(dmu) only                                        0.659
#     A(dmu)*(1/h_eff)^p, best fit                       0.529   corr +0.089
#     LEAVE-ONE-VESSEL-OUT:   2-param 0.775   vs   const 0.611
#
# The two-parameter law is WORSE out of sample than the constant and has no correlation with
# truth.  So a constant is the best estimator available, and the honest response is to bound
# its use rather than to dress it up as a law.  Do not replace this with a fitted function
# without beating 0.611 LOVO first.
#
# VALIDITY.  Anchored on WOUND vessels at gelation, i.e. a specific clot viscosity at
# essentially unoccluded geometry.  Applying it at high occlusion is extrapolation, and the
# sweep above says the error there is of order the value itself.

#: Measured post-gelation wall-shear ratio (wound vessels, at gelation).  See the block above
#: for what this does and does not license.
GELATION_SR_RATIO = 0.1226

#: Interquartile range of the measurement, carried so callers can reason about its spread
#: instead of treating the median as exact.
GELATION_SR_RATIO_IQR = (0.113, 0.136)

#: Case-median dispersion of `sr/sr0` over the synthetic severe-occlusion corpus.  NOT a
#: confidence interval on the constant -- it is the range the constant does not cover.
GELATION_SR_RATIO_SWEEP_RANGE = (0.004, 19.7)


def gelation_sr_ratio(default: float | None = None) -> float:
    """The gelation shear ratio, overridable at runtime.

    Reads `CLOT_GELATION_SR_RATIO` so a caller can substitute a value measured on THEIR
    population rather than silently inheriting the wound-vessel anchor.  Returns
    :data:`GELATION_SR_RATIO` when unset.
    """
    import os as _os

    if default is not None:
        return float(default)
    raw = (_os.environ.get("CLOT_GELATION_SR_RATIO") or "").strip()
    if raw:
        try:
            v = float(raw)
        except ValueError:
            return GELATION_SR_RATIO
        if v > 0.0:
            return v
    return GELATION_SR_RATIO


def oracle_blockage(data, bio_cfg, fields, *, hops: int = 3, ratio: float | None = None,
                    every: int = 1, phys_cfg=None):
    """ORACLE ``blockage`` callable: the best any flow corrector could ever do.

    NOT DEPLOY-LEGAL AND NOT MEANT TO BE.  At rollout step ``i`` it reads the **ground-truth**
    clot occupancy at pack time ``i`` and applies the measured gelation collapse
    ``sr <- sr * ratio`` there, re-differentiates along the wall with the SAME MLS operator
    the consumer uses (so no stencil mismatch enters), and re-evaluates the deposition gate.

    This is the upper bound for the whole corrector programme: perfect clot localisation
    **and** the correct shear response, with no model error at all.  If the deploy score does
    not move here, no corrector -- learned, analytic or otherwise -- can pay, and the
    clot->flow loop should be closed as a research direction rather than rebuilt.

    Hysteresis: an occluded node never loses the gate it already had.
    """
    from src.config import PhysicsConfig
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    ratio = gelation_sr_ratio(ratio)
    phys = phys_cfg if phys_cfg is not None else PhysicsConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, _ = build_mls_gradient(pos, ei, hops=hops)
    d_bar = float(data.d_bar.reshape(-1)[0])
    sr0 = np.asarray(fields.sr, dtype=np.float64)
    nt = int(data.y.shape[0])
    state = {"gate": None, "last": -(10 ** 9), "calls": 0, "occ": 0}

    def blockage(mat, gate0, i):
        if state["gate"] is not None and i - state["last"] < int(every):
            return state["gate"]
        ti = max(0, min(int(i), nt - 1))
        occ = gt_clot_phi_at_time(data, ti, phys).detach().cpu().numpy().reshape(-1) > 0.5
        if not occ.any():
            g = gate0
        else:
            sr = np.where(occ, sr0 * float(ratio), sr0)
            dsx = (Dx @ sr) / (d_bar * M_TO_CM)
            g = gate_from_shear(sr, dsx, bio_cfg, wall=wall)
            g = np.where(occ, np.maximum(g, gate0), g)
        state["gate"], state["last"] = g, i
        state["calls"] += 1
        state["occ"] = int(occ.sum())
        return g

    blockage.state = state
    return blockage


def predict_phi(
    data,
    bio_cfg,
    mode: str = "phi",
    *,
    hops: int = 3,
    da_scale: float = 1.0,
    time_index: int = 0,
    flow_source: str = "gt",
) -> tuple[torch.Tensor, T0Fields, np.ndarray | None]:
    """Binary wall-clot prediction ``phi_pred`` [N] plus the intermediates."""
    del flow_source  # reserved for callers that pass deploy-flow kwargs uniformly
    fields = t0_flow_fields(data, bio_cfg, hops=hops, time_index=time_index)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    if mode == "gate":
        pred = (fields.gate > 0) & wall
        return torch.tensor(pred.astype(np.float32)), fields, None
    mat = integrate_mat(data, bio_cfg, fields, da_scale=da_scale)
    pred = (mat >= float(bio_cfg.viscosity_mat_crit)) & wall
    return torch.tensor(pred.astype(np.float32)), fields, mat


#: Where the never-igniting wall nodes sit, relative to the median igniter onset, as a
#: fraction of the horizon; and how far they spread around it.  Fitted on FIT vessels,
#: leave-one-vessel-out selects (0.25, 0.6) on 8 of 13 folds and (0.35, 0.8) on 4.
STITCH_OFFSET = 0.25
STITCH_SPREAD = 0.6


def stitch_onset(
    onset: np.ndarray,
    ignited: np.ndarray,
    node_set: np.ndarray,
    sr: np.ndarray,
    n_times: int,
    *,
    offset: float = STITCH_OFFSET,
    spread: float = STITCH_SPREAD,
) -> np.ndarray:
    """Schedule the wall nodes the surface ODE never ignites, instead of one constant.

    A fifth (FIT) to nearly half (DEV) of the wall nodes that clot in GT never cross ``crit``
    in the ODE -- they all have ``gate == 0``, so their trajectory never moves -- and the
    scoring convention hands every one of them the SAME number, the median igniter onset.
    Two things are wrong with that number, both measured on 13 full-horizon cohort vessels:

    **It is in the wrong place.**  Those nodes' true median onset is **+0.267 of the horizon
    later** than the constant they are given (FIT mean; DEV +0.196), and the sign is the same
    on **13 of 13 vessels**, ranging +0.060 to +0.363.  Physically that is what it should be:
    these are the nodes outside the admission gate, reached later by growth.

    **It has no spread.**  PHASE6_RESULTS 12 priced the wall model's compressed onset
    distribution at about -0.13; a single constant is the extreme case of it.  ``sr`` at t=0
    orders these nodes' true onset at rank **+0.809 FIT / +0.658 DEV** -- better than hop
    distance to the nearest igniter (+0.50 / +0.34), which is why the obvious front-arrival
    model is the wrong shape and was measured and dropped.

    So: keep the ODE's own onset where it exists, and give the rest a shear-ordered spread
    about a shifted centre.  Mean-over-time wall deploy score, GT wall set so this isolates
    timing from the mask, 13 vessels:

        median-igniter constant (ships)              0.8164
        best degenerate null, all stitch at 0.75*T   0.8417
        offset only, no spread                       0.8478
        THIS, offset 0.25 spread 0.6                 0.8636
        the same spread with a RANDOM order          0.8339   <- worse than a plain shift
        GT on the stitch nodes (ceiling)             0.9011

    Leave-one-vessel-out, refitting ``(offset, spread)`` on the other 12 each time:
    **+0.0422, 95% CI [+0.0252, +0.0613], P(delta<=0) = 0.0000, positive on 12 of 13** --
    the lower bound clears the +-0.024 wall noise floor.  It collects 55% of the stitch prize.

    CAVEAT, and it is a real one.  Measured on a set held at the GT wall set.  On a
    *predicted* set (igniters plus 20-hop along-wall growth) the apparent gain is far larger,
    +0.176, but **94% of that is a degenerate null** -- simply not showing graph-grown nodes
    until the end of the horizon -- which is a precision effect on an over-grown mask, not
    this timing model.  Do not quote the predicted-set number as a timing result.

    ``onset`` and the return are in grid-step units; ``node_set`` is the mask being scheduled.
    """
    out = np.asarray(onset, dtype=np.float64).copy()
    ign = np.asarray(ignited, dtype=bool)
    sel = np.asarray(node_set, dtype=bool)
    live = sel & ign
    if not live.any():
        return out
    centre = float(np.median(out[live])) + float(offset) * float(n_times)
    stitch = sel & ~ign
    k = int(stitch.sum())
    if k == 0:
        return out
    out[stitch] = centre
    if spread > 0 and k > 1:
        rank = np.argsort(np.argsort(np.asarray(sr, dtype=np.float64)[stitch])) / (k - 1)
        out[stitch] = centre + float(spread) * (rank - 0.5) * float(n_times)
    out[stitch] = np.clip(out[stitch], 0.0, float(n_times) - 1.0)
    return out
