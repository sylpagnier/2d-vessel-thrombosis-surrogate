"""Wall-AP time-varying field via Damkohler sink + upwind graph renewal.

CONTEXT (docs/WOUND_PROGRESS.md §18).  The shipped model holds ``AP`` at its
spatially-uniform t=0 value.  COMSOL's own Transport of Diluted Species solve shows AP is
depleted to 18.9% of its inlet value on ``wound_patient003`` (CV 1.23 — 4× the dataset
maximum) because the wound sits in a stagnating zone where replenishment is slow and
consumption is fast.  A time-varying AP field is the missing input that drives the ODE
``Mat`` from its current 1.96× to the 20.22× crit needed by the off-wall readout.

WHAT FAILS WITHOUT THIS.  Sweeping ``da_scale_auto`` to 400 with frozen AP lifts wall
``Mat`` p90 to 16.5× crit but plateaus the far-field score at **0.6125** — because the
*pattern* of t=0 AP is wrong.  The frozen ApClosure gives a static quasi-steady correction
(AP_eq < AP0 at high-gate/low-shear nodes), but this correction never changes as the clot
grows and consumes more AP.  The time-varying model below starts at AP0 and depletes toward
the Damkohler equilibrium at the advective replenishment timescale.

MECHANISM.  At each solid-boundary node *i*, activated platelets obey a first-order
sink-plus-renewal balance:

    dAP_i / dt  =  −R_cons_i · AP_i  +  R_renew_i · AP0
                =  −(R_cons_i + R_renew_i) · AP_i  +  R_renew_i · AP0

    R_renew_i  =  renewal_scale / max(τ_i, Δt)          [1/s]
    R_cons_i   =  Da_i · R_renew_i                       [1/s]
    Da_i       =  C · gate_i · k_as / sr_i^q             (Damköhler, from SHIPPED ApClosure)
    τ_i        =  upwind advective residence time [s]    (src.clot_ml.transport)

Quasi-steady equilibrium: ``AP_eq_i / AP0 = 1 / (1 + Da_i)`` — reproduces the ApClosure
formula exactly, so the dynamic model subsumes the static correction.  The dynamics govern
*how fast* each node reaches equilibrium: fast flow (low τ) → quick replenishment;
stagnation (high τ) → slow, deep depletion.

LIMITS.
    renewal_scale = 0  →  R_renew = R_cons = 0  →  AP stays at AP0 forever
                          (bit-identical to passing no ``species`` / ``ap_closure=None``)
    renewal_scale = 1  →  full upwind renewal at the measured advective timescale

FIRST PASS — DECOUPLED.  The static t=0 gate enters ``Da_i``, so the full ``[T, N]``
AP trajectory is pre-computable before the Mat/Mas ODE runs.  Pass as::

    species = make_species_from_renewal(data, bio_cfg, fields, renewal=...)
    traj, t = integrate_mat_trajectory(..., species=species, ap_closure=None, ...)

The dynamic AP field supersedes the static ApClosure quasi-steady correction;
do NOT pass ``ap_closure`` alongside it or you will double-count.

INTEGRATE VIA `integrate_mat_trajectory`.  The convenience entry point is to pass
``wall_ap_renewal=WallApRenewal(...)`` and ``wall_ap_fields=fields`` to
``integrate_mat_trajectory``, which resolves the species internally and clears
``ap_closure``.

DEPLOY LEGALITY.  Transport uses ``fields.u`` / ``fields.v`` (t=0 ``u0_pred`` or GT at t=0
only).  No GT velocity or chemistry after t=0 enters the computation.

MEASURED ON ``wound_patient003`` (docs/WOUND_PROGRESS.md §18.2):
    GT-chem + gate + wash + da_scale_auto=123  →  far-field AUC 0.966, off 0.8512
    wall_ap_renewal + gate + wash + da_scale_auto=123  →  measured by diag_wall_ap_renewal.py

NEXT BUILD.  If far-field AUC on 003 ≥ 0.90, upwind renewal captures the mechanism and
the next step is calibration (``da_scale_auto=123``) + promotion.  If far-field AUC ≈ 0.86
(same as frozen), the GNN residual on the AP field is the fallback (§18.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core_physics.ap_closure import SHIPPED, ApClosure
from src.core_physics.physics_wall_model import T0Fields, wall_platelet_constants
from src.clot_ml.transport import residence_time as _upwind_residence_time

#: Safety floor on shear rate [1/s] for the Damköhler number (matches ap_closure.SR_FLOOR).
SR_FLOOR: float = 1e-3

#: Minimum advective residence time [s].  Below this the node is in fast flow and AP is
#: immediately replenished to AP0 — capping tau here is correct and avoids division by zero.
TAU_FLOOR_S: float = 1.0


@dataclass
class WallApRenewal:
    """Configuration for the time-varying wall-AP ODE.

    ``renewal_scale = 0``  →  frozen AP, bit-identical to the no-species baseline.
    ``renewal_scale = 1``  →  full upwind renewal at the measured advective timescale.

    Attributes
    ----------
    renewal_scale : float
        Dimensionless multiplier on the renewal rate.  0 disables the ODE entirely (frozen
        AP0).  Values > 1 are permitted and speed up the approach to equilibrium.
    closure : ApClosure | None
        The Damköhler balance used to set the quasi-steady AP equilibrium.  Defaults to
        ``SHIPPED`` (C=62.42, q=1, kernel='static') when ``None``.  Pass
        ``ApClosure(C=0)`` to eliminate the consumption term and use pure upwind relaxation
        toward AP0 everywhere.
    """

    renewal_scale: float = 0.0
    closure: ApClosure | None = None   # resolved to SHIPPED in __post_init__

    def __post_init__(self) -> None:
        if self.closure is None:
            self.closure = SHIPPED


def compute_wall_ap_trajectory(
    data,
    bio_cfg,
    fields: T0Fields,
    *,
    renewal: WallApRenewal | None = None,
    solid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-compute the time-varying wall AP field, returning ``[T, N]`` in CGS [plt/cm³].

    When ``renewal.renewal_scale == 0``, returns ``AP0`` broadcast to ``[T, N]``
    (no ODE integration; bit-identical to passing no ``species`` argument to
    ``integrate_mat_trajectory``).

    Parameters
    ----------
    data : torch_geometric.data.Data
        The pack.
    bio_cfg : BiochemConfig
        Kinetic constants.
    fields : T0Fields
        t=0 shear fields.  ``fields.u`` / ``fields.v`` are used for the upwind transport
        solve; if they are ``None`` (hand-built T0Fields), the function falls back to the
        t=0 GT velocity in ``data.y``.
    renewal : WallApRenewal, optional
        Defaults to ``WallApRenewal()`` (``renewal_scale=0`` → frozen AP).
    solid_mask : np.ndarray of bool, optional
        Union of ``mask_wall`` and ``mask_wound``.  Inferred from ``data`` when ``None``.
        Off-solid nodes keep AP0 throughout — their gate is 0, so their AP value does not
        affect ``integrate_mat_trajectory``'s source term.

    Returns
    -------
    np.ndarray
        Shape ``[T, N]``, CGS ``[plt/cm³]``.  Pass as ``species=(rp0, ap_traj)``
        to ``integrate_mat_trajectory`` with ``ap_closure=None``.
    """
    if renewal is None:
        renewal = WallApRenewal()

    _, ap0 = wall_platelet_constants(data, bio_cfg)
    t_arr = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    T = len(t_arr)
    N = int(data.num_nodes)

    # ---- fast path: frozen AP (renewal_scale = 0) --------------------------
    if renewal.renewal_scale == 0.0:
        return np.broadcast_to(ap0[None, :], (T, N)).copy()

    # ---- upwind advective residence time τ [s] ----------------------------
    u_ref = float(data.u_ref.reshape(-1)[0])          # m/s
    d_bar = float(data.d_bar.reshape(-1)[0])           # m

    # data.x[:, :2] are ND positions (divided by d_bar) → convert to [m]
    pos_m = data.x[:, :2].detach().cpu().numpy().astype(np.float64) * d_bar
    ei_np = data.edge_index.detach().cpu().numpy()

    if fields.u is not None:
        u_nd, v_nd = np.asarray(fields.u, np.float64), np.asarray(fields.v, np.float64)
    else:
        # fallback: t=0 GT velocity (oracle / hand-built T0Fields context)
        u_nd = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
        v_nd = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)

    u_ms = u_nd * u_ref        # ND → [m/s]
    v_ms = v_nd * u_ref

    horizon_s = max(float(t_arr[-1] - t_arr[0]), 1.0)           # [s]
    tau = _upwind_residence_time(pos_m, ei_np, u_ms, v_ms, horizon=horizon_s)  # [s]
    tau = np.maximum(tau, TAU_FLOOR_S)

    # ---- Damköhler number Da_i (static t=0 gate, SHIPPED closure) ---------
    cl = renewal.closure
    k_as_cgs = float(bio_cfg.k_as) * 100.0        # m/s → cm/s
    gate = np.asarray(fields.gate, dtype=np.float64)
    sr   = np.maximum(np.asarray(fields.sr, dtype=np.float64), SR_FLOOR)  # [1/s]
    Da   = cl.C * gate * k_as_cgs / np.power(sr, cl.q)     # dimensionless

    # ---- renewal / consumption rates [1/s] ---------------------------------
    R_renew = renewal.renewal_scale / tau    # AP renewal rate from upstream flow
    R_cons  = Da * R_renew                  # AP consumption rate at wall (Damköhler)
    R_total = R_renew + R_cons

    # ---- solid-boundary mask: wall ∪ wound ---------------------------------
    if solid_mask is not None:
        smask = np.asarray(solid_mask, dtype=bool)
    else:
        import torch
        smask = data.mask_wall.reshape(-1).bool().cpu().numpy()
        wnd = getattr(data, "mask_wound", None)
        if wnd is not None and torch.is_tensor(wnd) and wnd.numel() > 0:
            smask = smask | wnd.reshape(-1).bool().cpu().numpy()

    # ---- backward-Euler ODE integration ------------------------------------
    #   AP_i(t+h) = (AP_i(t) + h · R_renew_i · AP0_i) / (1 + h · R_total_i)
    #
    # Off-solid nodes: both rates are clamped to 0 → AP stays at AP0 exactly.
    # The backward-Euler scheme is unconditionally stable (same pattern as
    # washout_step in integrate_mat_trajectory).
    R_renew_s = np.where(smask, R_renew, 0.0)
    R_total_s = np.where(smask, R_total, 0.0)

    ap_traj = np.empty((T, N), dtype=np.float64)
    ap = ap0.copy()                 # initialise at uniform t=0 AP
    ap_traj[0] = ap

    for i in range(T - 1):
        h = t_arr[i + 1] - t_arr[i]
        ap = np.where(
            smask,
            (ap + h * R_renew_s * ap0) / (1.0 + h * R_total_s),
            ap0,     # off-solid: keep AP0 (gate=0 → no deposition → doesn't matter)
        )
        ap_traj[i + 1] = ap

    return ap_traj


def make_species_from_renewal(
    data,
    bio_cfg,
    fields: T0Fields,
    *,
    renewal: WallApRenewal | None = None,
    solid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(rp0, ap_traj)`` ready for the ``species`` argument of ``integrate_mat_trajectory``.

    ``rp0`` is the frozen t=0 constant ``[N]`` (no renewal model for RP — it is spatially
    uniform and shows no significant depletion in the cohort).

    ``ap_traj`` is the ``[T, N]`` time-varying AP field from
    :func:`compute_wall_ap_trajectory`.

    Usage::

        species = make_species_from_renewal(data, bio_cfg, fields,
                                             renewal=WallApRenewal(renewal_scale=1.0))
        traj, t = integrate_mat_trajectory(data, bio_cfg, gate,
                                            species=species, ap_closure=None, ...)

    Pass ``ap_closure=None`` — the dynamic field supersedes the static Damköhler
    correction and including both would double-count the quasi-steady term.
    """
    rp0, _ = wall_platelet_constants(data, bio_cfg)
    ap_traj = compute_wall_ap_trajectory(
        data, bio_cfg, fields, renewal=renewal, solid_mask=solid_mask)
    return rp0, ap_traj
