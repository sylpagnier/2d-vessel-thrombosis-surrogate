"""Time-resolved clot masks: the shipped SET, scheduled by the physics ODE's TIMING.

Measured motivation (`scripts/eda_timing_prize.py`, mean-over-time severity deploy score):

    frozen_model  (ships today)            wall 0.7921   off 0.5015
    frozen_oracle (perfect SET, no time)   wall 0.8190   off 0.5744
    physics_onset (zero-param ODE timing)  wall 0.8247   off 0.5015
    oracle_onset  (perfect timing)         wall 0.9897   off 1.0000

Two things follow.  Crude timing beats a *perfect* frozen set on the wall -- committing
everything at t=0 is worse than committing the wrong things at roughly the right times.
And the off-wall timing prize (+0.43) is completely untouched, because the ODE is a wall
object.

This module supplies both halves without training anything:

  wall     onset = the ODE's own first crossing of ``viscosity_mat_crit``
  off-wall onset = the time the node's OWNER wall trajectory crosses ``crit / attenuation``

The off-wall rule is the time-domain form of the measured 0.16 attenuation: if
``Mat_off(t) ~= att * Mat_owner(t)`` with ``att`` stable in time -- which
`scripts/eda_extrapolate.py` confirmed (0.004 -> 0.003 across the horizon) -- then an
off-wall node crosses ``crit`` exactly when its owner crosses ``crit / att``.  Same
trajectory, later threshold; no second model.

The SET is never taken from the ODE.  It stays whatever the caller supplies (the locked
GNN mask), because the GNN's set is materially better -- the ODE only supplies *when*.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_ATTENUATION = 0.16


#: MLS stencil width per flow source.  GT is differentiated at the consumer's own hops=3;
#: any RECONSTRUCTED field needs a wider stencil to keep its second derivative from being its
#: own sign flip.  `fem` is a converged field like `gt` -- it does NOT need the extra smoothing
#: and takes the GT treatment (hops=3).  Confirmed 2026-09-01 by diag: fem h3 gateJ 0.908 vs
#: h6 gateJ 0.062-0.67 (the shipped path scored 0.520 at h6 g3.0).
#: `pred` uses hops=6 to match `features.py` (features.py was 6, temporal.py was 4 -- aligned
#: here so the chemistry/ODE path and the feature builder use the same stencil for the same arm).
_FLOW_HOPS = {"gt": 3, "pred": 6, "fem": 3}


def _flow_hops(flow: str) -> int:
    """`CLOT_PRED_HOPS` overrides the stencil used for a RECONSTRUCTED field.

    The wide stencil exists to stop a noisy surrogate's second derivative flipping sign; it is
    not obviously right for an accurate field, and it differs from the hops=3 the LABELS were
    differentiated at.  Overridable so the two can be measured apart.
    """
    import os

    raw = os.environ.get("CLOT_PRED_HOPS", "").strip()
    if raw and flow != "gt":
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _FLOW_HOPS[flow]


def _first_crossing(traj: np.ndarray, thresh: float) -> np.ndarray:
    hot = traj >= thresh
    return np.where(hot.any(axis=0), hot.argmax(axis=0), -1)


def ode_trajectory(data, bio_cfg, *, flow: str = "gt", ap_closure: bool = True,
                   wake: bool = False, stall: bool = False, wound_source: bool = True,
                   wound_rate: tuple[float, float] | None = None):
    """The zero-parameter surface ODE's ``Mat`` trajectory ``[T, N]`` plus the time grid.

    ``wake`` closes the flow loop with :mod:`src.core_physics.gelation_wake`: committed
    tissue drops the shear its neighbours see, which opens their low-shear gate.  Worth
    **+0.0268 final / +0.0143 mean-over-time** on the wall domain over 26 vessels (better on
    18, worse on 7) measured on this trajectory directly.

    ``stall`` is the near-field alternative in :mod:`src.core_physics.near_stall`: committed
    SOLID (wall union wound) thickens the no-slip by ``STALL_HOPS`` (one corner shell) and
    neighbours take the ``mu1`` shear ratio on the low-shear branch only (``dsrx`` unscaled).
    The wound is a stall source from t=0.  Default OFF -- the wound *deploy* path unions
    ungated stall-wall ignitions into the shipped series without retraining the head.
    ``wake`` and ``stall`` cannot both be on: they are two operators for the same hook, not
    layers.

    ``wound_rate`` is the complement's fitted ``(G_pre, G_post)``.  ``wound_source`` gives the
    injured patch COMSOL's static ``srf2`` prefactor of 1; this runs it at the two-regime rate
    :func:`~src.clot_ml.wound.predict_wound_series` already uses, so the shared ODE and the
    complement stop disagreeing about the same patch by an order of magnitude
    (:func:`~src.clot_ml.wound.wound_rate_blockage`).  ``None`` reproduces the static
    prefactor bit-for-bit, and the knob is a structural no-op on a pack with no wound mask.

    ``wound_source`` includes the INJURED wall as a ``Mat`` source, and is ON by default
    because the alternative is wrong rather than conservative.  Without it the ODE
    integrates ``gate * mask_wall`` -- the healthy wall only -- so on a wound pack the wound
    contributes nothing, while carrying **50-88% of the vessel's total surface Mat**.  Every
    downstream transport feature inherits that hole: the advective source omits it, and the
    5.6-16.4% of the mesh whose nearest solid node is a wound node reads ``mat_owner_t = 0``
    for all time.  This is the same class of defect as WOUND_PROGRESS 6/6b (the wound encoded
    as lumen), one layer further up -- geometry and ownership were fixed there, the SOURCE
    was not.  Inert on any pack without a wound mask.

    **``wake`` and ``stall`` default to OFF, and that is not timidity.**  ``clot_gnn_v5``'s
    temporal head was trained on features derived from this function (``oon``,
    ``ode_wall_series``); switching the ODE underneath it is a train/deploy skew, not an
    improvement, until the head is retrained against it.  Turning either on is therefore a
    new artifact generation, not a flag flip.  With both False the trajectory is
    bit-identical to the shipped one.
    """
    if wake and stall:
        raise ValueError("ode_trajectory: pass at most one of wake=True, stall=True")
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import (
        deposition_gate, integrate_mat_trajectory, t0_flow_fields,
    )

    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio_cfg, hops=_flow_hops(flow), flow_source=flow)
    hook = make_rollout_hook(SHIPPED, bio_cfg, f.sr) if ap_closure else None
    gate = deposition_gate(data, f, wall=wall, wound_source=wound_source)
    blk = None
    if wake:
        from src.core_physics.gelation_wake import make_gelation_wake_blockage
        blk = make_gelation_wake_blockage(data, bio_cfg, f, wall)
    elif stall:
        from src.core_physics.near_stall import make_near_stall_blockage
        blk = make_near_stall_blockage(data, bio_cfg, f, wall=wall)
    if wound_rate is not None and wound_source:
        from src.clot_ml.wound import wound_rate_blockage
        blk = wound_rate_blockage(data, bio_cfg, g_pre=float(wound_rate[0]),
                                  g_post=float(wound_rate[1]), inner=blk)
    traj, t = integrate_mat_trajectory(data, bio_cfg, gate,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook,
                                       blockage=blk)
    return traj, np.asarray(t).reshape(-1)


def onset_from_ode(traj, mask, wall, pos, crit, *, attenuation=DEFAULT_ATTENUATION):
    """Per-node onset INDEX for the nodes in ``mask``; -1 elsewhere.

    Wall nodes take the ODE's own crossing of ``crit``.  Off-wall nodes take their owner's
    crossing of ``crit / attenuation``.  Masked nodes the ODE never ignites (the graph-grown
    ones) fall back to the median onset of those it does, which is the convention
    ``predict_wall_onset`` already uses.
    """
    T = traj.shape[0]
    on_w = _first_crossing(traj, crit)
    on_hi = _first_crossing(traj, crit / max(attenuation, 1e-9))

    widx = np.flatnonzero(wall)
    owner = widx[cKDTree(pos[wall]).query(pos)[1]] if len(widx) else np.zeros(len(wall), int)

    onset = np.full(len(wall), -1, dtype=int)
    onset[wall & mask] = on_w[wall & mask]
    off = (~wall) & mask
    onset[off] = on_hi[owner][off]

    # Fallback is taken from WALL-ignited nodes only.  Pooling wall and off-wall makes the
    # fallback depend on the off-wall rule, which silently changes the WALL score between
    # arms that should differ only off-wall.
    ignited = on_w[wall & mask & (on_w >= 0)]
    fallback = int(np.median(ignited)) if ignited.size else T - 1
    onset[mask & (onset < 0)] = fallback
    return onset


def mask_series(onset: np.ndarray, mask: np.ndarray, times) -> dict:
    """``time_index -> boolean mask``.  Nested by construction: clot never un-clots."""
    return {int(ti): mask & (onset >= 0) & (onset <= ti) for ti in times}


def union_ungated_stall_series(data, bio_cfg, series: dict, times, *, flow: str = "gt",
                               wound_rate: tuple[float, float] | None = None):
    """OR t=0-ungated wall that the stall ODE ignites into an existing series.

    Measured on ``clot_gnn_v5w`` (GT t=0 flow, ``STALL_HOPS = 2``, wound acting as a ``Mat``
    source): **+22 wall TP / 0 wall FP** on ``wound_patient003``, and a trickle of 8 nodes
    (6 TP / 2 FP) on each of ``wound_patient001/002``, whose near-wound wall is already
    gated.  Net wall deploy score +0.0246 / +0.0026 / +0.0060.

    Reached only through :func:`~src.clot_ml.locked.predict_temporal_v4_wound`, which
    short-circuits on ``has_wound``, so no cohort pack ever sees it -- the seeded stall is
    NOT inert on a no-wound pack (it fires on committed healthy wall too: 29 extra on
    ``patient012``), it is simply unreachable there.

    Does NOT add owner-basin lumen -- that is 450+ false positives on 003, and
    ``scripts/diag_wound_lumen_shell.py`` re-measured the whole shell family as losing even
    when seeded on the GT solid set.
    """
    from src.core_physics.physics_wall_model import t0_flow_fields

    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio_cfg, hops=_flow_hops(flow), flow_source=flow)
    ung = wall & (np.asarray(f.gate) * wall <= 0)
    if not ung.any():
        return series
    traj, _ = ode_trajectory(data, bio_cfg, flow=flow, stall=True, wound_rate=wound_rate)
    crit = float(bio_cfg.viscosity_mat_crit)
    T = int(traj.shape[0])
    out = dict(series)
    for ti in times:
        ti_c = int(np.clip(int(ti), 0, T - 1))
        key = int(ti)
        out[key] = np.asarray(out[key]) | (ung & (np.asarray(traj)[ti_c] >= crit))
    return out

