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

# Former environment overrides that nothing in the tree ever set and no doc
# named, so each always resolved to the value below.  Kept as named constants
# rather than inlined literals so the value stays greppable and explainable.
CLOT_PRED_HOPS = ""


DEFAULT_ATTENUATION = 0.16


#: MLS stencil width per flow source.  GT is differentiated at the consumer's own hops=3;
#: any RECONSTRUCTED field needs a wider stencil to keep its second derivative from being its
#: own sign flip.  `fem` is a converged field like `gt` -- it does NOT need the extra smoothing
#: and takes the GT treatment (hops=3).  Confirmed 2026-09-01 by diag: fem h3 gateJ 0.908 vs
#: h6 gateJ 0.062-0.67 (the shipped path scored 0.520 at h6 g3.0).
#: `pred` uses hops=6 to match `features.py` (features.py was 6, temporal.py was 4 -- aligned
#: here so the chemistry/ODE path and the feature builder use the same stencil for the same arm).
#: `rgp` is the FEM prior plus a band-localised residual, so it inherits `fem`'s stencil.
from src.core_physics.flow_sources import HOPS as _FLOW_HOPS


def _flow_hops(flow: str) -> int:
    """`CLOT_PRED_HOPS` overrides the stencil used for a RECONSTRUCTED field.

    The wide stencil exists to stop a noisy surrogate's second derivative flipping sign; it is
    not obviously right for an accurate field, and it differs from the hops=3 the LABELS were
    differentiated at.  Overridable so the two can be measured apart.
    """
    import os

    raw = CLOT_PRED_HOPS.strip()
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
    source): **+22 wall TP / 0 wall FP** on ``wound_comsol003``, and a trickle of 8 nodes
    (6 TP / 2 FP) on each of ``wound_comsol001/002``, whose near-wound wall is already
    gated.  Net wall deploy score +0.0246 / +0.0026 / +0.0060.

    Reached only through :func:`~src.clot_ml.locked.predict_temporal_v4_wound`, which
    short-circuits on ``has_wound``, so no cohort pack ever sees it -- the seeded stall is
    NOT inert on a no-wound pack (it fires on committed healthy wall too: 29 extra on
    ``comsol012``), it is simply unreachable there.

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



# --- strict-CV temporal features -------------------------------------------------
# Moved out of `scripts/eval_strict_temporal.py`. `clot_ml.locked.predict_temporal_v4`
# builds its features with these, so a promoted temporal artifact cannot be scored
# without them -- they are library code that happened to live in a script.

#: The owner-onset contour level the ODE wall series is read at, held in a
#: one-element list so callers can retune it in place.  `scripts/eval_strict_temporal.py`
#: sweeps it per fold and writes `ANCHOR_LEVEL[0]`; `ode_wall_series` reads it.
#: Shared mutable state, kept beside its reader rather than in a script.
ANCHOR_LEVEL = [1.0]

#: Per-anchor predicted owner-onset, keyed by anchor stem.  The timing head is fed the
#: MODEL's estimate of when a node's owner commits rather than the ODE's, which is biased
#: low (PHASE9 12.2: "crit/att is simply unreachable").  Populated by the caller before
#: `time_block` runs -- `scripts/eval_strict_temporal.py` fills it per fold.  Shared
#: mutable state, which is not lovely, but it lives with its reader now instead of being
#: a global in a script the library imported.
OWNER_PRED: dict = {}


def lag_features(V, a, oofs):
    """Node features for the lag regression, plus the PHYSICS' own predicted lag.

    The transport solve already answers the question the lag model is asking.  For each node
    it gives the grid step at which the advected field crosses `crit`, and for that node's
    owner the step at which the wall ODE does; their difference is the lag the physics
    predicts, with no fitting at all.  Handing the regression that residual target is much
    easier than making it rediscover the boundary-layer filling time from static features.
    """
    X = node_features(V, a, oofs)
    v, S = V[a], V[a]["S"]
    if v["tt"] is None:
        return X
    thr = float(np.log(2.0))
    n_t = len(v["times"])

    def first_cross(F):
        hot = F >= thr
        return np.where(hot.any(0), hot.argmax(0), n_t).astype(np.float32)

    t_adv = first_cross(v["tt"]["mat_adv_t"])          # when transport says THIS node fires
    t_own = first_cross(v["tt"]["mat_self_t"])[S["owner"]]   # ... and when its owner does
    return np.concatenate([X, t_adv[:, None], t_own[:, None],
                           (t_adv - t_own)[:, None]], axis=1)


def node_features(V, a, oofs):
    """Static per-node block: the cached features, the t=0 rate, the ODE onset, and EVERY
    arm's out-of-fold score as its own column (the head can weigh them itself)."""
    v, S = V[a], V[a]["S"]
    cols = [S["X"], np.log1p(np.maximum(v["r0"], 0))[:, None], (v["oon"] / v["T"])[:, None]]
    cols += [oofs[arm][a][:, None] for arm in sorted(oofs)]
    return np.concatenate(cols, axis=1)


def ode_wall_series(V, a, gm, n_t):
    """Wall commit series taken from the ODE's own crossing -- the physics anchor."""
    v, S = V[a], V[a]["S"]
    oi = np.searchsorted(np.asarray(v["times"]), v["oon_c"][ANCHOR_LEVEL[0]], side="left")
    M = np.zeros((n_t, len(S["wall"])), dtype=bool)
    for j in range(n_t):
        M[j] = gm & S["wall"] & (oi <= j)
    return M


def offwall_by_learned_lag(M_wall, gm, owner, wall, lag_per_node, commit_final=True,
                           times=None, horizon=None):
    """As :func:`offwall_by_lag` but with a per-node lag instead of a cohort constant.

    ``times``/``horizon`` switch the lag from WHOLE GRID STEPS to a continuous fraction of
    the run.  The quantised form is why refining the regression was measured EXACTLY neutral
    to four decimals: the lag is rounded to one of eleven steps, so a better prediction never
    crosses a step boundary and the mask does not change.  Continuous lags are added to the
    owner's commit TIME and compared against the real grid times, so a predicted 0.36 T and a
    0.44 T land on different steps where 4-vs-4 steps could not.
    """
    T, N = M_wall.shape
    won = np.full(N, T, dtype=int)
    for j in range(T - 1, -1, -1):
        won[M_wall[j]] = j
    if times is None or horizon is None:
        on_idx = np.clip(won[owner] + np.rint(lag_per_node).astype(int), 0, T)
        M = np.zeros((T, N), dtype=bool)
        for j in range(T):
            M[j] = gm & ~wall & (on_idx <= j)
    else:
        tt = np.asarray(times, dtype=float)
        big = float(tt[-1]) + 10.0 * float(horizon)
        own_t = np.where(won[owner] < T, tt[np.clip(won[owner], 0, T - 1)], big)
        on_t = own_t + np.asarray(lag_per_node, dtype=float) * float(horizon)
        M = np.zeros((T, N), dtype=bool)
        for j in range(T):
            M[j] = gm & ~wall & (on_t <= tt[j])
    if commit_final:
        M[-1] = gm & ~wall
    return np.maximum.accumulate(M, axis=0)


def series_masks(gm, P, th, commit_final=True, owner=None, wall=None):
    """Committed mask at each time, with the two constraints the production law implies.

    MONOTONE: `P` is already a cumulative maximum, so a node never un-clots -- `J0_Mat >= 0`
    and there is no sink (PHASE7 12.1 measured wall `Mat` to be the exact time-integral of
    its own nodal derivative, rank 0.999).

    COMMIT BY THE END: every node in the committed set is clot at the last timestep.  This
    is a coherence constraint, not a new model -- the set *is* the prediction of the final
    mask, so a node that is in the set but still below the time cut at `t_final` is the
    readout contradicting itself.  v3 had no such constraint and paid for it: its extra
    probability filter DELETES correct nodes at the last step, which is exactly why the
    time-conditioned arm reads FINAL off-wall 0.6514 against the frozen set's 0.7075 on
    the same set.  With this, the final mask equals the set by construction, so the
    temporal arm can no longer lose to frozen at `t_final`.

    Whether to enforce it is chosen per domain inside the fold, because the probability
    filter is not purely a cost: on the wall it also removes low-confidence set members and
    measured +0.014 FINAL there, while off-wall it deletes real clot and costs -0.064.

    OWNER PRECEDENCE: an off-wall node is fed by its nearest wall node -- PHASE7 3.1
    measured that an off-wall GT node's owner is itself GT-committed **99.9%** of the time,
    and PHASE7 1.1 says why (the wall flux is the only source, `Mat` is advected from it).
    So an off-wall node cannot be clot before its owner is.  `src/clot_ml/locked.py`'s
    shipped `enforce_owner_and_monotone` applies this and the strict evaluator did not.
    """
    M = gm[None, :] & (P >= th)
    if commit_final:
        M[-1] = gm
    M = np.maximum.accumulate(M, axis=0)
    if owner is not None and wall is not None:
        keep = np.zeros(M.shape[1], dtype=bool)
        for j in range(M.shape[0]):
            m = M[j] | keep
            m &= (wall | m[owner])
            M[j] = m
            keep = m
    return M


def time_block(V, a, j, sel=None):
    """The per-(node, time) columns for grid index ``j``, optionally on a node subset.

    v3 had two: the normalised query time and the ODE's fired/not-fired bit.  The rest are
    the time-resolved physics of `scripts/build_temporal_transport.py` -- the node's own
    ODE `Mat(t)`, its owner's, and the advected off-wall field `mat_adv(t)`, all as
    continuous log values.  The advected channel is the first time-varying quantity the
    head has ever been given off the wall.
    """
    v = V[a]
    ti = v["times"][j]
    idx = slice(None) if sel is None else np.asarray(sel, dtype=bool)
    ode_now = (v["oon"][idx] <= ti).astype(np.float32).reshape(-1, 1)
    cols = [np.full((len(ode_now), 1), ti / v["T"], dtype=np.float32), ode_now]
    if v["tt"] is not None:
        for k in ("mat_self_t", "mat_owner_t", "mat_adv_t"):
            cols.append(v["tt"][k][j][idx].reshape(-1, 1))
    # --- the OWNER'S OWN PREDICTED STATE ----------------------------------------------
    # The head sees `mat_owner_t`, the owner's ODE `Mat(t)` -- but the ODE is biased low,
    # which is exactly why PHASE9 12.2's owner-threshold timing rule collapsed ("crit/att is
    # simply unreachable").  The MODEL's estimate of when the owner commits is much better
    # than the ODE's, and it is available: run the head once, read off the wall onsets, feed
    # them back.  This is the timing analogue of the `log_mat_owner` channel.
    if a in OWNER_PRED:
        op = OWNER_PRED[a]
        cols.append(op[j][idx].reshape(-1, 1))
        cols.append(op[-1][idx].reshape(-1, 1))
    # --- the PER-VESSEL PHYSICS CLOCK -------------------------------------------------
    # `ti / T` is a wall-clock fraction and says nothing about how far THIS vessel has got.
    # PHASE9 13.5 measured exactly this: replacing the ODE's per-vessel onset histogram with
    # a pooled cohort growth curve costs -0.081 wall, and concluded "the ODE's contribution
    # is not its ordering -- it is its per-vessel time CALIBRATION".  13.6/13.9 then showed a
    # known per-vessel schedule is worth +0.036 wall / +0.125 off over the ODE's.  These two
    # scalars are the deployable form of that calibration: how far along its OWN growth this
    # vessel is at `ti`, by the ODE and by the advected field, broadcast to every node.
    for c in v["clock"]:
        cols.append(np.full((len(ode_now), 1), c[j], dtype=np.float32))
    return np.concatenate(cols, axis=1)
