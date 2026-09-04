"""The v4 feature block: advective transport + the indicator-gate physics variant.

Factored out of `scripts/build_clot_ml_cache_v4.py` so the SHIPPED model can rebuild exactly
the same channels at deploy time -- `src/clot_ml/locked.py`'s `build_sample` produces the 55
v3 channels, and a `clot_gnn_v4` member expects 68.  Single source of truth: the cache
builder imports from here.

See `docs/PHASE10_V4.md` 5 for what these are and why the boundary-outflow term in
`src/clot_ml/transport.py` is what makes them work.
"""
from __future__ import annotations
from src.utils.units import M_TO_CM

import numpy as np

from src.clot_ml.transport import transport_fields


__all__ = ["indicator_physics", "horizon_for", "new_channels", "augment_sample",
           "V4_CHANNELS"]


#: MLS stencil width per flow source -- must match ``features.build_features``, which widens
#: the stencil on the noisier predicted field.  Kept here so the v4 block and the v3 block
#: differentiate the same velocity with the same operator.
HOPS_FOR_FLOW = {"gt": 3, "pred": 6, "fem": 3}


def indicator_physics(data, bio, wall, hops=None, *, flow: str = "gt"):
    """The backbone rerun with the separation branch as an INDICATOR (see module docs).

    ``flow`` selects the velocity field, exactly as ``features.build_features`` does.  It
    used to be hardwired to ``"gt"``, which meant a ``flow="pred"`` sample carried four
    GT-derived channels (``gate_ind``, ``log_mat_phys_ind``, ``onset_phys_ind``,
    ``log_mat_ind_owner``) plus a ``log_mat_adv_ind`` transporting a GT source -- silently
    optimistic and uninterpretable.  ``flow="gt"`` reproduces the old behaviour exactly.
    """
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import (
        deposition_gate, integrate_mat_trajectory, t0_flow_fields,
    )

    hops = HOPS_FOR_FLOW.get(flow, 3) if hops is None else int(hops)
    f0 = t0_flow_fields(data, bio, hops=hops, flow_source=flow)
    sgt = float(bio.sgt) / M_TO_CM
    gate_ind = (f0.dsrx < sgt).astype(np.float64) + (f0.sr < float(bio.lss)).astype(np.float64)
    hook = make_rollout_hook(SHIPPED, bio, f0.sr)
    # Same union as the backbone (docs/WOUND_PROGRESS.md 14.6): the indicator gate applies
    # to the healthy wall, the wound deposits ungated.  No-op without a wound mask.
    gate_full = deposition_gate(data, gate=gate_ind, wall=wall)
    traj, _ = integrate_mat_trajectory(data, bio, gate_full,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    crit = float(bio.viscosity_mat_crit)
    hot = traj >= crit
    onset = np.where(hot.any(0), hot.argmax(0), traj.shape[0]).astype(np.float32) / traj.shape[0]
    return traj[-1], onset, gate_ind


def horizon_for(pos, u, v, solid):
    """Domain-crossing time at the bulk speed -- the natural time unit for the transport.

    ``solid`` is excluded from the median because no-slip nodes sit at ~zero speed and are not
    "bulk".  Pass the wall/wound UNION: a wound node is no-slip too, and counting its zero
    speed into the bulk median lengthens the horizon spuriously.
    """
    L = float(np.ptp(pos[:, 0]) + np.ptp(pos[:, 1]))
    spd = float(np.median(np.hypot(u, v)[~solid])) + 1e-12
    return L / spd


def new_channels(S, mat_ind, onset_ind, gate_ind, crit) -> dict:
    # The transport BOUNDARY is the wall/wound union, not the healthy-wall label: `Mat` enters
    # the domain through every no-slip surface, and seeding the source on `mask_wall` alone
    # leaves the injured segment contributing nothing to the advection operator
    # (MODEL_REVIEW_2026-08-22 5b.3/5b.5).  Older samples carry no `solid` key; on those --
    # and on every no-wound pack -- the union IS `wall`, so this falls back exactly.
    solid = S.get("solid", S["wall"])
    ei, owner = S["edge_index"], S["owner"]
    pos = S["pos"].astype(np.float64)
    u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
    mat_phys = S["mat_phys"].astype(np.float64)
    H = horizon_for(pos, u, v, solid)

    T = transport_fields(pos, ei, u, v, solid, mat_phys, horizon=H)
    Ti = transport_fields(pos, ei, u, v, solid, np.asarray(mat_ind, float), horizon=H)

    def rel(a):
        """Value relative to the owner wall node's -- the attenuation, made dimensionless.

        This is the quantity PHASE7 12.5 asks for: `Mat_off/Mat_owner` has median 0.16 on
        every vessel but spans 0.12-0.19 *within* one, and near a threshold that spread is
        the whole off-wall gap.  Here it is computed from the flow rather than assumed.
        """
        return np.log1p(np.maximum(a, 0) / np.maximum(a[owner], 1e-30))

    tau = np.maximum(T["tau"], 0.0)
    return {
        # --- (A) advective transport -------------------------------------------------
        "log_mat_adv": np.log1p(np.maximum(T["mat_adv"], 0) / crit).astype(np.float32),
        "log_mat_adv_ind": np.log1p(np.maximum(Ti["mat_adv"], 0) / crit).astype(np.float32),
        "log_tau": np.log1p(tau / max(H, 1e-30)).astype(np.float32),
        "log_mat_adv_n": np.log1p(np.maximum(T["mat_adv_n"], 0) / crit).astype(np.float32),
        "log_src_reach": np.log1p(np.maximum(T["src_reach"], 0) / max(H, 1e-30)).astype(np.float32),
        # the flow-computed attenuation, and the same for pure wall-contact dose
        "att_adv": rel(T["mat_adv"]).astype(np.float32),
        "att_reach": rel(T["src_reach"]).astype(np.float32),
        "tau_rel_owner": rel(tau).astype(np.float32),
        # an absolute off-wall Mat estimate: owner's backbone Mat times the computed
        # attenuation, which is the shipped 0.16 rule with the constant made per-node
        "log_mat_off_est": np.log1p(
            np.maximum(mat_phys[owner] * np.minimum(
                np.maximum(T["mat_adv"], 0) / np.maximum(T["mat_adv"][owner], 1e-30), 4.0),
                0) / crit).astype(np.float32),
        # --- (B) separation branch as an indicator ------------------------------------
        "log_mat_phys_ind": np.log1p(np.maximum(mat_ind, 0) / crit).astype(np.float32),
        "onset_phys_ind": np.asarray(onset_ind, np.float32),
        "log_mat_ind_owner": np.log1p(
            np.maximum(np.asarray(mat_ind)[owner], 0) / crit).astype(np.float32),
        "gate_ind": np.asarray(gate_ind, np.float32),
    }


def augment_sample(data, S: dict, bio, *, flow: str = "gt") -> tuple[np.ndarray, list[str]]:
    """Extend a v3 sample's ``X``/``cols`` with the v4 channels, in the cache's own order.

    ``flow`` must match the flow the v3 block in ``S`` was built with -- the transport
    channels read ``S["u"]``/``S["v"]`` and the indicator channels re-derive the gate, so a
    mismatch produces a sample that is half predicted and half GT.
    """
    crit = float(bio.viscosity_mat_crit)
    # `wall`, not the union: `indicator_physics` re-runs the gated healthy-wall DEPOSITION LAW
    # (`srf1`).  The union belongs to the geometry and to the transport boundary, both of
    # which are resolved inside `new_channels`.
    wall = S["wall"]
    mat_ind, onset_ind, gate_ind = indicator_physics(data, bio, wall, flow=flow)
    NC = new_channels(S, mat_ind, onset_ind, gate_ind, crit)
    order = sorted(NC)
    X = np.concatenate([S["X"]] + [NC[k].reshape(-1, 1) for k in order], axis=1)
    return X.astype(np.float32), [str(c) for c in S["cols"]] + order


#: the 13 added channel names, sorted -- the order the cache and the models use
V4_CHANNELS = sorted([
    "att_adv", "att_reach", "gate_ind", "log_mat_adv", "log_mat_adv_ind", "log_mat_adv_n",
    "log_mat_ind_owner", "log_mat_off_est", "log_mat_phys_ind", "log_src_reach", "log_tau",
    "onset_phys_ind", "tau_rel_owner"])
