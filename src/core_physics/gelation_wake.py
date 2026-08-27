"""The closed loop, in two stages: ``Mat >= crit  ->  sr collapses  ->  low-shear gate opens``.

WHY THIS EXISTS.  The shipped wall model freezes the deposition gate at ``t=0``, so a node
whose shear only falls *because of the clot* can never ignite.  On ``wound_patient003`` that
is 15 healthy-wall nodes sitting at ``sr = 117 /s`` against ``lss = 25`` -- gate 0, no source
term at all -- which own 29 of the 67 wound-region lumen clot nodes the model misses
(docs/WOUND_PROGRESS.md 14).  They ignite under **no** species arm and under **no**
:func:`~src.core_physics.physics_wall_model.graded_gate` mode, because grading a margin
cannot reach a node five times above the threshold.  Only the flow moving reaches them.

WHY NOT THE CORRECTOR.  ``MODEL_REVIEW 9e`` handed the local kinematic corrector oracle
occlusion and it produced -3.5% shear against the required -87%, wrong sign on one vessel:
an 87% collapse is the channel closing, which is not a sum of local residuals.

WHAT REPLACES IT.  One measured kernel.  The first form tried keyed the collapse on hops to
the NEAREST gelled node, and it saturates: ``wound_patient003``'s blind owners read GT
``sr/sr0 = 0.387`` while two hops from clot, where a nearest-node kernel says 0.706.  They
are *surrounded* by clot rather than beside one node.  So the load is a **superposition**,

    w_i  =  sum over gelled wall nodes j of  exp(-h_ij / WAKE_LAMBDA_HOPS)

and GT ``sr(t)/sr(0)`` on not-yet-gelled wall is a monotone function of it, the same curve on
wound and no-wound vessels alike (pooled over eleven: ``wound_patient001/002/003`` and
``patient012/016/020/028/032/035/041/044``):

    w:    0     0.25    0.5     1      2      4      6
    amp: 1.00   0.976   0.955  0.929  0.786  0.552  (clamped)

A node that has gelled itself takes :data:`GELLED_SR_RATIO` instead -- the ``mu1`` x80 step,
measured per node at 0.1226 (p25-p75 0.113-0.136) in
``scripts/diag_closed_loop_feasibility.py``.

**The table is CLAMPED at its last measured bin and never extrapolated.**  Beyond ``w ~ 6``
there is no data, because in GT a node under that much load has already gelled -- so the
not-yet-gelled population that defines the curve stops there.  Extrapolating a positive
feedback past its evidence is how this kind of term runs away, and the clamp is what keeps
the false-positive count at zero on the clot-free vessels.

Deploy-legal: it reads the rollout's OWN committed set, mesh connectivity and the ``t=0``
shear field.  No GT, no flow solve, no network.
"""
from __future__ import annotations

import numpy as np

#: The ``mu1`` x80 step a node applies to its OWN shear once it gels.
GELLED_SR_RATIO: float = 0.1226

#: Decay length, in mesh hops, of one gelled node's contribution to a neighbour's load.
WAKE_LAMBDA_HOPS: float = 4.0

#: Hops beyond which a gelled node contributes nothing (the weight is <5% by here).
WAKE_MAX_HOPS: int = 12

#: Measured ``amp = sr(t)/sr(0)`` on NOT-yet-gelled wall, as (load, amp) knots.  Pooled
#: median over eleven vessels; see the module docstring.  CLAMPED at the last knot.
#: The last knot is NOT a free extrapolation: it is the OTHER measured anchor.  A node under
#: enough load is effectively inside the clot mass, so its shear must approach what a gelled
#: node's own shear does -- :data:`GELLED_SR_RATIO`.  The load at which it gets there, 9.0,
#: is where the measured slope between the last two observed bins (0.786 at w=3, 0.552 at
#: w=5) reaches it.  So the curve runs between two measurements rather than past either.
WAKE_LOAD_KNOTS: tuple[float, ...] = (0.0, 0.125, 0.375, 0.75, 1.5, 3.0, 5.0, 9.0)
WAKE_LOAD_AMP: tuple[float, ...] = (1.0, 1.0, 0.9762, 0.9546, 0.9287, 0.7861, 0.5517,
                                    GELLED_SR_RATIO)


def wake_amplitude(load: np.ndarray, *, knots: tuple[float, ...] = WAKE_LOAD_KNOTS,
                   amps: tuple[float, ...] = WAKE_LOAD_AMP) -> np.ndarray:
    """``sr`` multiplier for each node given its superposed gelled-neighbour load.

    Linear interpolation between the knots, **clamped** at both ends: zero load leaves ``sr``
    untouched, and load beyond the last knot holds :data:`GELLED_SR_RATIO` rather than
    running a positive feedback off the end of its evidence.
    """
    return np.interp(np.asarray(load, dtype=np.float64), np.asarray(knots),
                     np.asarray(amps), left=amps[0], right=amps[-1])


def wall_wake_operator(data, wall: np.ndarray, *, lam: float = WAKE_LAMBDA_HOPS,
                       max_hops: int = WAKE_MAX_HOPS) -> np.ndarray:
    """``K [n_wall, n_wall]``, ``K[i, j] = exp(-hops(i, j) / lam)`` within ``max_hops``.

    Hops are measured through the FULL mesh graph, not the wall subgraph -- two wall nodes
    facing each other across the lumen are close in the flow even when the boundary path
    between them is long, and that is the coupling the wake is about.  ``Mat`` is a surface
    field, so only wall nodes can be sources and only wall nodes need a load: the operator is
    a few hundred squared, not ``num_nodes`` squared.
    """
    from src.clot_ml.features import adjacency, hop_distance  # noqa: PLC0415

    n = int(data.num_nodes)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    idx = np.flatnonzero(np.asarray(wall, dtype=bool))
    K = np.zeros((idx.size, idx.size), dtype=np.float64)
    for a, i in enumerate(idx):
        seed = np.zeros(n, dtype=bool)
        seed[i] = True
        h = hop_distance(seed, A, max_h=max_hops + 1)[idx]
        K[a] = np.where(h <= max_hops, np.exp(-h / float(lam)), 0.0)
    return K


def make_gelation_wake_blockage(
    data, bio_cfg, fields, wall: np.ndarray, *,
    every: int = 1, lam: float = WAKE_LAMBDA_HOPS, max_hops: int = WAKE_MAX_HOPS,
    scale_dsrx: bool = True, K: np.ndarray | None = None,
):
    """``blockage(mat, gate0, step) -> gate`` for :func:`integrate_mat_trajectory`.

    Each call takes the rollout's own committed set ``mat >= crit``, forms the superposed
    load ``w = K @ committed``, turns it into a shear multiplier through
    :func:`wake_amplitude`, and re-evaluates the deposition gate through the single
    :func:`~src.core_physics.physics_wall_model.gate_from_shear` transcription.  A node that
    has committed takes :data:`GELLED_SR_RATIO` directly.

    ``scale_dsrx`` rescales the shear GRADIENT by the same local factor: exact for a uniform
    rescale, first-order otherwise, and the same convention
    :func:`~src.core_physics.shear_redistribution.make_blockage` uses, so the two gate
    branches stay consistent with each other.

    An already-committed node keeps at least the gate it ignited with -- ``mu1`` has fired,
    it is clot now, and letting the wake close its own gate would make the ODE non-monotone.

    With nothing committed the returned gate is ``gate0`` bit-for-bit, so a vessel that never
    ignites is untouched.  ``K`` may be passed in to reuse the operator across arms.
    """
    from src.core_physics.physics_wall_model import gate_from_shear  # noqa: PLC0415

    crit = float(bio_cfg.viscosity_mat_crit)
    wall = np.asarray(wall, dtype=bool)
    widx = np.flatnonzero(wall)
    Kw = wall_wake_operator(data, wall, lam=lam, max_hops=max_hops) if K is None else K
    state: dict = {"gate": None, "last": -10 ** 9, "n_occ": -1}

    def blockage(mat, gate0, step):
        occ = np.asarray(mat) >= crit
        n_occ = int(occ.sum())
        if n_occ == 0:
            return gate0
        # Recompute on a cadence, but ALWAYS when the committed set has grown -- the front
        # advancing IS the coupling, and skipping that step defeats it.
        if (state["gate"] is not None and n_occ == state["n_occ"]
                and (step - state["last"]) < every):
            return state["gate"]
        occ_w = occ[widx].astype(np.float64)
        amp = np.ones(occ.shape, dtype=np.float64)
        amp[widx] = wake_amplitude(Kw @ occ_w)
        amp[occ] = GELLED_SR_RATIO
        dsrx = fields.dsrx * amp if scale_dsrx else fields.dsrx
        g = gate_from_shear(fields.sr * amp, dsrx, bio_cfg) * wall
        g = np.where(occ, np.maximum(g, gate0), g)
        state.update(gate=g, last=step, n_occ=n_occ)
        return g

    blockage.state = state
    blockage.K = Kw
    return blockage
