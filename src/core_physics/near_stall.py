"""Near-field stall: committed solid thickens the no-slip; 1-corner-shell neighbours feel ``mu1``.

WHY THIS EXISTS.  Hop-wake cannot open ``wound_comsol003``'s blind owners even as a
GT-oracle: they sit at ``sr = 118 /s`` against ``lss = 25``.  A hop disk of
:data:`GELLED_SR_RATIO` is the on-node ``mu1`` step, not a far-field superposition.

WHAT THE MEASUREMENTS SAY (``scripts/diag_stall_deploy_oracle.py``,
``scripts/diag_nonlocal_flow_gate.py``).  The GT gate that opens those blinds is the
**B-branch only**.  Scaling ``dsrx`` is the wrong lever (it *closes* A-gates on
``wound_comsol001``: 63 ignitions -> 45).  A 1-hop stencil does not march -- newly
gelled hop-1 nodes do not pull hop-2 blinds over the truncated horizon.  ``STALL_HOPS = 2``
is one corner shell on these quadratic meshes (PHASE7_FINDINGS 8).  Occupancy that waits
for the wall ODE's ``Mat >= crit`` on the wound starves the kernel until the flash at
step 49 (GT wound gels at step 2-6); the wound is 100% GT clot, so it is a stall source
from t=0.

Default OFF in :func:`~src.clot_ml.temporal.ode_trajectory`.  The wound deploy path may
OR ungated stall-wall ignitions into the shipped series: +12 wall TP / 0 FP on 003,
inert on 001/002 (no ungated near-wound wall) and on every no-wound pack.
"""
from __future__ import annotations

import numpy as np

from src.core_physics.gelation_wake import GELLED_SR_RATIO

#: One corner shell on a quadratic mesh.  Hop-1 does not march; hops=4 is a disk (4 wall FP).
STALL_HOPS: int = 2


def near_stall_amplitude(hops: np.ndarray, *, hops_cut: int = STALL_HOPS,
                         amp: float = GELLED_SR_RATIO) -> np.ndarray:
    """``sr`` multiplier: :data:`GELLED_SR_RATIO` inside the stall stencil, else 1."""
    h = np.asarray(hops, dtype=np.float64)
    out = np.ones(h.shape, dtype=np.float64)
    out[h <= float(hops_cut)] = float(amp)
    return out


def make_near_stall_blockage(
    data, bio_cfg, fields, *,
    wall: np.ndarray | None = None, solid: np.ndarray | None = None,
    hops: int = STALL_HOPS, every: int = 1, scale_dsrx: bool = False,
    sr_ratio: float = GELLED_SR_RATIO, seed_wound: bool = True,
):
    """``blockage(mat, gate0, step) -> gate`` for :func:`integrate_mat_trajectory`.

    Sources are the SOLID boundary (wound included).  The gate rewrite is HEALTHY wall
    only -- ``srf2`` on the wound is already ungated and must not be replaced by a
    stalled ``srf1``.  Occupied nodes keep at least the gate they ignited with, so the
    ODE stays monotone.

    ``scale_dsrx`` defaults OFF: the GT blinds open on the low-shear indicator, and
    scaling ``dsrx`` destroys the separation branch on 001.  ``seed_wound`` defaults ON:
    the injured patch is a stall source from t=0 (100% GT clot), not after the wall ODE
    happens to cross ``crit``.

    With nothing committed (and no wound seed) this returns ``gate0`` without copying.
    """
    from src.clot_ml.features import adjacency, hop_distance  # noqa: PLC0415
    from src.core_physics.physics_wall_model import gate_from_shear  # noqa: PLC0415
    from src.data_gen.lib.mesh_wls import solid_boundary_nodes  # noqa: PLC0415

    crit = float(bio_cfg.viscosity_mat_crit)
    wall = (data.mask_wall.reshape(-1).bool().cpu().numpy() if wall is None
            else np.asarray(wall, dtype=bool))
    solid = (solid_boundary_nodes(data) if solid is None
             else np.asarray(solid, dtype=bool))
    n = int(data.num_nodes)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    state: dict = {"gate": None, "last": -10 ** 9, "n_occ": -1}
    
    wnd = None
    if seed_wound:
        from src.clot_ml.wound import wound_mask  # noqa: PLC0415
        wnd = wound_mask(data)

    def blockage(mat, gate0, step):
        occ = (np.asarray(mat) >= crit) & solid
        if wnd is not None:
            occ = occ | wnd
        n_occ = int(occ.sum())
        if n_occ == 0:
            return gate0
        if (state["gate"] is not None and n_occ == state["n_occ"]
                and (step - state["last"]) < every):
            return state["gate"]
        dist = hop_distance(occ, A, max_h=int(hops))
        amp = near_stall_amplitude(dist, hops_cut=int(hops), amp=float(sr_ratio))
        g = np.asarray(gate0, dtype=np.float64).copy()

        # The low-shear branch only.  Scaling `dsrx` by the same factor CLOSES separation
        # gates rather than opening stagnation ones -- on `wound_comsol001` it takes the
        # ODE from 63 ignitions to 45 -- so `scale_dsrx` stays off by default and exists to
        # keep that negative result reproducible.
        dsrx = fields.dsrx * amp if scale_dsrx else fields.dsrx
        g_wall = gate_from_shear(fields.sr * amp, dsrx, bio_cfg, wall=wall)
        g[wall] = g_wall[wall]
        g = np.where(occ, np.maximum(g, gate0), g)
        state.update(gate=g, last=step, n_occ=n_occ)
        return g

    blockage.state = state
    return blockage
