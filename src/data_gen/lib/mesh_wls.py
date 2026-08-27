"""Shared 2nd-order WLS operators and Gmsh boundary mask extraction for graph builders (DRY)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def solid_boundary_mask(
    mask_wall: torch.Tensor, mask_wound: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Every no-slip solid boundary node: healthy wall **plus** wound (COMSOL ``uni1``).

    ``mask_wall`` is the *healthy* wall label (COMSOL ``dif1``) and is carved disjoint from
    ``mask_wound`` (``sel1``) so the two deposition laws stay separable -- ``srf1``/``fl1``
    are gated, ``srf2``/``fl2`` are not.  That split is correct for physics and wrong for
    geometry: a wound node is still wall.  Anything that asks "how far is this node from the
    vessel wall" -- SDF, wall normals, hydraulic width, the no-slip constraint, WSS masking --
    must use this union, or the injured segment is encoded as open lumen and every
    wall-derived feature on it is measured to the nearest *un-wounded* node instead.

    No-wound packs are unaffected: ``mask_wound`` is empty and this returns ``mask_wall``.
    """
    if mask_wound is None:
        return mask_wall
    if not torch.is_tensor(mask_wound) or mask_wound.numel() == 0:
        return mask_wall
    return mask_wall | mask_wound.to(device=mask_wall.device, dtype=mask_wall.dtype)


def solid_boundary_nodes(data) -> np.ndarray:
    """:func:`solid_boundary_mask` read off a pack, as a flat numpy bool array.

    The one place anything outside the graph builders should ask "is this node on a solid
    boundary".  Both stored masks may carry a trailing singleton dimension and the wound mask
    may be absent entirely, so the reshape/`getattr` dance is not optional -- and repeating it
    per call site is how `src/clot_ml/features.py` ended up measuring geometry against
    ``mask_wall`` alone long after the builders had been fixed (MODEL_REVIEW_2026-08-22 5b.3).
    """
    wall = data.mask_wall.reshape(-1).bool()
    wound = getattr(data, "mask_wound", None)
    if torch.is_tensor(wound) and wound.numel():
        wound = wound.reshape(-1).bool()
    else:
        wound = None
    return solid_boundary_mask(wall, wound).cpu().numpy()


def boundary_normals_from_graph(
    pos: np.ndarray,
    mask_solid: np.ndarray,
    edge_index: np.ndarray,
    *,
    orient_targets: Optional[np.ndarray] = None,
    max_hops: int = 3,
) -> np.ndarray:
    """Unit inward wall normals at solid-boundary nodes, computed from the GRAPH alone.

    WHY THIS EXISTS.  The normals used to come from Gmsh **line** cells, and the COMSOL
    ``.msh`` exports contain only ``triangle6`` -- no line cells at all.  So that branch
    never ran on any pack in the cohort, the KD-tree fallback handed a wall node *itself* as
    its nearest boundary neighbour, and ``wall_normal`` came out identically **zero at every
    wall node** on every pack, wound and no-wound alike (WOUND_PROGRESS 8).

    Reading the normals off the triangulation would fix it only where the mesh still exists,
    and the COMSOL exports for the three wound runs are gone.  The boundary of a 2-D vessel
    is a 1-D curve that the graph already resolves, so nothing else is needed: take each
    solid node's solid neighbours, fit the local tangent, and rotate it a quarter turn.

    Method
    ------
    * Restrict the graph to solid-solid edges.  On a quadratic (P2) mesh this is the boundary
      polyline through corner *and* mid-side nodes.
    * The local tangent is the leading eigenvector of the covariance of the neighbour
      offsets -- a total-least-squares line fit, which is stable for both the degree-2
      interior of the polyline and the higher-degree junctions where wall meets inlet.
      Neighbourhoods are grown hop by hop (up to ``max_hops``) until at least two distinct
      neighbours are available, so isolated boundary nodes still get a normal.
    * The normal is the perpendicular, oriented **into the lumen** -- toward
      ``orient_targets[i]`` when given (the centerline point nearest node ``i``), else toward
      the centroid of the non-solid nodes.  This matches the sign convention the old
      line-cell branch used.

    Returns ``[N, 2]`` float64, unit length on solid nodes and exactly zero elsewhere; the
    caller decides what non-solid nodes get (historically the KD-tree offset).
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    solid = np.asarray(mask_solid, dtype=bool).reshape(-1)
    n = len(pos)
    out = np.zeros((n, 2), dtype=np.float64)
    if not solid.any():
        return out

    ei = np.asarray(edge_index)
    src, dst = ei[0], ei[1]
    keep = solid[src] & solid[dst]
    a, b = src[keep], dst[keep]
    # symmetric adjacency over solid nodes only
    adj: Dict[int, set] = {}
    for i, j in zip(a.tolist(), b.tolist()):
        if i == j:
            continue
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    if orient_targets is None:
        interior = ~solid
        centre = pos[interior].mean(axis=0) if interior.any() else pos.mean(axis=0)
        targets = np.broadcast_to(centre, (n, 2))
    else:
        targets = np.asarray(orient_targets, dtype=np.float64).reshape(n, 2)

    # Some solid nodes are ISOLATED in the solid subgraph -- degree 0, no solid neighbour to
    # fit a tangent from (12 of 539 on `patient008`, mid-domain, not at inlet/outlet).  They
    # are still on the boundary curve, so fall back to their nearest solid nodes by position:
    # same total-least-squares fit, just a geometric neighbourhood instead of a topological
    # one.  Still mesh-free.
    solid_idx = np.flatnonzero(solid)
    _kdt = None

    def _fallback_neighbours(i: int, k: int = 6) -> list:
        nonlocal _kdt
        if _kdt is None:
            from scipy.spatial import cKDTree
            _kdt = cKDTree(pos[solid_idx])
        kq = min(k + 1, len(solid_idx))
        _, near = _kdt.query(pos[i], k=kq)
        return [int(solid_idx[j]) for j in np.atleast_1d(near) if int(solid_idx[j]) != i]

    for i in solid_idx.tolist():
        nb = set(adj.get(i, ()))
        # grow the neighbourhood until the tangent is determined
        hop = 1
        while len(nb) < 2 and hop < max_hops:
            grown = set(nb)
            for j in nb:
                grown |= adj.get(j, set())
            grown.discard(i)
            if grown == nb:
                break
            nb = grown
            hop += 1
        if len(nb) < 2:
            nb |= set(_fallback_neighbours(i))
        if not nb:
            continue
        d = pos[sorted(nb)] - pos[i]
        d = d[np.linalg.norm(d, axis=1) > 1e-15]
        if len(d) == 0:
            continue
        if len(d) == 1:
            tangent = d[0]
        else:
            # leading eigenvector of the offset covariance = best-fit line direction
            cov = d.T @ d
            w, V = np.linalg.eigh(cov)
            tangent = V[:, int(np.argmax(w))]
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-15:
            continue
        tangent = tangent / t_norm
        nrm = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        if float(nrm @ (targets[i] - pos[i])) < 0.0:
            nrm = -nrm
        out[i] = nrm
    return out


def node_type_one_hot(
    mask_solid: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    *,
    device=None,
) -> torch.Tensor:
    """The 4-channel ``node_type_*`` one-hot: ``[interior, solid, inlet, outlet]``.

    These four channels were a literal ``torch.zeros((N, 4))`` in
    ``build_kinematics_node_x_tensor`` -- declared in ``KINE_X_SCHEMA``, consumed by
    ``RGP_DEQ`` through ``NodeFeat.REST``, and never populated on any pack
    (WOUND_PROGRESS 8).  Four dead input channels.

    Exactly one channel is 1 per node.  Priority is inlet > outlet > solid > interior, so a
    node where the wall meets the inlet is labelled inlet -- it is a Dirichlet velocity node,
    which is what downstream code needs to know about it.  ``mask_solid`` should be
    :func:`solid_boundary_mask`'s union, so a wound node reads as boundary rather than lumen.
    """
    solid = mask_solid.reshape(-1).bool()
    inlet = mask_inlet.reshape(-1).bool().to(solid.device)
    outlet = mask_outlet.reshape(-1).bool().to(solid.device)
    n = solid.numel()
    oh = torch.zeros((n, 4), dtype=torch.float32,
                     device=device if device is not None else solid.device)
    is_in = inlet
    is_out = outlet & ~is_in
    is_solid = solid & ~is_in & ~is_out
    oh[~(is_in | is_out | is_solid), 0] = 1.0     # interior / lumen
    oh[is_solid, 1] = 1.0
    oh[is_in, 2] = 1.0
    oh[is_out, 3] = 1.0
    return oh


def precompute_wls_operators(edge_index: torch.Tensor, num_nodes: int, pos_tensor: torch.Tensor):
    """
    2nd-order polynomial WLS on edges; returns ``V``, ``W``, ``M_inv``.
    """
    row, col = edge_index
    pos_diff = pos_tensor[col, :2] - pos_tensor[row, :2]
    dx, dy = pos_diff[:, 0], pos_diff[:, 1]

    dist_sq = dx**2 + dy**2 + 1e-8

    dx2 = 0.5 * dx**2
    dxy = dx * dy
    dy2 = 0.5 * dy**2

    V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)
    W = 1.0 / dist_sq

    V_unsqueezed = V.unsqueeze(2)
    V_T_unsqueezed = V.unsqueeze(1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

    M_e_flat = M_e.view(-1, 25)
    out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
    row_exp = row.view(-1, 1).expand_as(M_e_flat)
    M_flat = out.scatter_add_(0, row_exp, M_e_flat)

    M = M_flat.view(num_nodes, 5, 5)
    M_inv = rank_aware_pinv_sym(M)

    return V, W, M_inv


#: Relative eigenvalue floor for the 5-term WLS normal matrix.  Directions below this are
#: unresolved by the node's own stencil and are truncated rather than inverted.
WLS_RCOND = 1e-5


def rank_aware_pinv_sym(M: torch.Tensor, rcond: float = WLS_RCOND) -> torch.Tensor:
    """Per-node pseudo-inverse of a symmetric normal matrix, truncating what it cannot resolve.

    RGP_DEQ_REPAIR_PLAN.md D5.  The previous form was ``pinv(M + 1e-6*I, rcond=1e-5)``: an
    absolute ridge under a *relative* truncation threshold, which is unsound because whether
    the lifted null directions survive ``rcond`` depends on the node's own ``lambda_max``.
    Dropping the ridge and truncating relative to each node's own best-resolved direction
    removes that coupling -- a direction is inverted when the stencil resolves it and is
    exactly zero otherwise.

    **Be precise about what this buys.**  Measured, it buys correctness, not accuracy: on
    ``patient020`` the two operators agree to within 3% in norm, and on the synthetic P2 mesh
    in ``test_rgp_deq_repair`` they produce identical per-node ranks at every scale from 1.0 to
    1e-3.  The ridge was **not** the active defect on these meshes.  The active defect is the
    rank deficiency itself -- ~74.6% of biochem nodes are P2 mid-side vertices whose 5-term
    normal matrix has rank 2, and recovering a known quadratic through them gives a relative
    error of 0.72 against 4.2e-16 on the full-rank rows.  Truncation states that honestly;
    :func:`math_operators._fill_rank_deficient_rows` is what actually repairs it.
    """
    evals, evecs = torch.linalg.eigh(M.double())
    lam_max = evals.abs().amax(dim=1, keepdim=True).clamp(min=1e-300)
    keep = (evals.abs() / lam_max) > float(rcond)
    safe = torch.where(keep, evals, torch.ones_like(evals))
    inv = torch.where(keep, 1.0 / safe, torch.zeros_like(evals))
    out = evecs @ torch.diag_embed(inv) @ evecs.transpose(1, 2)
    return out.to(M.dtype)


def rebuild_wls_operators_from_graph(data):
    """Recompute ``V``, ``W``, ``M_inv`` from a pack's OWN ``edge_index`` and node positions.

    RGP_DEQ_REPAIR_PLAN.md B13.  The operators stored on the biochem packs do not correspond
    to the edge lists stored alongside them.  Measured maxima of ``|M_inv|``:

    ```
    pack          stored     rebuilt (either inversion)
    patient001    1.000e+06        5.6e+03
    patient012    6.658e+05        2.1e+04
    patient020    7.001e+05        2.8e+04
    patient041    7.243e+05        1.7e+04
    patient044    6.256e+05        2.9e+04
    ```

    ``1.000e+06`` is exactly ``1/epsilon`` for the old ``M + 1e-6*I``: the signature of a node
    whose ``M`` was *empty*, i.e. built from an edge list that did not include it.  Rebuilding
    from the pack's own graph drops the operator by 20-60x, and with it the derived channels:

    ```
    pack          width_d2 stored   width_d2 rebuilt      (training p95: 73.8)
    patient020         4.784e+04              21.8
    patient012         1.046e+05             183.8
    patient041         1.019e+05             273.1
    patient044         1.773e+05             258.2
    ```

    So ``kinematics_inference.clamped_width_priors`` -- which forces these into the training
    range at every call site -- is a workaround for a **stale operator**, not for the collinear
    stencils its docstring blames.  A rebuild puts them there on their own.

    **Not wired in by default.**  This changes every flow-derived quantity downstream, and the
    plan's own rule is that flow changes are judged on wall ``dsrx`` correlation, gate union
    Jaccard and oracle-F1 -- not on the operator norm.  Make it the default only after that
    measurement, not because the numbers above look better.
    """
    pos = data.x[:, :2]
    V, W, M_inv = precompute_wls_operators(data.edge_index, int(data.num_nodes), pos)
    return V, W, M_inv


def gmsh_line_boundary_masks(mesh, num_nodes: int, tags: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inlet / outlet / wall / wound boolean masks from Gmsh line physical tags."""
    mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
    mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
    mask_wall = torch.zeros(num_nodes, dtype=torch.bool)
    mask_wound = torch.zeros(num_nodes, dtype=torch.bool)

    line_cells = []
    line_tags = []

    t_in = tags["Inlet"]
    t_out = tags["Outlet_1"]
    t_wall = tags["Walls"]
    t_wound = tags.get("Wound")

    try:
        if "line" in mesh.cells_dict:
            line_cells = mesh.cells_dict["line"]
            line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
        elif hasattr(mesh, "get_cells_type"):
            line_cells = mesh.get_cells_type("line")
            line_tags = mesh.get_cell_data("gmsh:physical", "line")
    except Exception:
        pass

    if len(line_tags) == 0 or len(line_cells) == 0:
        raise ValueError(
            "gmsh_line_boundary_masks: no Gmsh line cells with physical tags were found "
            "(expected mesh.cells_dict['line'] and cell_data_dict['gmsh:physical']['line'], "
            "or meshio equivalents). Re-export the .msh with tagged inlet, outlet, and wall curves."
        )
    if len(line_tags) != len(line_cells):
        raise ValueError(
            f"gmsh_line_boundary_masks: line_tags length ({len(line_tags)}) != line_cells length "
            f"({len(line_cells)}); mesh file is inconsistent."
        )

    def _line_node_indices(cell) -> np.ndarray:
        arr = np.asarray(cell, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return arr
        if (arr < 0).any() or (arr >= num_nodes).any():
            bad = arr[(arr < 0) | (arr >= num_nodes)]
            raise ValueError(
                "gmsh_line_boundary_masks: line references node index outside "
                f"[0, {num_nodes - 1}] (bad values: {bad[:16]!r}{'...' if bad.size > 16 else ''}). "
                "Re-export or repair the mesh."
            )
        return arr

    for i, tag in enumerate(line_tags):
        if isinstance(line_cells, list) and not isinstance(line_cells[0], (int, float, np.integer)):
            nodes = line_cells[i]
        else:
            nodes = line_cells[i]
        idx = _line_node_indices(nodes)

        if tag == t_in:
            mask_inlet[idx] = True
        elif tag == t_out:
            mask_outlet[idx] = True
        elif tag == t_wall:
            mask_wall[idx] = True
        elif t_wound is not None and tag == t_wound:
            mask_wound[idx] = True

    mask_wall = mask_wall & (~mask_wound)
    mask_inlet = mask_inlet & (~mask_wall) & (~mask_wound)
    mask_outlet = mask_outlet & (~mask_wall) & (~mask_wound)

    unique_tags = sorted({int(t) for t in line_tags})
    tag_msg = f"Unique gmsh:physical line tags present in mesh: {unique_tags}. " f"Expected Inlet={t_in}, Outlet_1={t_out}, Walls={t_wall}."

    if not bool(mask_inlet.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no inlet nodes** matched VesselConfig.TAGS['Inlet']. "
            + tag_msg
            + " Fix Gmsh physical names/IDs or update TAGS, then re-export."
        )
    if not bool(mask_outlet.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no outlet nodes** matched VesselConfig.TAGS['Outlet_1']. "
            + tag_msg
            + " Fix Gmsh physical names/IDs or update TAGS, then re-export."
        )
    if not bool(mask_wall.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no wall nodes** matched VesselConfig.TAGS['Walls']. "
            + tag_msg
            + " Without wall tags, surface species and wall residuals are undefined. "
            "Re-export the mesh with wall boundary curves under the expected physical group."
        )
    if bool((mask_inlet & mask_outlet).any()):
        overlap = int((mask_inlet & mask_outlet).sum().item())
        raise ValueError(
            f"gmsh_line_boundary_masks: {overlap} node(s) are both inlet and outlet after wall "
            "carving (tags overlap on shared vertices). Fix boundary curve tagging in Gmsh."
        )

    return mask_inlet, mask_outlet, mask_wall, mask_wound


__all__ = [
    "precompute_wls_operators",
    "gmsh_line_boundary_masks",
    "solid_boundary_mask",
    "solid_boundary_nodes",
]
