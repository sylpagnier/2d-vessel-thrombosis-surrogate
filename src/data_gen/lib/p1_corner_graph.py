"""Collapse a P2 (``triangle6``) pack onto its P1 corner graph, and lift a field back.

**Why** (RGP_DEQ_REPAIR_PLAN.md §6.1, T2 Route B).  Stage-A has never seen a P2 mesh --
measured over the synthetic corpus, **0.0% of training nodes are degree-2** against **74.5%**
of every biochem deploy mesh, and the deploy meshes are 5x larger (N median 14830 vs 2983).
That is the largest single train/deploy gap in the stack, and it is not a modelling problem:
mid-side nodes carry no independent geometric information.  Each one is the exact midpoint of
two corners, its 2-neighbour stencil is rank-deficient by construction, and every derivative
the pipeline takes at one has to be filled from its corners anyway
(``math_operators._fill_rank_deficient_rows``).

A P2 mesh's corner subgraph -- corner-to-corner adjacency taken *through* the mid-sides -- is
exactly the original P1 triangulation, and it lands the deploy mesh inside the training
distribution rather than asking the model to generalise to a discretisation it has never seen:

    14830 nodes x 23.4% corners ~= 3470,  against a synthetic corpus median of 2983 [1957-4323]

**What is legitimate here, and what is not.**  Lifting a *prediction* back onto the mid-side
nodes by averaging its two corners is fine: it is our own output, and the mid-side node is
geometrically the midpoint.  Interpolating *labels* the same way would not be -- a P2 finite
element solution's mid-side value is genuinely not the mean of its corners -- which is why the
reverse route (synthesising P2 training data from P1 meshes) needs real new CFD and this one
does not.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

#: Channel layout of ``kine_x_v1_18ch`` (see `legal_priors` for the same constants).
COL_XY = slice(0, 2)
COL_SDF = 2
COL_WALL_NORMAL = slice(4, 6)
COL_WIDTH = 15
COL_WIDTH_D1 = 16
COL_WIDTH_D2 = 17


class CornerMap(NamedTuple):
    """How a P1 corner graph relates to the P2 pack it came from."""

    corner_ids: torch.Tensor      # [Nc] indices into the original pack
    midside_ids: torch.Tensor     # [Nm] indices into the original pack
    midside_ends: torch.Tensor    # [Nm, 2] the two ORIGINAL corner ids each mid-side bridges
    num_nodes_full: int


def _degree(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    deg = torch.zeros(n, dtype=torch.long, device=edge_index.device)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long,
                                                device=edge_index.device))
    return deg


def identify_midside_nodes(data, *, max_cos: float = -0.9):
    """Degree-2 nodes lying *between* their two neighbours (edge vectors anti-parallel).

    The test is anti-parallelism, **not** an exact-midpoint check.  COMSOL places boundary
    mid-side nodes on the true curved geometry rather than on the chord -- that is the entire
    point of a ``triangle6`` element -- so a strict midpoint tolerance silently misclassifies
    every mid-side node on a curved wall as a corner.  Measured on ``comsol001``, deviation
    from the chord midpoint relative to segment length:

    ```
                       median     p99      max     nodes
    off-wall deg-2     0.0e+00  0.0e+00  0.0e+00    6775
    wall     deg-2     1.9e-03      --   1.4e-02     268
    ```

    A 1e-5 tolerance keeps 96.0% of degree-2 nodes and leaves corners at 28.8%; the true corner
    fraction is 25.8%, and the 3% difference is exactly the curved wall.  Anti-parallelism
    catches all of them: the two edge vectors of a degree-2 node run in opposite directions
    whether or not the node sits on the chord.

    Degree 2 alone would be nearly sufficient here, but the direction test is what makes this a
    statement about ``triangle6`` topology rather than about connectivity, so a genuinely P1
    mesh with a degenerate 2-neighbour corner is not silently collapsed.
    """
    n = int(data.num_nodes)
    row, col = data.edge_index
    pos = data.x[:, COL_XY]
    deg = _degree(data.edge_index, n)

    order = torch.argsort(row, stable=True)
    r_s, c_s = row[order], col[order]
    counts = torch.bincount(r_s, minlength=n)
    ptr = torch.zeros(n + 1, dtype=torch.long, device=row.device)
    ptr[1:] = counts.cumsum(0)

    cand = (deg == 2).nonzero().reshape(-1)
    if cand.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=row.device)
        return torch.zeros(n, dtype=torch.bool, device=row.device), empty.reshape(0, 2)

    a = c_s[ptr[cand]]
    b = c_s[ptr[cand] + 1]
    v1 = pos[a] - pos[cand]
    v2 = pos[b] - pos[cand]
    cos = torch.nn.functional.cosine_similarity(v1, v2, dim=1)
    is_mid = cos < float(max_cos)

    mask = torch.zeros(n, dtype=torch.bool, device=row.device)
    mask[cand[is_mid]] = True
    ends = torch.stack([a[is_mid], b[is_mid]], dim=1)
    return mask, ends


def build_corner_graph(data, *, recompute_width_derivs: bool = True):
    """Return ``(p1_data, CornerMap)``.  A pack with no mid-side nodes is returned unchanged."""
    from src.data_gen.lib.mesh_wls import precompute_wls_operators

    n = int(data.num_nodes)
    mid_mask, ends = identify_midside_nodes(data)
    if not bool(mid_mask.any()):
        ids = torch.arange(n, device=data.x.device)
        return data, CornerMap(ids, ids[:0], ends.reshape(0, 2), n)

    corner_ids = (~mid_mask).nonzero().reshape(-1)
    midside_ids = mid_mask.nonzero().reshape(-1)

    remap = torch.full((n,), -1, dtype=torch.long, device=data.x.device)
    remap[corner_ids] = torch.arange(corner_ids.numel(), device=data.x.device)

    # Corner-to-corner adjacency through each mid-side: this IS the P1 triangulation's edge set.
    e_new = remap[ends]                                   # [Nm, 2]
    # Plus any corner-corner edge the pack already carried, so nothing is silently dropped.
    row, col = data.edge_index
    both_corner = (~mid_mask[row]) & (~mid_mask[col])
    if bool(both_corner.any()):
        direct = torch.stack([remap[row[both_corner]], remap[col[both_corner]]], dim=1)
        e_new = torch.cat([e_new, direct], dim=0)

    und = torch.cat([e_new, e_new.flip(1)], dim=0)
    und = torch.unique(und, dim=0)
    edge_index = und.t().contiguous()

    out = data.__class__()
    x = data.x[corner_ids].clone()
    pos = x[:, COL_XY]
    delta = pos[edge_index[0]] - pos[edge_index[1]]
    out.x = x
    out.edge_index = edge_index
    out.edge_attr = torch.cat([delta, delta.norm(dim=1, keepdim=True)], dim=1)
    out.num_nodes = int(corner_ids.numel())

    for name in ("mask_inlet", "mask_outlet", "mask_wall", "mask_wound"):
        m = getattr(data, name, None)
        if torch.is_tensor(m) and m.reshape(-1).numel() == n:
            setattr(out, name, m.reshape(-1)[corner_ids])
    for name in ("u_ref", "d_bar", "graph_stem", "geometry_level", "config_id"):
        v = getattr(data, name, None)
        if v is not None:
            setattr(out, name, v)
    for name in ("u_inlet_bc", "mu_inlet_bc", "mu_wall_bc"):
        v = getattr(data, name, None)
        if torch.is_tensor(v) and v.shape[0] == n:
            setattr(out, name, v[corner_ids])

    V, W, M_inv = precompute_wls_operators(edge_index, out.num_nodes, pos)
    out.V, out.W, out.M_inv = V, W, M_inv

    if recompute_width_derivs:
        # `width_d1`/`width_d2` are the ONLY unnormalised inputs the encoder takes, and on the
        # P2 packs they read 1e4-1e5 against a training p95 of 73.8.  They are derivatives of
        # `width_nd`, so on the corner graph they must be re-derived, not carried over: a value
        # computed with the P2 operator is meaningless on P1 connectivity.
        _set_width_derivatives(out, V, W, M_inv)

    return out, CornerMap(corner_ids, midside_ids, ends, n)


def _set_width_derivatives(g, V, W, M_inv) -> None:
    """``width_d1``/``width_d2`` as directional derivatives along the local flow tangent."""
    from src.data_gen.lib.node_feature_assembly import flow_direction_from_wall_normals
    from src.utils.math_operators import wls_derivatives

    n = int(g.num_nodes)
    width = g.x[:, COL_WIDTH].reshape(-1, 1)
    dir_x, dir_y = flow_direction_from_wall_normals(g.x[:, COL_WALL_NORMAL], g.x[:, COL_XY])

    def _grad(f):
        d = wls_derivatives(f, g.edge_index, n, V, W, M_inv)
        return d[:, 0, 0], d[:, 1, 0]

    gx, gy = _grad(width)
    d1 = gx * dir_x + gy * dir_y
    gx2, gy2 = _grad(d1.reshape(-1, 1))
    d2 = gx2 * dir_x + gy2 * dir_y
    g.x[:, COL_WIDTH_D1] = d1.to(g.x.dtype)
    g.x[:, COL_WIDTH_D2] = d2.to(g.x.dtype)


def lift_to_full_mesh(values: torch.Tensor, cmap: CornerMap) -> torch.Tensor:
    """Scatter a corner-graph field back onto the full P2 node set.

    Mid-side nodes take the mean of the two corners they bridge, which is exact to 2nd order
    for a field sampled at the midpoint.  ``values`` is ``[Nc]`` or ``[Nc, C]``.
    """
    single = values.dim() == 1
    v = values.reshape(values.shape[0], -1)
    full = torch.zeros(cmap.num_nodes_full, v.shape[1], dtype=v.dtype, device=v.device)
    full[cmap.corner_ids] = v
    if cmap.midside_ids.numel():
        full[cmap.midside_ids] = 0.5 * (full[cmap.midside_ends[:, 0]] + full[cmap.midside_ends[:, 1]])
    return full.reshape(-1) if single else full


__all__ = [
    "CornerMap",
    "build_corner_graph",
    "identify_midside_nodes",
    "lift_to_full_mesh",
]
