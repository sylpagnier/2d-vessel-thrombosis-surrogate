"""Elevate a P1 (``triangle``) kinematics graph to the P2 (``triangle6``) topology COMSOL emits.

**Direction of travel** (RGP_DEQ_REPAIR_PLAN.md §8).  The biochem anchor pipeline is expensive
and is not going to change, so it defines the deployment domain and training has to match it.
The alternative -- collapsing deployment onto its corner subgraph (`p1_corner_graph`, §7.2) --
was measured and only fixed shear *amplitude*, so it is kept as a diagnostic, not as the plan.

**What has to match, measured on `patient020`:**

```
corners 5009   midsides 14699   midside fraction 0.7458   midsides/corners 2.935
directed edges 58796:  corner-midside 100.00%   corner-corner 0.00%   midside-midside 0.00%
corner degree median 6.0
```

A mid-side node connects **only** to the two corners of its parent edge.  There are no
corner-corner edges at all -- so elevation is not "add nodes and keep the old edges", it is
"replace every P1 edge with two half-edges".  Getting that wrong reproduces the degree
distribution incorrectly and re-opens the gap this module exists to close.

**Why the labels can be interpolated.**  Measured on the deploy packs, a true P2 mid-side value
against the mean of its two corners:

```
channel   mean rel err   p95      (field scale)
u             0.2-1.0%   0.7-6.3%
v             0.3-2.2%   1.1-5.3%
mu_eff        1.3-3.7%   6.3-10.8%
```

1-2% against a model whose own error is ~15-20%, so interpolated supervision is well inside the
noise floor and no new CFD is needed.  This is a statement about *these* fields on *these*
meshes, not a general claim about P2 finite elements -- re-measure it if the mesh density or
the physics changes materially.
"""

from __future__ import annotations

import torch

COL_XY = slice(0, 2)
COL_SDF = 2
COL_SHEAR_POT = 3
COL_WALL_NORMAL = slice(4, 6)
COL_NODE_TYPE = slice(6, 10)
COL_RHEO = 10
COL_PRIOR = slice(11, 15)
COL_WIDTH = 15
COL_WIDTH_D1 = 16
COL_WIDTH_D2 = 17


def undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Unique ``[E, 2]`` undirected pairs (i < j) from a symmetric directed edge index."""
    a = edge_index.min(dim=0).values
    b = edge_index.max(dim=0).values
    keep = a != b
    pairs = torch.stack([a[keep], b[keep]], dim=1)
    return torch.unique(pairs, dim=0)


def elevate_to_p2(data, *, interpolate_labels: bool = True, keep_wls: bool = True):
    """Return a new graph with a mid-side node inserted on every edge.

    Node ordering is ``[all original corners] + [one mid-side per undirected edge]`` so the
    original node indices are preserved -- anything keyed on them stays valid.

    ``keep_wls=False`` drops the ``V``/``W``/``M_inv`` stencil arrays once the width
    derivatives have been re-derived from them.  They are **47% of an elevated graph's memory**
    (3.9 MB of 8.4 MB on a 24k-node vessel) and nothing downstream reads them:
    ``graph_gradient_operators`` defaults to MLS and rebuilds from positions + connectivity,
    touching ``data.G_x`` only under ``BIOCHEM_GRAD_OPERATOR=legacy``.  Keeping them is also the
    hazard B13 describes -- a stored operator that no longer matches the graph beside it.
    """
    from src.data_gen.lib.mesh_wls import precompute_wls_operators

    n = int(data.num_nodes)
    pairs = undirected_edges(data.edge_index)
    m = int(pairs.shape[0])
    a, b = pairs[:, 0], pairs[:, 1]
    mid_ids = torch.arange(n, n + m, device=data.x.device)
    total = n + m

    # Only corner-midside half-edges, matching COMSOL exactly (0% corner-corner).
    src = torch.cat([a, mid_ids, b, mid_ids])
    dst = torch.cat([mid_ids, a, mid_ids, b])
    edge_index = torch.stack([src, dst], dim=0)

    x = torch.zeros(total, data.x.shape[1], dtype=data.x.dtype, device=data.x.device)
    x[:n] = data.x
    # Every per-node channel is a smooth geometric or physical field sampled on the mesh, so the
    # edge midpoint value is the corner mean to 2nd order.  `wall_normal` is the exception --
    # averaging two unit vectors does not give a unit vector -- so it is renormalised below, and
    # `width_d1/d2` are re-derived from the new topology rather than averaged at all.
    x[n:] = 0.5 * (data.x[a] + data.x[b])
    # Renormalise the NEW rows only.  Averaging two unit vectors does not give a unit vector,
    # but the original corner rows must stay bit-identical: elevation adds nodes, it does not
    # get to quietly rewrite the mesh it was handed.
    wn = x[n:, COL_WALL_NORMAL]
    norm = wn.norm(dim=1, keepdim=True)
    x[n:, COL_WALL_NORMAL] = torch.where(norm > 1e-8, wn / norm.clamp(min=1e-8), wn)

    out = data.__class__()
    out.x = x
    out.edge_index = edge_index
    out.num_nodes = total
    delta = x[edge_index[0], 0] - x[edge_index[1], 0], x[edge_index[0], 1] - x[edge_index[1], 1]
    d2 = torch.stack(delta, dim=1)
    out.edge_attr = torch.cat([d2, d2.norm(dim=1, keepdim=True)], dim=1)

    # A mid-side node inherits a boundary label only when BOTH its parents carry it: a node
    # bridging a wall corner and an interior corner lies in the interior.
    for name in ("mask_inlet", "mask_outlet", "mask_wall", "mask_wound"):
        msk = getattr(data, name, None)
        if not (torch.is_tensor(msk) and msk.reshape(-1).numel() == n):
            continue
        flat = msk.reshape(-1).bool()
        setattr(out, name, torch.cat([flat, flat[a] & flat[b]]))

    # NOTE: `is_anchor` is deliberately NOT copied here -- it is rebuilt below as a per-node
    # mask so mid-side nodes carry no fabricated supervision.
    for name in ("u_ref", "d_bar", "graph_stem", "geometry_level", "config_id",
                 "is_clinical_anchor", "x_schema", "y_schema", "channel_schema_version",
                 "x_channel_names", "y_channel_names", "re_actual", "bend_sign", "mesh_path"):
        v = getattr(data, name, None)
        if v is not None:
            setattr(out, name, v)

    for name in ("u_inlet_bc", "mu_inlet_bc", "mu_wall_bc"):
        v = getattr(data, name, None)
        if torch.is_tensor(v) and v.shape[0] == n:
            setattr(out, name, torch.cat([v, 0.5 * (v[a] + v[b])], dim=0))

    y = getattr(data, "y", None)
    if torch.is_tensor(y) and y.shape[0] == n and interpolate_labels:
        mid_y, n_true = _true_midside_labels(data, x[n:, COL_XY], y, a, b)
        out.y = torch.cat([y, mid_y], dim=0)
        out.p2_midside_true = int(n_true)
        ym = getattr(data, "y_valid_mask", None)
        if torch.is_tensor(ym) and ym.shape[0] == n:
            out.y_valid_mask = torch.cat([ym, ym[a] & ym[b]], dim=0)

    # SUPERVISE THE TRUE LABELS ONLY.  `anchor_node_mask` broadcasts a graph-level
    # ``is_anchor=[1]`` flag to every node, so without this the data term would be computed on
    # interpolated mid-side values at 74.5% of the mesh -- teaching the model that a mid-side
    # value IS the mean of its corners.  That is a 1-2% bias (measured), and it is a bias in
    # exactly the direction that makes the field smoother than a real P2 solution.
    #
    # Mid-side nodes are not unsupervised: continuity, momentum, BC and the wall-band shear
    # terms all still act on them.  They simply do not get a fabricated data label.
    anc = getattr(data, "is_anchor", None)
    if anc is not None:
        flag = anc.reshape(-1) if torch.is_tensor(anc) else torch.tensor([bool(anc)])
        if flag.numel() == n:
            corner_anchor = flag.bool()
        else:                                   # graph-level [1] flag
            corner_anchor = torch.full((n,), bool(flag.any()), dtype=torch.bool)
        out.is_anchor = torch.cat(
            [corner_anchor, torch.zeros(m, dtype=torch.bool, device=corner_anchor.device)]
        )

    V, W, M_inv = precompute_wls_operators(edge_index, total, x[:, COL_XY])
    _set_width_derivatives(out, V, W, M_inv)
    if keep_wls:
        out.V, out.W, out.M_inv = V, W, M_inv
    # The P1 graph this came from, so a caller can supervise on the CORNER subgraph -- where the
    # labels are COMSOL's own rather than an interpolation of them.
    out.p1_edge_index = data.edge_index
    out.p1_num_nodes = n
    return out



def _true_midside_labels(data, mid_xy_nd, y, a, b, *, tol: float = 1e-4):
    """``(mid_labels, n_true)`` -- COMSOL's own mid-side values where the pack carries them.

    **Why the corner mean is not good enough.**  A mid-side value set to the mean of its two
    corners makes the field piecewise-linear along the half-edge *by construction*, so a
    quadratic fit through a stencil containing it reads ~zero curvature.  `dsrx` -- the gate
    branch that decides ~91% of firing wall nodes at the FIT median -- is exactly that
    curvature.  Measured two independent ways: re-elevating a decimated native-P2 deploy pack
    costs **6.2x** of wall `dsrx` spread, and asking COMSOL directly on the generation box
    recovers **2.2-4.4x** (RGP_DEQ_REPAIR_PLAN.md §16.5, PILOT_COHORT_RUNBOOK.md §7.3).

    Matched by POSITION, not by index: the generator, the mesher and this function each build
    the edge list their own way, and an ordering contract between three modules is exactly the
    kind of assumption that fails silently.  Falls back to the corner mean per node, so a pack
    with no probe set -- every pack generated before 2026-08-28, and every one solved at
    ``order_fluid=1``, where COMSOL's interpolant IS the corner mean -- is unchanged.
    """
    lin = 0.5 * (y[a] + y[b])
    probe_xy = getattr(data, "p2_probe_xy_nd", None)
    probe_y = getattr(data, "p2_probe_y", None)
    if not (torch.is_tensor(probe_xy) and torch.is_tensor(probe_y)) or probe_xy.numel() == 0:
        return lin, 0
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(probe_xy.detach().cpu().numpy()).query(
            mid_xy_nd.detach().cpu().numpy()
        )
        hit = torch.as_tensor(dist, dtype=torch.float32) <= tol
        if not bool(hit.any()):
            return lin, 0
        got = probe_y[torch.as_tensor(idx, dtype=torch.long)]
        out = lin.clone()
        c = min(out.shape[1], got.shape[1])
        out[hit, :c] = got[hit, :c].to(out.dtype)
        # No-slip is a boundary condition, not a sample: a wall mid-side is exactly zero even
        # if the solver's interpolant there reads 1e-9.
        mw = getattr(data, "mask_wall", None)
        if torch.is_tensor(mw) and mw.reshape(-1).numel() == y.shape[0]:
            flat = mw.reshape(-1).bool()
            out[(flat[a] & flat[b]), 0:2] = 0.0
        return out, int(hit.sum())
    except Exception:
        return lin, 0

def _set_width_derivatives(g, V, W, M_inv) -> None:
    """Re-derive ``width_d1``/``width_d2`` on the NEW connectivity.

    These are the only unnormalised inputs the encoder takes, and they are derivatives, so a
    value computed on P1 connectivity is meaningless once the graph is P2.  Carrying them over
    is how a nominally-aligned corpus stays misaligned on the two channels that matter most.
    """
    from src.data_gen.lib.node_feature_assembly import flow_direction_from_wall_normals
    from src.utils.math_operators import wls_derivatives

    n = int(g.num_nodes)
    dir_x, dir_y = flow_direction_from_wall_normals(g.x[:, COL_WALL_NORMAL], g.x[:, COL_XY])

    def grad(f):
        d = wls_derivatives(f, g.edge_index, n, V, W, M_inv)
        return d[:, 0, 0], d[:, 1, 0]

    gx, gy = grad(g.x[:, COL_WIDTH].reshape(-1, 1))
    d1 = gx * dir_x + gy * dir_y
    gx2, gy2 = grad(d1.reshape(-1, 1))
    g.x[:, COL_WIDTH_D1] = d1.to(g.x.dtype)
    g.x[:, COL_WIDTH_D2] = (gx2 * dir_x + gy2 * dir_y).to(g.x.dtype)


__all__ = ["elevate_to_p2", "undirected_edges"]
