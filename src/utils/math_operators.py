import torch


def scatter_add(src, index, dim=0, dim_size=None):
    """Standalone replacement for torch_scatter.scatter_add."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out_size = list(src.size())
    out_size[dim] = dim_size
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index = index.view(view_shape).expand_as(src)
    return out.scatter_add_(dim, index, src)


def _effective_upwind_weights(row, V, W, boundary_mask, boundary_normals):
    """
    Build one-sided/upwind WLS edge weights for boundary rows.
    For boundary node i with outward normal n_i, keep edges (i->j) with (x_j-x_i)·n_i <= 0.
    """
    w_eff = W.clone()
    if boundary_mask is None or boundary_normals is None:
        return w_eff
    if boundary_mask.numel() == 0 or (not boundary_mask.any()):
        return w_eff

    bmask = boundary_mask.view(-1).bool()
    normals = boundary_normals
    if normals.dim() != 2 or normals.shape[1] != 2:
        return w_eff

    edge_on_boundary = bmask[row]
    if not edge_on_boundary.any():
        return w_eff

    n_row = normals[row]
    n_norm = torch.linalg.norm(n_row, dim=1)
    valid_normal = n_norm > 1e-12
    dot = V[:, 0] * n_row[:, 0] + V[:, 1] * n_row[:, 1]
    # Keep tangential and interior-pointing displacements, suppress outward-pointing edges.
    keep = (dot <= 1e-12) | (~edge_on_boundary) | (~valid_normal)
    w_eff = torch.where(keep, w_eff, torch.zeros_like(w_eff))
    return w_eff


def _recompute_boundary_minv(row, V, w_eff, M_inv, boundary_mask):
    """Rebuild local normal equations for boundary rows after one-sided edge filtering."""
    if boundary_mask is None or boundary_mask.numel() == 0 or (not boundary_mask.any()):
        return M_inv

    M_inv_eff = M_inv.clone()
    v_col = V.unsqueeze(2)
    v_row = V.unsqueeze(1)
    M_e = w_eff.view(-1, 1, 1) * torch.bmm(v_col, v_row)
    M_flat = scatter_add(M_e.view(-1, 25), row, dim=0, dim_size=M_inv.shape[0]).view(-1, 5, 5)

    bmask = boundary_mask.view(-1).bool()
    if bmask.shape[0] != M_flat.shape[0]:
        return M_inv_eff

    # RGP_DEQ_REPAIR_PLAN.md D5.  Boundary rows are the MOST likely to be rank-deficient
    # (one-sided, often collinear stencils), so ridging them before `pinv` is worst exactly
    # where it matters.  Truncate instead; see `mesh_wls.rank_aware_pinv_sym`.
    from src.data_gen.lib.mesh_wls import rank_aware_pinv_sym

    M_inv_eff[bmask] = rank_aware_pinv_sym(M_flat[bmask]).to(M_inv_eff.dtype)
    return M_inv_eff


def wls_derivatives(
    field,
    edge_index,
    num_nodes,
    V,
    W,
    M_inv,
    boundary_mask=None,
    boundary_normals=None,
):
    """Compute WLS derivatives [x, y, xx, xy, yy] for nodal fields."""
    row, col = edge_index

    if M_inv.dim() == 4 and M_inv.shape[1] == 1:
        M_inv = M_inv.squeeze(1)

    u = field if field.dim() == 2 else field.unsqueeze(-1)
    if u.dim() != 2:
        raise ValueError(f"wls_derivatives expects [N] or [N,C], got {tuple(field.shape)}")
    if u.shape[0] != int(num_nodes):
        raise ValueError(f"wls_derivatives expected N={num_nodes}, got {u.shape[0]}")

    w_eff = _effective_upwind_weights(row, V, W, boundary_mask, boundary_normals)
    M_inv_eff = _recompute_boundary_minv(row, V, w_eff, M_inv, boundary_mask)

    du = u[col] - u[row]
    b_e = w_eff.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), du.unsqueeze(1))
    channels = u.shape[1]
    b_flat = scatter_add(b_e.view(-1, 5 * channels), row, dim=0, dim_size=num_nodes)
    b = b_flat.view(num_nodes, 5, channels)
    out = torch.bmm(M_inv_eff, b)
    return _fill_rank_deficient_rows(out, M_inv_eff, row, col, num_nodes)


#: A node's WLS row is usable only if its stencil resolves all 5 Taylor terms.
_WLS_FULL_RANK = 5

#: Which rows of a given operator are rank-deficient is a property of the MESH, not of the
#: field being differentiated, but finding it costs an eigendecomposition of N 5x5 matrices.
#: Training differentiates the same graphs thousands of times, so memoise on the operator's
#: identity.  Bounded because a curriculum can cycle through a few hundred graphs.
_DEFICIENT_CACHE: dict = {}
_DEFICIENT_CACHE_MAX = 512


def _deficient_rows(M_inv_eff):
    key = (M_inv_eff.data_ptr(), tuple(M_inv_eff.shape), str(M_inv_eff.device),
           str(M_inv_eff.dtype))
    hit = _DEFICIENT_CACHE.get(key)
    if hit is not None and hit.shape[0] == M_inv_eff.shape[0]:
        return hit
    evals = torch.linalg.eigvalsh(M_inv_eff.detach().double())
    lam_max = evals.abs().amax(dim=1, keepdim=True).clamp(min=1e-300)
    rank = ((evals.abs() / lam_max) > 1e-10).sum(dim=1)
    out = rank < _WLS_FULL_RANK
    if len(_DEFICIENT_CACHE) >= _DEFICIENT_CACHE_MAX:
        _DEFICIENT_CACHE.clear()
    _DEFICIENT_CACHE[key] = out
    return out


def _fill_rank_deficient_rows(deriv, M_inv_eff, row, col, num_nodes):
    """Replace derivative rows the stencil cannot resolve with a neighbour average.

    RGP_DEQ_REPAIR_PLAN.md D5.  COMSOL exports ``triangle6``, so ~74.6% of every biochem mesh
    node is a P2 mid-side vertex of degree 2 whose two edge vectors are exactly antiparallel.
    Its 5-term normal matrix has **rank 2 of 5** -- measured on ``patient020``: 14699 nodes at
    rank 2, 4679 at rank 5.  Recovering a known quadratic through the operator gives a relative
    error of **4.2e-16 on the rank-5 rows and 0.72 on the rank-2 rows**: full-rank nodes are
    exact to machine precision, deficient ones carry no usable derivative at all.

    Truncation (:func:`mesh_wls.rank_aware_pinv_sym`) makes those rows honestly incomplete
    rather than noisy, but incomplete is still unusable, and every downstream shear/continuity
    term reads them.  A mid-side node lies exactly on the segment between its two corner
    neighbours, so averaging the neighbours' full-rank derivatives is 2nd-order exact.  This is
    the same repair that took the potential-flow direction from cos 0.65 to 0.99 (§1f/§1g).

    A P1 mesh has no deficient rows and this is a no-op.
    """
    deficient = _deficient_rows(M_inv_eff)
    if not bool(deficient.any()):
        return deriv

    out = deriv
    good = (~deficient).to(deriv.dtype)
    for _ in range(3):
        if not bool(deficient.any()):
            break
        flat = out.reshape(num_nodes, -1)
        acc = scatter_add(flat[col] * good[col].unsqueeze(1), row, dim=0, dim_size=num_nodes)
        cnt = scatter_add(good[col], row, dim=0, dim_size=num_nodes)
        fillable = deficient & (cnt > 0)
        if not bool(fillable.any()):
            break
        filled = torch.where(
            fillable.unsqueeze(1), acc / cnt.clamp(min=1.0).unsqueeze(1), flat
        )
        out = filled.view_as(deriv)
        good = good.clone()
        good[fillable] = 1.0
        deficient = deficient & ~fillable
    return out


def sparse_gradient(field, G_x, G_y):
    """Compute sparse gradient components for a scalar nodal field."""
    col = field.unsqueeze(1) if field.dim() == 1 else field
    gx = torch.sparse.mm(G_x, col).squeeze(1)
    gy = torch.sparse.mm(G_y, col).squeeze(1)
    return gx, gy
