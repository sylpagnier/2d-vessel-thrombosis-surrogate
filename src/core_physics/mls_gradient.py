"""Moving-least-squares gradient operators on the unstructured vessel graph.

WHY THIS EXISTS.  The packs ship ``data.G_x`` / ``data.G_y`` sparse operators that every
flow-derived feature (shear rate, shear gradient, both COMSOL deposition gates) is built
on.  Audited 2026-08-09: ``G_x`` has a **median of one non-zero per row** and
``G_x(f=x)`` returns **0 across the interior** (it is only linearly consistent on wall
rows).  So ``gamma_si`` and ``dshear_ds`` as computed in the repo are not derivatives of
the velocity field -- measured against COMSOL's own ``spf.sr`` / ``d(spf.sr,x)`` on
patient007 they rank-correlate 0.19 and 0.00 respectively.

This module replaces them with a weighted moving-least-squares (MLS) fit over
**graph** neighbourhoods (2-hop by default, so the stencil follows the mesh and never
jumps the lumen), using a quadratic basis.  The gradient rows are exact for linear
fields by construction and one-sided-safe at the wall.

Returned operators are ``scipy.sparse`` matrices so a field is differentiated with a
single matvec, and they depend only on node positions + connectivity -- i.e. they are
**deploy-legal** (geometry only, no GT solution).

ROOT CAUSE, for the record.  ``G_x``/``G_y`` are assembled in
``src/data_gen/lib/mesh_to_graph_biochem.py`` as a weighted-least-squares gradient from a
**1-hop** stencil against a **5-term quadratic basis** ``[dx, dy, dx^2, dx*dy, dy^2]``.
Mean node degree in the packs is ~3 (51904 edges / 17413 nodes on patient007), so the
moment matrix is rank-deficient at most interior nodes and its inverse saturates at the
pipeline's 5e6 clamp -- ``M_inv`` row-norm has median exactly 5.0e6 across the interior.
The assembly is structurally correct; the stencil is simply smaller than the basis.  That
is why the operator survives at the wall (denser local connectivity) and dies inside.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch


def _khop_neighbors(edge_index: np.ndarray, n: int, hops: int) -> list[np.ndarray]:
    A = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n)
    ).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    R = A.copy()
    acc = A.copy()
    for _ in range(hops - 1):
        R = (R @ A).astype(bool).astype(np.int8)
        acc = ((acc + R) > 0).astype(np.int8)
    acc.setdiag(1)
    acc = acc.tocsr()
    return [acc.indices[acc.indptr[i]:acc.indptr[i + 1]] for i in range(n)]


def build_mls_operators(
    pos: np.ndarray,
    edge_index: np.ndarray,
    *,
    hops: int = 2,
    min_pts: int = 8,
    order: int = 2,
    ridge: float = 1e-10,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix | None]:
    """Sparse ``(Dx, Dy, L)`` for this graph; ``L`` is ``None`` unless ``order == 2``.

    The same quadratic fit that gives the gradient already carries the second-order
    coefficients, so the Laplacian costs nothing extra.  With the basis written in the
    scaled offset ``dn = (p_j - p_i) / h``, ``f ~ c0 + c1 dnx + c2 dny + c3 dnx^2 +
    c4 dnx dny + c5 dny^2``, so ``f_xx + f_yy = 2 (c3 + c5) / h^2``.

    ``pos`` is [N,2] in whatever length unit the caller wants the derivative in.
    Stencils are graph ``hops``-neighbourhoods, grown one hop at a time until at least
    ``min_pts`` points are available (needed to condition the quadratic basis).
    """
    n = pos.shape[0]
    nbrs = _khop_neighbors(edge_index, n, hops)
    wide = None
    rows_x, cols_x, vals_x = [], [], []
    rows_y, cols_y, vals_y = [], [], []
    rows_l, cols_l, vals_l = [], [], []
    nb_basis = 6 if order == 2 else 3
    for i in range(n):
        idx = nbrs[i]
        if idx.size < max(min_pts, nb_basis):
            if wide is None:
                wide = _khop_neighbors(edge_index, n, hops + 1)
            idx = wide[i]
        d = pos[idx] - pos[i]
        h = np.sqrt((d ** 2).sum(1)).max()
        if h <= 0:
            continue
        dn = d / h
        cols = [np.ones(len(idx)), dn[:, 0], dn[:, 1]]
        if order == 2:
            cols += [dn[:, 0] ** 2, dn[:, 0] * dn[:, 1], dn[:, 1] ** 2]
        B = np.stack(cols, 1)
        r = np.sqrt((dn ** 2).sum(1))
        w = np.exp(-(2.0 * r) ** 2)
        w[r == 0] = 1.0
        Bw = B * w[:, None]
        G = B.T @ Bw + ridge * np.eye(B.shape[1])
        try:
            C = np.linalg.solve(G, Bw.T)        # [nb, k] coefficient rows
        except np.linalg.LinAlgError:
            C = np.linalg.pinv(G) @ Bw.T
        gx = C[1] / h
        gy = C[2] / h
        rows_x.append(np.full(len(idx), i)); cols_x.append(idx); vals_x.append(gx)
        rows_y.append(np.full(len(idx), i)); cols_y.append(idx); vals_y.append(gy)
        if order == 2:
            rows_l.append(np.full(len(idx), i)); cols_l.append(idx)
            vals_l.append(2.0 * (C[3] + C[5]) / (h * h))
    Dx = sp.coo_matrix((np.concatenate(vals_x),
                        (np.concatenate(rows_x), np.concatenate(cols_x))), shape=(n, n)).tocsr()
    Dy = sp.coo_matrix((np.concatenate(vals_y),
                        (np.concatenate(rows_y), np.concatenate(cols_y))), shape=(n, n)).tocsr()
    L = None
    if order == 2 and rows_l:
        L = sp.coo_matrix((np.concatenate(vals_l),
                           (np.concatenate(rows_l), np.concatenate(cols_l))), shape=(n, n)).tocsr()
    return Dx, Dy, L


def build_mls_gradient(pos: np.ndarray, edge_index: np.ndarray, **kw
                       ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """``(Dx, Dy)`` only -- see :func:`build_mls_operators`."""
    Dx, Dy, _ = build_mls_operators(pos, edge_index, **kw)
    return Dx, Dy


def shear_rate_2d(ux, uy, vx, vy):
    """COMSOL ``spf.sr`` for a 2D incompressible flow: sqrt(2 eij eij) with eij symmetric."""
    return np.sqrt(2.0 * ux ** 2 + 2.0 * vy ** 2 + (uy + vx) ** 2)


# ---------------------------------------------------------------------------
# Runtime provider: drop-in replacement for ``data.G_x`` / ``data.G_y``
# ---------------------------------------------------------------------------

DEFAULT_HOPS = 3

# Two bounded LRUs. At hops=3 an operator pair is ~17 MB for a 17k-node graph, so caching
# all 26 training vessels as device tensors would be ~444 MB -- material on the 4 GB card
# that standing constraint 5.5 is about. The MLS fit itself (~0.7 s) is what is worth
# keeping, so the scipy factors get the larger cache and only a couple of graphs are held
# as device tensors. Rebuilding costs ~18 s per 26-vessel epoch against a 21-25 min epoch.
_SCIPY_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_DEV_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()


def _cache_cap(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name) or default), 1)
    except ValueError:
        return default


def _lru_get(cache: "OrderedDict", key):
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
    return hit


def _lru_put(cache: "OrderedDict", key, val, cap: int):
    cache[key] = val
    cache.move_to_end(key)
    while len(cache) > cap:
        cache.popitem(last=False)


def gradient_operator_mode() -> str:
    """``mls`` (default) | ``legacy``.  Env: ``BIOCHEM_GRAD_OPERATOR``.

    ``legacy`` restores the packs' ``G_x``/``G_y`` verbatim so any pre-2026-08-09 number
    can be reproduced bit-for-bit.
    """
    return (os.environ.get("BIOCHEM_GRAD_OPERATOR") or "mls").strip().lower()


def gradient_operator_hops() -> int:
    try:
        return int(os.environ.get("BIOCHEM_GRAD_HOPS") or DEFAULT_HOPS)
    except ValueError:
        return DEFAULT_HOPS


def node_positions(data) -> np.ndarray:
    """[N,2] node coordinates in the pack's non-dimensional length unit.

    ``siren_pos`` when present, else the ``x_nd``/``y_nd`` channels of ``data.x``
    (``kine_x_v1_18ch`` channels 0-1), which several packs carry instead.
    """
    pos = getattr(data, "siren_pos", None)
    if pos is not None:
        return pos.detach().cpu().numpy().astype(np.float64)
    return data.x[:, 0:2].detach().cpu().numpy().astype(np.float64)


def _to_torch_sparse(m: sp.csr_matrix, device, dtype) -> torch.Tensor:
    coo = m.tocoo()
    idx = torch.from_numpy(np.stack([coo.row, coo.col])).long()
    val = torch.from_numpy(coo.data).to(dtype)
    return torch.sparse_coo_tensor(idx, val, size=m.shape).coalesce().to(device=device)


def graph_gradient_operators(data, *, device=None, dtype=None, hops: int | None = None):
    """``(G_x, G_y)`` for this graph, in the SAME non-dimensional length unit as the
    packs' operators, so callers need no rescaling.

    Returns the packs' own tensors under ``BIOCHEM_GRAD_OPERATOR=legacy``.  Otherwise
    returns MLS operators built from positions + connectivity, cached per graph.
    """
    dev = device if device is not None else (
        data.G_x.device if getattr(data, "G_x", None) is not None else torch.device("cpu"))
    dt = dtype if dtype is not None else torch.float32
    if gradient_operator_mode() == "legacy":
        return data.G_x.to(device=dev), data.G_y.to(device=dev)

    h = gradient_operator_hops() if hops is None else int(hops)

    # `_compute_derivatives` is called 10-15 times per training step, and BOTH the signature and
    # the fallback path used to read `node_positions(data)` and `data.edge_index` -- each a
    # `.detach().cpu().numpy()`, i.e. a device SYNC plus a ~1.5 MB D2H copy, before the cache was
    # even consulted.  On this cohort that pinned GPU utilisation near 10%.  Two memos fix it
    # without changing a single returned value:
    #   1. the resolved operators, on the object itself -- kills the repeats inside one step;
    #   2. the LRU key from `graph_stem` + node count, so a lookup ACROSS steps needs no copy.
    memo_key = f"_grad_ops_h{h}_{dev}_{dt}"
    memo = getattr(data, memo_key, None)
    if memo is not None:
        return memo

    stem = getattr(data, "graph_stem", None)
    if isinstance(stem, (list, tuple)) and len(stem) == 1:
        stem = stem[0]
    n_nodes = int(getattr(data, "num_nodes", 0) or 0)
    stem_key = (str(stem), n_nodes, h, str(dev), str(dt)) if stem else None
    if stem_key is not None:
        hit = _lru_get(_DEV_CACHE, stem_key)
        if hit is not None:
            try:
                setattr(data, memo_key, hit)
            except Exception:
                pass
            return hit

    # Both the coordinates and the connectivity are required to build an MLS stencil.
    # Anything missing either (synthetic fixtures, packs built before siren_pos) falls
    # back to the shipped operators rather than raising.
    try:
        pos = node_positions(data)
        ei = data.edge_index.detach().cpu().numpy()
    except (AttributeError, TypeError, IndexError):
        return data.G_x.to(device=dev), data.G_y.to(device=dev)

    # Signature avoids hashing the whole coordinate array on every call.
    sig = (int(pos.shape[0]), int(ei.shape[1]), h,
           float(pos[0, 0]), float(pos[-1, 1]), float(pos[:, 0].max()))
    dev_key = sig + (str(dev), str(dt))
    hit = _lru_get(_DEV_CACHE, dev_key)
    if hit is not None:
        return hit
    factors = _lru_get(_SCIPY_CACHE, sig)
    if factors is None:
        factors = build_mls_operators(pos, ei, hops=h)
        _lru_put(_SCIPY_CACHE, sig, factors, _cache_cap("BIOCHEM_GRAD_CACHE_CPU", 12))
    ops = (_to_torch_sparse(factors[0], dev, dt), _to_torch_sparse(factors[1], dev, dt))
    cap = _cache_cap("BIOCHEM_GRAD_CACHE_DEV", 3)
    _lru_put(_DEV_CACHE, dev_key, ops, cap)
    if stem_key is not None:
        _lru_put(_DEV_CACHE, stem_key, ops, cap)
    try:
        setattr(data, memo_key, ops)
    except Exception:
        pass
    return ops


def graph_laplacian_operator(data, *, device=None, dtype=None, hops: int | None = None):
    """``L`` for this graph, in the SAME non-dimensional length unit as the gradients.

    The packs also ship ``data.Laplacian``, and it has the same defect as ``G_x``/``G_y``:
    on the corner-edge packs 74.2% of its rows are empty, and elsewhere it recovers
    ``lap(x^2 + y^2)`` as **1.79 against an exact 4.0** -- the rank-deficient 1-hop stencil
    under-resolves the second-order terms.  Returns the packs' own tensor under
    ``BIOCHEM_GRAD_OPERATOR=legacy``.
    """
    dev = device if device is not None else (
        data.Laplacian.device if getattr(data, "Laplacian", None) is not None else torch.device("cpu"))
    dt = dtype if dtype is not None else torch.float32
    if gradient_operator_mode() == "legacy":
        return data.Laplacian.to(device=dev)

    h = gradient_operator_hops() if hops is None else int(hops)
    memo_key = f"_lap_op_h{h}_{dev}_{dt}"
    memo = getattr(data, memo_key, None)
    if memo is not None:
        return memo

    try:
        pos = node_positions(data)
        ei = data.edge_index.detach().cpu().numpy()
    except (AttributeError, TypeError, IndexError):
        return data.Laplacian.to(device=dev)

    sig = (int(pos.shape[0]), int(ei.shape[1]), h,
           float(pos[0, 0]), float(pos[-1, 1]), float(pos[:, 0].max()))
    dev_key = sig + (str(dev), str(dt), "lap")
    hit = _lru_get(_DEV_CACHE, dev_key)
    if hit is not None:
        return hit
    # Shares `_SCIPY_CACHE` with the gradient provider, so whichever runs first pays the fit.
    factors = _lru_get(_SCIPY_CACHE, sig)
    if factors is None or len(factors) < 3 or factors[2] is None:
        factors = build_mls_operators(pos, ei, hops=h)
        _lru_put(_SCIPY_CACHE, sig, factors, _cache_cap("BIOCHEM_GRAD_CACHE_CPU", 12))
    op = _to_torch_sparse(factors[2], dev, dt)
    _lru_put(_DEV_CACHE, dev_key, op, _cache_cap("BIOCHEM_GRAD_CACHE_DEV", 3))
    try:
        setattr(data, memo_key, op)
    except Exception:
        pass
    return op


def clear_operator_cache() -> None:
    _SCIPY_CACHE.clear()
    _DEV_CACHE.clear()
