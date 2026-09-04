"""Helpers shared by the diagnostic probes.

Every probe needs the anchor packs and most need the wall band, so those live
here rather than being re-derived per probe. `_wall_band` was byte-identical in
four probes and `_gt_inlet` in two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.paths import anchor_packs_dir, get_project_root


def repo_root() -> Path:
    return get_project_root()


def biochem_packs_dir() -> Path:
    """The COMSOL-anchor packs. Alias kept because several probes call it by this name."""
    return anchor_packs_dir()


def wall_band(data, hops: int = 3) -> np.ndarray:
    """Nodes within ``hops`` graph steps of the wall, as a boolean mask."""
    n = int(data.num_nodes)
    row, col = data.edge_index
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n, dtype=torch.bool)
        acc.index_put_((row,), band[col], accumulate=False)
        band = band | acc
    return band.numpy()


def gt_inlet(data):
    """COMSOL's own inlet ``(u, v)`` at t=0, or ``None`` when there is no ground truth.

    Returns ``None`` for parametrically generated vessels (``research_synthetic``),
    which have no COMSOL solve behind them, and for packs whose ``y`` is absent,
    empty, or all zeros -- so a caller can branch on "is there a reference" without
    knowing the pack's provenance.
    """
    if bool(getattr(data, "research_synthetic", False)):
        return None
    y = getattr(data, "y", None)
    if y is None or not torch.is_tensor(y) or y.numel() == 0 or y.shape[1] == 0:
        return None
    cand = y[0, :, 0:2].detach().cpu().numpy()
    if np.isfinite(cand).all() and float(np.abs(cand).max()) > 0.0:
        return cand
    return None
