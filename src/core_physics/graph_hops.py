"""Hop distance from the vessel wall, over the mesh graph.

Extracted from ``species_pushforward_continuous`` when the species stack was
retired: this BFS is the one piece of that module the shipped deploy pipeline
and the research sweeps still need, and keeping it there meant a 4,800-line
retired module sat in the product's import closure to supply 25 lines of graph
traversal.
"""

from __future__ import annotations

import torch

#: Nodes the BFS never reaches (disconnected, or outside the band) are given this
#: sentinel rather than -1, so downstream comparisons like ``hops <= k`` treat
#: them as "far away" instead of "adjacent to the wall".
UNREACHED_HOPS = 99


def compute_hop_distances(
    edge_index: torch.Tensor,
    wall_mask_band: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """BFS to compute the exact hop distance from the wall for all band nodes."""
    dev = edge_index.device
    hops = torch.full((num_nodes,), -1, dtype=torch.long, device=dev)
    wall_m = wall_mask_band.to(device=dev).bool()
    hops[wall_m] = 0
    row, col = edge_index
    current_mask = wall_m.clone()
    current_hop = 0
    while True:
        neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
        neighbor_mask[col[current_mask[row]]] = True
        next_mask = neighbor_mask & (hops == -1)
        if not next_mask.any():
            break
        current_hop += 1
        hops[next_mask] = current_hop
        current_mask = next_mask
    hops[hops == -1] = UNREACHED_HOPS
    return hops
