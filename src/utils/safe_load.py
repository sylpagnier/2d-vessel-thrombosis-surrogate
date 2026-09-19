"""Load a PyG graph from a file the user did not build: no pickle code execution.

``torch.load(..., weights_only=False)`` unpickles arbitrary objects, so a crafted ``.pt`` runs
code on load.  Research scripts read the project's own artifacts and keep the permissive form;
anything a customer hands the app (web upload, inbox, retrain folder) goes through here, which
admits tensors, containers and the four PyG classes a graph pack actually contains -- measured
over the customer demo and every anchor graph, nothing else appears.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

GRAPH_SAFE_GLOBALS = [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]

# CVE-2025-32434: before torch 2.6 a crafted file could run code even with weights_only=True.
MIN_SAFE_TORCH = (2, 6)


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
    return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)


def load_untrusted_graph(path: Path | str, map_location: str = "cpu"):
    """``torch.load`` restricted to tensors and PyG graph classes (``weights_only=True``)."""
    if _torch_version() < MIN_SAFE_TORCH:
        raise RuntimeError(
            f"torch {torch.__version__} cannot load untrusted files safely; "
            f"upgrade to torch>={'.'.join(map(str, MIN_SAFE_TORCH))}."
        )
    with torch.serialization.safe_globals(GRAPH_SAFE_GLOBALS):
        return torch.load(path, map_location=map_location, weights_only=True)
