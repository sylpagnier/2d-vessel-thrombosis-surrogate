"""Shared bootstrap for diagnostic CLIs."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    from src.utils.paths import get_project_root

    return get_project_root()


def biochem_packs_dir() -> Path:
    return repo_root() / "data" / "processed" / "graphs_biochem_anchors"


def bootstrap() -> Path:
    """Ensure repo root and ``scripts/`` are importable (legacy eval_strict helpers)."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scripts = root / "scripts"
    if scripts.is_dir() and str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return root
