"""Biochem deploy baseline trainer (archived).

See ``src/archive/mat_growth/train_biochem_gnn.py``.
"""
from __future__ import annotations

import runpy
import warnings

warnings.warn(
    "train_biochem_gnn is archived under src/archive/mat_growth/.",
    DeprecationWarning,
    stacklevel=2,
)

from src.archive.mat_growth.train_biochem_gnn import *  # noqa: F403

if __name__ == "__main__":
    runpy.run_module(
        "src.archive.mat_growth.train_biochem_gnn",
        run_name="__main__",
        alter_sys=True,
    )
