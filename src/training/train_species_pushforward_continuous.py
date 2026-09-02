"""Mat-growth species pushforward trainer (archived).

See ``src/archive/mat_growth/train_species_pushforward_continuous.py``.
"""
from __future__ import annotations

import runpy
import warnings

warnings.warn(
    "train_species_pushforward_continuous is archived; use locked checkpoints or clot_ml_0.",
    DeprecationWarning,
    stacklevel=2,
)

from src.archive.mat_growth.train_species_pushforward_continuous import *  # noqa: F403

if __name__ == "__main__":
    runpy.run_module(
        "src.archive.mat_growth.train_species_pushforward_continuous",
        run_name="__main__",
        alter_sys=True,
    )
