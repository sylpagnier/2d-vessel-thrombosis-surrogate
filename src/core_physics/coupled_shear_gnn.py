"""Compatibility shim -- implementation lives in ``src/archive/corrector_era/``.

The local kinematic corrector is used only by the legacy mat-growth / biochem deploy path.
Default customer and research sweeps use frozen RGP-DEQ t=0 flow + ``clot_ml_0``.
"""
from src.archive.corrector_era.coupled_shear_gnn import *  # noqa: F403
