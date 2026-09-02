"""Compatibility shim -- implementation lives in ``src/archive/corrector_era/``.

Used only by the legacy mat-growth / biochem deploy path. Default customer and research
sweeps use frozen RGP-DEQ t=0 flow + ``clot_ml_0``.
"""
from src.archive.corrector_era.corrector_coupling import *  # noqa: F403
