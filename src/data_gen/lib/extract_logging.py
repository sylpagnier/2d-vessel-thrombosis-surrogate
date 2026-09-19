"""Quiet COMSOL/mph extract chatter; user-facing status stays on print().

Default extract logs were a JVM/session dump plus one [NEW]/[OK] per export file.
Call ``quiet_comsol_extract_logs`` before ``mph.start()``. Verbose:
``BIOCHEM_EXTRACT_VERBOSE=1`` or ``--verbose``.
"""

from __future__ import annotations

import logging
import os

_MPH_LOGGERS = ("mph",)
_EXPORT_PREFIXES = (
    "mph",
    "src.data_gen.lib.biochem_comsol",
)
_EXPORT_LOGGERS = (
    "src.data_gen.lib.biochem_comsol_auto_export",
    "src.data_gen.lib.biochem_comsol_mesh_export",
    "src.data_gen.lib.biochem_comsol_mph_export",
    "src.data_gen.lib.biochem_comsol_datasets",
)

# CLI ``--verbose`` sticks for later quiet() calls in the same process (e.g. mph.start).
_verbose_forced: bool | None = None


def extract_verbose_requested() -> bool:
    if _verbose_forced is not None:
        return _verbose_forced
    return (os.environ.get("BIOCHEM_EXTRACT_VERBOSE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _iter_extract_loggers():
    names = set(_MPH_LOGGERS + _EXPORT_LOGGERS)
    for name in logging.Logger.manager.loggerDict:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _EXPORT_PREFIXES):
            names.add(name)
    for name in sorted(names):
        yield logging.getLogger(name)


def quiet_comsol_extract_logs(*, verbose: bool | None = None) -> None:
    """Drop mph session/JVM INFO and per-file export INFO unless verbose.

    Pass ``verbose=True`` from ``--verbose``. Omit (or pass None) to honor
    ``BIOCHEM_EXTRACT_VERBOSE``. Do not pass ``verbose=False`` from the CLI
    when the flag is absent -- that would ignore the env override.
    """
    global _verbose_forced
    if verbose is not None:
        _verbose_forced = bool(verbose)
        if verbose:
            os.environ["BIOCHEM_EXTRACT_VERBOSE"] = "1"
    level = logging.INFO if extract_verbose_requested() else logging.WARNING
    for logger in _iter_extract_loggers():
        logger.setLevel(level)
