"""Diagnostic probes (ad-hoc analysis CLIs)."""

from src.tools.diagnostics.registry import DIAGNOSTICS, resolve_main

__all__ = ["DIAGNOSTICS", "resolve_main"]
