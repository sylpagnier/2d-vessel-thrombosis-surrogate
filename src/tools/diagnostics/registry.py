"""Diagnostic probe registry (single source of truth)."""

from __future__ import annotations

from typing import Callable

# slug -> import path of module with main(argv) -> int
DIAGNOSTICS: dict[str, str] = {
    "clot-free-headroom": "src.tools.diagnostics.clot_free_headroom",
    "fem-error-indicators": "src.tools.diagnostics.fem_error_indicators",
    "fem-prior-headroom": "src.tools.diagnostics.fem_prior_headroom",
    "fem-warm-start": "src.tools.diagnostics.fem_warm_start",
    "field-calibration": "src.tools.diagnostics.field_calibration",
    "geometry-class-recal": "src.tools.diagnostics.geometry_class_recal",
    "local-fem-accuracy": "src.tools.diagnostics.local_fem_accuracy",
    "lumen-001-vs-007": "src.tools.diagnostics.lumen_001_vs_007",
    "physics-gate-support": "src.tools.diagnostics.physics_gate_support",
    "prior-vs-model-gate": "src.tools.diagnostics.prior_vs_model_gate",
    "residual-head-audit": "src.tools.diagnostics.residual_head_audit",
    "pi-flux-interaction": "src.tools.diagnostics.pi_flux_interaction",
    "wound-composition": "src.tools.diagnostics.wound_composition",
    "wound-p003-causes": "src.tools.diagnostics.wound_p003_causes",
}


def resolve_main(slug: str) -> Callable[[list[str] | None], int]:
    key = str(slug).strip().lower()
    if key not in DIAGNOSTICS:
        known = ", ".join(sorted(DIAGNOSTICS))
        raise KeyError(f"Unknown diagnostic {slug!r}; known: {known}")
    import importlib

    mod = importlib.import_module(DIAGNOSTICS[key])
    main = getattr(mod, "main", None)
    if main is None:
        raise AttributeError(f"{DIAGNOSTICS[key]} has no main()")
    return main


__all__ = ["DIAGNOSTICS", "resolve_main"]
