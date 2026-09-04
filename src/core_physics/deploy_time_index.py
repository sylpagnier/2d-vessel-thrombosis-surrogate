"""Which time index a deploy evaluation reports at.

Extracted from ``species_pushforward_continuous``.  These five functions are
pure index arithmetic over a graph's macro-step count plus one runtime knob, and
they are the *only* thing the shipped ``clot_ml`` feature builder needed from
that 190 KB species module -- so importing them dragged the whole retired species
stack into the product's import closure.

``species_pushforward_continuous`` re-exports them, so existing importers keep
working.
"""

from __future__ import annotations

import os

#: The short-horizon deploy checkpoint the pre-2026-08 runs reported at.  Kept
#: because ``default_deploy_metric_times`` still probes it as a mid-timeline point.
LEGACY_CAPPED_DEPLOY_HORIZON = 53


def graph_last_time_index(n_times: int) -> int:
    """Last macro-step index for a graph with ``n_times`` knots (0-based)."""
    return max(int(n_times) - 1, 0)


def legacy_capped_deploy_time_index(n_times: int) -> int:
    """Legacy short-horizon deploy checkpoint."""
    return min(LEGACY_CAPPED_DEPLOY_HORIZON, graph_last_time_index(n_times))


def deploy_horizon_steps() -> int:
    """Explicit deploy horizon cap, or 0 for "run to the graph's last step"."""
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(int(rt.rollout.deploy_horizon), 0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_HORIZON") or "0").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 0


def deploy_eval_use_full_timeline() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.rollout.deploy_eval_full)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on", "full", "last")


def deploy_eval_time_index(n_times: int) -> int:
    """Deploy metric time index: per-graph last step unless explicitly capped."""
    last = graph_last_time_index(n_times)
    if deploy_eval_use_full_timeline():
        return last
    h = deploy_horizon_steps()
    if h > 0:
        return min(h, last)
    return last


def resolve_deploy_eval_time_index(n_times: int, *, time_index: int | None = None) -> int:
    """Resolve eval index from an explicit override or the deploy convention."""
    if time_index is not None:
        return max(0, min(int(time_index), graph_last_time_index(n_times)))
    return deploy_eval_time_index(n_times)


def default_deploy_metric_times(n_times: int) -> list[int]:
    """Standard deploy eval grid: t=0, mid probes, per-graph last."""
    last = graph_last_time_index(n_times)
    candidates = (0, 27, legacy_capped_deploy_time_index(n_times), last)
    return sorted({max(0, min(int(t), last)) for t in candidates})
