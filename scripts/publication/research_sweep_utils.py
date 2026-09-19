"""Shared helpers for research-sweep publication figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.publication.config import CONFIG, RESEARCH_SWEEP_DATA_DIR


def sweep_output_dir(sweep_id: str) -> Path:
    return CONFIG.research_sweep_root / sweep_id


def _resolved_model(name: str) -> str:
    """Artifact directory a model NAME designates today, following the shipped pointer."""
    try:
        from src.clot_ml.artifacts import LEGACY_NAMES, pointer
    except ImportError:
        return name
    if name in LEGACY_NAMES:
        return str(pointer().get("name") or name)
    return name


def load_sweep_summary(sweep_id: str) -> dict[str, Any]:
    path = sweep_output_dir(sweep_id) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sweep summary: {path}. Run scripts/run_research_sweep.py --sweep {sweep_id} first."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))

    # `run_research_sweep.py` writes every model's sweeps to ONE root, but the figures are
    # profile-scoped.  So the raw data can silently belong to a different generation than the
    # figures being built from it: on 2026-09-06 this root held `DeployClotS_0` output from
    # the split sequence, while the shipped figure set was still the 2026-09-04 build against
    # `DeployClot2_0`.  Regenerating the shipped figures would have swapped the model under
    # them without changing a filename, a caption, or a single line of output.
    want = _resolved_model(CONFIG.clot_ml_model)
    got = _resolved_model(str(summary.get("clot_model") or summary.get("model") or ""))
    if want and got and want != got:
        raise SystemExit(
            f"sweep data under {path.parent} was generated with clot model {got!r}, but the "
            f"active '{CONFIG.profile}' profile builds figures for {want!r}.\n"
            f"Regenerating would mislabel the model. Either re-run the sweeps for this "
            f"profile:\n"
            f"    python scripts/run_research_sweep.py --all --clot-model {want}\n"
            f"or build the figures under the profile the data belongs to (--profile).")
    return summary


def summary_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten sweep summary arms into a metrics table."""
    rows = summary.get("arms") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def collect_sweep_metrics(sweep_ids: list[str]) -> pd.DataFrame:
    """Load and tag multiple sweep summaries."""
    frames: list[pd.DataFrame] = []
    for sid in sweep_ids:
        summary = load_sweep_summary(sid)
        df = summary_to_dataframe(summary)
        if df.empty:
            continue
        df.insert(0, "sweep_id", sid)
        df.insert(1, "sweep_axis", summary.get("axis", ""))
        df.insert(2, "sweep_title", summary.get("title", ""))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_research_sweep_metrics(df: pd.DataFrame, name: str = "research_sweep_metrics.csv") -> Path:
    RESEARCH_SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RESEARCH_SWEEP_DATA_DIR / name
    df.to_csv(out, index=False)
    return out
