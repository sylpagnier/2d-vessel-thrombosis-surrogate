"""Shared helpers for research-sweep publication figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.publication.config import CONFIG, RESEARCH_SWEEP_DATA_DIR


def sweep_output_dir(sweep_id: str) -> Path:
    return CONFIG.research_sweep_root / sweep_id


def load_sweep_summary(sweep_id: str) -> dict[str, Any]:
    path = sweep_output_dir(sweep_id) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sweep summary: {path}. Run scripts/run_research_sweep.py --sweep {sweep_id} first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


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
