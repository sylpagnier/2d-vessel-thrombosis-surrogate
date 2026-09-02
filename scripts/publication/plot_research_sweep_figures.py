"""Plot research sweep sensitivity curves (publication style)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, FIG_DIR, RESEARCH_SWEEP_DATA_DIR
from scripts.publication.research_sweep_utils import load_sweep_summary, summary_to_dataframe
from scripts.publication.utils import setup_matplotlib_style

# Primary metrics for line plots (must exist in summary rows).
PRIMARY_METRICS = (
    "max_occlusion_pct_final",
    "vessel_clot_pct_final",
    "wall_clot_pct_final",
    "wound_region_clot_pct_final",
    "wound_lumen_clot_pct_final",
)


def _plot_sweep(summary: dict, out_dir: Path) -> None:
    sid = str(summary.get("id", "sweep"))
    axis = str(summary.get("axis", "axis_value"))
    df = summary_to_dataframe(summary)
    if df.empty or "axis_value" not in df.columns:
        print(f"[WARN] Skip {sid}: no axis_value column")
        return

    df = df.sort_values("axis_value").reset_index(drop=True)
    metrics = [m for m in PRIMARY_METRICS
               if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().any()]
    if not metrics:
        print(f"[WARN] Skip {sid}: no known metrics in summary")
        return

    # A handful of distinct axis values (e.g. a 0/1 on-off toggle) reads as a
    # near-invisible flat line with orphan single points where a metric is
    # only defined for one category; a bar per category is legible instead.
    categorical = df["axis_value"].nunique(dropna=True) <= 3

    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 2.2 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    x_pos = np.arange(len(df))
    for ax, metric in zip(axes, metrics):
        y = pd.to_numeric(df[metric], errors="coerce")
        valid = y.notna().to_numpy()
        if categorical:
            ax.bar(x_pos[valid], y[valid], color=CONFIG.color_model, width=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{v:g}" for v in df["axis_value"]])
        else:
            ax.plot(df["axis_value"][valid], y[valid], marker="o",
                    color=CONFIG.color_model, linewidth=2)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(axis.replace("_", " "))
    fig.suptitle(str(summary.get("title", sid)), fontsize=CONFIG.font_size + 1)
    out_path = out_dir / f"research_{sid}_sensitivity.{CONFIG.fig_format}"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [OK] Saved {out_path}")


def _plot_wound_panel(out_dir: Path) -> None:
    """Combined wound sweeps panel (16-20) when data exists."""
    wound_ids = list(CONFIG.research_wound_sweeps)
    metric = "wound_region_clot_pct_final"
    fig, ax = plt.subplots(figsize=(7, 4))
    any_line = False
    for sid in wound_ids:
        try:
            summary = load_sweep_summary(sid)
        except FileNotFoundError:
            continue
        df = summary_to_dataframe(summary)
        if df.empty or metric not in df.columns:
            continue
        df = df.sort_values("axis_value")
        if df["axis_value"].nunique(dropna=True) <= 3:
            # A binary/categorical toggle (e.g. wound on/off) doesn't belong
            # on a shared "vs continuous parameter" line chart — it gets its
            # own bar chart from _plot_sweep instead.
            continue
        ax.plot(
            df["axis_value"],
            pd.to_numeric(df[metric], errors="coerce"),
            marker="o",
            linewidth=2,
            label=sid.replace("_", " "),
        )
        any_line = True
    if not any_line:
        plt.close()
        return
    ax.set_xlabel("arm axis value")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Wound sweeps: wound-region clot coverage")
    ax.legend(fontsize=CONFIG.font_size - 2)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"research_wound_panel.{CONFIG.fig_format}"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  [OK] Saved {out_path}")


def main() -> None:
    print("[i] Plotting research sweep figures")
    setup_matplotlib_style()
    out_dir = FIG_DIR / "research_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = RESEARCH_SWEEP_DATA_DIR / "research_sweep_metrics.csv"
    if metrics_path.is_file():
        df = pd.read_csv(metrics_path)
        df.to_csv(out_dir / "table_research_sweeps.csv", index=False)
        print(f"  [OK] Wrote {out_dir / 'table_research_sweeps.csv'}")

    sweep_ids = list(CONFIG.research_geometry_sweeps) + list(CONFIG.research_wound_sweeps)
    for sid in sweep_ids:
        try:
            summary = load_sweep_summary(sid)
        except FileNotFoundError:
            continue
        _plot_sweep(summary, out_dir)

    _plot_wound_panel(out_dir)
    print("[OK] Research sweep figures complete")


if __name__ == "__main__":
    main()
