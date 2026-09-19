"""The critical-point figure: when does a geometry cross into clot-friendly territory.

Reads the already-computed 3x3 interaction grid (bendiness x stenosis occlusion,
``12_bend_x_stenosis`` in ``table_research_sweeps.csv``) and renders it as a heatmap, because
it is the one sweep in the corpus that varies two geometric axes jointly and therefore the
only one that can show a *joint* transition rather than a single-axis curve.

This does not fit or assert a decision boundary -- there are only 9 grid cells and no COMSOL
ground truth at any of them (same caveat as the rest of chapter 3's geometry-response
sweeps). What the figure shows, and only this, is the surrogate's own predicted response
surface. The stenosis axis is strength as defined everywhere else: the percentage reduction of
the vessel width at the stenosis centre.

The figure used to carry a hard-coded "cliff between 0.50 and 0.75 at every bend level" line. That
was true of the 2026-09-04 data and false after the 2026-09-14 graph-builder alignment (a mild bend
already carries ~30% wall clot with no stenosis), so the annotation is now computed: each row's
largest step is found, and a line is drawn only when every row puts it in the same interval.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.publication.config import CONFIG, RESEARCH_SWEEP_FIG_DIR
from scripts.publication.utils import setup_matplotlib_style

METRIC = "wall_clot_pct_final"
SWEEP_ID = "12_bend_x_stenosis"

# Row order (bendiness, weakest to strongest) and the occlusion values swept at each.
BEND_ORDER = ("straight", "arc_mild", "arc_strong")
BEND_LABEL = {"straight": "straight", "arc_mild": "mild bend", "arc_strong": "strong bend"}
OCC_ORDER = (0.0, 0.5, 0.75)


def main() -> int:
    setup_matplotlib_style()
    csv_path = RESEARCH_SWEEP_FIG_DIR / "table_research_sweeps.csv"
    if not csv_path.is_file():
        print(f"[critical-point] missing {csv_path}; run plot_research_sweep_figures.py first")
        return 1
    df = pd.read_csv(csv_path)
    sub = df[df["sweep_id"] == SWEEP_ID].copy()
    if sub.empty:
        print(f"[critical-point] no rows for {SWEEP_ID}")
        return 1

    sub["bend"] = sub["name"].str.extract(r"^(straight|arc_mild|arc_strong)")
    sub["occ"] = pd.to_numeric(sub["axis_value"], errors="coerce")

    grid = np.full((len(BEND_ORDER), len(OCC_ORDER)), np.nan)
    for i, bend in enumerate(BEND_ORDER):
        for j, occ in enumerate(OCC_ORDER):
            row = sub[(sub["bend"] == bend) & np.isclose(sub["occ"], occ)]
            if not row.empty:
                grid[i, j] = pd.to_numeric(row[METRIC], errors="coerce").iloc[0]

    if np.isnan(grid).any():
        print("[critical-point] WARNING: missing cell(s) in the 3x3 grid:")
        print(grid)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(grid, cmap="magma", aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(OCC_ORDER)))
    ax.set_xticklabels([f"{100 * o:.0f}%" for o in OCC_ORDER])
    ax.set_yticks(range(len(BEND_ORDER)))
    ax.set_yticklabels([BEND_LABEL[b] for b in BEND_ORDER])
    ax.set_xlabel("stenosis: width reduction at centre")
    ax.set_ylabel("vessel bendiness")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                color = "white" if grid[i, j] < 0.6 * np.nanmax(grid) else "black"
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                         color=color, fontsize=CONFIG.font_size)

    # Each row's largest step; annotate only a location every bend level agrees on.
    steps = np.nanargmax(np.diff(grid, axis=1), axis=1)
    if len(set(steps.tolist())) == 1:
        k = int(steps[0])
        ax.axvline(k + 0.5, color="cyan", lw=1.5, ls="--")
        note = (f"dashed line: every bend level's largest rise is between "
                f"{100 * OCC_ORDER[k]:.0f}% and {100 * OCC_ORDER[k + 1]:.0f}% narrowing")
    else:
        note = "bend levels disagree on where the largest rise is; no single step is marked"

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"{METRIC.replace('_', ' ')} (%)")
    ax.set_title(
        "Predicted clot burden vs. bendiness x stenosis strength\n"
        "(model response only -- no COMSOL ground truth at these grid points)",
        loc="left", fontsize=CONFIG.font_size - 1)
    fig.text(0.5, 0.005, note, ha="center", va="bottom", color="0.15",
             fontsize=CONFIG.font_size - 2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = RESEARCH_SWEEP_FIG_DIR / f"critical_point.{CONFIG.fig_format}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[critical-point] wrote {out_path}")
    print(pd.DataFrame(grid, index=BEND_ORDER, columns=OCC_ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
