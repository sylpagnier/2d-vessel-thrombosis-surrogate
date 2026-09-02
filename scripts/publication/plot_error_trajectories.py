"""Limitations figure: does a mid-run error recover, or does it compound?

Reads the per-timestep wall / off-wall scores already exported by
generate_fig3_4_data.py and generate_fig6_data.py (fig34_metrics.csv,
fig6_metrics.csv) -- no new inference, this just re-frames data we already
have as an explicit divergence-vs-convergence claim for the limitations
section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.utils import setup_matplotlib_style  # noqa: E402

VESSEL_COLORS = {
    "patient005": "#c44e52",
    "patient012": "#4c72b0",
    "patient020": "#55a868",
    "patient041": "#8172b2",
    "patient014": "#dd8452",
}

# (vessel, metric, label, xy(time, score), xytext-offset) -- exact values read off
# fig34_metrics.csv / fig6_metrics.csv, not eyeballed off the rendered figure.
CALLOUTS = [
    dict(vessel="patient014", metric="wall", t=40, note="single-frame collapse to 0.0 at "
         "t=40,\nfully recovers to 0.98 by t=100", dx=20, dy=0.32),
    dict(vessel="patient020", metric="off", t=160, note="off-wall dips to 0.14 near t=160,\n"
         "recovers to 0.53 by t=200", dx=-100, dy=0.30),
    dict(vessel="patient014", metric="off", t=200, note="off-wall keeps declining to 0.07 --\n"
         "no recovery within the horizon", dx=-135, dy=0.22),
]


def _load() -> pd.DataFrame:
    frames = []
    for name in ("fig34_metrics.csv", "fig6_metrics.csv"):
        p = DATA_DIR / name
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("[ERR] no metrics CSVs found; run generate_fig3_4_data.py / "
                          "generate_fig6_data.py first")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["vessel", "time"]).sort_values(["vessel", "time"])


def main() -> None:
    print("[i] Plotting error trajectories (divergence vs. convergence)")
    setup_matplotlib_style()
    df = _load()
    vessels = [v for v in VESSEL_COLORS if v in df["vessel"].unique()]

    fig, (ax_w, ax_o) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)

    for v in vessels:
        vdf = df[df["vessel"] == v].sort_values("time")
        color = VESSEL_COLORS[v]
        if "wall" in vdf.columns and vdf["wall"].notna().any():
            ax_w.plot(vdf["time"], vdf["wall"], color=color, lw=2.0, label=v)
        if "off" in vdf.columns and vdf["off"].notna().any():
            ax_o.plot(vdf["time"], vdf["off"], color=color, lw=2.0, label=v)

    for ax, metric, title in ((ax_w, "wall", "wall score over time"),
                              (ax_o, "off", "off-wall score over time")):
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("deploy score")
        ax.set_title(title, loc="left", fontsize=CONFIG.font_size)
        ax.grid(True, alpha=0.25)
        for c in CALLOUTS:
            if c["metric"] != metric:
                continue
            vdf = df[df["vessel"] == c["vessel"]].sort_values("time")
            row = vdf.iloc[(vdf["time"] - c["t"]).abs().argsort()[:1]]
            if row.empty or metric not in row.columns or pd.isna(row[metric].iloc[0]):
                continue
            xy = (float(row["time"].iloc[0]), float(row[metric].iloc[0]))
            ax.annotate(
                c["note"], xy=xy, xytext=(xy[0] + c["dx"], xy[1] + c["dy"]),
                fontsize=CONFIG.font_size - 3, color="0.15",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8, shrinkA=2, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.75", alpha=0.92),
            )

    ax_o.set_xlabel("timestep")
    ax_w.legend(fontsize=CONFIG.font_size - 3, ncol=len(vessels), loc="lower right",
               frameon=False)
    fig.suptitle("Does an error at one timestep compound or recover?", fontsize=CONFIG.font_size + 2,
                fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIG_DIR / f"error_trajectories.{CONFIG.fig_format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
