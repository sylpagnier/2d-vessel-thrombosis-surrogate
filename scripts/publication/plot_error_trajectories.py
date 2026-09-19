"""Figure 0.6b: an error at one timestep usually recovers — unless it is the last timestep.

  (a) wall and (b) off-wall BATC over time for comsol014 and comsol037 (the cohort's faint lines were
      removed on review, 2026-09-15; `dip_stats` still counts every out-of-fold vessel).
  (c) comsol014, model vs ground truth at t = 40 (the wall dip), t = 60 (recovered) and t = 200
      (the final frame, where off-wall false positives are still growing).
  (d) comsol037 and (e) comsol040, counter-examples: a late off-wall error (t = 160) that has
      recovered by the final frame.

Panel labels only; the argument and its numbers are the caption's (docs/publication/FIGURES.md
0.6b).  `dip_stats` is printed so the caption's counts come from the same series.

Reads `data/oof_batc_series.json` (`example_vessels.oof_batc_series`, shipped BATC config).

    python scripts/publication/plot_error_trajectories.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG, FIG_DIR
from scripts.publication.example_vessels import oof_batc_series, oof_series, score_tag, zoom_for
from scripts.publication.pub_style import apply_style, plot_clot_field

FOCUS = "comsol014"
FOCUS_COLOUR = "#dd6b20"
SNAP_TIMES = (40, 60, 200)
#: Counter-examples to comsol014's growing final-frame error: an off-wall error late in the run
#: that has recovered by t = 200 (comsol037 misses all 10 GT nodes at t = 160, BATC 0.00 -> 0.93;
#: comsol040 is at 0.55 at t = 160, 0.97 final).  Picked by `late_recoveries`.
RECOVERERS = {"comsol037": (160, 200), "comsol040": (160, 200)}
#: drawn beside comsol014 in (a)/(b): the late off-wall miss that recovers
COMPARE, COMPARE_COLOUR = "comsol037", "#2b6cb0"
DIP, RECOVER = 0.6, 0.9
MAP_KW = dict(s_lumen=4, s_clot_wall=14, s_clot_off=14, s_wall_open=7)


def dip_stats(series: dict, dom: str) -> tuple[int, int, int]:
    """(vessels that dip below DIP mid-run, of which end >= RECOVER, that end at their lowest)."""
    dipped = recovered = final_low = 0
    for s in series.values():
        y = np.asarray(s[dom]["score"], dtype=float)
        mid = y[1:-1]
        if mid.size and mid.min() < DIP:
            dipped += 1
            if y[-1] >= RECOVER:
                recovered += 1
        if y[-1] < DIP and (not mid.size or y[-1] <= mid.min()):
            final_low += 1
    return dipped, recovered, final_low


def late_recoveries(series: dict, dom: str = "off", after: int = 120) -> list:
    """Vessels whose score drops below DIP at some t >= ``after`` before the last frame, yet end >= RECOVER."""
    out = []
    for stem, s in series.items():
        y, t = s[dom]["score"], s["times"]
        late = [y[k] for k in range(len(t) - 1) if t[k] >= after]
        if late and min(late) < DIP and y[-1] >= RECOVER:
            out.append(stem)
    return out


def _snapshots(fig, grid, series: dict, cells: list, off_wall_zoom: bool = False) -> None:
    """Model (top) over ground truth (bottom), one column per (vessel, timestep, panel label).

    Columns of one vessel share a zoom; a vessel's first column carries its panel label.
    ``off_wall_zoom`` frames only the off-wall clot, for errors too small to read at vessel scale.
    """
    cache, zooms = {}, {}
    for stem, _, _ in cells:
        if stem not in cache:
            S = cache[stem] = oof_series(stem)
            ks = [list(S["times"]).index(t) for s, t, _ in cells if s == stem]
            dom = S["off"] if off_wall_zoom else True
            zooms[stem] = zoom_for(S, [(S["pred"][k] | S["gt"][k]) & dom for k in ks],
                                   aspect=1.5, scale=1.6 if off_wall_zoom else 1.0)
    seen = set()
    for c, (stem, t, label) in enumerate(cells):
        S = cache[stem]
        k = list(S["times"]).index(t)
        for r, (field, name) in enumerate(((S["pred"][k], "Model"), (S["gt"][k], "Ground truth"))):
            ax = fig.add_subplot(grid[r, c])
            plot_clot_field(ax, S["pos"], field.astype(float), wall=S["wall"],
                            zoom_limits=zooms[stem], **MAP_KW)
            if r == 0:
                kk = series[stem]["times"].index(t)
                score_tag(ax, f"BATC  wall {series[stem]['wall']['score'][kk]:.2f} · "
                              f"off-wall {series[stem]['off']['score'][kk]:.2f}")
                lead = f"({label}) {stem} · " if stem not in seen else ""
                ax.set_title(f"{lead}t = {t}" + (" (final)" if t == S["times"][-1] else ""),
                             loc="left", fontsize=10)
                seen.add(stem)
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5)


def main() -> None:
    apply_style()
    series = oof_batc_series()

    fig = plt.figure(figsize=(12.0, 12.2))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.9, 1.6], hspace=0.16,
                             left=0.07, right=0.99, top=0.97, bottom=0.02)
    top = outer[0].subgridspec(1, 2, wspace=0.06)
    for i, (dom, label) in enumerate((("wall", "(a) wall"), ("off", "(b) off-wall"))):
        ax = fig.add_subplot(top[0, i])
        f = series[FOCUS]
        ax.plot(f["times"], f[dom]["score"], "-o", color=FOCUS_COLOUR, lw=2.2, ms=3.5,
                label=FOCUS, zorder=5)
        g = series[COMPARE]
        ax.plot(g["times"], g[dom]["score"], "-s", color=COMPARE_COLOUR, lw=2.0, ms=3.5,
                label=f"{COMPARE} (recovers)", zorder=4)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_ylabel("BATC")
            ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        else:
            ax.tick_params(labelleft=False)
        print(f"[i] {dom}: dipped/recovered/final-low = {dip_stats(series, dom)}; "
              f"late dip (t>=120) recovered by the final frame: {late_recoveries(series, dom)}")

    _snapshots(fig, outer[1].subgridspec(2, 3, hspace=0.06, wspace=0.04), series,
               [(FOCUS, t, "c") for t in SNAP_TIMES])
    # counter-examples: a late off-wall error that has recovered by the final frame
    _snapshots(fig, outer[2].subgridspec(2, 4, hspace=0.06, wspace=0.04), series,
               [(stem, t, label) for stem, label in zip(RECOVERERS, "de") for t in RECOVERERS[stem]],
               off_wall_zoom=True)

    for ext in (CONFIG.fig_format, "png"):
        out = FIG_DIR / f"error_trajectories.{ext}"
        fig.savefig(out, dpi=CONFIG.dpi)
        print(f"[OK] Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
