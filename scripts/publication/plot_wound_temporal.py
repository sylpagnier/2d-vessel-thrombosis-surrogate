"""Figures 0.5a-c: tracked over the whole horizon — intact and injured vessels.

0.5a  `wound_temporal`  total clot mass (clotted nodes, % of the vessel -- the app's
      `vessel_clot_pct`), predicted vs ground truth over each vessel's horizon, EVERY frame.
      (a) 27 out-of-fold intact vessels: median and IQR band of each.  (b)-(g) the 6 injured
      vessels, one panel each.  Series: `example_vessels.clot_mass_series`.
0.5b  `wound_example_comsol003`, 0.5c `wound_example_comsol006`
      one injured vessel each: model beside ground truth at clot onset, two intermediate frames
      and the FINAL frame, with the wound drawn as the app draws it (faint band and dashed cut lines) and BATC on\n      every model panel.

Protocols differ and the caption must say so: intact is strict out-of-fold; injured is
leave-one-vessel-out on the complement's two scalars, with a GNN base that never saw a wound.
Panel labels only; the argument is in docs/publication/FIGURES.md 0.5.

    python scripts/publication/plot_wound_temporal.py
"""
from __future__ import annotations

import argparse
import json

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.example_vessels import (WOUND_COLOUR, clot_mass_series, mark_wound,
                                                 score_tag, wall_with_wound, wound_batc_series,
                                                 wound_series, zoom_for)
from scripts.publication.pub_style import apply_style, plot_clot_field

#: injured example vessels and how far to zoom out around the wound region
EXAMPLES = {"wound_comsol003": 1.8, "wound_comsol006": 1.1}
GRID = np.linspace(0, 1, 21)
MAP_KW = dict(s_lumen=5, s_clot_wall=22, s_clot_off=22, s_wall_open=10)


def _curve_panels(fig_path: str, mass: dict) -> None:
    """Total clot mass (% of vessel), predicted vs ground truth, over each vessel's horizon."""
    series = {"model": CONFIG.color_model, "ground truth": CONFIG.color_gt}
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, 5, wspace=0.32, hspace=0.42)

    a = fig.add_subplot(gs[:, :2])
    for key, (name, colour) in zip(("pred", "gt"), series.items()):
        C = np.asarray([np.interp(GRID, np.asarray(s["times"], float) / max(s["times"][-1], 1),
                                  np.asarray(s[key], float)) for s in mass["intact"].values()])
        a.fill_between(GRID, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                       color=colour, alpha=0.16, lw=0)
        a.plot(GRID, np.median(C, 0), color=colour, lw=2.2, label=f"{name} (median, IQR)")
    a.set_title(f"(a) intact · {len(mass['intact'])} out-of-fold vessels", loc="left", fontsize=9.5)
    a.set_ylabel("total clot mass (% of vessel)")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    axes = [a]

    for i, (stem, s) in enumerate(mass["wound"].items()):
        b = fig.add_subplot(gs[i // 3, 2 + i % 3])
        x = np.asarray(s["times"], float) / max(s["times"][-1], 1)
        for key, colour in zip(("pred", "gt"), series.values()):
            b.plot(x, s[key], "-", color=colour, lw=1.6)
        b.set_title(f"({'bcdefg'[i]}) injured · {stem.replace('wound_comsol', 'w')}",
                    loc="left", fontsize=9)
        b.set_ylim(bottom=0)
        axes.append(b)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    a.set_ylim(bottom=0)
    a.set_xlabel("fraction of horizon", fontsize=8.5)
    for ax in axes[4:]:
        ax.set_xlabel("fraction of horizon", fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.11)
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{fig_path}.{ext}", dpi=CONFIG.dpi)
    plt.close(fig)
    print(f"[i] wrote {fig_path}.{CONFIG.fig_format} / .png")


def _example(fig_path: str, stem: str, z, series: dict, label: str, scale: float) -> None:
    W = wound_series(stem, z)
    onset = next(i for i, n in enumerate(series["w_reg"]["n_gt"]) if n > 0)
    last = len(series["times"]) - 1
    ks = sorted({onset, onset + (last - onset) // 3, onset + 2 * (last - onset) // 3, last})
    region = W["domains"]["w_reg"] | W["domains"]["w_lum"]
    zoom = zoom_for(W, [region] + [(W["pred"][k] | W["gt"][k]) & region for k in ks],
                    aspect=2.0, scale=scale)

    fig, axes = plt.subplots(len(ks), 2, figsize=(11.0, 2.75 * len(ks) + 0.5))
    for r, k in enumerate(ks):
        t = int(series["times"][k])
        for c, (field, name) in enumerate(((W["pred"][k], "model"), (W["gt"][k], "ground truth"))):
            a = axes[r, c]
            # drawn as the app draws it: wound boundary as wall, faint band, dashed cut lines
            plot_clot_field(a, W["pos"], field.astype(float), wall=wall_with_wound(W),
                            zoom_limits=zoom, **MAP_KW)
            mark_wound(a, W)
            if c == 0:
                score_tag(a, f"BATC  wound region {series['w_reg']['score'][k]:.2f} · "
                             f"lumen {series['w_lum']['score'][k]:.2f} · "
                             f"wall {series['wall']['score'][k]:.2f}")
            if r == 0:
                a.set_title(name, fontsize=10.5)
            if c == 0:
                a.set_ylabel(f"t = {t}" + (" (final)" if k == last else ""), fontsize=10)
    fig.text(0.01, 0.995, f"({label}) {stem}", ha="left", va="top", fontsize=11)
    fig.legend(handles=[mlines.Line2D([], [], color=WOUND_COLOUR, ls=(0, (5, 3)), lw=1.4,
                                      label="wound")],
               loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{fig_path}.{ext}", dpi=CONFIG.dpi)
    plt.close(fig)
    print(f"[i] wrote {fig_path}.{CONFIG.fig_format} / .png  (t = "
          f"{[int(series['times'][k]) for k in ks]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    apply_style()
    wound = wound_batc_series()
    mass = clot_mass_series()
    _curve_panels(str(FIG_DIR / "wound_temporal"), mass)
    final = {g: {s: (v["pred"][-1], v["gt"][-1]) for s, v in mass[g].items()} for g in mass}
    print("[i] final clot mass (pred, gt) %:", json.dumps(final))
    z = np.load(DATA_DIR / "wound_series_fem.npz", allow_pickle=True)
    for label, (stem, scale) in zip("bc", EXAMPLES.items()):
        _example(str(FIG_DIR / f"wound_example_{stem.replace('wound_', '')}"), stem, z,
                 wound[stem], label, scale)
    final = {s: {d: round(wound[s][d]["score"][-1], 4) for d in ("w_reg", "w_lum")} for s in wound}
    print("[i] wound final BATC:", json.dumps(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
