"""Wound A/B figure: wound_comsol005 vs. its matched no-wound twin comsol048.

Same vessel geometry (identical node bounding box), one row per variant, zoomed
to the wound-region window (mapped onto both since the coordinate systems are
identical). Illustrative example -- see generate_wound_ab_data.py's docstring
for the caveat on held-out status; the wound section is frozen in
PUBLICATION_NOTES.md pending §7.0 Q1-3.
"""
from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.pub_style import (  # noqa: E402
    apply_style, plot_clot_field, plot_clot_error_map, error_legend_handles,
    clot_zoom_limits, row_height,
)

PANEL_W = 6.2
PAIR = ("wound_comsol005", "comsol048")
ROW_LABEL = {"wound_comsol005": "wound_comsol005  (WOUND)",
            "comsol048": "comsol048  (no wound, matched geometry)"}


def _load(stem: str) -> dict:
    return torch.load(DATA_DIR / f"wound_ab_{stem}.pt", map_location="cpu", weights_only=False)


def main() -> None:
    print("[i] Plotting Wound A/B example")
    apply_style()

    paths = [DATA_DIR / f"wound_ab_{s}.pt" for s in PAIR]
    if not all(p.is_file() for p in paths):
        print("[WARN] missing wound A/B data; run generate_wound_ab_data.py first")
        return

    d_wound, d_now = _load(PAIR[0]), _load(PAIR[1])
    pos = d_wound["pos"]  # identical geometry -- shared across both rows

    wound_region = d_wound["wound_doms"]["region"] | d_wound["wound_doms"]["lumen"]
    zoom = clot_zoom_limits(pos, wound_region, d_wound["wall"], pad_frac=0.6)
    panel_h = row_height(pos, PANEL_W, zoom_limits=zoom)

    top_margin, bottom_margin, mid_pad = 0.7, 0.15, 0.32
    fig_h = 2 * panel_h + top_margin + bottom_margin + mid_pad
    fig = plt.figure(figsize=(3 * PANEL_W + 1.5, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.03, right=0.90, top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
        hspace=0.30, wspace=0.06,
    )

    for row, (stem, d) in enumerate(((PAIR[0], d_wound), (PAIR[1], d_now))):
        last_t = d["times"][-1]
        fd = d["frames"][last_t]
        sc = d["scores"][last_t]
        pred_phi, gt_phi = fd["pred_phi"], fd["gt_phi"]
        pred_b, gt_b = fd["pred_mask"], fd["gt_mask"]

        if d["is_wound"]:
            w_score, o_score = sc.get("w_reg"), sc.get("w_lum")
            model_title = "Model  (wound region / lumen scores)"
        else:
            w_score, o_score = sc.get("wall"), sc.get("off")
            model_title = "Model  (wall / off-wall scores)"

        ax_pred = fig.add_subplot(gs[row, 0])
        ax_gt   = fig.add_subplot(gs[row, 1])
        ax_err  = fig.add_subplot(gs[row, 2])

        plot_clot_field(ax_pred, d["pos"], pred_phi, wall=d["wall"], zoom_limits=zoom,
                        title=f"{model_title}\n{ROW_LABEL[stem]}  (t = {last_t}/{d['T_total']-1})",
                        wall_score=w_score, off_score=o_score)
        plot_clot_field(ax_gt, d["pos"], gt_phi, wall=d["wall"], zoom_limits=zoom,
                        title=f"Ground truth\n{ROW_LABEL[stem]}  (t = {last_t}/{d['T_total']-1})")
        plot_clot_error_map(ax_err, d["pos"], pred_b, gt_b, wall=d["wall"], zoom_limits=zoom,
                            title="Error")
        if row == 0:
            ax_err.legend(handles=error_legend_handles(), loc="upper left",
                          bbox_to_anchor=(1.03, 1), fontsize=8)

    fig.suptitle(
        "Wound A/B — same vessel geometry, with and without the wound boundary condition\n"
        "illustrative example, not a scored OOF result — see caveat in generator docstring",
        fontsize=11, fontweight="bold", y=0.995,
    )

    out_path = FIG_DIR / f"wound_ab.{CONFIG.fig_format}"
    plt.savefig(out_path)
    plt.close()
    print(f"  [OK] Saved {out_path}")


if __name__ == "__main__":
    main()
