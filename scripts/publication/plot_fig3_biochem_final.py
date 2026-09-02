"""Plot Figure 3: Biochem clot_ml_0 — final-time clot maps.

3-panel per vessel: [Model] [GT] [Error map]
Annotations included for wall/off-wall scores. No colorbar for phi.
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.pub_style import (
    apply_style, CLOT_THRESHOLD,
    plot_clot_field, plot_clot_error_map, error_legend_handles,
    clot_zoom_limits, row_height,
)

PANEL_W = 6.2  # inches, per column


def main():
    print("[i] Plotting Figure 3: Biochem Final Time")
    apply_style()

    metrics_path = DATA_DIR / "fig34_metrics.csv"
    if metrics_path.exists():
        df_all = pd.read_csv(metrics_path)
    else:
        df_all = pd.DataFrame()

    for stem in CONFIG.fig3_vessels:
        data_path = DATA_DIR / f"fig34_{stem}_biochem.pt"
        if not data_path.exists():
            continue

        d      = torch.load(data_path, map_location="cpu", weights_only=False)
        pos    = d["pos"]
        wall   = d.get("wall", None)
        times  = d["times"]
        last_t = times[-1]
        fd     = d["frames"][last_t]

        pred_phi = np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64)
        gt_phi   = np.asarray(fd.get("gt_phi",   fd.get("gt_mask")),   dtype=np.float64)
        pred_b   = pred_phi >= CLOT_THRESHOLD
        gt_b     = gt_phi   >= CLOT_THRESHOLD

        # Extract scores
        w_score, o_score = None, None
        if not df_all.empty:
            v_df = df_all[(df_all["vessel"] == stem) & (df_all["time"] == last_t)]
            if not v_df.empty:
                w_score = float(v_df["wall"].iloc[0]) if "wall" in v_df.columns else None
                o_score = float(v_df["off"].iloc[0]) if "off" in v_df.columns else None

        zoom = clot_zoom_limits(pos, gt_b | pred_b, wall)
        panel_h = row_height(pos, PANEL_W, zoom_limits=zoom)
        top_margin, bottom_margin = 0.6, 0.2  # inches, for suptitle + panel title / bottom pad
        fig_h = panel_h + top_margin + bottom_margin
        fig = plt.figure(figsize=(3 * PANEL_W + 1.5, fig_h), facecolor="white")
        gs  = gridspec.GridSpec(
            1, 3, figure=fig,
            left=0.02, right=0.90, top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
            wspace=0.06,
        )

        ax_pred = fig.add_subplot(gs[0])
        ax_gt   = fig.add_subplot(gs[1])
        ax_err  = fig.add_subplot(gs[2])

        plot_clot_field(ax_pred, pos, pred_phi, wall=wall,
                        title="Model Prediction", zoom_limits=zoom,
                        wall_score=w_score, off_score=o_score)
        plot_clot_field(ax_gt, pos, gt_phi, wall=wall,
                        title="Ground Truth", zoom_limits=zoom)
        plot_clot_error_map(ax_err, pos, pred_b, gt_b, wall=wall,
                            title="Error Map", zoom_limits=zoom)

        ax_err.legend(
            handles=error_legend_handles(),
            loc="upper left", bbox_to_anchor=(1.03, 1), fontsize=8,
        )

        fig.suptitle(
            f"{stem}  —  Final time (t = {last_t})  [OOF fold {d.get('fold', '?')}, flow={d.get('flow', 'gt')}]",
            fontsize=11, fontweight="bold", y=0.97,
        )

        out_path = FIG_DIR / f"fig3_{stem}_final.{CONFIG.fig_format}"
        plt.savefig(out_path)
        plt.close()
        print(f"  [OK] Saved {out_path}")


if __name__ == "__main__":
    main()
