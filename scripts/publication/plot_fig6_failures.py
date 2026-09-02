"""Plot Figure 6: Known failure analysis.

3 temporal snapshots per vessel (early / mid / final).
No colorbars. Legend moved outside to avoid blocking the vessel.
"""
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

FAILURE_CAPTION = {
    "patient005": (
        "patient005 — visually plausible but poor deploy score; "
        "subtle spatial errors compound into low Jaccard"
    ),
    "patient014": (
        "patient014 — tracks well early/mid, then diverges badly late; "
        "heavily penalised by the temporal window scoring"
    ),
}


def main():
    print("[i] Plotting Figure 6: Failure Analysis")
    apply_style()

    metrics_path = DATA_DIR / "fig6_metrics.csv"
    df_all = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    for stem in CONFIG.fig6_vessels:
        data_path = DATA_DIR / f"fig6_{stem}_failures.pt"
        if not data_path.exists():
            continue

        d     = torch.load(data_path, map_location="cpu", weights_only=False)
        pos   = d["pos"]
        wall  = d.get("wall", None)
        times = d["times"]
        v_df  = df_all[df_all["vessel"] == stem] if not df_all.empty else pd.DataFrame()

        frame_ts = [
            times[max(1, len(times) // 6)],
            times[len(times) // 2],
            times[-1],
        ]

        # One zoom window, shared across all three time rows.
        clot_union = np.zeros(len(pos), dtype=bool)
        for t in frame_ts:
            fd = d["frames"][t]
            clot_union |= (np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64) >= CLOT_THRESHOLD)
            clot_union |= (np.asarray(fd.get("gt_phi", fd.get("gt_mask")), dtype=np.float64) >= CLOT_THRESHOLD)
        zoom = clot_zoom_limits(pos, clot_union, wall)
        panel_h = row_height(pos, PANEL_W, zoom_limits=zoom)

        n_rows = len(frame_ts)
        top_margin, bottom_margin = 0.55, 0.15
        fig_h = n_rows * panel_h + top_margin + bottom_margin
        fig = plt.figure(figsize=(3 * PANEL_W + 1.5, fig_h), facecolor="white")
        gs  = gridspec.GridSpec(
            n_rows, 3, figure=fig,
            left=0.02, right=0.90, top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
            hspace=0.20, wspace=0.06,
        )

        for i, t in enumerate(frame_ts):
            fd = d["frames"][t]
            pred_phi = np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64)
            gt_phi   = np.asarray(fd.get("gt_phi",   fd.get("gt_mask")),   dtype=np.float64)
            pred_b   = pred_phi >= CLOT_THRESHOLD
            gt_b     = gt_phi   >= CLOT_THRESHOLD

            w_score, o_score = None, None
            if not v_df.empty:
                frame_df = v_df[v_df["time"] == t]
                if not frame_df.empty:
                    w_score = float(frame_df["wall"].iloc[0]) if "wall" in frame_df.columns else None
                    o_score = float(frame_df["off"].iloc[0]) if "off" in frame_df.columns else None

            ax_pred = fig.add_subplot(gs[i, 0])
            ax_gt   = fig.add_subplot(gs[i, 1])
            ax_err  = fig.add_subplot(gs[i, 2])

            plot_clot_field(ax_pred, pos, pred_phi, wall=wall,
                            title=f"Model  (t = {t})", zoom_limits=zoom,
                            wall_score=w_score, off_score=o_score)
            plot_clot_field(ax_gt, pos, gt_phi, wall=wall,
                            title=f"GT  (t = {t})", zoom_limits=zoom)
            plot_clot_error_map(ax_err, pos, pred_b, gt_b, wall=wall,
                                title=f"Error  (t = {t})", zoom_limits=zoom)

            # Place legend outside the bounding box
            if i == 0:
                ax_err.legend(
                    handles=error_legend_handles(),
                    loc="upper left", bbox_to_anchor=(1.03, 1),
                    fontsize=8,
                )

        caption = FAILURE_CAPTION.get(stem, stem)
        fold = d.get("fold", "?")
        flow = d.get("flow", "gt")
        fig.suptitle(
            f"Failure analysis (OOF fold {fold}, flow={flow})  —  {caption}",
            fontsize=11, fontweight="bold", y=0.975,
        )

        out_path = FIG_DIR / f"fig6_{stem}_failures.{CONFIG.fig_format}"
        plt.savefig(out_path)
        plt.close()
        print(f"  [OK] Saved {out_path}")


if __name__ == "__main__":
    main()
