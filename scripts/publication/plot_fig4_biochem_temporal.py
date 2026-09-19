"""Plot Figure 4: Biochem clot_ml_0 — temporal evolution.

No colorbar. Model panels annotated with wall/off-wall scores.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.pub_style import (
    apply_style, CLOT_THRESHOLD,
    plot_clot_field, plot_clot_error_map, error_legend_handles,
    clot_zoom_limits, row_height,
)

WALL_LINE_COLOR    = "#c0392b"
OFFWALL_LINE_COLOR = "#2980b9"
FRAME_MARKER_COLOR = "#555555"

PANEL_W = 6.2  # inches, per column
TIMELINE_H = 3.0  # inches


def main():
    print("[i] Plotting Figure 4: Biochem Temporal")
    apply_style()

    metrics_path = DATA_DIR / "fig34_metrics.csv"
    if not metrics_path.exists():
        print("[WARN] Metrics CSV missing.")
        return
    df_all = pd.read_csv(metrics_path)

    for stem in CONFIG.fig4_vessels:
        data_path = DATA_DIR / f"fig34_{stem}_biochem.pt"
        if not data_path.exists():
            continue

        d     = torch.load(data_path, map_location="cpu", weights_only=False)
        pos   = d["pos"]
        wall  = d.get("wall", None)
        times = d["times"]

        v_df = df_all[df_all["vessel"] == stem].sort_values("time").reset_index(drop=True)
        if v_df.empty:
            continue

        frame_ts = [
            times[max(1, len(times) // 6)],
            times[len(times) // 2],
            times[-1],
        ]

        # One zoom window, shared across all three time rows, so clot growth
        # is legible instead of a sliver on the full vessel silhouette.
        clot_union = np.zeros(len(pos), dtype=bool)
        for t in frame_ts:
            fd = d["frames"][t]
            clot_union |= (np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64) >= CLOT_THRESHOLD)
            clot_union |= (np.asarray(fd.get("gt_phi", fd.get("gt_mask")), dtype=np.float64) >= CLOT_THRESHOLD)
        zoom = clot_zoom_limits(pos, clot_union, wall)
        panel_h = row_height(pos, PANEL_W, zoom_limits=zoom)

        top_margin, bottom_margin, mid_pad = 0.55, 0.15, 0.35
        fig_h = 3 * panel_h + TIMELINE_H + top_margin + bottom_margin + mid_pad
        fig = plt.figure(figsize=(3 * PANEL_W + 1.5, fig_h), facecolor="white")
        gs = gridspec.GridSpec(
            4, 3, figure=fig,
            left=0.03, right=0.90, top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
            hspace=0.22, wspace=0.06,
            height_ratios=[panel_h, panel_h, panel_h, TIMELINE_H],
        )

        first_err_ax = None

        for i, t in enumerate(frame_ts):
            fd = d["frames"][t]
            pred_phi = np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64)
            gt_phi   = np.asarray(fd.get("gt_phi",   fd.get("gt_mask")),   dtype=np.float64)
            pred_b   = pred_phi >= CLOT_THRESHOLD
            gt_b     = gt_phi   >= CLOT_THRESHOLD

            # Extract scores
            w_score, o_score = None, None
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
            if i == 0:
                first_err_ax = ax_err

        # Timeline
        ax_time = fig.add_subplot(gs[3, :])
        ax_time.set_facecolor("white")
        for sp in ax_time.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#aaaaaa")

        is_wound = bool(v_df["is_wound"].iloc[0]) if "is_wound" in v_df.columns else False

        if is_wound:
            for col, lab, clr in [
                ("w_reg", "Wound region",  WALL_LINE_COLOR),
                ("w_lum", "Wound lumen",   OFFWALL_LINE_COLOR),
                ("far",   "Far field",     "#27ae60"),
            ]:
                if col in v_df.columns:
                    ax_time.plot(v_df["time"], v_df[col], color=clr, lw=2.0, label=lab)
        else:
            if "wall" in v_df.columns:
                ax_time.plot(v_df["time"], v_df["wall"], color=WALL_LINE_COLOR,
                             lw=2.0, label="Wall score")
            if "off" in v_df.columns:
                ax_time.plot(v_df["time"], v_df["off"], color=OFFWALL_LINE_COLOR,
                             lw=2.0, label="Off-wall score")

        for t in frame_ts:
            ax_time.axvline(x=t, color=FRAME_MARKER_COLOR, lw=1.0, ls="--", alpha=0.6)

        ax_time.set_ylim(0, 1)
        ax_time.set_xlabel("Timestep", fontsize=9)
        ax_time.set_ylabel("Domain score", fontsize=9)
        ax_time.grid(True, alpha=0.3, color="#cccccc")
        ax_time.legend(fontsize=9, loc="upper left")
        ax_time.set_title("Domain scores over time  (dashed = sampled frames)", fontsize=10)

        # Error legend — anchored to the first row, matching fig6's convention
        if first_err_ax is not None:
            first_err_ax.legend(handles=error_legend_handles(),
                               loc="upper left", bbox_to_anchor=(1.03, 1), fontsize=8)

        fig.suptitle(
            f"{stem}  —  Temporal clot evolution  [OOF fold {d.get('fold', '?')}, flow={d.get('flow', 'gt')}]",
            fontsize=12, fontweight="bold", y=0.98,
        )

        out_path = FIG_DIR / f"fig4_{stem}_temporal.{CONFIG.fig_format}"
        plt.savefig(out_path)
        plt.close()
        print(f"  [OK] Saved {out_path}")


if __name__ == "__main__":
    main()
