"""Plot Figure 1: Flow comparison — vertically stacked.

Layout (3 rows × 2 cols):
  [RGP-DEQ Velocity]    [RGP-DEQ Error]
  [FEM Velocity]        [FEM Error]
  [GT Velocity]         (empty)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.pub_style import (
    apply_style, VEL_CMAP, ERR_CMAP,
    style_colorbar, plot_scalar_field, row_height,
)

PANEL_W = 6.0  # inches, per column


def main():
    print("[i] Plotting Figure 1: Flow Comparisons (Stacked)")
    apply_style()

    for stem in CONFIG.fig1_vessels:
        data_path = DATA_DIR / f"fig1_{stem}_flow.pt"
        if not data_path.exists():
            print(f"  [WARN] Missing {data_path.name}. Run generate_fig1_data.py first.")
            continue

        d    = torch.load(data_path, map_location="cpu", weights_only=False)
        pos  = d["pos"]
        wall = d.get("wall", None)

        mag_rgp = np.sqrt(d["u_rgp"] ** 2 + d["v_rgp"] ** 2)
        mag_fem = np.sqrt(d["u_fem"] ** 2 + d["v_fem"] ** 2)
        mag_gt  = np.sqrt(d["u_gt"]  ** 2 + d["v_gt"]  ** 2)

        vmax = float(np.nanpercentile(mag_gt, 99))

        err_rgp = np.abs(mag_rgp - mag_gt)
        err_fem = np.abs(mag_fem - mag_gt)
        emax    = float(np.nanpercentile(np.concatenate([err_rgp, err_fem]), 99))

        panel_h = row_height(pos, PANEL_W)
        top_margin, bottom_margin, row_pad = 0.55, 0.1, 0.3
        fig_h = 3 * panel_h + 2 * row_pad + top_margin + bottom_margin
        fig_w = 2 * PANEL_W + 1.6
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = gridspec.GridSpec(
            3, 3, figure=fig,
            width_ratios=[1, 1, 0.05], # extra col for colorbars
            height_ratios=[panel_h, panel_h, panel_h],
            left=0.02, right=0.88, top=1 - top_margin / fig_h, bottom=bottom_margin / fig_h,
            wspace=0.08, hspace=row_pad / panel_h,
        )

        ax_rgp_vel = fig.add_subplot(gs[0, 0])
        ax_rgp_err = fig.add_subplot(gs[0, 1])
        ax_fem_vel = fig.add_subplot(gs[1, 0])
        ax_fem_err = fig.add_subplot(gs[1, 1])
        ax_gt_vel  = fig.add_subplot(gs[2, 0])

        # velocity panels
        sc_vel = plot_scalar_field(ax_rgp_vel, pos, mag_rgp, vmin=0, vmax=vmax,
                                   cmap=VEL_CMAP, wall=wall, title="RGP-DEQ  |U|")
        plot_scalar_field(ax_fem_vel, pos, mag_fem, vmin=0, vmax=vmax,
                          cmap=VEL_CMAP, wall=wall, title="In-house FEM  |U|")
        plot_scalar_field(ax_gt_vel, pos, mag_gt, vmin=0, vmax=vmax,
                          cmap=VEL_CMAP, wall=wall, title="Ground Truth (COMSOL)  |U|")

        # error panels
        sc_err = plot_scalar_field(ax_rgp_err, pos, err_rgp, vmin=0, vmax=emax,
                                   cmap=ERR_CMAP, wall=wall, title="RGP-DEQ vs GT  |error|")
        plot_scalar_field(ax_fem_err, pos, err_fem, vmin=0, vmax=emax,
                          cmap=ERR_CMAP, wall=wall, title="FEM vs GT  |error|")

        # colorbars: split the narrow 3rd gridspec column into a velocity cbar
        # (top half) and an error cbar (bottom half), so placement tracks the
        # figure's own layout instead of hardcoded absolute figure fractions.
        cbar_gs = gs[0:3, 2].subgridspec(2, 1, hspace=1.2)
        cbar_vel_ax = fig.add_subplot(cbar_gs[0])
        cbar_err_ax = fig.add_subplot(cbar_gs[1])

        cbar_vel = fig.colorbar(sc_vel, cax=cbar_vel_ax)
        style_colorbar(cbar_vel, label="|U| (ND)")

        cbar_err = fig.colorbar(sc_err, cax=cbar_err_ax)
        style_colorbar(cbar_err, label="|pred − GT| (ND)")

        fig.suptitle(
            f"Flow field comparison  —  {stem}",
            fontsize=11, fontweight="bold", y=0.98,
        )

        out_path = FIG_DIR / f"fig1_{stem}_flow.{CONFIG.fig_format}"
        plt.savefig(out_path)
        plt.close()
        print(f"  [OK] Saved {out_path}")


if __name__ == "__main__":
    main()
