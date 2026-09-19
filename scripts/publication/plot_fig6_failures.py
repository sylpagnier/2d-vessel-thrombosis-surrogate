"""Figure 0.6: where BATC and the eye disagree — comsol005 at final time.

`docs/publication/STORY.md` 0.3 says BATC must be shown with its failure mode: a vessel whose
prediction looks right to an expert eye can still score poorly.  comsol005 is that case.  The
mural thrombus is in the right place and the wall scores ~1.0, but the off-wall domain carries only
a handful of GT nodes, so a small compact cluster of lumen false positives next to them is most of
that domain's prediction and pulls its BATC to about half.

Final time: a whole-vessel strip locating the cluster, then model | ground truth | error map zoomed
onto it.  The mid-run behaviour of the same family of
vessels -- errors that appear at one timestep and then recover, or appear at the last one -- is
figure 0.6b (`plot_error_trajectories.py`).

> **RE-SCORED 2026-09-13.** This figure used to print wall 0.988 / off-wall 0.262 and call them
> BATC.  Those were BATC_0: `score_deploy` follows `CLOT_SCORE` mode, whose default is the legacy
> guiding score.  The numbers here come from `example_vessels` with the shipped `BATC` config.

Reads `data/fig6_comsol005_failures.pt` (strict OOF, `generate_fig6_data.py`) and
`data/oof_batc_series.json` (`example_vessels.oof_batc_series`).

    python scripts/publication/plot_fig6_failures.py
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.example_vessels import oof_batc_series
from scripts.publication.pub_style import (
    CLOT_THRESHOLD, apply_style, clot_zoom_limits, error_legend_handles, plot_clot_error_map,
    plot_clot_field,
)

STEM = "comsol005"


def main():
    apply_style()
    d = torch.load(DATA_DIR / f"fig6_{STEM}_failures.pt", map_location="cpu", weights_only=False)
    pos, wall, t = d["pos"], d.get("wall"), d["times"][-1]
    fd = d["frames"][t]
    pred_phi = np.asarray(fd.get("pred_phi", fd.get("pred_mask")), dtype=np.float64)
    gt_phi = np.asarray(fd.get("gt_phi", fd.get("gt_mask")), dtype=np.float64)
    pred_b, gt_b = pred_phi >= CLOT_THRESHOLD, gt_phi >= CLOT_THRESHOLD

    ser = oof_batc_series()[STEM]
    k = ser["times"].index(int(t))
    w_score, o_score = ser["wall"]["score"][k], ser["off"]["score"][k]

    # Two scales.  The vessel is long and thin, so at full extent the off-wall cluster is a few
    # pixels: a full-length strip locates it, and the three panels below zoom onto it.
    import matplotlib.patches as mpatches

    from scripts.publication.example_vessels import load_pack, zoom_for
    from src.clot_ml.wound import solid_mask

    G = dict(pos=pos, wall=np.asarray(wall, dtype=bool), n=len(pos))
    off = ~np.asarray(solid_mask(load_pack(STEM))).reshape(-1).astype(bool)
    zoom = zoom_for(G, [(pred_b | gt_b) & off], aspect=1.55)
    x0, x1, y0, y1 = zoom
    pad = 0.3 * max(x1 - x0, y1 - y0)
    zoom = (x0 - pad, x1 + pad, y0 - pad / 1.55, y1 + pad / 1.55)
    full = clot_zoom_limits(pos, np.ones(len(pos), dtype=bool), wall, pad_frac=0.02)

    fig = plt.figure(figsize=(15.5, 5.6), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.02, right=0.88, top=0.95, bottom=0.03,
                           height_ratios=[0.6, 2.2], hspace=0.10, wspace=0.05)
    ax_full = fig.add_subplot(gs[0, :])
    plot_clot_field(ax_full, pos, pred_phi, wall=wall, zoom_limits=full,
                    title=f"(a) model, whole vessel · t = {t}",
                    wall_score=w_score, off_score=o_score)
    ax_full.add_patch(mpatches.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                         fill=False, ec="#c2185b", lw=1.6, zorder=10))
    ax_pred, ax_gt, ax_err = (fig.add_subplot(gs[1, i]) for i in range(3))
    plot_clot_field(ax_pred, pos, pred_phi, wall=wall, title="(b) model", zoom_limits=zoom,
                    s_lumen=14, s_clot_wall=40, s_clot_off=40, s_wall_open=22)
    plot_clot_field(ax_gt, pos, gt_phi, wall=wall, title="(c) ground truth", zoom_limits=zoom,
                    s_lumen=14, s_clot_wall=40, s_clot_off=40, s_wall_open=22)
    plot_clot_error_map(ax_err, pos, pred_b, gt_b, wall=wall, title="(d) error", zoom_limits=zoom,
                        s_bg=12, s_err=34)
    ax_err.legend(handles=error_legend_handles(), loc="upper left", bbox_to_anchor=(1.03, 1),
                  fontsize=8.5)

    for ext in (CONFIG.fig_format, "png"):
        out = FIG_DIR / f"fig6_{STEM}_failures.{ext}"
        fig.savefig(out, dpi=CONFIG.dpi, bbox_inches="tight")
        print(f"  [OK] Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
