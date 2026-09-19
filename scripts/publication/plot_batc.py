"""Figure 0.3: why BATC — model vs ground truth on three held-out vessels, strict F1 vs BATC.

One column per out-of-fold vessel at final time: the model's clot field, the ground truth, and
the two scores computed on exactly those masks (`example_vessels.scores`).  No text beyond panel
labels -- the argument lives in the caption (docs/publication/FIGURES.md 0.3):

    comsol041 off-wall   every error is a near miss        strict F1 0.68, BATC 0.96
    comsol010 off-wall   a 12-node clot                     strict F1 0.38, BATC 0.79
    comsol028 wall       a real miss BATC does not rescue   strict F1 0.57, BATC 0.66

Writes `figures/batc.{pdf,png}`, `data/batc_examples.json` (all three scores, for the ledger)
and `data/batc.json` (STORY 0.3's worked recall table).

    python scripts/publication/plot_batc.py
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
import matplotlib.patches as mpatches

from scripts.publication.example_vessels import oof_series, score_tag, scores, zoom_for
from scripts.publication.pub_style import apply_style, plot_clot_field

#: (stem, domain, short label, zoom-out factor) -- (b) and (c) zoom out so the clot is seen in
#: its vessel, which is what shows the prediction is in the right place
EXAMPLES = [
    ("comsol041", "off", "near misses", 1.0),
    ("comsol010", "off", "small clot", 1.4),
    ("comsol028", "wall", "real miss", 1.4),
]
DOMAIN_LABEL = {"wall": "wall", "off": "off-wall"}
BARS = (("strict_f1", "strict F1", "#9aa3a6"), ("batc", "BATC", CONFIG.color_model))
MAP_KW = dict(s_lumen=4, s_clot_wall=18, s_clot_off=18, s_wall_open=8)


def _write_worked_cases() -> None:
    """`data/batc.json`: STORY 0.3's worked recall table, computed by the shipped metric config.

    STORY quotes these and the `batc.case*` ledger rows read this file, so the script that owns
    the metric's figure keeps producing it.
    """
    from src.clot_ml.severity_metric import BATC, BATC_0

    def recall_eff(n_gt: float, found: float, cfg) -> float:
        tau = min(cfg.tau_abs, cfg.rho * float(n_gt))
        return min(1.0, float(found) / max(float(n_gt) - tau, 1.0))

    meta = {
        "batc": dict(BATC.__dict__), "batc_0": dict(BATC_0.__dict__),
        "panel_a_cases": [{"n_gt": g, "found": f,
                           "unadjusted": round(recall_eff(g, f, BATC_0), 4),
                           "batc": round(recall_eff(g, f, BATC), 4)}
                          for g, f in ((15, 10), (150, 100), (4, 1))],
    }
    (DATA_DIR / "batc.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"[i] wrote {DATA_DIR / 'batc.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_style()
    fig = plt.figure(figsize=(11.0, 7.0))
    gs = fig.add_gridspec(4, len(EXAMPLES), height_ratios=[0.55, 1, 1, 0.55], hspace=0.14,
                          wspace=0.06, left=0.06, right=0.99, top=0.95, bottom=0.05)

    record = []
    for c, (stem, dom, label, zscale) in enumerate(EXAMPLES):
        S = oof_series(stem)
        pred, gt = S["pred"][-1], S["gt"][-1]
        sc = scores(pred, gt, S[dom], S["D"])
        record.append(dict(stem=stem, domain=dom, time=int(S["times"][-1]), fold=S["fold"],
                           **{k: (None if v != v else round(v, 4)) if isinstance(v, float) else v
                              for k, v in sc.items()}))
        zoom = zoom_for(S, [(pred | gt) & S[dom]], aspect=1.7, scale=zscale)

        # whole vessel, small, with the zoom window boxed -- so the close-ups read as a window
        ov = fig.add_subplot(gs[0, c])
        pos = S["pos"]
        pad = 0.03 * max(pos[:, 0].ptp(), pos[:, 1].ptp())
        # the vessel AND the zoom box, so the window is drawn whole even where it overhangs
        full = (min(pos[:, 0].min() - pad, zoom[0]), max(pos[:, 0].max() + pad, zoom[1]),
                min(pos[:, 1].min() - pad, zoom[2]), max(pos[:, 1].max() + pad, zoom[3]))
        plot_clot_field(ov, pos, pred.astype(float), wall=S["wall"], zoom_limits=full,
                        s_lumen=1, s_clot_wall=5, s_clot_off=5, s_wall_open=2)
        ov.set_title(f"({'abc'[c]}) {stem} · {DOMAIN_LABEL[dom]} · {label}", fontsize=9.5,
                     loc="left")
        if c == 0:
            ov.set_ylabel("whole vessel", fontsize=9)

        for r, (field, name) in enumerate(((pred, "Model"), (gt, "Ground truth")), start=1):
            ax = fig.add_subplot(gs[r, c])
            plot_clot_field(ax, S["pos"], field.astype(float), wall=S["wall"], zoom_limits=zoom,
                            **MAP_KW)
            for sp in ax.spines.values():
                sp.set_edgecolor("#c2185b")
                sp.set_linewidth(1.0)
            if r == 1:
                score_tag(ax, f"BATC {DOMAIN_LABEL[dom]} {sc['batc']:.2f}")
                # box the window the close-up ACTUALLY shows: equal-aspect axes widen their own
                # limits, so the requested zoom is not what is drawn
                ax.apply_aspect()
                (zx0, zx1), (zy0, zy1) = ax.get_xlim(), ax.get_ylim()
                ov.add_patch(mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                                ec="#c2185b", lw=1.4, zorder=10))
                ov.set_xlim(min(full[0], zx0), max(full[1], zx1))
                ov.set_ylim(min(full[2], zy0), max(full[3], zy1))
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5)

        bx = fig.add_subplot(gs[3, c])
        y = np.arange(len(BARS))[::-1]
        vals = [sc[k] for k, _l, _c in BARS]
        bx.barh(y, vals, color=[col for _k, _l, col in BARS], height=0.62)
        for yy, v in zip(y, vals):
            bx.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
        bx.set_yticks(y)
        bx.set_yticklabels([lab for _k, lab, _c in BARS], fontsize=9)
        bx.set_xlim(0, 1.12)
        bx.set_xticks([0, 0.5, 1.0])
        bx.tick_params(axis="x", labelsize=8)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)
        if c:
            bx.tick_params(labelleft=False)

    stem = args.out or str(FIG_DIR / "batc")
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{stem}.{ext}", dpi=CONFIG.dpi)
    print(f"[i] wrote {stem}.{CONFIG.fig_format} / .png")
    out_json = DATA_DIR / "batc_examples.json"
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[i] wrote {out_json}")
    _write_worked_cases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
