"""README figure: one out-of-fold vessel at final time, model vs ground truth, with its scores.

Scores are BATC_0 (the README table's setting) and node-exact strict F1, on the wall and the
off-wall interior separately, on exactly the masks drawn.  `--list` scores every OOF vessel so the
example can be chosen near the cohort median rather than cherry-picked.

    python scripts/publication/plot_readme_example.py --list
    python scripts/publication/plot_readme_example.py --stem comsol041
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG
from scripts.publication.example_vessels import oof_series, strict_f1
from scripts.publication.oof_data import ensure_oof_series, load_oof_archive
from scripts.publication.pub_style import apply_style, plot_clot_field
from src.clot_ml.severity_metric import BATC_0, dilation_operator, severity_components

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "oof_example.png"
DOMAINS = (("wall", "Wall"), ("off", "Off-wall"))


def vessel_scores(S: dict) -> dict:
    pred, gt = S["pred"][-1], S["gt"][-1]
    D0 = dilation_operator(S["ei"], S["n"], hops=BATC_0.relax_hops)
    out = {}
    for dom, _ in DOMAINS:
        c = severity_components(pred, gt, D0, S[dom], cfg=BATC_0)
        out[dom] = dict(batc0=float("nan") if c["empty_gt"] else float(c["score"]),
                        f1=strict_f1(pred, gt, S[dom]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="comsol041")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        archive = load_oof_archive(ensure_oof_series(CONFIG))
        for stem in sorted(archive.vessels):
            sc = vessel_scores(oof_series(stem, archive))
            print(f"{stem}  " + "  ".join(
                f"{d}: BATC0 {sc[d]['batc0']:.2f} F1 {sc[d]['f1']:.2f}" for d, _ in DOMAINS))
        return 0

    apply_style()
    S = oof_series(args.stem)
    pred, gt = S["pred"][-1], S["gt"][-1]
    sc = vessel_scores(S)
    pos = S["pos"]
    pad = 0.03 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
    lim = (pos[:, 0].min() - pad, pos[:, 0].max() + pad, pos[:, 1].min() - pad, pos[:, 1].max() + pad)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6))
    for ax, field, name in zip(axes, (pred, gt), ("Model (held out)", "COMSOL ground truth")):
        plot_clot_field(ax, pos, field.astype(float), wall=S["wall"], zoom_limits=lim,
                        s_lumen=1.5, s_clot_wall=8, s_clot_off=8, s_wall_open=3)
        ax.set_title(name, fontsize=10, loc="left")
    tag = "   ".join(f"{lab}: BATC₀ {sc[d]['batc0']:.2f} · F1 {sc[d]['f1']:.2f}"
                     for d, lab in DOMAINS)
    fig.text(0.5, 0.015, tag, ha="center", va="bottom", fontsize=10,
             bbox=dict(fc="white", ec="#c9cfcd", lw=0.6, pad=3))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT, dpi=150)
    print(f"[i] wrote {OUT}  ({args.stem}, fold {S['fold']}, t={int(S['times'][-1])})  {tag.replace('₀', '_0')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
