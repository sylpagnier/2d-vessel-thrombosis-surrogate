"""The ablation ladder figure: what each layer of physics, and each design choice, is worth.

Two panels, one per scoring domain.  Each row is an arm, drawn as its held-out cohort mean
with the paired 95% bootstrap interval of its DELTA against the reference arm anchored at the
reference's score -- so the eye reads the interval as "does this arm's bar reach A4", which is
the question, rather than as a confidence interval on the arm's own mean, which is wider and
not what any claim rests on.

Run `scripts/publication/generate_ablation_data.py` first; this only draws its JSON.

    python scripts/publication/plot_ablation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()

from scripts.publication.config import FIG_DIR  # noqa: E402
from scripts.publication.pub_style import apply_style  # noqa: E402

DATA = REPO / "outputs/ablation/ablation_report.json"

#: rows, top to bottom, and how they are labelled.  The two ladders are drawn in one figure
#: on purpose: "which physics" and "which architecture" are the same question asked of
#: different parts of the stack, and a reader comparing the two bar lengths is doing exactly
#: the comparison the paper wants them to do.
ROWS: tuple[tuple[str, str], ...] = (
    ("A0", "A0  geometry only (physics arch ON)"),
    ("A0_pure", "A0$_{pure}$  geometry, doors shut (flow still in edges)"),
    ("A0_fresh", "A0$_{fresh}$  solved fresh: no physics anywhere"),
    ("A1", "A1  + raw flow"),
    ("A1p", "A1+ + flow aggregates"),
    ("A1_pure", "A1$_{pure}$  generic mesh GNN"),
    ("A2", "A2  + law's gate groups"),
    ("A3", "A3  + deposition ODE"),
    ("A4", "A4  + transport solve  (shipped)"),
    ("B_iso", "B  isotropic messages"),
    ("B_nomp", "B  no MP layers"),
    ("B_mlp", "B  no graph at all"),
    ("B_nobase", "B  no physics base"),
    ("B_noseed", "B  no physics seed"),
    ("B_r1", "B  one round"),
    ("B_bce", "B  BCE loss (no metric)"),
)

PHYS_C = "#2c6fbb"     # the physics-conditioning ladder
ARCH_C = "#b8532a"     # the architecture ladder
REF_C = "#1a1a1a"

#: the pipeline's own noise floor at three seeds, measured on disjoint seeds of one arm
#: (`scripts/eval_significance.py --floor`; DEPLOYCLOT / go_deployclot_split.sh).  A bar
#: inside this band of the reference is not a result, and the figure says so.
NOISE = {"wall": 0.005, "off": 0.045}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(DATA),
                    help="ablation JSON to draw (one exists per ladder/cache)")
    ap.add_argument("--suffix", default="",
                    help="appended to the figure filename, so two ladders do not collide")
    args = ap.parse_args()
    data = Path(args.report)
    if not data.is_file():
        print("[WARN] %s missing; run generate_ablation_data.py first" % data)
        return
    d = json.loads(data.read_text(encoding="utf-8"))
    arms = d["arms"]
    ref = d["ref"]
    rows = [(k, lab) for k, lab in ROWS if k in arms]
    if not rows:
        print("[WARN] no ladder arms in %s" % DATA)
        return

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.34 * len(rows) + 1.9), sharey=True)
    y = np.arange(len(rows))[::-1]

    for ax, dom, title in ((axes[0], "wall", "wall domain"),
                           (axes[1], "off", "off-wall domain")):
        r0 = arms[ref]["auto"][dom]
        ax.axvspan(r0 - NOISE[dom], r0 + NOISE[dom], color="#dddddd", zorder=0,
                   label="pipeline noise floor")
        ax.axvline(r0, color=REF_C, lw=1.0, zorder=1)
        for yi, (k, _lab) in zip(y, rows):
            a = arms[k]
            v = a["auto"][dom]
            c = ARCH_C if k.startswith("B_") else PHYS_C
            ax.barh(yi, v, height=0.62, color=c, alpha=0.85 if k != ref else 1.0,
                    edgecolor=REF_C if k == ref else "none", linewidth=0.8, zorder=2)
            dd = a.get("delta", {}).get("auto")
            if dd:
                lo, hi = r0 + dd[dom]["lo"], r0 + dd[dom]["hi"]
                ax.plot([lo, hi], [yi, yi], color=REF_C, lw=1.1, zorder=3,
                        solid_capstyle="butt")
                ax.plot([lo, hi], [yi, yi], "|", color=REF_C, ms=4, zorder=3)
            ax.text(min(v, 0.985) + 0.008, yi, "%.3f" % v, va="center", ha="left",
                    fontsize=6.5, color=REF_C, zorder=4)
        ax.set_xlim(0, 1.06)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("held-out severity score")
        ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _k, lab in rows], fontsize=7.5)
    axes[0].legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.suptitle("Physics conditioning (blue) and architecture (orange) ablations, "
                 "n=%d held-out vessels; bars against %s (black line), "
                 "whiskers = paired 95%% CI of the difference  [cache %s]"
                 % (d["n_carrying"], ref, d.get("cache", "?")), fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(FIG_DIR) / ("fig_ablation_ladder%s.png" % args.suffix)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("[OK] wrote %s" % out)


if __name__ == "__main__":
    main()
