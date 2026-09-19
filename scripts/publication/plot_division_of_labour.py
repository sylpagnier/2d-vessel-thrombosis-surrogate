"""Figure: the division of labour — which domain wants physics, and which wants learning.

**This is the figure for the paper's lead claim** (`docs/PAPER.md` §1, §5). Every other
ablation figure in this package illustrates a *null*; this one illustrates the spine.

Two panels, and each answers a question a reviewer actually asks.  A third panel of paired
per-vessel deltas was dropped from the figure on review (2026-09-13); the intervals it drew are
quoted in STORY 1.1 and the ledger instead:

  (a) **Does the answer change between domains?**  Four predictors on one axis, ordered by
      how much they know, drawn once per scoring domain.  The eye reads the *reversal*: pure
      physics BEATS the physics-free network at the wall and LOSES to it in the lumen.  That
      swap is the finding, and it is invisible in a table because it lives between two rows of
      two different columns.

      Note carefully what the panel does and does not claim.  Physics is **not** the best
      predictor at the wall -- the shipped model is, by +0.0252, which is not significant.
      The reversal that carries the paper is the green-versus-blue pair, not the top of the
      ordering, and the annotations say exactly that.

  (b) **Where does the physics actually enter?**  What each of the three doors — readout,
      architecture, conditioning — is worth *to a physics-free model* at the wall.
      **Deliberately NOT drawn as a waterfall.**  The doors are not additive: the +0.25
      readout effect is measured on the naive arm and is worth ~0.00 to the shipped model,
      which already has the physics.  A stacked bar summing to the total would be a claim the
      data does not support, so each door is drawn as its own independent measurement with the
      arm it was measured on named on the bar.


Reads `outputs/ablation/physics_vs_learned_fem.json` (the SHIPPED `v5_fem` generation) plus the
readout rows of the verified claims ledger.  Draws only; computes nothing new.

    python scripts/publication/plot_division_of_labour.py
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

DATA = REPO / "outputs/ablation/physics_vs_learned_fem.json"
#: the SAME protocol with a depth-matched MeshGraphNet-STYLE control in the `naive` slot
#: (`A_mgn`, 15 message-passing layers).  Optional: the figure draws without it, and says so.
#: It exists to answer the reviewer's first objection to panel (a) -- "your baseline is shallow"
#: -- inside the figure rather than three sections away.
MGN = REPO / "outputs/ablation/physics_vs_learned_mgn.json"
LEDGER = REPO / "outputs/ablation/claims_ledger.json"

# The four points of §5's axis, in the order "how much does it know about the problem".
# Labels are the manuscript's, NOT the code's arm names: `plain` is explicitly not "a plain
# GNN" (PAPER.md §5), and calling it one in a figure is the same overstatement the prose
# spends a paragraph forbidding.
LADDER = [
    ("naive", "naive mesh GNN\n(geometry only)", "#b0b0b0", 1.6, 6),
    # NAMING DISCIPLINE: this is a depth-matched control in the naive slot, NOT a MeshGraphNet
    # reimplementation, and it must never be labelled as one on a figure a reader can quote.
    ("mgn", "the same, MeshGraphNet-style\ndepth-matched control (15 layers)", "#8a8a8a", 1.3, 5),
    ("plain", "physics-informed architecture\n(no physics conditioning)", "#2b7bba", 2.6, 8),
    ("phys_tuned", "pure physics\n(same fitted cut)", "#2e8b57", 2.6, 8),
    ("full", "shipped model\n(both)", "#d95f02", 1.6, 6),
]

DOMAINS = [("wall", "Wall"), ("off", "Lumen (off-wall)")]

#: Test ids spell the arm as `phystuned`; the per-vessel records spell it `phys_tuned`.
#: Splitting a test id on "_minus_" therefore yields a key that is NOT in `per_vessel`, and
#: the lookup fails silently to an empty dot cloud.  Map explicitly.
ARM_KEY = {"phystuned": "phys_tuned", "phys": "phys", "plain": "plain",
           "full": "full", "naive": "naive"}


def _ledger() -> dict[str, float]:
    rows = json.loads(LEDGER.read_text(encoding="utf-8"))
    return {r["id"]: r["artifact"] for r in rows}


def panel_a(ax, d: dict) -> None:
    """Four predictors x two domains; the green/blue reversal is the point."""
    means = d["means"]
    xs = np.arange(len(DOMAINS))
    # Label offsets per (series, domain): the four annotations collide differently at each
    # end, and `full` in particular must drop BELOW its point at the lumen end or it lands
    # under the legend.
    # The two GREY CONTROL series are offset HORIZONTALLY, the four ladder series vertically.
    # Vertical offsets cannot work for the controls: at the wall `naive` (0.683) sits above
    # `mgn` (0.638) but in the lumen it sits BELOW it (0.485 vs 0.521), so a fixed up/down pair
    # makes the two labels swap apparent order at one end and collide with `plain` at the other.
    # Pushing them clear of the line entirely is the only arrangement that holds at both ends.
    offs = {
        ("naive", 0): (-36, 8), ("naive", 1): (34, -14),
        ("mgn", 0): (-36, -14), ("mgn", 1): (34, 8),
        ("plain", 0): (0, -15), ("plain", 1): (0, -15),
        ("phys_tuned", 0): (0, 10), ("phys_tuned", 1): (0, 10),
        ("full", 0): (0, 10), ("full", 1): (0, -15),
    }

    for key, label, colour, lw, ms in LADDER:
        if key not in means[DOMAINS[0][0]]:
            continue                      # MGN arm not run on this cache -- draw without it
        ys = [means[dom][key] for dom, _ in DOMAINS]
        # The depth-matched control is drawn DASHED and muted on purpose: it is a control, not
        # a rung of the axis, and it must not read as a fifth predictor in the ladder.
        style = "--o" if key == "mgn" else "-o"
        ax.plot(xs, ys, style, color=colour, linewidth=lw, markersize=ms,
                markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=offs[(key, int(x))], ha="center", fontsize=7.5, color=colour,
                        fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in DOMAINS], fontsize=9)
    ax.set_xlim(-0.30, len(DOMAINS) - 0.70)
    ax.set_ylabel("held-out severity score")
    ax.set_ylim(0.40, 1.36)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) held-out score by domain", loc="left", fontweight="bold")

    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=6.1, handlelength=1.2,
              borderpad=0.32, labelspacing=0.35, columnspacing=1.0)


def panel_b(ax, d: dict, led: dict[str, float]) -> None:
    """The three doors, each as an independent measurement -- NOT a waterfall."""
    naive_plain = led["rd.naive.plain_wall"]          # physics barred everywhere
    naive_auto = led["rd.naive.auto_wall"]            # physics available at the cut only
    plain_wall = d["means"]["wall"]["plain"]
    full_wall = d["means"]["wall"]["full"]

    doors = [
        ("readout",
         naive_auto - naive_plain, "measured on the naive arm", "#8e44ad"),
        ("architecture",
         plain_wall - naive_plain, "naive → ablated architecture", "#2b7bba"),
        ("conditioning",
         full_wall - plain_wall, "ablated → shipped", "#d95f02"),
    ]

    ys = np.arange(len(doors))[::-1]
    for y, (label, val, prov, colour) in zip(ys, doors):
        ax.barh(y, val, height=0.46, color=colour, alpha=0.85, zorder=3)
        ax.text(val + 0.007, y, f"+{val:.3f}", va="center", fontsize=8.5,
                fontweight="bold", color=colour)

    ax.set_yticks(ys)
    ax.set_yticklabels([lbl for lbl, _, _, _ in doors], fontsize=9)
    ax.set_xlabel("wall score gained")
    ax.set_xlim(0, 0.315)
    ax.set_ylim(-0.55, len(doors) - 0.35)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) wall gain per physics entry point", loc="left", fontweight="bold")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    apply_style()
    d = json.loads(DATA.read_text(encoding="utf-8"))
    led = _ledger()
    # Splice the depth-matched control into `means` only.  It is deliberately NOT added to
    # `per_vessel` or `tests`: panels (b) and (c) price the physics doors and the paired
    # deltas along the four-point axis, and a control does not belong in either.
    if MGN.is_file():
        mg = json.loads(MGN.read_text(encoding="utf-8"))["means"]
        for dom, _ in DOMAINS:
            d["means"][dom]["mgn"] = mg[dom]["naive"]

    fig = plt.figure(figsize=(10.0, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.075, right=0.975, top=0.92, bottom=0.13)
    panel_a(fig.add_subplot(gs[0, 0]), d)
    panel_b(fig.add_subplot(gs[0, 1]), d, led)

    out = args.out or (FIG_DIR / "division_of_labour.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {out}")
    print(f"[ok] wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
