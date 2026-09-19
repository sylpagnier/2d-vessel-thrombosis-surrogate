"""Figure: biochem (clot_ml_0) architecture -- physics-first, GNN residual, coupled by rounds.

Slide figure. Pipeline (verified against the shipped deploy path, not the retired
`biochem_gnn`/GraphSAGE model docs/BIOCHEM_GNN.md documents as dead):

  1. t=0 flow solve  -- steady-state Carreau FEM solve (`src/core_physics/local_fem_solver.py`),
                         run ONCE at t=0, not re-solved per timestep
                         (`scripts/promote_clot_ml_0.py` pins flow="fem").
  2. Physics backbone -- closed-form shear gate (`src/clot_ml/features.py` -- low-shear +
                         separation branches) integrated by a zero-parameter deposition ODE
                         (`src/core_physics/clot_temporal_growth_rules.py`,
                         `integrate_mat_trajectory`) into a first-pass field `mat_phys`.
  3. Spatial GNN      -- an ENSEMBLE of 9 `ClotGNN` networks (`src/clot_ml/gnn.py`): three
                         configurations (v5a/v5b/v5c) x three seeds, each 4 anisotropic
                         message-passing layers at hidden 64 (the class default of 6 is not what
                         ships). `locked.predict_scores` runs all 9 and averages their per-node
                         probabilities. Each network's zero-init regression head predicts a
                         residual r, so `reg = mat_phys + r` -- an untrained head reproduces
                         the physics backbone exactly (same pattern as RGP-DEQ's prior+residual).
                         Drawn as a stack of cards: an earlier version put "9-member ensemble"
                         in a caption under a dashed bracket, and readers took the box for one net.
  4. Coupling         -- NOT a DEQ equilibrium solve: `rollout()` (`src/clot_ml/gnn.py`)
                         reruns the same weight-shared GNN for a fixed number of refinement
                         rounds (3 for v5a/v5c, 5 for v5b), each round recomputing flow-mediated
                         occlusion channels from the PREVIOUS round's Mat guess before rerunning
                         it -- this is where coupling (self-shadowing, upstream/downstream
                         occlusion) enters. Every member runs its own rounds before averaging.
  5. Temporal model   -- not a recurrent net: a gradient-boosted ensemble of 4 onset classifiers
                         ("temporal head") and 3 lag regressors (`temporal.pkl`,
                         `scripts/promote_clot_gnn_v4_temporal.py`), each group averaged, fit on
                         top of the GNN ensemble's spatial output and anchored to the ODE's own
                         first-crossing times (`src/clot_ml/temporal.py`).

No header/title text -- drops straight into a slide with its own title. Conceptual figure,
not a data-backed publication panel -- no JSON artifact is written.

    python scripts/publication/plot_biochem_architecture.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils.paths import get_project_root

REPO = get_project_root()

from scripts.publication.arch_diagram import (  # noqa: E402
    BLUE, GREEN, PURPLE, ORANGE, RED, GREY, NEUTRAL, box, arrow,
)
from scripts.publication.config import FIG_DIR  # noqa: E402
from scripts.publication.pub_style import apply_style  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    apply_style()
    # Same wide-short canvas as plot_rgp_architecture.py so the two slide figures match.
    fig_w, fig_h = 15.0, 4.6
    xlim = 112
    ylim = xlim / (fig_w / fig_h)
    V = ylim / 53.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.axis("off")

    y0, h = 13 * V, 26 * V
    top = y0 + h
    mid_off = 4 * V

    # -- Mesh input -----------------------------------------------------------
    box(ax, (1, y0 + 4 * V), 11, h - 8 * V, fc=NEUTRAL, ec="#9aa5ab", title="mesh",
        body="vessel\ngeometry", title_c="#222222", title_fs=11, body_fs=8.2)
    arrow(ax, (12.3, y0 + h / 2 - mid_off), (15.2, y0 + h / 2 - mid_off))

    # -- 1: t=0 flow solve --------------------------------------------------------
    box(ax, (15.5, y0), 16, h, fc=BLUE, ec="#1c5d8f",
        title="1 · t=0 flow solve",
        body="local FEM\nsteady Carreau NS\nsolved once, ~5 s")
    arrow(ax, (31.8, y0 + h / 2 - mid_off), (34.9, y0 + h / 2 - mid_off))

    # -- 2: physics backbone --------------------------------------------------------
    box(ax, (35.2, y0), 17, h, fc=GREEN, ec="#1c5c3a",
        title="2 · Physics backbone",
        body="shear gate\n(low-shear + separation)\n$\\rightarrow$ deposition ODE\n+ advective transport")
    arrow(ax, (52.5, y0 + h / 2 - mid_off), (54.8, y0 + h / 2 - mid_off))

    # -- 2b: the wound branch, drawn ON the physics backbone and nowhere else -------
    # This is the whole of STORY.md Leg 0.2: injury is a boundary-condition SWITCH on one
    # term of the backbone, not a second network and not a parallel pipeline. Drawing it
    # hanging off box 2 is the point -- the GNN, the coupling rounds and the temporal head
    # are untouched, and on a pack with no wound mask the artifact is bit-identical.
    wb_x, wb_w = 33.4, 19.8
    wb_y, wb_h = 0.9 * V, 11.0 * V
    ax.add_patch(FancyBboxPatch(
        (wb_x, wb_y), wb_w, wb_h, boxstyle="round,pad=0,rounding_size=1.6",
        fc="#f6f2fa", ec=PURPLE, lw=1.4, linestyle=(0, (4, 2.5)), zorder=2,
    ))
    ax.text(wb_x + wb_w / 2, wb_y + wb_h - 2.2 * V, "2b · wound branch",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color=PURPLE)
    ax.text(wb_x + wb_w / 2, wb_y + wb_h - 5.6 * V,
            "ungated srf2 + 2 fitted rates",
            ha="center", va="center", fontsize=8.4, color="#222222")
    ax.text(wb_x + wb_w / 2, wb_y + wb_h - 8.6 * V,
            "no wound mask $\\Rightarrow$ bit-identical output",
            ha="center", va="center", fontsize=8.2, color=PURPLE, style="italic")
    arrow(ax, (wb_x + wb_w / 2, wb_y + wb_h), (wb_x + wb_w / 2, y0 - 0.3 * V),
          color=PURPLE, lw=1.5, style="-|>")

    # -- 3: spatial GNN ensemble, drawn as a stack of cards ------------------------
    # The stack IS the claim: nine separately trained networks whose outputs are averaged.
    # A caption under a dashed bracket was read as decoration and the box as one network.
    gnn_x, gnn_w = 55.1, 17
    eq_x0, eq_x1 = gnn_x, gnn_x + gnn_w
    for k in (2, 1):  # back cards first, offset down-right so the loop above stays clear
        ax.add_patch(FancyBboxPatch(
            (gnn_x + 1.0 * k, y0 - 1.5 * V * k), gnn_w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            fc="#b98acb" if k == 2 else "#a468bd", ec="#5e2d70", lw=1.1, zorder=2.6 + 0.1 * (2 - k),
        ))
    box(ax, (gnn_x, y0), gnn_w, h, fc=PURPLE, ec="#5e2d70",
        title="3 · Spatial GNN ensemble",
        body="9 networks, outputs averaged\n4 message-passing layers\nedges carry t=0 flow\n= physics + residual",
        title_fs=9.8)
    ax.text(gnn_x + 1.0 + gnn_w / 2, y0 - 3.0 * V - 1.2 * V,
            "3 configs × 3 seeds", ha="center", va="top", fontsize=8.6, color=GREY, style="italic")

    # Refinement-round loop, drawn as a self-loop over the spatial-GNN box (exits the
    # right quarter, re-enters the left quarter -- same up/across/down shape as the
    # DEQ loop in plot_rgp_architecture.py, just landing on one box instead of two).
    loop_y = top + 4.0 * V
    loop_left, loop_right = gnn_x + gnn_w * 0.24, gnn_x + gnn_w * 0.76
    arrow(ax, (loop_right, top - 0.3 * V), (loop_right, loop_y), color=RED, lw=1.8, style="-", rad=0)
    arrow(ax, (loop_right, loop_y), (loop_left, loop_y), color=RED, lw=1.8, style="-", rad=0)
    arrow(ax, (loop_left, loop_y), (loop_left, top - 0.3 * V), color=RED, lw=1.8,
          style="-|>", rad=0)
    ax.text((eq_x0 + eq_x1) / 2, loop_y + 1.0 * V,
            "4 · occlusion feedback rounds (×3 or 5, per network)",
            ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=RED)

    arrow(ax, (75.3, y0 + h / 2 - mid_off), (78.0, y0 + h / 2 - mid_off))

    # -- 5: temporal model -----------------------------------------------------------
    box(ax, (78.3, y0), 17, h, fc=ORANGE, ec="#a3480a",
        title="5 · Temporal model",
        body="gradient-boosted ensemble\n4 onset + 3 lag models\nanchored to ODE\ncrossing time")
    arrow(ax, (95.6, y0 + h / 2 - mid_off), (98.7, y0 + h / 2 - mid_off))

    # -- Output -------------------------------------------------------------------
    box(ax, (99.0, y0 + 4 * V), 11, h - 8 * V, fc=NEUTRAL, ec="#9aa5ab",
        title="clot", body="mask(t)", title_c="#222222", title_fs=11, body_fs=9)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    out = args.out or (FIG_DIR / "biochem_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
