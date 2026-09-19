"""Shared drawing primitives for the slide architecture diagrams
(plot_rgp_architecture.py, plot_biochem_architecture.py, plot_biochem_detail.py).

Factored out once a third figure needed the same box/arrow/palette code -- see
[[feedback_code_organization]] (reuse/extend over forking one-off variants).
"""
from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#2b7bba"
GREEN = "#2e8b57"
PURPLE = "#8e44ad"
ORANGE = "#d95f02"
RED = "#e74c3c"
GREY = "#555555"
NEUTRAL = "#cfd8dc"


def box(ax, xy, w, h, *, fc, ec, title, body="", title_c="white", body_c="#222222",
        title_fs=11.5, body_fs=8.6, lw=1.3, alpha=1.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3, mutation_aspect=1,
    ))
    # Offsets are fractions of h (tuned against a 26-unit-tall box) so title/body placement
    # stays proportionally right whether h is rescaled for a wider/shorter canvas or this is
    # one of the smaller mesh/output boxes.
    ax.text(x + w / 2, y + h * (1 - 3.6 / 26), title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=title_c, zorder=4, alpha=alpha)
    if body:
        ax.text(x + w / 2, y + h * (0.5 - 1.6 / 26), body, ha="center", va="center",
                fontsize=body_fs, color=body_c, zorder=4, linespacing=1.6, alpha=alpha)
    return x, y, w, h


def arrow(ax, p0, p1, *, color=GREY, lw=1.8, style="-|>", rad=0.0, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=0, shrinkB=0,
        alpha=alpha,
    ))
