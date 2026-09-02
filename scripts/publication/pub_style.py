"""
pub_style.py  —  Publication figure design system (v3).

Key design decisions:
  - White background, academic grey axes, sans-serif font
  - Clot field: wall-attached clots = CIRCLES, off-wall clots = SQUARES
  - Error map: FP=red, FN=blue only (TP folded into neutral background —
    the panel highlights mistakes, not agreement); wall = small dots on the
    boundary curve, off-wall = larger dark-edged squares
  - No clot-phi colourbar (removed by request)
  - Panels labeled with wall/off-wall deploy scores where available
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
VEL_CMAP  = "turbo"     # COMSOL-like blue-to-red (perceptually uniform)
ERR_CMAP  = "Reds"
CLOT_CMAP = "YlOrRd"

# ---------------------------------------------------------------------------
# Fixed colours
# ---------------------------------------------------------------------------
LUMEN_COLOR   = "#cccccc"   # open lumen nodes
WALL_COLOR    = "#1a1a1a"   # wall strip (dark)
TP_COLOR      = "#2ecc71"   # true positive
FP_COLOR      = "#e74c3c"   # false positive
FN_COLOR      = "#3498db"   # false negative
BG_COLOR      = "#e8e8e8"   # background (neither clotted nor GT)

CLOT_THRESHOLD = 0.45

# Font sizes
TITLE_SZ = 9
LABEL_SZ = 8
TICK_SZ  = 7


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#aaaaaa",
        "axes.linewidth":    0.7,
        "text.color":        "#111111",
        "axes.labelcolor":   "#333333",
        "axes.labelsize":    LABEL_SZ,
        "axes.titlesize":    TITLE_SZ,
        "axes.titlepad":     4,
        "xtick.color":       "#555555",
        "ytick.color":       "#555555",
        "xtick.labelsize":   TICK_SZ,
        "ytick.labelsize":   TICK_SZ,
        "legend.fontsize":   LABEL_SZ,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "font.family":       "sans-serif",
        "font.size":         LABEL_SZ,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------
def style_ax(ax, *, title: str = "") -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#cccccc")
    if title:
        ax.set_title(title, fontsize=TITLE_SZ, pad=3, wrap=True)


def style_colorbar(cbar, *, label: str) -> None:
    cbar.set_label(label, fontsize=LABEL_SZ)
    cbar.ax.tick_params(labelsize=TICK_SZ)


# ---------------------------------------------------------------------------
# Layout helpers — vessel geometries are long and thin (aspect 2-7:1); sizing
# panels to that aspect (instead of a fixed square-ish cell) removes the dead
# whitespace above/below every panel row.
# ---------------------------------------------------------------------------
def vessel_aspect(pos: np.ndarray) -> float:
    """Width/height of the node bounding box (used to size panel rows)."""
    x = np.asarray(pos)[:, 0]
    y = np.asarray(pos)[:, 1]
    w = float(x.max() - x.min())
    h = float(y.max() - y.min())
    return max(w / max(h, 1e-9), 0.15)


def row_height(pos: np.ndarray, panel_w: float, *, min_h: float = 1.1, max_h: float = 5.5,
                zoom_limits=None) -> float:
    """Panel height (inches) for a row of width panel_w, matched to the plotted
    aspect ratio — the full vessel bounding box, or a zoom box if one is given."""
    if zoom_limits is not None:
        xmin, xmax, ymin, ymax = zoom_limits
        aspect = max((xmax - xmin) / max(ymax - ymin, 1e-9), 0.15)
    else:
        aspect = vessel_aspect(pos)
    return float(np.clip(panel_w / aspect, min_h, max_h))


# ---------------------------------------------------------------------------
# Zoom helpers
# ---------------------------------------------------------------------------
def clot_zoom_limits(
    pos: np.ndarray,
    clot_mask: np.ndarray,
    wall: np.ndarray | None = None,
    *,
    pad_frac: float = 0.30,
    n_hops: int = 3,
    hop_scale: float = 1.0,
):
    """
    Compute axis limits zoomed to the clot bounding box.

    n_hops controls how much extra context to add beyond the clot bounding box.
    The 'hop' size is estimated from the median nearest-node distance.
    """
    c = np.asarray(clot_mask, dtype=bool).ravel()
    if not c.any():
        return None

    pts = pos[c]
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()

    # Estimate node spacing from all nodes
    idx = np.random.choice(len(pos), min(200, len(pos)), replace=False)
    sample = pos[idx]
    # Quick approx: median of distances to sample centroid in each axis
    hop_x = float(np.median(np.diff(np.sort(sample[:, 0])))) if len(sample) > 1 else 1e-4
    hop_y = float(np.median(np.diff(np.sort(sample[:, 1])))) if len(sample) > 1 else 1e-4
    hop = max(hop_x, hop_y) * hop_scale

    pad_x = max((xmax - xmin) * pad_frac, n_hops * hop * 3)
    pad_y = max((ymax - ymin) * pad_frac, n_hops * hop * 3)

    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def apply_zoom(ax, limits) -> None:
    if limits is None:
        return
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Score annotation helper
# ---------------------------------------------------------------------------
def annotate_scores(ax, *, wall_score: float | None, off_score: float | None) -> None:
    """
    Add wall / off-wall deploy scores as text inside the panel lower-left.
    Scores are expected in [0, 1]. None or NaN scores (undefined when a
    domain is empty at that timestep) are silently omitted rather than
    rendered as the literal text "nan".
    """
    def _valid(v: float | None) -> bool:
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    parts = []
    if _valid(wall_score):
        parts.append(f"wall={wall_score:.3f}")
    if _valid(off_score):
        parts.append(f"off={off_score:.3f}")
    if not parts:
        return
    ax.text(
        0.03, 0.04, "  ".join(parts),
        transform=ax.transAxes,
        fontsize=7, color="#333333",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
        zorder=20,
    )


# ---------------------------------------------------------------------------
# Scalar field (velocity, error, pressure)
# ---------------------------------------------------------------------------
def plot_scalar_field(
    ax,
    pos: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str = VEL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    wall: np.ndarray | None = None,
    s_lumen: float = 6.0,
    s_wall:  float = 8.0,
    zoom_limits=None,
    wall_score: float | None = None,
    off_score:  float | None = None,
) -> ScalarMappable:
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanpercentile(vals, 99))

    ax.clear()
    style_ax(ax, title=title)

    if wall is not None:
        w = np.asarray(wall, dtype=bool).ravel()
        fluid_m = ~w
    else:
        w = np.zeros(len(pos), dtype=bool)
        fluid_m = np.ones(len(pos), dtype=bool)

    # Fluid nodes
    sc = ax.scatter(
        pos[fluid_m, 0], pos[fluid_m, 1],
        c=vals[fluid_m], cmap=cmap,
        s=s_lumen, vmin=vmin, vmax=vmax,
        linewidths=0, rasterized=True, zorder=2,
    )
    # Wall nodes (same colourmap, thin dark edge)
    if w.any():
        ax.scatter(
            pos[w, 0], pos[w, 1],
            c=vals[w], cmap=cmap,
            s=s_wall, vmin=vmin, vmax=vmax,
            linewidths=0.5, edgecolors=WALL_COLOR,
            rasterized=True, zorder=4,
        )

    if zoom_limits is not None:
        apply_zoom(ax, zoom_limits)

    annotate_scores(ax, wall_score=wall_score, off_score=off_score)
    return sc


# ---------------------------------------------------------------------------
# Clot field  (wall=circles, off-wall=squares)
# ---------------------------------------------------------------------------
def plot_clot_field(
    ax,
    pos: np.ndarray,
    phi: np.ndarray,
    *,
    wall: np.ndarray | None = None,
    title: str = "",
    threshold: float = CLOT_THRESHOLD,
    zoom_limits=None,
    s_lumen: float = 5.0,
    s_clot_wall:  float = 14.0,   # circles — wall-attached clot
    s_clot_off:   float = 14.0,   # squares — off-wall clot
    s_wall_open:  float = 8.0,
    wall_score: float | None = None,
    off_score:  float | None = None,
) -> ScalarMappable:
    """
    Rendering:
      - Open lumen   → grey circles
      - Off-wall clot → YlOrRd SQUARES (■)
      - Wall-attached clot → YlOrRd CIRCLES (●)  on top of dark wall strip
      - Open wall nodes → dark strip
    """
    vals  = np.asarray(phi, dtype=np.float64).ravel()
    clot  = vals >= threshold
    open_ = ~clot

    if wall is not None:
        w = np.asarray(wall, dtype=bool).ravel()
    else:
        w = np.zeros(len(pos), dtype=bool)

    ax.clear()
    style_ax(ax, title=title)

    # Invisible scatter to carry the norm for external colorbar reference
    norm = Normalize(vmin=0.0, vmax=1.0)
    sc_ref = ScalarMappable(cmap=CLOT_CMAP, norm=norm)
    sc_ref.set_array([])

    # 1. Open lumen (not wall)
    open_lumen = open_ & ~w
    if open_lumen.any():
        ax.scatter(pos[open_lumen, 0], pos[open_lumen, 1],
                   c=LUMEN_COLOR, s=s_lumen,
                   linewidths=0, rasterized=True, zorder=1)

    # 2. Open wall nodes — dark strip
    open_wall = open_ & w
    if open_wall.any():
        ax.scatter(pos[open_wall, 0], pos[open_wall, 1],
                   c=WALL_COLOR, s=s_wall_open,
                   linewidths=0, rasterized=True, zorder=3)

    # 3. Off-wall clot — SQUARES
    clot_off = clot & ~w
    if clot_off.any():
        c_phi = np.clip(vals[clot_off], 0.0, 1.0)
        sizes = s_clot_off + 12.0 * np.power(c_phi, 1.5)
        ax.scatter(pos[clot_off, 0], pos[clot_off, 1],
                   c=c_phi, cmap=CLOT_CMAP,
                   s=sizes, vmin=0.0, vmax=1.0,
                   linewidths=0, marker="s",
                   rasterized=True, zorder=5)

    # 4. Wall-attached clot — CIRCLES
    clot_wall = clot & w
    if clot_wall.any():
        c_phi = np.clip(vals[clot_wall], 0.0, 1.0)
        sizes = s_clot_wall + 12.0 * np.power(c_phi, 1.5)
        ax.scatter(pos[clot_wall, 0], pos[clot_wall, 1],
                   c=c_phi, cmap=CLOT_CMAP,
                   s=sizes, vmin=0.0, vmax=1.0,
                   linewidths=0.3, edgecolors=WALL_COLOR,
                   marker="o", rasterized=True, zorder=6)

    if zoom_limits is not None:
        apply_zoom(ax, zoom_limits)

    annotate_scores(ax, wall_score=wall_score, off_score=off_score)
    return sc_ref


# ---------------------------------------------------------------------------
# Error map (TP/FP/FN)
# ---------------------------------------------------------------------------
def plot_clot_error_map(
    ax,
    pos: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    wall: np.ndarray | None = None,
    title: str = "",
    zoom_limits=None,
    s_bg:  float = 4.0,
    s_err: float = 12.0,
    wall_score: float | None = None,
    off_score:  float | None = None,
) -> None:
    pred_b = np.asarray(pred, dtype=bool).ravel()
    gt_b   = np.asarray(gt,   dtype=bool).ravel()

    if wall is not None:
        w = np.asarray(wall, dtype=bool).ravel()
    else:
        w = np.zeros(len(pos), dtype=bool)

    ax.clear()
    style_ax(ax, title=title)

    tp  = pred_b  &  gt_b
    fp  = pred_b  & ~gt_b
    fn  = ~pred_b &  gt_b
    # This panel is for errors, not confirmation — TP is folded into the same
    # neutral tone as true negatives instead of getting its own highlight
    # color, so FP/FN aren't buried under a sea of green.
    ok  = ~fp & ~fn

    # Background / correctly-classified nodes (small, neutral)
    if ok.any():
        ax.scatter(pos[ok, 0], pos[ok, 1],
                   c=BG_COLOR, s=s_bg, linewidths=0,
                   rasterized=True, zorder=1)

    # Wall strip: coloured circles sitting exactly on the wall curve, no edge —
    # reads as a continuous coloured line tracing the vessel boundary.
    for mask_cls, col in [
        (ok  & w, "#aaaaaa"),
        (fn  & w, FN_COLOR),
        (fp  & w, FP_COLOR),
    ]:
        if mask_cls.any():
            ax.scatter(pos[mask_cls, 0], pos[mask_cls, 1],
                       c=col, s=s_err * 0.8, linewidths=0, marker="o",
                       rasterized=True, zorder=3)

    # Off-wall (lumen) error nodes: larger squares with a dark edge, drawn
    # above the wall strip — reads as clots floating off the boundary line
    # rather than more dots on it, which is the wall/off-wall distinction
    # that matters here.
    nw = ~w
    for mask_cls, col, zo in [
        (fn & nw, FN_COLOR, 4),
        (fp & nw, FP_COLOR, 5),
    ]:
        if mask_cls.any():
            ax.scatter(pos[mask_cls, 0], pos[mask_cls, 1],
                       c=col, s=s_err * 1.6, marker="s",
                       linewidths=0.5, edgecolors="#222222",
                       rasterized=True, zorder=zo)

    if zoom_limits is not None:
        apply_zoom(ax, zoom_limits)

    annotate_scores(ax, wall_score=wall_score, off_score=off_score)


def error_legend_handles() -> list:
    """Legend entries that match plot_clot_error_map's markers exactly —
    each label names both the error class and the domain (wall vs off-wall),
    so reading the legend doesn't require cross-referencing color against
    shape separately."""
    def _dot(color, label):
        return mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                              markersize=7, markeredgewidth=0, label=label)

    def _sq(color, label):
        return mlines.Line2D([], [], color=color, marker="s", linestyle="None",
                              markersize=8, markeredgecolor="#222222",
                              markeredgewidth=0.8, label=label)

    return [
        _dot(FP_COLOR, "Wall · FP"),
        _dot(FN_COLOR, "Wall · FN"),
        _sq(FP_COLOR, "Off-wall · FP"),
        _sq(FN_COLOR, "Off-wall · FN"),
    ]
