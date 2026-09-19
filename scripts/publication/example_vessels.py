"""Shared loaders and scorers for the per-vessel example panels (figures 0.3, 0.5, 0.6, 0.6b).

Every example panel needs the same four things for a vessel -- node positions, the wall mask, the
edge list, and predicted/GT masks over time -- and the same three scores on them.  They were
being re-derived per script with different metric settings (fig 0.5 scored BATC_0 while fig 0.6
scored BATC), which is how two figures in one paper came to quote numbers that cannot sit side by
side.  One module, one definition of each score.

SCORES, all computed on the same masks and the same domain:

    strict F1   node-exact F1: a prediction counts only on the exact GT node.  No tolerance,
                no graces -- what a reader means by "just use F1".
    no grace    BATC's k-hop tolerance and shape term, with every burden grace zeroed.
    BATC        the reported metric (`severity_metric.BATC`).

    python -c "from scripts.publication.example_vessels import *"
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from src.clot_ml.severity_metric import BATC, dilation_operator, severity_components

#: BATC with every burden grace zeroed, tolerance and shape weighting unchanged
NO_GRACE = replace(BATC, tau_abs=0.0, rho=0.0, tau_fp_abs=0.0, rho_fp=0.0)

SCORE_NAMES = (("strict_f1", "strict F1"), ("no_grace", "no grace"), ("batc", "BATC"))


from src.clot_ml.evaluate import strict_f1  # noqa: E402  (one shared definition)


def scores(pred: np.ndarray, gt: np.ndarray, domain: np.ndarray, D) -> dict:
    """The three scores plus the counts behind them, for one domain."""
    b = severity_components(pred, gt, D, domain, cfg=BATC)
    ng = severity_components(pred, gt, D, domain, cfg=NO_GRACE)
    return dict(strict_f1=strict_f1(pred, gt, domain),
                no_grace=float("nan") if b["empty_gt"] else float(ng["score"]),
                batc=float("nan") if b["empty_gt"] else float(b["score"]),
                n_gt=int((gt & domain).sum()), n_pred=int((pred & domain).sum()),
                fp=int(b["fp"]), fn=int(b["fn"]))


def batc_score(pred, gt, domain, D) -> float:
    """BATC on one domain; NaN where the domain carries no GT clot (pre-onset)."""
    c = severity_components(pred, gt, D, domain, cfg=BATC)
    return float("nan") if c["empty_gt"] else float(c["score"])


#: overlay classes: (key, label, colour, size multiplier, zorder)
OVERLAY = (
    ("hit", "clot, predicted exactly", "#8c1d18", 1.0, 5),
    ("near", f"miss within {BATC.relax_hops} hops (forgiven)", "#f0a202", 1.0, 4),
    ("fp", "false positive, beyond tolerance", "#c2185b", 1.35, 6),
    ("fn", "missed clot, beyond tolerance", "#1f78b4", 1.35, 6),
)


def overlay_classes(pred, gt, domain, D) -> dict:
    """Split a domain's prediction/GT disagreement into exact, forgiven and real errors."""
    p, g = pred & domain, gt & domain
    near_g = (D @ g.astype(np.int8)) > 0
    near_p = (D @ p.astype(np.int8)) > 0
    return dict(hit=p & g,
                near=(p & ~g & near_g) | (g & ~p & near_p),
                fp=p & ~g & ~near_g,
                fn=g & ~p & ~near_p)


def plot_overlay(ax, G: dict, pred, gt, domain, *, zoom=None, title="", s=10.0):
    """Mesh nodes as a scatter: lumen faint, wall dark, then the four overlay classes."""
    from scripts.publication.pub_style import apply_zoom, style_ax

    pos, wall = G["pos"], G["wall"]
    style_ax(ax, title=title)
    ax.scatter(pos[~wall, 0], pos[~wall, 1], s=s * 0.18, c="#d7dcdc", linewidths=0,
               rasterized=True, zorder=1)
    ax.scatter(pos[wall, 0], pos[wall, 1], s=s * 0.35, c="#6b7275", linewidths=0,
               rasterized=True, zorder=2)
    # A list of domains is classified PER DOMAIN and merged, exactly as BATC scores them: an
    # off-wall false positive beside wall clot is not "forgiven" by clot in the other domain.
    doms = domain if isinstance(domain, (list, tuple)) else [domain]
    cls = None
    for dm in doms:
        c = overlay_classes(pred, gt, dm, G["D"])
        cls = c if cls is None else {k: cls[k] | c[k] for k in cls}
    for key, _label, colour, mult, z in OVERLAY:
        m = cls[key]
        if m.any():
            ax.scatter(pos[m, 0], pos[m, 1], s=s * mult, c=colour, linewidths=0,
                       marker="s" if key in ("fp", "fn") else "o", rasterized=True, zorder=z)
    if zoom is not None:
        apply_zoom(ax, zoom)
    return cls


def overlay_legend_handles(keys=("hit", "near", "fp", "fn")) -> list:
    import matplotlib.lines as mlines

    return [mlines.Line2D([], [], color=c, marker="s" if k in ("fp", "fn") else "o",
                          linestyle="None", markersize=6, label=lab)
            for k, lab, c, _m, _z in OVERLAY if k in keys]


def zoom_for(G: dict, masks, aspect: float | None = None, scale: float = 1.0):
    """One zoom window around every clot node in `masks`, shared across a panel group.

    ``aspect`` (width / height) widens the shorter side to match the panel it is drawn in, so a
    tall clot in a wide panel shows its surroundings instead of leaving the row empty.
    ``scale`` > 1 zooms out about the window's centre, so the clot is seen in its vessel.
    """
    from scripts.publication.pub_style import clot_zoom_limits

    union = np.zeros(G["n"], dtype=bool)
    for m in masks:
        union |= np.asarray(m, dtype=bool)
    lim = clot_zoom_limits(G["pos"], union, G["wall"])
    if lim is None:
        return lim
    x0, x1, y0, y1 = lim
    if scale != 1.0:
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
        x0, x1, y0, y1 = cx - hw, cx + hw, cy - hh, cy + hh
    if aspect is None:
        return x0, x1, y0, y1
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        grow = (aspect * h - w) / 2
        x0, x1 = x0 - grow, x1 + grow
    else:
        grow = (w / aspect - h) / 2
        y0, y1 = y0 - grow, y1 + grow
    return x0, x1, y0, y1


def load_pack(stem: str):
    from scripts.publication.utils import get_pack_path

    data = torch.load(get_pack_path(stem), map_location="cpu", weights_only=False)
    data.graph_stem = stem
    return data


def geometry(data) -> dict:
    """Positions, wall / off-wall domains and the BATC dilation operator for one pack."""
    from src.clot_ml.wound import solid_mask

    n = int(data.num_nodes)
    ei = data.edge_index.detach().cpu().numpy()
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid = np.asarray(solid_mask(data)).reshape(-1).astype(bool)
    return dict(pos=data.x[:, 0:2].cpu().numpy(), wall=wall, off=~solid, wound=solid & ~wall,
                ei=ei, n=n, D=dilation_operator(ei, n, hops=BATC.relax_hops))


#: colour of the wound band and its cut lines, as the app draws them (its `--muted` token)
WOUND_COLOUR = "#6b7275"


def wall_with_wound(G: dict) -> np.ndarray:
    """Wall mask with the wound boundary OR'd in -- the app treats the wound as wall for drawing.

    `mask_wall` and `mask_wound` are disjoint, so without this the injured boundary renders as a
    gap in the wall and its clot as off-wall squares (src/tools/customer_predict_web.py).
    """
    w = G.get("wound")
    return G["wall"] if w is None else (G["wall"] | w)


def mark_wound(ax, G: dict) -> bool:
    """The app's wound marker: a faint band across the vessel plus two dashed cut lines.

    Transcribed from `customer_predict_web.py`'s canvas renderer: split the wound nodes at their
    mean height into the two walls, fill the strip between them very lightly, and draw one dashed
    line across the vessel at each axial end of the band, on top of every marker.  Returns whether
    the vessel has a wound at all.
    """
    w = G.get("wound")
    if w is None or not np.any(w):
        return False
    pts = G["pos"][w]
    mid = pts[:, 1].mean()
    top = pts[pts[:, 1] >= mid]
    bot = pts[pts[:, 1] < mid]
    if len(top) and len(bot):
        top = top[np.argsort(top[:, 0])]
        bot = bot[np.argsort(bot[:, 0])]
        poly = np.vstack([top, bot[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color=WOUND_COLOUR, alpha=0.16, lw=0, zorder=1.5)
        for a, b in ((top[0], bot[0]), (top[-1], bot[-1])):
            ax.plot([a[0], b[0]], [a[1], b[1]], ls=(0, (5, 3)), lw=1.4, color=WOUND_COLOUR,
                    alpha=0.9, zorder=8)
    return True


def score_tag(ax, text: str) -> None:
    """A small BATC readout in the panel's lower-left corner."""
    ax.text(0.015, 0.03, text, transform=ax.transAxes, fontsize=8, color="#222222",
            ha="left", va="bottom", zorder=12,
            bbox=dict(fc="white", ec="#c9cfcd", lw=0.6, pad=2.2, alpha=0.92))


def oof_series(stem: str, archive=None) -> dict:
    """Out-of-fold predicted masks with GT at the same timesteps, for an intact vessel."""
    from scripts.eval_wound_complement import gt_series
    from scripts.publication.config import CONFIG
    from scripts.publication.oof_data import ensure_oof_series, load_oof_archive
    from src.config import PhysicsConfig

    archive = archive or load_oof_archive(ensure_oof_series(CONFIG))
    oof = archive.get(stem)
    if oof is None:
        raise KeyError(f"{stem} not in the OOF archive")
    data = load_pack(stem)
    times = [int(t) for t in oof.times.tolist()]
    gts = gt_series(data, PhysicsConfig(phase="biochem"), times)
    return dict(stem=stem, times=np.asarray(times), fold=oof.fold, flow=oof.flow,
                pred=np.asarray(oof.masks, dtype=bool),
                gt=np.stack([np.asarray(gts[t], dtype=bool).reshape(-1) for t in times]),
                **geometry(data))


def _domain_series(pred, gt, domain, D) -> dict:
    """Per-timestep BATC on one domain, with the GT burden so pre-onset frames can be told apart.

    ``score`` follows the shipped metric everywhere, including frames where the domain holds no
    GT clot yet -- there it grades false-positive restraint (`empty_gt_fp_tol`), which is what
    the rollout is being judged on before onset.  ``n_gt`` lets a plot mark those frames.
    """
    out = dict(score=[], n_gt=[], n_pred=[])
    for k in range(pred.shape[0]):
        c = severity_components(pred[k], gt[k], D, domain, cfg=BATC)
        out["score"].append(round(float(c["score"]), 5))
        out["n_gt"].append(int(c["n_gt"]))
        out["n_pred"].append(int(c["n_pred"]))
    return out


def oof_batc_series(regenerate: bool = False) -> dict:
    """BATC over time, wall and off-wall, for every out-of-fold intact vessel (cached json)."""
    import json

    from scripts.publication.config import CONFIG, DATA_DIR
    from scripts.publication.oof_data import ensure_oof_series, load_oof_archive

    path = DATA_DIR / "oof_batc_series.json"
    if path.is_file() and not regenerate:
        return json.loads(path.read_text(encoding="utf-8"))
    archive = load_oof_archive(ensure_oof_series(CONFIG))
    out = {}
    for stem in sorted(archive.vessels):
        S = oof_series(stem, archive)
        out[stem] = dict(times=S["times"].tolist(), fold=S["fold"],
                         wall=_domain_series(S["pred"], S["gt"], S["wall"], S["D"]),
                         off=_domain_series(S["pred"], S["gt"], S["off"], S["D"]))
        print(f"  [oof] {stem}", flush=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def wound_batc_series(regenerate: bool = False) -> dict:
    """BATC over time on the wound domains for the leave-one-vessel-out wound runs (cached)."""
    import json

    from scripts.publication.config import DATA_DIR

    path = DATA_DIR / "wound_batc_series.json"
    if path.is_file() and not regenerate:
        return json.loads(path.read_text(encoding="utf-8"))
    z = np.load(DATA_DIR / "wound_series_fem.npz", allow_pickle=True)
    stems = json.loads(str(z["meta"][0]))["vessels"]
    out = {}
    for stem in stems:
        W = wound_series(stem, z)
        out[stem] = dict(times=W["times"].tolist(),
                         **{dom: _domain_series(W["pred"], W["gt"], W["domains"][dom], W["D"])
                            for dom in ("wall", "w_reg", "w_lum")})
        print(f"  [wound] {stem}", flush=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def _clot_mass_pct(masks: np.ndarray, data) -> list:
    """Clotted nodes as % of the vessel per frame -- the app's `vessel_clot_pct` (inlet/outlet excluded)."""
    edge = (data.mask_inlet.reshape(-1).bool() | data.mask_outlet.reshape(-1).bool()).cpu().numpy()
    interior = ~edge
    return [round(100.0 * float((m & interior).sum()) / float(interior.sum()), 5) for m in masks]


def clot_mass_series(regenerate: bool = False) -> dict:
    """Predicted and GT total clot mass over time: intact out-of-fold and injured LOVO (cached json)."""
    import json

    from scripts.publication.config import CONFIG, DATA_DIR
    from scripts.publication.oof_data import ensure_oof_series, load_oof_archive

    path = DATA_DIR / "clot_mass_series.json"
    if path.is_file() and not regenerate:
        return json.loads(path.read_text(encoding="utf-8"))
    out = {"intact": {}, "wound": {}}
    archive = load_oof_archive(ensure_oof_series(CONFIG))
    for stem in sorted(archive.vessels):
        S = oof_series(stem, archive)
        data = load_pack(stem)
        out["intact"][stem] = dict(times=S["times"].tolist(), pred=_clot_mass_pct(S["pred"], data),
                                   gt=_clot_mass_pct(S["gt"], data))
        print(f"  [mass] {stem}", flush=True)
    z = np.load(DATA_DIR / "wound_series_fem.npz", allow_pickle=True)
    for stem in json.loads(str(z["meta"][0]))["vessels"]:
        W = wound_series(stem, z)
        data = load_pack(stem)
        out["wound"][stem] = dict(times=W["times"].tolist(), pred=_clot_mass_pct(W["pred"], data),
                                  gt=_clot_mass_pct(W["gt"], data))
        print(f"  [mass] {stem}", flush=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def wound_series(stem: str, z) -> dict:
    """Leave-one-vessel-out wound masks from `wound_series_fem.npz`, with the pack geometry."""
    data = load_pack(stem)
    doms = {k.split("|")[2]: np.asarray(z[k], dtype=bool)
            for k in z.files if k.startswith(f"domain|{stem}|")}
    return dict(stem=stem, times=np.asarray(z[f"times|{stem}"]),
                pred=np.asarray(z[f"masks|{stem}"], dtype=bool),
                gt=np.asarray(z[f"gt|{stem}"], dtype=bool), domains=doms, **geometry(data))
