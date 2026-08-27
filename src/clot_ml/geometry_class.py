"""Geometry classes: aneurysm / stenosis / baseline, with an explicit abstain.

Stenoses and aneurysms are the class that matters most, and `patient039`-`patient044` are
in it.  Rather than hard-code that list, the class is **measured** from the mesh's own lumen
width and then checked against it -- so it transfers to an unlabelled vessel.

TWO SCALARS, both dimensionless (normalised by the vessel's own median width, so calibre
does not matter), measured over wall nodes **more than 12 hops from the inlet and outlet**
so the cut ends cannot masquerade as pathology, and locally averaged along the wall so a
single bad node cannot:

    bulge      p98(smoothed width) / median      a local dilatation
    narrowing  p2 (smoothed width) / median      a local constriction

RECALIBRATED 2026-08-22 against the repaired `width_nd` (roadmap item A2;
`scripts/diag_geometry_class_recal.py` reproduces every number here).  The channel is
sphere-marched along `wall_normal`, which was identically zero at every wall node on every
pack until the repair, so both cuts had been fitted against a degenerate statistic.  The
docstring's own prediction -- *"fix the channel and the abstain goes away"* -- held:
**unusable width_nd is now 0 of 45, was 9 of 34.**  What the repair did to the two cuts is
not symmetric, and this is the important part:

    aneurysm   bulge >= 2.0       STILL SEPARATES, and by more than before.
                                  designated 2.15 / 2.23 / 2.71 against a cohort maximum of
                                  1.61 (patient013).  Gap 0.54, cut sits in the middle of it.

    stenosis   narrowing <= 0.40  NO LONGER SEPARATES ANYTHING.  The three designated
                                  stenoses read 0.52 / 0.53 / 0.58 and `patient012`, a
                                  baseline, reads 0.51 -- BELOW all three.  Unsmoothed it is
                                  the same story (012 p2/med 0.510 against 041's 0.538).
                                  No threshold on this statistic can recover the labels.

So the measured stenosis branch is **effectively dead**: it fires only on a vessel far more
severe than any labelled one.  It is left in place because it costs nothing and would still
catch a severe unlabelled case, but nothing should be inferred from it NOT firing, and
`NARROWING_STENOSIS` must **not** be retuned -- refitting a cut whose classes overlap is not
calibration, it is coin-flipping (PHASE10 13.3, `clot-cohort-noise-floor`).

CONSEQUENCE: `classify` now takes ``USER_DESIGNATED`` as **authoritative** wherever it exists,
and falls back to the measured statistics only for an unlabelled vessel -- which is the job
the statistics were introduced for ("so it transfers to an unlabelled vessel").  Before this,
the designation was consulted only when the width channel was unusable, so once the repair
made every vessel usable all three designated stenoses silently reclassified to `baseline`
and the priority-class reporting axis lost them.

THE ISOLATED-NODE BUG, found by A2 and fixed here.  The along-wall smoothing averaged over a
node's SELECTED NEIGHBOURS ONLY.  A selected node with no selected neighbour therefore divided
0 by `max(0, 1)` and smoothed to **exactly 0**, which then set the 2nd percentile.  That is
`patient008`'s `narrowing = 0.0000` -- 12 isolated nodes, and its raw wall width never goes
below 0.61.  The window now includes the node itself, which is both the standard estimator and
divide-by-zero-free.  It moves `patient008` 0.0000 -> 0.8533 and the designated stenoses
0.46/0.48/0.51 -> 0.54/0.52/0.58; the aneurysm statistic barely moves.

CONSEQUENCE FOR THE PROTOCOL, worth stating plainly: **DEV (040/041/044) is entirely
priority-class and FIT is entirely baseline.**  Every FIT-vs-DEV difference in
`docs/PHASE9_ML.md` is therefore confounded with a geometry-class difference, and neither
split can certify the other.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.data_gen.lib.mesh_wls import solid_boundary_nodes

BULGE_ANEURYSM = 2.0
NARROWING_STENOSIS = 0.40
BOUNDARY_HOPS = 12
# Usable-width guard: outside this band the channel is not measuring anatomy.
WIDTH_OK_LO, WIDTH_OK_HI = 0.40, 5.0

USER_DESIGNATED = {
    "patient039": "aneurysm", "patient040": "aneurysm", "patient043": "aneurysm",
    "patient041": "stenosis", "patient042": "stenosis", "patient044": "stenosis",
}
PRIORITY = ("aneurysm", "stenosis", "stenosis+aneurysm")


def _adj(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def width_stats(data) -> dict:
    ch = {c: i for i, c in enumerate(data.x_channel_names.split(","))}
    nan = dict(bulge=float("nan"), narrowing=float("nan"), usable=False)
    if "width_nd" not in ch:
        return nan
    x = data.x.detach().cpu().numpy()
    w = x[:, ch["width_nd"]].astype(np.float64)
    # the wall/wound UNION: this is a geometry question, and a wound node is wall
    # (MODEL_REVIEW_2026-08-22 5b.5(1)).  Identical to `mask_wall` on every no-wound pack.
    wall = solid_boundary_nodes(data)
    n = len(wall)
    A = _adj(data.edge_index.detach().cpu().numpy(), n)

    io_ = np.zeros(n, bool)
    for k in ("mask_inlet", "mask_outlet"):
        m = getattr(data, k, None)
        if m is not None:
            io_ |= m.reshape(-1).bool().cpu().numpy()
    far = np.ones(n, bool)
    if io_.any():
        cur, d = io_.copy(), np.full(n, 99, np.int16)
        d[cur] = 0
        for h in range(1, BOUNDARY_HOPS + 1):
            nxt = ((A @ cur.astype(np.int8)) > 0) & ~cur
            if not nxt.any():
                break
            d[nxt] = h
            cur = cur | nxt
        far = d > BOUNDARY_HOPS

    sel = wall & far & (w > 0)
    if int(sel.sum()) < 30:
        return nan
    # Smooth over the node's selected neighbours AND ITSELF.  Neighbours-only divided by
    # `max(cnt, 1)`, so a selected node with no selected neighbour smoothed to exactly 0 and
    # set the 2nd percentile -- `patient008` read `narrowing = 0.0000` from 12 such nodes
    # while its raw wall width never went below 0.61.  See the module docstring.
    selv = sel.astype(np.float64)
    cnt = np.asarray(A @ selv).reshape(-1) + selv
    sm = (np.asarray(A @ np.where(sel, w, 0.0)).reshape(-1) + np.where(sel, w, 0.0)) \
        / np.maximum(cnt, 1.0)
    ws, sml = w[sel], sm[sel]
    med = float(np.median(ws))
    if med <= 0:
        return nan
    lo, hi = float(np.percentile(ws, 5) / med), float(np.percentile(ws, 95) / med)
    usable = (WIDTH_OK_LO <= lo) and (hi <= WIDTH_OK_HI)
    return dict(bulge=float(np.percentile(sml, 98) / med),
                narrowing=float(np.percentile(sml, 2) / med),
                width_median=med, usable=bool(usable))


def classify(stats: dict, anchor: str | None = None) -> str:
    """Geometry class, human designation first.

    ``USER_DESIGNATED`` is AUTHORITATIVE where it exists: it is a human reading of the
    geometry, and the measured statistics exist to extend that reading to vessels nobody has
    labelled -- not to overrule it.  Deferring to the measurement instead is what silently
    reclassified all three designated stenoses to `baseline` when the `width_nd` repair made
    the channel usable (see the module docstring).  Use :func:`measured_class` for the
    label-free answer, and :func:`disagreements` to see where the two differ.
    """
    if anchor and anchor in USER_DESIGNATED:
        return USER_DESIGNATED[anchor]
    if not stats.get("usable", False):
        return "unknown"
    b, nr = stats.get("bulge", np.nan), stats.get("narrowing", np.nan)
    an = b == b and b >= BULGE_ANEURYSM
    st = nr == nr and nr <= NARROWING_STENOSIS
    if an and st:
        return "stenosis+aneurysm"
    if an:
        return "aneurysm"
    if st:
        return "stenosis"
    return "baseline"


def measured_class(stats: dict) -> str:
    """What the statistics alone say -- no designation, no fallback.

    This is the honest answer for an unlabelled vessel, and the thing to compare the
    designation against.  Note the module docstring: the stenosis branch does not currently
    separate the labelled stenoses from `patient012`, so a `baseline` from this function is
    evidence about the BULGE only.
    """
    return classify(dict(stats), None)


def disagreements(rows: dict) -> dict:
    """``{anchor: (designated, measured)}`` wherever a labelled vessel is not reproduced.

    ``rows`` maps anchor -> the dict :func:`width_stats` returns.  Kept as a function rather
    than a printout so a test can assert on the CURRENT disagreement set and fire when it
    changes -- in either direction.  A future `width_nd` improvement that makes the stenosis
    branch work again should not pass silently.
    """
    out = {}
    for a, s in rows.items():
        want = USER_DESIGNATED.get(a)
        if want is None:
            continue
        got = measured_class(s)
        if got != want:
            out[a] = (want, got)
    return out


def is_priority(cls: str) -> bool:
    return cls in PRIORITY


def classify_cohort(anchors, load) -> dict:
    out = {}
    for a in anchors:
        d = load(a)
        if d is None:
            continue
        s = width_stats(d)
        out[a] = dict(cls=classify(s, a), **s)
    return out
