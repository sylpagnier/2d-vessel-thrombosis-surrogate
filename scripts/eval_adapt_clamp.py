"""Bound the adaptive cut to the support it was actually fitted on.

MOTIVATION, from the readout's own construction -- no held-out vessel is involved.

`eval_strict.tune_adapt` picks one slope `b` by maximising the mean score over the selection
vessels, whose statistic `stat = mean score in the domain` spans some interval [lo, hi].  It
then returns `(b, med)` and **throws the interval away**.  `apply_adapt` computes
`off = b * (stat - med)` with no bound, so for a vessel outside [lo, hi] the cut is displaced
by an amount no labelled vessel ever validated, and the displacement grows without limit as
`stat` departs.  With `b = -0.6` and the four `resid` cuts, a `stat` far enough above `med`
drives every cut to the 0.02 floor, i.e. commits the entire domain.

The guardrail is the obvious one: clamp `stat` to [lo, hi] before forming the offset.  Inside
the support it is an EXACT no-op -- byte-identical masks -- so it cannot flatter anything the
cohort already measures.  Outside it, the cut is held at the most extreme perturbation the
fit actually saw instead of continuing a line nothing supports.

This script establishes three things on the 19-vessel pool alone:

  1. HOW MUCH EXPOSURE EXISTS.  Under the shipped 5-fold protocol, how often is a held-out
     vessel already exterior to its own selection set's [lo, hi], and how big is the
     unvalidated part of its offset.
  2. THAT THE CLAMP IS A NO-OP WHERE THE COHORT CAN SEE IT.  Interior vessels must be
     unchanged to four decimals; if they are not, the implementation is wrong.
  3. WHAT IT DOES WHERE IT BINDS.  An EXTERIOR-CV partition -- folds assigned by `stat`
     extremity rather than geometry, so the held-out vessels are precisely the ones the
     selection set does not bracket.  This is the only pool-only way to put the
     extrapolation regime under test; ordinary geometry-stratified CV mostly does not.

    python scripts/eval_adapt_clamp.py --tags v5a,v5b,v5c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import (  # noqa: E402
    B_GRID, FAMILIES, GRID, attach_physics, load_cache, load_scores, vessel_stat,
)
from src.clot_ml.geometry_splits import classes_for  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def wall_of(S):
    return S["wall"]


def off_of(S):
    return ~S["wall"]


DOMAINS = (("wall", wall_of), ("off", off_of))


def tune_adapt_support(cache, vs, anchors, sc, family, th, dom_of):
    """`eval_strict.tune_adapt`, but also returning the SUPPORT the slope was fitted on.

    Identical selection arithmetic -- same `B_GRID`, same objective, same `med` -- so the
    chosen `b` is bit-for-bit what the shipped path chooses.  The only addition is `(lo, hi)`.
    """
    apply_ = FAMILIES[family][1]
    sv = {a: vessel_stat(cache[a], sc[a], dom_of(cache[a])) for a in anchors}
    med = float(np.median([sv[a] for a in anchors])) if anchors else 0.0
    best = None
    for b in B_GRID:
        vals = []
        for a in anchors:
            S = cache[a]
            d = dom_of(S)
            off = b * (sv[a] - med)
            x = vs[a].score(apply_(S, sc[a], tuple(np.clip(np.array(th) + off, 0.02, 0.98)))
                            & d, d)
            if x == x:
                vals.append(x)
        q = float(np.mean(vals)) if vals else -1e9
        if best is None or q > best[0]:
            best = (q, float(b))
    lo, hi = (float(min(sv.values())), float(max(sv.values()))) if sv else (0.0, 0.0)
    return best[1], med, lo, hi


def apply_adapt_clamped(S, sc, family, th, dom_of, b, med, lo=None, hi=None):
    """`eval_strict.apply_adapt` with the statistic held inside the fitted support."""
    stat = vessel_stat(S, sc, dom_of(S))
    used = stat if lo is None else float(min(max(stat, lo), hi))
    off = b * (used - med)
    return FAMILIES[family][1](S, sc, tuple(np.clip(np.array(th) + off, 0.02, 0.98)))


def pick_family(cache, vs, sel, sel_sc):
    """One family for both domains on the whole selection set -- eval_strict's default."""
    best = None
    for fam, (tune, apply_) in FAMILIES.items():
        th = tune(cache, vs, sel, sel_sc, GRID)
        vals = []
        for a in sel:
            S = cache[a]
            pr = apply_(S, sel_sc[a], th)
            for d in (S["wall"], ~S["wall"]):
                v = vs[a].score(pr & d, d)
                if v == v:
                    vals.append(v)
        q = float(np.mean(vals))
        if best is None or q > best[0]:
            best = (q, fam, th)
    return best[1], best[2]


def run(cache, vs, pool, folds, oof, label):
    """One partition, both arms.  Returns per-vessel rows."""
    rows = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        sel_sc = {a: oof[a] for a in sel}
        fam, th = pick_family(cache, vs, sel, sel_sc)
        par = {dk: tune_adapt_support(cache, vs, sel, sel_sc, fam, th, dom_of)
               for dk, dom_of in DOMAINS}
        for a in held:
            S, w = cache[a], cache[a]["wall"]
            r = dict(fold=k, family=fam, partition=label)
            masks = {}
            for arm in ("unclamped", "clamped"):
                m = {}
                for dk, dom_of in DOMAINS:
                    b, med, lo, hi = par[dk]
                    bounds = (None, None) if arm == "unclamped" else (lo, hi)
                    m[dk] = apply_adapt_clamped(S, oof[a], fam, th, dom_of, b, med, *bounds)
                pr = (w & m["wall"]) | (~w & m["off"])
                masks[arm] = pr
                r[arm] = dict(wall=vs[a].score(pr & w, w), off=vs[a].score(pr & ~w, ~w),
                              n_pred=int(pr.sum()))
            r["identical"] = bool((masks["unclamped"] == masks["clamped"]).all())
            for dk, dom_of in DOMAINS:
                b, med, lo, hi = par[dk]
                stat = vessel_stat(S, oof[a], dom_of(S))
                clamped = float(min(max(stat, lo), hi))
                r[dk + "_stat"] = stat
                r[dk + "_support"] = [lo, hi]
                r[dk + "_exterior"] = float(stat - clamped)     # 0 if inside
                r[dk + "_off_unvalidated"] = float(b * (stat - clamped))
            rows[a] = r
    return rows


def summarise(rows, pool, title):
    def agg(arm, dom, keep):
        v = [rows[a][arm][dom] for a in pool if keep(a) and rows[a][arm][dom] == rows[a][arm][dom]]
        return (float(np.mean(v)) if v else float("nan")), len(v)

    ext = lambda a: abs(rows[a]["wall_exterior"]) > 0 or abs(rows[a]["off_exterior"]) > 0  # noqa: E731
    print("\n=== %s ===" % title)
    print("%-22s %18s %18s" % ("subset", "wall (uncl -> clamp)", "off (uncl -> clamp)"))
    for nm, keep in (("all", lambda a: True), ("interior only", lambda a: not ext(a)),
                     ("EXTERIOR only", ext)):
        n = sum(1 for a in pool if keep(a))
        if not n:
            print("%-22s %18s %18s   (n=0)" % (nm, "-", "-"))
            continue
        uw, nw = agg("unclamped", "wall", keep)
        cw, _ = agg("clamped", "wall", keep)
        uo, no = agg("unclamped", "off", keep)
        co, _ = agg("clamped", "off", keep)
        print("%-22s %8.4f -> %.4f %8.4f -> %.4f   (n=%d, off n=%d)"
              % (nm, uw, cw, uo, co, n, no))
    n_ident = sum(1 for a in pool if rows[a]["identical"])
    print("masks bit-identical between arms: %d/%d" % (n_ident, len(pool)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--n-exterior-folds", type=int, default=5)
    ap.add_argument("--save", default="outputs/eval_adapt_clamp.json")
    args = ap.parse_args()

    cfg = DEFAULT if args.metric == "severity" else LEGACY
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), cfg) for a in pool}
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}

    # ---- 1. the shipped geometry-stratified partition -------------------------------
    geo = run(cache, vs, pool, folds, oof, "geometry")
    summarise(geo, pool, "geometry-stratified 5-fold (the shipped protocol)")

    # ---- 2. EXTERIOR-CV: folds assigned by `stat` extremity --------------------------
    # Sort by the wall statistic and deal the extremes into their own folds, so each fold's
    # held-out vessels are the ones its selection set does NOT bracket.  Same protocol, same
    # arithmetic, only the partition differs -- and both arms share it, so the comparison is
    # clean.  Nothing here uses a vessel outside the 19-vessel pool.
    order = sorted(pool, key=lambda a: vessel_stat(cache[a], oof[a], cache[a]["wall"]))
    k = max(int(args.n_exterior_folds), 2)
    ext_folds = {}
    for i in range(k // 2):
        ext_folds[i] = [order[i], order[-(i + 1)]]          # one low, one high extreme
    rest = [a for a in order if not any(a in v for v in ext_folds.values())]
    for j, chunk in enumerate(np.array_split(np.array(rest, dtype=object), max(k - k // 2, 1))):
        ext_folds[k // 2 + j] = [str(x) for x in chunk]
    ext_folds = {kk: v for kk, v in ext_folds.items() if v}
    ex = run(cache, vs, pool, ext_folds, oof, "exterior")
    summarise(ex, pool, "EXTERIOR-CV (extremes held out; the regime the clamp targets)")

    # ---- 3. exposure detail ----------------------------------------------------------
    print("\nper-vessel exposure (EXTERIOR-CV partition; blank = inside the fitted support)")
    print("  %-12s %6s %7s %-16s %9s %9s %8s" % ("vessel", "cls", "stat", "support",
                                                 "outside", "unvalid.", "d wall"))
    for a in sorted(pool, key=lambda a: -abs(ex[a]["wall_exterior"])):
        r = ex[a]
        if abs(r["wall_exterior"]) == 0 and abs(r["off_exterior"]) == 0:
            continue
        lo, hi = r["wall_support"]
        print("  %-12s %6s %7.3f [%.3f,%.3f]  %+9.3f %+9.3f %+8.4f"
              % (a, classes.get(a, "?")[:6], r["wall_stat"], lo, hi, r["wall_exterior"],
                 r["wall_off_unvalidated"], r["clamped"]["wall"] - r["unclamped"]["wall"]))

    out = REPO / args.save
    out.write_text(json.dumps(dict(geometry=geo, exterior=ex, tags=args.tags), default=float))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
