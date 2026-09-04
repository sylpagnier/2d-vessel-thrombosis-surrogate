"""How big does a difference have to be, on this cohort, before it means anything?

Every table in `docs/PHASE9_ML.md` compares cohort means over 19 vessels, several of which
carry fewer than 15 off-wall GT nodes, and differences of 0.01-0.03 are routinely read as
wins.  Nothing in the project has ever put an interval on one.  This does, two ways:

  * **paired bootstrap over vessels** -- resample the 19 vessels with replacement and
    recompute the paired difference, which is the right unit because both arms are scored
    on the same vessels;
  * **the seed floor** -- the spread between individual configurations of the *same* arm,
    which bounds from below what any feature or readout change has to beat.

    python scripts/eval_significance.py --a cv5a,cv5b,cv5c --b v4a,v4b --cache gt

**The floor is a property of the cohort, so it must be re-measured whenever the cohort or the
features change.**  The +/-0.024 wall / +/-0.091 off-wall figure quoted throughout PHASE10 was
measured at n=19 on the pre-repair features; the 2026-08-22 cohort is 23 clot-carrying
vessels on rebuilt features and inherits none of it.

CLOT-FREE VESSELS: they take part in readout SELECTION exactly as `eval_strict.py` runs it
(their false-positive branch, `empty_gt="score"`), and are then excluded from every reported
mean and from the bootstrap.  A vessel that scores 1.0000 for every arm carries no spread, so
folding them into a NOISE FLOOR would shrink it for free -- which is the one thing a floor
must not do.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root


from scripts.eval_strict import FAMILIES, GRID, BoundScorer, load_scores  # noqa: E402
from src.clot_ml.data import (  # noqa: E402
    attach_physics, load_cache, off_domain, wall_domain,
)
from src.clot_ml.severity_metric import DEFAULT  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402


def nested_rows(cache, tags):
    """Per-vessel held-out (wall, off) under the strict protocol of eval_strict.py.

    Returns ``(carrying, rows)`` -- the CLOT-CARRYING pool and the rows.  Clot-free vessels
    stay in ``sel`` so the readout is chosen exactly as `eval_strict.py` chooses it, and are
    then dropped: they score 1.0000 for everything and would flatten the floor.
    """
    pool, folds, sc = load_scores(tags)
    pool = [a for a in pool if a in cache]
    carrying = [a for a in pool if a not in CLOT_FREE]
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}
    vs = {a: BoundScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                         len(cache[a]["wall"]), DEFAULT,
                         "score" if a in CLOT_FREE else "nan") for a in pool}
    out = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        best = None
        for fam, (tune, apply_) in FAMILIES.items():
            th = tune(cache, vs, sel, {a: oof[a] for a in sel}, GRID)
            vals = []
            for a in sel:
                S = cache[a]
                for d in (wall_domain(S), off_domain(S)):
                    x = vs[a].score(apply_(S, oof[a], th) & d, d)
                    if x == x:
                        vals.append(x)
            q = float(np.mean(vals))
            if best is None or q > best[0]:
                best = (q, fam, th)
        _, fam, th = best
        for a in held:
            if a in CLOT_FREE:
                continue
            S = cache[a]
            w, o = wall_domain(S), off_domain(S)
            pr = FAMILIES[fam][1](S, oof[a], th)
            out[a] = (vs[a].score(pr & w, w), vs[a].score(pr & o, o))
    return carrying, out


def boot(pool, A, B, i, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    keep = [a for a in pool if A[a][i] == A[a][i] and B[a][i] == B[a][i]]
    d = np.array([B[a][i] - A[a][i] for a in keep])
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), \
        float((bs <= 0).mean()), len(keep)


def _floor_report(cache, pool, floors) -> None:
    """The seed/config floor: the spread between single members of the SAME arm.

    This is the number every later claim is measured against -- a feature or readout change
    has to beat the spread that three configurations of the *same* model already show, or it
    is not distinguishable from re-rolling the dice.  Also reports the per-vessel spread,
    because a cohort-mean floor hides which vessels are actually unstable.
    """
    print("\nSEED / CONFIG FLOOR -- single members of one arm, same protocol")
    print("n = %d clot-carrying vessels, tags = %s\n" % (len(pool), ",".join(floors)))
    rs = {}
    for t in floors:
        _, r = nested_rows(cache, [t])
        rs[t] = r
        print("  %-8s wall %.4f  off %.4f"
              % (t, np.nanmean([r[a][0] for a in pool if a in r]),
                 np.nanmean([r[a][1] for a in pool if a in r])))
    print()
    for i, dom in ((0, "wall"), (1, "off")):
        v = [np.nanmean([rs[t][a][i] for a in pool if a in rs[t]]) for t in floors]
        per = []
        for a in pool:
            x = [rs[t][a][i] for t in floors if a in rs[t] and rs[t][a][i] == rs[t][a][i]]
            if len(x) == len(floors):
                per.append(max(x) - min(x))
        print("  %-5s cohort-mean spread %.4f (min %.4f max %.4f) | per-vessel spread "
              "median %.4f p90 %.4f max %.4f on n=%d"
              % (dom, max(v) - min(v), min(v), max(v),
                 float(np.median(per)) if per else float("nan"),
                 float(np.percentile(per, 90)) if per else float("nan"),
                 float(max(per)) if per else float("nan"), len(per)))
    print("\n  A change smaller than the cohort-mean spread is not a result.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="", help="baseline tags, comma separated")
    ap.add_argument("--b", default="",
                    help="candidate tags.  Omit BOTH to measure only the floor, which is "
                         "what a fresh re-baseline needs -- there is nothing to compare "
                         "against yet, and the floor is what every later comparison is "
                         "judged against.")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--floor", default="v5a,v5b,v5c",
                    help="single tags of the SAME arm, to bound the seed/config floor")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    floors = [t for t in args.floor.split(",") if t]
    if not (args.a and args.b):
        pool, _ = nested_rows(cache, floors[:1])
        _floor_report(cache, pool, floors)
        return 0
    pool, A = nested_rows(cache, args.a.split(","))
    _, B = nested_rows(cache, args.b.split(","))

    print("\nPAIRED DIFFERENCE  (%s)  ->  (%s),  n=%d clot-carrying vessels\n"
          % (args.a, args.b, len(pool)))
    print("%-6s %8s %8s %18s %10s %4s" % ("dom", "base", "cand", "diff [95% CI]", "P(<=0)", "n"))
    for i, dom in ((0, "wall"), (1, "off")):
        m, lo, hi, p, n = boot(pool, A, B, i)
        ba = np.nanmean([A[a][i] for a in pool])
        bb = np.nanmean([B[a][i] for a in pool])
        print("%-6s %8.4f %8.4f  %+.4f [%+.4f,%+.4f] %8.3f %4d"
              % (dom, ba, bb, m, lo, hi, p, n))

    if len(floors) > 1:
        _floor_report(cache, pool, floors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
