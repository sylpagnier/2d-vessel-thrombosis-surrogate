"""Variance-reduce the readout cut, because its effective sample size is not 19.

MOTIVATION, measured on the pool alone -- no held-out vessel is involved.

`eval_strict.tune_resid` chooses two cuts per domain by taking the argmax of the mean
severity over the selection vessels, on a 33x33 grid.  Two facts about that objective, both
measurable on the 19-vessel pool:

  * **11 of 19 vessels have a flat response.**  Sweeping the whole cut grid moves their
    score by less than 0.05 (7 of them by less than 0.05 across every cut in `GRID`); the
    median vessel has only 4.8% of its wall nodes in the ambiguous band 0.05 < p < 0.95.
    A vessel with a flat response contributes a near-constant to the objective and therefore
    does not move the argmax at all.
  * **So the cut is set by ~8 vessels, not 19** -- and those 8 average 0.860 out-of-fold
    against the flat group's 0.960, i.e. the cut is determined by precisely the vessels the
    model handles worst.  `corr(cut sensitivity, out-of-fold wall) = -0.39`.

An argmax over 1089 grid points fitted on an effective n of 8, on a cohort whose measured
noise floor is +-0.024 wall at n=19 (docs/PHASE10_V4.md 2), is a high-variance estimator.
`docs/PHASE9_ML.md` already identifies variance reduction by ensembling as one of the two
levers that reliably pays on this cohort; this applies it to the readout instead of the
model.  **Bagged selection:** resample the selection vessels with replacement, take the
argmax on each resample, and use the coordinate-wise median.  It is the same estimator with
its sampling noise averaged down -- no new degrees of freedom, no new statistic, and it
reduces to the control as B -> 1.

The script reports the control and the bagged arm under the identical strict protocol, the
paired vessel bootstrap of the difference against the cohort's noise floor, and -- as direct
evidence of the under-determination -- the bootstrap spread of the chosen cut itself.

    python scripts/eval_readout_bagged.py --tags v5a,v5b,v5c --boot 400
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
    GRID, attach_physics, load_cache, load_scores, readout_resid, tune_adapt, vessel_stat,
)
from src.clot_ml.geometry_splits import classes_for  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
NG = len(GRID)


def wall_of(S):
    return S["wall"]


def off_of(S):
    return ~S["wall"]


DOMAINS = (("wall", wall_of), ("off", off_of))


def cut_surface(cache, vs, pool, oof):
    """``tab[dom][a]`` = [NG, NG] severity of the `resid` readout at every (keep, add) pair.

    Precomputing it makes every arm below a pure re-reduction of the same numbers, so the
    control is guaranteed to reproduce `tune_resid` exactly rather than approximately.
    """
    tab = {}
    for dk, dom_of in DOMAINS:
        tab[dk] = {}
        for a in pool:
            S = cache[a]
            d, ph, s = dom_of(S), S["phys_mask"], oof[a]
            M = np.full((NG, NG), np.nan)
            for i, tk in enumerate(GRID):
                keep = d & ph & (s >= tk)
                for j, ta in enumerate(GRID):
                    M[i, j] = vs[a].score(keep | (d & ~ph & (s >= ta)), d)
            tab[dk][a] = M
    return tab


def argmax_mean(tab_dom, anchors, weights=None):
    """Argmax of the (optionally resample-weighted) mean over anchors, NaNs skipped."""
    stack = np.stack([tab_dom[a] for a in anchors])           # [n, NG, NG]
    w = np.ones(len(anchors)) if weights is None else np.asarray(weights, float)
    ok = ~np.isnan(stack)
    num = np.nansum(stack * w[:, None, None], axis=0)
    den = (ok * w[:, None, None]).sum(axis=0)
    mean = np.where(den > 0, num / np.maximum(den, 1e-9), -1e9)
    i, j = np.unravel_index(int(np.argmax(mean)), mean.shape)
    return float(GRID[i]), float(GRID[j])


def bagged_cut(tab_dom, anchors, boot, rng):
    """Coordinate-wise median of the argmax over `boot` bootstrap resamples of `anchors`."""
    picks = []
    n = len(anchors)
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        picks.append(argmax_mean(tab_dom, anchors, w))
    P = np.array(picks)
    return (float(np.median(P[:, 0])), float(np.median(P[:, 1]))), P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--boot", type=int, default=400, help="bootstrap resamples for bagging")
    ap.add_argument("--pair-boot", type=int, default=4000, help="paired vessel bootstrap")
    ap.add_argument("--no-adapt", action="store_true", help="drop the adaptive slope")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="outputs/eval_readout_bagged.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = DEFAULT if args.metric == "severity" else LEGACY
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), cfg) for a in pool}
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}

    print("tabulating the cut surface (%d x %d per vessel per domain) ..." % (NG, NG))
    tab = cut_surface(cache, vs, pool, oof)

    # ---- how under-determined is the cut, on the whole pool? -------------------------
    print("\nHOW WELL DETERMINED IS THE CUT?  bootstrap over the 19 pool vessels, %d draws"
          % args.boot)
    for dk, _ in DOMAINS:
        ctrl = argmax_mean(tab[dk], pool)
        _, P = bagged_cut(tab[dk], pool, args.boot, np.random.default_rng(args.seed))
        print("  %-5s argmax on the full pool = (%.2f, %.2f)" % (dk, *ctrl))
        for c, nm in ((0, "keep"), (1, "add")):
            q = np.percentile(P[:, c], [5, 25, 50, 75, 95])
            print("        %-4s cut  5/25/50/75/95 pct = %.2f %.2f %.2f %.2f %.2f   "
                  "(IQR %.2f, %d distinct values chosen)"
                  % (nm, *q, q[3] - q[1], len(np.unique(P[:, c]))))

    # ---- strict 5-fold, control vs bagged -------------------------------------------
    rows = {a: dict(cls=classes.get(a, "?")) for a in pool}
    chosen = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        th = {}
        th["control"] = tuple(list(argmax_mean(tab["wall"], sel))
                              + list(argmax_mean(tab["off"], sel)))
        bw, _ = bagged_cut(tab["wall"], sel, args.boot, rng)
        bo, _ = bagged_cut(tab["off"], sel, args.boot, rng)
        th["bagged"] = tuple(list(bw) + list(bo))
        chosen[k] = {arm: [float(x) for x in t] for arm, t in th.items()}
        for arm, t in th.items():
            slopes = {}
            if not args.no_adapt:
                for dk, dom_of in DOMAINS:
                    slopes[dk] = tune_adapt(cache, vs, sel, {a: oof[a] for a in sel},
                                            "resid", t, dom_of)
            for a in held:
                S, w = cache[a], cache[a]["wall"]
                m = {}
                for dk, dom_of in DOMAINS:
                    if args.no_adapt:
                        m[dk] = readout_resid(S, oof[a], t)
                    else:
                        b, med = slopes[dk]
                        off = b * (vessel_stat(S, oof[a], dom_of(S)) - med)
                        m[dk] = readout_resid(
                            S, oof[a], tuple(np.clip(np.array(t) + off, 0.02, 0.98)))
                pr = (w & m["wall"]) | (~w & m["off"])
                rows[a][arm] = dict(wall=vs[a].score(pr & w, w),
                                    off=vs[a].score(pr & ~w, ~w), n_pred=int(pr.sum()))
        print("  fold %d  control=%s  bagged=%s"
              % (k, ",".join("%.2f" % x for x in th["control"]),
                 ",".join("%.2f" % x for x in th["bagged"])), flush=True)

    def col(arm, dom):
        return np.array([rows[a][arm][dom] for a in pool], float)

    print("\n%-24s %9s %9s" % ("arm", "wall", "off"))
    for arm in ("control", "bagged"):
        w, o = col(arm, "wall"), col(arm, "off")
        print("%-24s %9.4f %9.4f" % (arm, np.nanmean(w), np.nanmean(o)))

    # ---- paired vessel bootstrap, the project's standard test ------------------------
    print("\npaired vessel bootstrap (%d draws), bagged - control:" % args.pair_boot)
    rb = np.random.default_rng(args.seed + 1)
    for dom in ("wall", "off"):
        d = col("bagged", dom) - col("control", dom)
        keep = ~np.isnan(d)
        d = d[keep]
        n = len(d)
        bs = np.array([d[rb.integers(0, n, n)].mean() for _ in range(args.pair_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        floor = 0.024 if dom == "wall" else 0.091
        print("  %-5s %+.4f  [%+.4f, %+.4f]   P(diff<=0) = %.3f   n=%d   noise floor %.3f"
              % (dom, d.mean(), lo, hi, float((bs <= 0).mean()), n, floor))

    print("\nper vessel (wall):")
    print("  %-12s %8s %8s %8s" % ("vessel", "control", "bagged", "delta"))
    for a in sorted(pool, key=lambda a: rows[a]["control"]["wall"]):
        c, b = rows[a]["control"]["wall"], rows[a]["bagged"]["wall"]
        print("  %-12s %8.4f %8.4f %+8.4f" % (a, c, b, b - c))

    out = REPO / args.save
    out.write_text(json.dumps(dict(rows=rows, chosen=chosen, tags=args.tags,
                                   boot=args.boot), default=float))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
