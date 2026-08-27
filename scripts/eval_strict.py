"""Strictly-nested evaluation of a clot score, at the FINAL time point.

WHY THIS EXISTS.  Every number in `docs/PHASE9_ML.md` is out-of-fold in the *model*, but
not in the *readout*.  Two leaks survive:

  1. `scripts/train_time_conditioned.py` commits its set with **hard-coded** cuts
     `score >= 0.73` (wall) and `>= 0.92` (off-wall).  Those two constants were chosen by
     looking at the whole 19-vessel pool, so every vessel's reported score is read out with
     a rule that saw that vessel's answer.
  2. Where thresholds *are* tuned, they are tuned on the fold's own training vessels using
     **that fold's model**, which has those vessels in its training set.  In-sample scores
     are overconfident, so the selected cut is biased, and PHASE9 3 already recorded what
     in-sample selection did to this cohort once (FIT wall 0.90 in-sample against DEV 0.83).

The fix needs no retraining.  `run_phase9_cv.py` saves, for every fold, that fold's model's
score on *every* vessel, so an out-of-fold score exists for all 19.  To evaluate held-out
fold `k`, thresholds are selected on the **out-of-fold** scores of the vessels not in `k`.
Those scores come from other folds' models, and none of them ever saw a vessel of `k`.  So
no quantity used to produce a vessel's number was fitted with that vessel visible -- neither
the weights nor the readout.

Reported at the LAST time point, which is `cache["y"]` (`resolve_deploy_eval_time_index` is
the per-graph last index), per domain, under the severity metric.

    python scripts/eval_strict.py --tags cv5a,cv5b,cv5c --cache gt
    python scripts/eval_strict.py --tags v4a --cache v4
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

from src.clot_ml.data import (  # noqa: E402
    attach_physics, load_cache, wall_domain, off_domain,
)
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GRID = np.round(np.linspace(0.02, 0.98, 33), 4)


class BoundScorer:
    """`SeverityScorer` with one vessel's empty-GT convention baked in.

    The convention is a property of the VESSEL, not of the call site, and the two cases are
    genuinely different:

    * a clot-CARRYING vessel whose off-wall domain happens to hold no GT -- 6 of 19 -- must
      stay ``nan`` and drop out of the mean, or every off-wall number in the project silently
      changes meaning;
    * a CLOT-FREE vessel has no GT anywhere by design, and its evidence is entirely about
      false positives, so it scores ``1/(1 + n_pred/8)``.

    Binding it here means the tuners, the family chooser and the reporting loop all get the
    right one without threading a mode argument through five signatures -- and cannot get it
    wrong for some call sites and right for others.

    REPORTING and SELECTION can use different empty-GT conventions, and after 2026-08-22
    they do.  Reporting must stay ``nan`` on an empty domain or every off-wall number in the
    project changes meaning.  **Selection has no such constraint**, and throwing the cell away
    there is a measured mistake: six clot-carrying vessels have no off-wall GT, so a cut that
    sprays 414 false positives on `patient014` costs the tuner exactly nothing.  See
    ``empty_gt_select``.
    """

    def __init__(self, ei, gt, n, cfg, empty_gt: str, empty_gt_select: str | None = None):
        self._s = SeverityScorer(ei, gt, n, cfg)
        self.empty_gt = empty_gt
        #: convention used while TUNING.  Defaults to the reporting one, which reproduces
        #: every pre-2026-08-22 selection exactly.
        self.empty_gt_select = empty_gt_select or empty_gt

    def score(self, pred, domain=None) -> float:
        return self._s.score(pred, domain, empty_gt=self.empty_gt)

    def sel(self, pred, domain=None) -> float:
        """The score a THRESHOLD TUNER should see.  Never used for a reported number."""
        return self._s.score(pred, domain, empty_gt=self.empty_gt_select)

    def components(self, pred, domain=None) -> dict:
        return self._s.components(pred, domain)


# ---------------------------------------------------------------------------
# readouts.  Both families are offered and the choice is made INSIDE the fold.
# ---------------------------------------------------------------------------
def readout_plain(S, sc, th):
    """One cut per domain."""
    tw, to = th
    w = S["wall"]
    return (w & (sc >= tw)) | (~w & (sc >= to))


def readout_resid(S, sc, th):
    """Separate cuts for keeping a physics-positive node and adding a physics-negative one.

    Wall error is two opposite failure modes (PHASE7 10.3: weak-separation false positives
    on 018/019/025 against ungated false negatives on 012/028) and one cut cannot serve
    both.  This is the readout `train_clot_gnn.py` already uses; it is offered here so the
    comparison between feature sets is not confounded by the readout family.
    """
    kw, aw, ko, ao = th
    w, ph = S["wall"], S["phys_mask"]
    return ((w & ph & (sc >= kw)) | (w & ~ph & (sc >= aw))
            | (~w & ph & (sc >= ko)) | (~w & ~ph & (sc >= ao)))


def tune_plain(cache, vs, anchors, sc, grid):
    out = []
    for dom_of in (wall_domain, off_domain):
        best, bt = -1e9, float(grid[0])
        for t in grid:
            vals = [vs[a].sel(dom_of(cache[a]) & (sc[a] >= t), dom_of(cache[a]))
                    for a in anchors]
            vals = [v for v in vals if v == v]
            if vals and np.mean(vals) > best:
                best, bt = float(np.mean(vals)), float(t)
        out.append(bt)
    return tuple(out)


def tune_resid(cache, vs, anchors, sc, grid):
    out = []
    for dom_of in (wall_domain, off_domain):
        best, pair = -1e9, (float(grid[0]), float(grid[0]))
        for tk in grid:
            for ta in grid:
                vals = []
                for a in anchors:
                    S = cache[a]
                    d, ph = dom_of(S), S["phys_mask"]
                    pr = (d & ph & (sc[a] >= tk)) | (d & ~ph & (sc[a] >= ta))
                    v = vs[a].sel(pr, d)
                    if v == v:
                        vals.append(v)
                if vals and np.mean(vals) > best:
                    best, pair = float(np.mean(vals)), (float(tk), float(ta))
        out.extend(pair)
    return tuple(out)


FAMILIES = {"plain": (tune_plain, readout_plain), "resid": (tune_resid, readout_resid)}


# ---------------------------------------------------------------------------
# `--select-empty-gt score`: MEASURED AND REJECTED, 2026-08-22.  Do not re-derive.
# ---------------------------------------------------------------------------
# The motivation was sound and the diagnosis behind it is correct: six clot-carrying vessels
# have NO off-wall GT, they score `nan`, and they therefore cannot penalise a cut -- while
# actually receiving 414 / 267 / 235 / 84 / 39 / 33 false positives under the chosen one.  A
# cut that sprays there is free to the tuner.  Grading those cells with the metric's own
# false-positive branch during SELECTION ONLY (reporting stays `nan`) costs no parameter.
#
# It makes off-wall WORSE:
#
#     select empty-GT as nan   (default)   wall 0.9008   off 0.5812
#     select empty-GT as score             wall 0.9008   off 0.5382
#
# The reason is scale, not principle.  `1/(1 + n_pred/8)` is ~1.0 whenever little is
# committed, while a recall-bearing off-wall cell sits near 0.6 -- so mixing them in one mean
# makes "commit nothing" the cheapest way to raise the average.  The tuner duly moves the
# add-threshold to 0.89 and under-commits everywhere: `patient029` 0.2906 -> 0.0000,
# `patient012` 0.8140 -> 0.3880, `patient044` 0.8610 -> 0.5850.
#
# This is the same scale-mixing that keeps the clot-free vessels out of the reported mean
# (MODEL_REVIEW 8c) -- the error here was letting it into the OBJECTIVE instead.  A weighted
# version would work only by fitting a weight, which is the selection this cohort cannot
# support.  The real lever remains per-vessel cut placement (MODEL_REVIEW 8f, Phase C5).


# ---------------------------------------------------------------------------
# per-vessel adaptivity: PERTURB the cohort cut, do not replace it
# ---------------------------------------------------------------------------
#: label-free vessel statistic the cut is allowed to lean on.  `mean` (the mean score in the
#: domain) is used rather than a tail quantile because it is the most robust of the four
#: measured, and because `scripts/eval_fusion_calib.py` found q90/mean/physfrac all move the
#: score the same way while q99 does not -- a tail statistic is the fragile choice at n=19.
ADAPT_STAT = "mean"
B_GRID = np.round(np.linspace(-1.2, 1.2, 13), 3)


def vessel_stat(S, sc, dom, name=ADAPT_STAT):
    d = np.asarray(dom, dtype=bool)
    v = np.asarray(sc, dtype=np.float64)[d]
    if v.size == 0:
        return 0.0
    if name == "mean":
        return float(v.mean())
    if name == "q90":
        return float(np.quantile(v, 0.90))
    if name == "physfrac":
        return float((S["phys_mask"] & d).sum() / max(d.sum(), 1))
    raise ValueError(name)


def tune_adapt(cache, vs, anchors, sc, family, th, dom_of, return_support=False):
    """One slope `b` per domain, on top of an already-chosen family and thresholds.

    `b = 0` reproduces the cohort readout EXACTLY, so this can only move away from it if the
    statistic pays on the selection vessels.  That is the difference from
    `src/clot_ml/calibration.py`'s rules, which substitute the constant outright and all
    lose: measured, substitution reads 0.78-0.88 wall against 0.907, while perturbation
    reads 0.9097/0.7194 against 0.9016/0.7136.

    ``return_support`` additionally returns ``(lo, hi)``, the range of the vessel statistic
    the slope was actually fitted over.  Outside it the linear form has never been checked
    against a labelled vessel and `off` grows without bound, so `apply_adapt` can be asked to
    hold the statistic inside it -- see that function.  Default is off, so every existing
    caller keeps its exact two-value return.
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
            x = vs[a].sel(apply_(S, sc[a], tuple(np.clip(np.array(th) + off, 0.02, 0.98)))
                          & d, d)
            if x == x:
                vals.append(x)
        q = float(np.mean(vals)) if vals else -1e9
        if best is None or q > best[0]:
            best = (q, float(b))
    if not return_support:
        return best[1], med
    lo, hi = (float(min(sv.values())), float(max(sv.values()))) if sv else (med, med)
    return best[1], med, lo, hi


def apply_adapt(S, sc, family, th, dom_of, b, med, lo=None, hi=None):
    """Perturb the cohort cut by the fitted slope on this vessel's own statistic.

    ``lo``/``hi`` bound the statistic to the support `tune_adapt` fitted over.  Inside that
    support clamping is an EXACT no-op -- verified bit-identical on all 19 pool vessels by
    `scripts/eval_adapt_clamp.py`, under both the geometry-stratified partition and one that
    deliberately holds the extremes out -- so it cannot flatter anything the cohort measures.
    Outside it, the cut is held at the most extreme perturbation the fit actually saw rather
    than continuing a line no labelled vessel supports.  Both default to ``None``, which
    reproduces the unbounded behaviour exactly.
    """
    stat = vessel_stat(S, sc, dom_of(S))
    if lo is not None and hi is not None:
        stat = float(min(max(stat, float(lo)), float(hi)))
    off = b * (stat - med)
    return FAMILIES[family][1](S, sc, tuple(np.clip(np.array(th) + off, 0.02, 0.98)))

#: free scalars per domain, per family -- `resid` has twice `plain`'s
N_PARAMS = {"plain": 1, "resid": 2}


def pick_family(cache, vs, anchors, sc, grid, dom_of, inner=3, seed=0):
    """Choose the readout family by INNER cross-validation.  **MEASURED WORSE -- not default.**

    The motivation was sound: comparing families on the set their parameters were fitted on
    is biased toward the bigger family (`resid` has two scalars per domain against
    `plain`'s one), and `resid` does win every selection and then lose on held-out wall.

    Removing that bias costs more than the bias did.  Strictly nested on `cv5a,cv5b,cv5c`:

        family on the whole selection set   wall 0.9024   off 0.7075
        family by inner 3-fold (this)       wall 0.8920   off 0.6688

    and the same inversion on `v4a,v4b` (0.9092/0.6889 -> 0.8959/0.6764).  The inner split
    leaves 11-12 vessels to choose on, and at that size the selection noise exceeds the
    bias being removed -- consistent with `scripts/eval_significance.py`, where the
    seed/config floor alone is 0.024 wall / 0.091 off-wall.

    Kept, documented, and OFF by default (``--family-select inner``), so nobody re-derives
    it as an obvious improvement.
    """
    rng = np.random.default_rng(seed)
    order = list(anchors)
    rng.shuffle(order)
    parts = [order[i::inner] for i in range(inner)]
    best = None
    for fam, (tune, apply_) in FAMILIES.items():
        vals = []
        for iv in parts:
            itr = [a for a in order if a not in iv]
            if not itr or not iv:
                continue
            th = tune(cache, vs, itr, sc, grid)
            for a in iv:
                S = cache[a]
                d = dom_of(S)
                x = vs[a].sel(apply_(S, sc[a], th) & d, d)
                if x == x:
                    vals.append(x)
        q = float(np.mean(vals)) if vals else -1e9
        key = (q, -N_PARAMS[fam])
        if best is None or key > best[0]:
            best = (key, fam)
    fam = best[1]
    return fam, FAMILIES[fam][0](cache, vs, anchors, sc, grid)


# ---------------------------------------------------------------------------
def load_scores(tags: list[str]) -> tuple[list[str], dict, dict]:
    """-> pool, held-out map ``fold -> anchors``, ``{(fold, anchor): score}`` seed-averaged."""
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True) for t in tags]
    pool = [str(x) for x in zs[0]["pool"]]
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    sc = {}
    for k in folds:
        for a in pool:
            key = "%d|%s" % (k, a)
            sc[(k, a)] = np.mean([z[key] for z in zs], axis=0)
    return pool, folds, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--family", default="auto", choices=["auto", "plain", "resid"])
    ap.add_argument("--adapt", action="store_true",
                    help="perturb the cohort cut by a fitted slope on a vessel statistic")
    ap.add_argument("--family-select", default="pooled", choices=["pooled", "inner"],
                    help="'pooled' picks one family on the whole selection set (default, "
                         "measured best); 'inner' picks per domain by inner CV (worse)")
    ap.add_argument("--clot-free", default="score", choices=["score", "exclude"],
                    help="'score' (default) puts the 8 empty-GT vessels into readout "
                         "selection under the false-positive branch and reports them on "
                         "their own row; 'exclude' drops them entirely, which reproduces "
                         "every number published before 2026-08-22")
    ap.add_argument("--select-empty-gt", default="nan", choices=["nan", "score"],
                    help="what an EMPTY DOMAIN is worth while TUNING the cut. 'nan' "
                         "(default) throws the cell away; 'score' grades it with the "
                         "metric's false-positive branch. MEASURED AND REJECTED "
                         "2026-08-22: off-wall 0.5812 -> 0.5382. Kept so it is not "
                         "re-derived -- see the note above `FAMILIES`.")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cfg = DEFAULT if args.metric == "severity" else LEGACY
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    if args.clot_free == "exclude":
        pool = [a for a in pool if a not in CLOT_FREE]
        folds = {k: [a for a in held if a in pool] for k, held in folds.items()}
    free = [a for a in pool if a in CLOT_FREE]
    carrying = [a for a in pool if a not in CLOT_FREE]
    classes = classes_for(pool, PACKS)
    vs = {a: BoundScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                         len(cache[a]["wall"]), cfg,
                         "score" if a in CLOT_FREE else "nan",
                         empty_gt_select=("score" if a in CLOT_FREE
                                          else args.select_empty_gt)) for a in pool}
    fold_of = {a: k for k, held in folds.items() for a in held}

    # the honest score for every vessel: the model of the fold that held it out
    oof = {a: sc[(fold_of[a], a)] for a in pool}

    rows, chosen = {}, {}
    for k, held in sorted(folds.items()):
        # selection set: OTHER vessels, each carrying ITS OWN out-of-fold score, so nothing
        # used to pick the readout was produced by a model that had seen fold k
        sel = [a for a in pool if a not in held]
        sel_sc = {a: oof[a] for a in sel}
        spec = {}
        if args.family != "auto":
            th = FAMILIES[args.family][0](cache, vs, sel, sel_sc, GRID)
            spec = {d: (args.family, th) for d in ("wall", "off")}
        elif args.family_select == "inner":
            for dk, dom_of in (("wall", wall_domain),
                               ("off", off_domain)):
                spec[dk] = pick_family(cache, vs, sel, sel_sc, GRID, dom_of)
        else:
            # DEFAULT: one family for both domains, chosen on the whole selection set.
            # Both refinements of this -- per-domain choice and inner-CV choice -- were
            # measured and both LOSE (see pick_family's docstring).  At n=19 every extra
            # selection layer costs more variance than the bias it removes.
            best = None
            for fam, (tune, apply_) in FAMILIES.items():
                th = tune(cache, vs, sel, sel_sc, GRID)
                vals = []
                for a in sel:
                    S = cache[a]
                    pr = apply_(S, sel_sc[a], th)
                    for d in (wall_domain(S), off_domain(S)):
                        v = vs[a].sel(pr & d, d)
                        if v == v:
                            vals.append(v)
                q = float(np.mean(vals))
                if best is None or q > best[0]:
                    best = (q, fam, th)
            spec = {d: (best[1], best[2]) for d in ("wall", "off")}
        chosen[k] = spec
        slopes = {}
        if args.adapt:
            for dk, dom_of in (("wall", wall_domain),
                               ("off", off_domain)):
                slopes[dk] = tune_adapt(cache, vs, sel, sel_sc, spec[dk][0], spec[dk][1],
                                        dom_of)
        for a in held:
            S = cache[a]
            w, offd = wall_domain(S), off_domain(S)
            if args.adapt:
                pw = apply_adapt(S, oof[a], spec["wall"][0], spec["wall"][1],
                                 wall_domain, *slopes["wall"])
                po = apply_adapt(S, oof[a], spec["off"][0], spec["off"][1],
                                 off_domain, *slopes["off"])
            else:
                pw = FAMILIES[spec["wall"][0]][1](S, oof[a], spec["wall"][1])
                po = FAMILIES[spec["off"][0]][1](S, oof[a], spec["off"][1])
            pr = (w & pw) | (offd & po)
            rows[a] = dict(wall=vs[a].score(pr & w, w),
                           off=vs[a].score(pr & offd, offd),
                           full=vs[a].score(pr, None),
                           n_pred=int(pr.sum()),
                           n_pred_wall=int((pr & w).sum()),
                           n_pred_off=int((pr & offd).sum()),
                           free=bool(a in CLOT_FREE), cls=classes.get(a, "?"))
        print("  fold %d  %s  ->  %s" % (
            k, " ".join("%s:%s(%s)" % (d, spec[d][0],
                        ",".join("%.2f" % x for x in spec[d][1])) for d in ("wall", "off")),
            " ".join("%s w%.3f o%s" % (a[-3:], rows[a]["wall"],
                     ("%.3f" % rows[a]["off"]) if rows[a]["off"] == rows[a]["off"] else "-")
                     for a in held)), flush=True)

    # physics backbone under the same protocol (it has nothing to tune)
    phys = {a: dict(
        wall=vs[a].score(cache[a]["phys_mask"] & wall_domain(cache[a]), wall_domain(cache[a])),
        off=vs[a].score(cache[a]["phys_mask"] & off_domain(cache[a]), off_domain(cache[a])))
        for a in pool}

    def agg(sub, src):
        return (np.nanmean([src[a]["wall"] for a in sub]),
                np.nanmean([src[a]["off"] for a in sub]))

    # The recall-bearing means are over CLOT-CARRYING vessels only.  A clot-free vessel's
    # score is a false-positive grading on a different scale, and averaging the two produces
    # a number that means nothing -- `wall_cohort_splits` calls this out and it is the one
    # rule the 2026-08-22 cohort change depends on.  They get their own row instead.
    groups = [("ALL", carrying),
              ("baseline", [a for a in carrying if not is_priority(classes.get(a, ""))]),
              ("PRIORITY", [a for a in carrying if is_priority(classes.get(a, ""))])]
    print("\nFINAL TIME POINT, strictly nested (%s metric, tags=%s, cache=%s)\n"
          % (args.metric, args.tags, args.cache))
    print("%-10s %3s | %9s %9s | %9s %9s" % ("group", "n", "wall", "off", "ph wall", "ph off"))
    for name, sub in groups:
        if not sub:
            continue
        mw, mo = agg(sub, rows)
        pw, po = agg(sub, phys)
        print("%-10s %3d | %9.4f %9.4f | %9.4f %9.4f" % (name, len(sub), mw, mo, pw, po))

    if free:
        mw, mo = agg(free, rows)
        pw, po = agg(free, phys)
        print("%-10s %3d | %9.4f %9.4f | %9.4f %9.4f   <- FALSE POSITIVES ONLY, "
              "1/(1+n_pred/%.0f); never averaged with the rows above"
              % ("clot-free", len(free), mw, mo, pw, po, DEFAULT.empty_gt_fp_tol))
        print("%-10s %3s | %9d %9d | %9d %9d   <- nodes committed (0 is perfect)"
              % ("", "", sum(rows[a]["n_pred_wall"] for a in free),
                 sum(rows[a]["n_pred_off"] for a in free),
                 sum(int((cache[a]["phys_mask"] & wall_domain(cache[a])).sum()) for a in free),
                 sum(int((cache[a]["phys_mask"] & off_domain(cache[a])).sum())
                     for a in free)))

    print("\nper vessel")
    for a in sorted(rows):
        r = rows[a]
        print("  %-11s %-9s wall %.4f  off %6s  (phys %.4f / %s)%s"
              % (a, r["cls"][:9], r["wall"],
                 ("%.4f" % r["off"]) if r["off"] == r["off"] else "n/a", phys[a]["wall"],
                 ("%.4f" % phys[a]["off"]) if phys[a]["off"] == phys[a]["off"] else "n/a",
                 ("   CLOT-FREE  %d wall / %d off nodes committed"
                  % (r["n_pred_wall"], r["n_pred_off"])) if r["free"] else ""))

    if args.save:
        Path(args.save).write_text(json.dumps(
            dict(rows=rows, phys=phys, chosen={str(k): v for k, v in chosen.items()},
                 tags=args.tags, cache=args.cache, metric=args.metric),
            indent=2, default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
