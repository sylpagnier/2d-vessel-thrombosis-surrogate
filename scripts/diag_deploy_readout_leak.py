"""Does the strict-CV number actually estimate the SHIPPED artifact?  No -- and this is why.

`scripts/eval_strict.py` fixed PHASE10 1's leak 2 by selecting every readout scalar on the
OUT-OF-FOLD scores of the selection vessels.  That is the right way to *measure* a design.
But the artifact that ships cannot do it: `scripts/promote_clot_gnn_v4_temporal.py` fits the
readout on the full 19-vessel pool using the shipped ensemble's own IN-SAMPLE scores,
because after training on all 19 there is no held-out vessel left to calibrate against.

So the shipped readout is tuned on one score distribution (in-sample: sharp, confident) and
applied to another (a new vessel: flatter).  The strict CV never measures that mismatch --
it tunes and applies on the same, out-of-fold, distribution.

This script runs both procedures on the SAME folds, the SAME weights and the SAME cached
scores, so the only difference is which distribution the readout scalars were fitted on:

    strict  -- select on oof scores of the selection vessels   (what 0.9176 / 0.7366 measures)
    deploy  -- select on THIS fold model's IN-SAMPLE scores of  (what actually ships)
               the selection vessels

Both arms then predict the held-out vessel with the identical out-of-fold score array, so
any difference is attributable to readout calibration alone.

    python scripts/diag_deploy_readout_leak.py --tags v5a,v5b,v5c --adapt
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
    FAMILIES, GRID, apply_adapt, attach_physics, load_cache, load_scores, tune_adapt,
)
from src.clot_ml.geometry_splits import classes_for  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
WALL_OF, OFF_OF = (lambda S: S["wall"]), (lambda S: ~S["wall"])


def fit_spec(cache, vs, sel, sel_sc, adapt):
    """One family for both domains on the whole selection set, then optional per-domain
    slope -- `eval_strict.py`'s default `pooled` path, factored out so both arms share it."""
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
    _, fam, th = best
    slopes = {}
    if adapt:
        for dk, dom_of in (("wall", WALL_OF), ("off", OFF_OF)):
            slopes[dk] = tune_adapt(cache, vs, sel, sel_sc, fam, th, dom_of)
    return fam, th, slopes


def predict(S, sc_a, fam, th, slopes, adapt):
    if adapt:
        pw = apply_adapt(S, sc_a, fam, th, WALL_OF, *slopes["wall"])
        po = apply_adapt(S, sc_a, fam, th, OFF_OF, *slopes["off"])
    else:
        pw = po = FAMILIES[fam][1](S, sc_a, th)
    return (S["wall"] & pw) | (~S["wall"] & po)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--save", default="outputs/diag_deploy_readout_leak.json")
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

    rows: dict[str, dict] = {a: dict(cls=classes.get(a, "?")) for a in pool}
    specs: dict[int, dict] = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        arms = {
            # every selection vessel carries ITS OWN out-of-fold score
            "strict": {a: oof[a] for a in sel},
            # ... against THIS fold's model scoring its own training vessels, in-sample
            "deploy": {a: sc[(k, a)] for a in sel},
        }
        specs[k] = {}
        for arm, sel_sc in arms.items():
            fam, th, slopes = fit_spec(cache, vs, sel, sel_sc, args.adapt)
            specs[k][arm] = dict(family=fam, th=[float(x) for x in th],
                                 slopes={d: [float(x) for x in s] for d, s in slopes.items()})
            for a in held:
                S, w = cache[a], cache[a]["wall"]
                # IDENTICAL score array in both arms -- only the readout scalars differ
                pr = predict(S, oof[a], fam, th, slopes, args.adapt)
                rows[a][arm] = dict(wall=vs[a].score(pr & w, w),
                                    off=vs[a].score(pr & ~w, ~w), n_pred=int(pr.sum()),
                                    n_gt_wall=int(((cache[a]["y"] > 0.5) & w).sum()))
        print("  fold %d  strict th=%s  deploy th=%s" % (
            k, ",".join("%.2f" % x for x in specs[k]["strict"]["th"]),
            ",".join("%.2f" % x for x in specs[k]["deploy"]["th"])), flush=True)

    def agg(arm, dom, subset=None):
        v = [rows[a][arm][dom] for a in pool
             if (subset is None or classes.get(a) == subset) and rows[a][arm][dom] == rows[a][arm][dom]]
        return float(np.mean(v)) if v else float("nan"), len(v)

    print("\n%-28s %9s %9s" % ("readout fitted on...", "wall", "off"))
    for arm, label in (("strict", "OUT-OF-FOLD scores (reported)"),
                       ("deploy", "IN-SAMPLE scores (shipped)")):
        w, nw = agg(arm, "wall")
        o, no = agg(arm, "off")
        print("%-28s %9.4f %9.4f   (n=%d/%d)" % (label, w, o, nw, no))
    dw = agg("deploy", "wall")[0] - agg("strict", "wall")[0]
    do = agg("deploy", "off")[0] - agg("strict", "off")[0]
    print("%-28s %+9.4f %+9.4f" % ("cost of the deploy-time leak", dw, do))

    print("\nper vessel (wall):")
    print("  %-12s %6s %8s %8s %8s" % ("vessel", "cls", "strict", "deploy", "delta"))
    for a in sorted(pool, key=lambda x: rows[x]["deploy"]["wall"]):
        s_, d_ = rows[a]["strict"]["wall"], rows[a]["deploy"]["wall"]
        print("  %-12s %6s %8.4f %8.4f %+8.4f" % (a, rows[a]["cls"][:6], s_, d_, d_ - s_))

    out = REPO / args.save
    out.write_text(json.dumps(dict(rows=rows, specs=specs, tags=args.tags,
                                   adapt=bool(args.adapt)), default=float))
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
