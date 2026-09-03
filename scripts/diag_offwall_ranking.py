"""Compare off-wall RANKING quality across CV arms, separately from the threshold.

DEPLOYCLOT.md 20 and 23 closed the readout: no per-vessel cut rule beats the shipped one by
more than noise, and handing the model the exact true burden is worth +0.05.  What 24 then
found is that the bound belongs to the shipped FIELD, not to the task -- a gradient-boosted
tree on the same 69 features orders the boundary better.  So the quantity to track when
changing the model is the ordering, and the quantity that decides whether an ordering is
USABLE is whether one cohort cut lands in the right place on every vessel.

This prints both, per CV tag:

    P@n_gt          precision when you take exactly the true number of off-wall nodes.
                    Pure ranking: no threshold, no burden estimate.
    guid @ burden   what that same prefix scores -- the ceiling for the ordering.
    guid / resid    what the shipped four-cut readout actually gets, cuts fitted per fold
                    on the training vessels.  The gap to the line above is cuttability.
    GT p50 spread   the range, across vessels, of the median score of the vessel's OWN GT
                    off-wall nodes.  This is the cuttability diagnostic: the C0 GNN holds
                    it inside 0.95-0.99 and one constant works; the GBM spreads it over
                    0.11-0.93 and no constant can.

Everything is `guiding` (DEPLOYCLOT.md 0) and strictly out-of-fold.

    python scripts/diag_offwall_ranking.py --tags dc_fem_c0 dc_fem_cfw025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_strict import GRID, load_scores, readout_resid, tune_resid  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.severity_metric import LEGACY, SeverityScorer  # noqa: E402


def measure(tag: str, cache: dict) -> dict:
    pool, folds, sc = load_scores([tag])
    fo = {a: k for k, held in folds.items() for a in held}
    ves = [a for a in pool if a in cache
           and (np.asarray(cache[a]["y"]) > 0.5)[~np.asarray(cache[a]["wall"], bool)].any()]
    VS = {a: SeverityScorer(cache[a]["edge_index"], np.asarray(cache[a]["y"]) > 0.5,
                            len(cache[a]["wall"]), LEGACY) for a in ves}
    P, B, R, g50, per = [], [], [], [], {}
    for k, held in sorted(folds.items()):
        tr = [a for a in ves if a not in held]
        te = [a for a in held if a in ves]
        if not te:
            continue
        th = tune_resid(cache, VS, tr, {a: sc[(fo[a], a)] for a in tr}, GRID)
        for a in te:
            d = ~np.asarray(cache[a]["wall"], bool)
            y = (np.asarray(cache[a]["y"]) > 0.5)[d]
            n = int(y.sum())
            s = sc[(fo[a], a)]
            idx = np.flatnonzero(d)
            p = float(y[np.argsort(-s[d])[:n]].mean())
            mb = np.zeros(len(d), bool)
            mb[idx[np.argsort(-s[d])[:n]]] = True
            b = float(VS[a].score(mb, d))
            r = float(VS[a].score(readout_resid(cache[a], s, th) & d, d))
            P.append(p), B.append(b), R.append(r)
            g50.append(float(np.median(s[d][y])))
            per[a] = dict(p_at_ngt=p, guid_burden=b, guid_resid=r, gt_p50=g50[-1], n_gt=n)
    return dict(tag=tag, n=len(P), p_at_ngt=float(np.mean(P)),
                guid_burden=float(np.mean(B)), guid_resid=float(np.mean(R)),
                gt_p50_lo=float(min(g50)), gt_p50_hi=float(max(g50)), per_vessel=per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--cache", default="v5_fem")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    rows = []
    for t in args.tags:
        try:
            rows.append(measure(t, cache))
        except FileNotFoundError:
            print(f"[skip] {t}: no scores in outputs/phase9_scores", flush=True)

    print(f"\n{'CV tag':26s}{'n':>4s}{'P@nGT':>8s}{'guid@burden':>13s}{'guid/resid':>12s}"
          f"{'GT p50 spread':>16s}")
    for r in rows:
        print(f"{r['tag']:26s}{r['n']:4d}{r['p_at_ngt']:8.3f}{r['guid_burden']:13.4f}"
              f"{r['guid_resid']:12.4f}{r['gt_p50_lo']:9.2f}-{r['gt_p50_hi']:.2f}")
    if len(rows) > 1:
        b = rows[0]
        print()
        for r in rows[1:]:
            print(f"  {r['tag']} vs {b['tag']}:  P@nGT {r['p_at_ngt'] - b['p_at_ngt']:+.3f}   "
                  f"guid@burden {r['guid_burden'] - b['guid_burden']:+.4f}   "
                  f"guid/resid {r['guid_resid'] - b['guid_resid']:+.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2, default=float),
                                  encoding="utf-8")
        print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
