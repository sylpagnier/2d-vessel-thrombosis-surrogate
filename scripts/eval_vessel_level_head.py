"""Is the per-vessel decision LEARNABLE from vessel-level features?  The architecture test.

Every readout in this project is a NODE-level model plus a COHORT-level constant.  That is
the structural reason a new vessel can fail: the network conditions on local features, and
the one quantity that has to adapt per vessel -- where to cut -- is a number fitted on the
training cohort and carried over unchanged.  `docs/PHASE10_V4.md` 4 measured the size of the
prize (per-vessel ORACLE cut 0.9447 against the cohort cut's 0.9024) and then tried five
HAND-WRITTEN rules for recovering it (quantile, rel_max, phys-anchored, gap, nested pick).
All five lost.  None of them LEARNED the cut; each was a fixed formula of one statistic.

This asks the question that was skipped: regress the oracle cut on VESSEL-LEVEL features --
a set-level head over the whole graph -- and evaluate leave-one-vessel-out.  Nothing here is
per-node; the node model is left exactly as it is and only the decision becomes a function of
the vessel instead of a constant.

Every predictor is LABEL-FREE at apply time (score distribution shape, physics-mask size,
shear/geometry summaries, ensemble disagreement).  The TARGET is the oracle cut, which uses
labels -- legitimately, since it is only ever read on training vessels.

Ceilings and controls reported together, because the honest question is not "does it beat the
cohort cut" but "how much of the oracle gap is recoverable at all":

    cohort      one constant, tuned out-of-fold        (the shipped behaviour)
    learned     the set-level head, leave-one-vessel-out
    oracle      per-vessel best cut                    (the ceiling, uses labels)

    python scripts/eval_vessel_level_head.py --tags v5a,v5b,v5c
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

from eval_strict import GRID, attach_physics, load_cache  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402

SCORES = REPO / "outputs/phase9_scores"


def vessel_features(S, sc, spread):
    """Label-free summaries of one vessel.  All computable at deploy time."""
    w = S["wall"].astype(bool)
    s = sc[w]
    ph = S["phys_mask"].astype(bool) & w
    q = np.percentile(s, [10, 25, 50, 75, 90, 95, 99])
    return np.array([
        s.mean(), s.std(), *q,
        float(ph.sum()) / max(w.sum(), 1),          # physics burden fraction
        float((s > 0.30).mean()), float((s > 0.50).mean()), float((s > 0.90).mean()),
        float(((s > 0.30) & (s < 0.90)).mean()),    # band occupancy
        float(np.median(S["sr"][w])), float(np.percentile(S["sr"][w], 10)),
        float((S["gate"][w] > 0).mean()),
        float(np.median(S["spd"][~w])), float(w.sum()),
        float(spread[w].mean()) if spread is not None else 0.0,
    ], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--save", default="outputs/eval_vessel_level_head.json")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestRegressor

    cfg = DEFAULT if args.metric == "severity" else LEGACY
    tags = args.tags.split(",")
    zs = {t: np.load(SCORES / f"{t}.npz", allow_pickle=True) for t in tags}
    z0 = zs[tags[0]]
    cache = attach_physics(load_cache(args.cache))
    pool = [str(x) for x in z0["pool"] if str(x) in cache]
    folds = {int(k.split("|")[1]): [str(x) for x in z0[k]]
             for k in z0.files if k.startswith("held|")}
    fold_of = {a: k for k, h in folds.items() for a in h}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), cfg) for a in pool}

    oof, spread = {}, {}
    for a in pool:
        per = [zs[t]["%d|%s" % (fold_of[a], a)] for t in tags]
        oof[a] = np.mean(per, axis=0)
        spread[a] = np.std(per, axis=0)

    # per-vessel oracle cut and the score at every candidate cut
    curve, oracle_cut, oracle_val = {}, {}, {}
    for a in pool:
        w = cache[a]["wall"].astype(bool)
        v = np.array([vs[a].score(w & (oof[a] >= t), w) for t in GRID])
        curve[a] = v
        i = int(np.argmax(v))
        oracle_cut[a], oracle_val[a] = float(GRID[i]), float(v[i])

    X = {a: vessel_features(cache[a], oof[a], spread[a]) for a in pool}

    rows = {}
    for a in pool:                                   # leave ONE VESSEL out
        tr = [b for b in pool if b != a]
        Xtr = np.stack([X[b] for b in tr])
        ytr = np.array([oracle_cut[b] for b in tr])
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0)
        rf.fit(Xtr, ytr)
        t_hat = float(rf.predict(X[a][None, :])[0])
        j = int(np.argmin(np.abs(GRID - t_hat)))
        # cohort control: the single best constant on the SAME training vessels
        best = max(range(len(GRID)), key=lambda k: float(np.mean([curve[b][k] for b in tr])))
        rows[a] = dict(learned_cut=t_hat, learned=float(curve[a][j]),
                       cohort_cut=float(GRID[best]), cohort=float(curve[a][best]),
                       oracle_cut=oracle_cut[a], oracle=oracle_val[a])
        print("  %-12s learned cut %.2f -> %.4f   cohort %.2f -> %.4f   oracle %.2f -> %.4f"
              % (a, t_hat, rows[a]["learned"], rows[a]["cohort_cut"], rows[a]["cohort"],
                 oracle_cut[a], oracle_val[a]), flush=True)

    L = np.array([rows[a]["learned"] for a in pool])
    C = np.array([rows[a]["cohort"] for a in pool])
    O = np.array([rows[a]["oracle"] for a in pool])
    print("\n%-28s %8s %8s" % ("", "mean", "sd"))
    for nm, v in (("cohort constant (shipped)", C), ("LEARNED vessel-level head", L),
                  ("per-vessel oracle (ceiling)", O)):
        print("%-28s %8.4f %8.4f" % (nm, v.mean(), v.std(ddof=1)))
    gap = O.mean() - C.mean()
    got = L.mean() - C.mean()
    print("\noracle gap available: %+.4f   recovered by the learned head: %+.4f  (%.0f%%)"
          % (gap, got, 100 * got / gap if gap else 0.0))
    rb = np.random.default_rng(0)
    d = L - C
    bs = np.array([d[rb.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("paired bootstrap learned - cohort: %+.4f [%+.4f, %+.4f]  P(diff<=0)=%.3f  floor 0.024"
          % (d.mean(), lo, hi, float((bs <= 0).mean())))
    cc = np.corrcoef([rows[a]["learned_cut"] for a in pool],
                     [rows[a]["oracle_cut"] for a in pool])[0, 1]
    print("corr(predicted cut, oracle cut) = %+.3f   <- is the decision learnable at all?" % cc)

    (REPO / args.save).write_text(json.dumps(rows, default=float))
    print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
