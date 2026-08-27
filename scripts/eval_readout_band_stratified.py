"""Choose the wall readout on the vessels where the cut ACTUALLY BINDS.

`scripts/diag_score_field_shape.py` found the root cause of the SEALED wall shortfall: on the
19-vessel pool only 2.7% of wall nodes lie in the band the cut passes through (0.30 < p <
0.90), so an absolute threshold slices near-empty space and its position is close to free --
oracle-minus-shipped is +0.006 in-sample.  A vessel whose band is occupied (p001: 13.5%) puts
the cut somewhere the calibration set never tested, and there the position costs +0.239.

That means the pool comparison which selected `resid_adapt` over `expected_tuned` for the
wall was made almost entirely on vessels where the two CANNOT differ.  Averaging over 19
vessels of which 14 are indifferent buries whatever signal the other 5 carry.

This re-runs that comparison under the identical strict protocol, then reports it stratified
by **band occupancy** -- a per-vessel, LABEL-FREE covariate, identified by root-cause analysis
before this test rather than fitted here, and computed from the same out-of-fold scores used
to predict.  The stratum is small (n=5 at the pool's top) and the cohort noise floor is
+-0.024 wall, so this is powered to detect only a large effect; that limitation is the
finding if the effect is small.

The two arms differ in exactly the way the root cause implicates:

    resid_adapt     an ABSOLUTE cut (perturbed per vessel) -- undefined behaviour when the
                    band is occupied, because nothing pinned where in the band it should sit
    expected_tuned  rank the domain and commit the prefix that maximises the EXPECTED
                    severity, using the model's own p in place of GT -- a per-vessel BUDGET,
                    so band occupancy changes the answer instead of being ignored

    python scripts/eval_readout_band_stratified.py --tags v5a,v5b,v5c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_expected_score_readout import GAMMA, KSCALE, expected_curve  # noqa: E402
from eval_strict import (  # noqa: E402
    GRID, attach_physics, apply_adapt, load_cache, load_scores, tune_adapt, tune_resid,
)
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402
from src.clot_ml.softmetric import dilation_operator, to_torch_sparse  # noqa: E402

BAND = (0.30, 0.90)


def wall_of(S):
    return S["wall"]


def band_occupancy(sc, dom):
    return float(((sc[dom] > BAND[0]) & (sc[dom] < BAND[1])).mean())


def expected_mask(sc, dom, Dt, dev, gamma, kscale):
    ks, vals = expected_curve(sc, dom, Dt, dev, gamma)
    if len(ks) < 2:
        return np.zeros(len(sc), bool)
    k = int(np.clip(round(ks[int(np.argmax(vals))] * kscale), 1, ks[-1]))
    order = np.flatnonzero(dom)[np.argsort(-sc[dom])]
    m = np.zeros(len(sc), bool)
    m[order[:k]] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v5a,v5b,v5c")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--metric", default="severity", choices=["severity", "legacy"])
    ap.add_argument("--top", type=int, default=5, help="vessels in the band-occupied stratum")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--save", default="outputs/eval_readout_band_stratified.json")
    args = ap.parse_args()

    cfg = DEFAULT if args.metric == "severity" else LEGACY
    dev = torch.device("cpu")
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), cfg) for a in pool}
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc_all[(fold_of[a], a)] for a in pool}
    Dt = {a: to_torch_sparse(dilation_operator(cache[a]["edge_index"],
                                               len(cache[a]["wall"]), 2), dev) for a in pool}

    band = {a: band_occupancy(oof[a], cache[a]["wall"].astype(bool)) for a in pool}
    ranked = sorted(pool, key=lambda a: -band[a])
    hi = set(ranked[:args.top])
    print("band occupancy (out-of-fold, wall):")
    for a in ranked:
        print("   %-12s %5.2f%%%s" % (a, 100 * band[a], "   <- cut binds" if a in hi else ""))

    rows = {a: dict(band=band[a], hi=a in hi) for a in pool}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        sel_sc = {a: oof[a] for a in sel}
        th = tune_resid(cache, vs, sel, sel_sc, GRID)
        b, med = tune_adapt(cache, vs, sel, sel_sc, "resid", th, wall_of)
        # expected_tuned: its two scalars chosen in-fold on the same selection vessels
        best = None
        for g in GAMMA:
            for ks_ in KSCALE:
                v = [vs[a].score(expected_mask(sel_sc[a], cache[a]["wall"].astype(bool),
                                               Dt[a], dev, g, ks_),
                                 cache[a]["wall"].astype(bool)) for a in sel]
                v = [x for x in v if x == x]
                q = float(np.mean(v)) if v else -1e9
                if best is None or q > best[0]:
                    best = (q, g, ks_)
        _, g_, ks_ = best
        for a in held:
            S, w = cache[a], cache[a]["wall"].astype(bool)
            m_ra = apply_adapt(S, oof[a], "resid", th, wall_of, b, med) & w
            m_ex = expected_mask(oof[a], w, Dt[a], dev, g_, ks_)
            rows[a]["resid_adapt"] = vs[a].score(m_ra, w)
            rows[a]["expected_tuned"] = vs[a].score(m_ex, w)
            rows[a]["oracle"] = max(vs[a].score(w & (oof[a] >= t), w) for t in GRID)
        print("  fold %d  th=%s b=%+.2f | expected gamma=%.2f kscale=%.2f"
              % (k, ",".join("%.2f" % x for x in th), b, g_, ks_), flush=True)

    def agg(keep, arm):
        v = [rows[a][arm] for a in pool if keep(a) and rows[a][arm] == rows[a][arm]]
        return float(np.mean(v)) if v else float("nan")

    print("\n%-26s %8s %8s %9s %8s" % ("stratum", "resid_ad", "expected", "delta", "oracle"))
    for nm, keep in (("all 19", lambda a: True),
                     ("band EMPTY (cut free)", lambda a: not rows[a]["hi"]),
                     ("band OCCUPIED (binds)", lambda a: rows[a]["hi"])):
        n = sum(1 for a in pool if keep(a))
        ra, ex = agg(keep, "resid_adapt"), agg(keep, "expected_tuned")
        print("%-26s %8.4f %8.4f %+9.4f %8.4f   (n=%d)"
              % (nm, ra, ex, ex - ra, agg(keep, "oracle"), n))

    rb = np.random.default_rng(0)
    print("\npaired vessel bootstrap, expected_tuned - resid_adapt:")
    for nm, keep in (("all 19", lambda a: True), ("band OCCUPIED", lambda a: rows[a]["hi"])):
        d = np.array([rows[a]["expected_tuned"] - rows[a]["resid_adapt"]
                      for a in pool if keep(a)])
        d = d[~np.isnan(d)]
        bs = np.array([d[rb.integers(0, len(d), len(d))].mean() for _ in range(args.boot)])
        lo, hi_ = np.percentile(bs, [2.5, 97.5])
        print("  %-14s %+.4f  [%+.4f, %+.4f]   P(diff<=0) = %.3f   n=%d   floor 0.024"
              % (nm, d.mean(), lo, hi_, float((bs <= 0).mean()), len(d)))

    print("\nper vessel (sorted by band occupancy):")
    print("  %-12s %7s %10s %10s %9s" % ("vessel", "band%", "resid_ad", "expected", "delta"))
    for a in ranked:
        r = rows[a]
        print("  %-12s %6.2f%% %10.4f %10.4f %+9.4f"
              % (a, 100 * r["band"], r["resid_adapt"], r["expected_tuned"],
                 r["expected_tuned"] - r["resid_adapt"]))

    (REPO / args.save).write_text(json.dumps(rows, default=float))
    print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
