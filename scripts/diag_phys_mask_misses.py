"""What physics is MISSING?  Characterise the wall clot the backbone mask fails to reach.

`src/clot_ml/data.physics_mask` seeds on `gate > 0` and grows through wall nodes admissible
by `sr < 2*lss`, then into the lumen.  It is a CONNECTED growth: a clot-bearing region that
is admissible but separated from every seed by an inadmissible (high-shear) barrier can never
be reached, no matter how favourable its own local physics is.  In a stenosis that is exactly
the post-throat recirculation zone -- low shear, but fenced off from the upstream seeds by the
high-shear throat.

This measures whether that story is true on the 19 training vessels, before anything is
built.  For every wall node it separates:

    TP   GT clot, reached by the mask
    FN   GT clot, NOT reached          <- the physics the model is missing
    FP   reached but not GT
    TN   neither

and reports, for each group, the local hydrodynamics the mask ignores: shear, speed, the hop
distance to the nearest gate seed, pressure, vorticity, and wall-normal/tangential velocity.
If the FN group is low-shear and admissible but FAR from any seed, the deficiency is
topological (the growth cannot get there) rather than a bad admission criterion.

MEASURED (2026-08-21, 19-vessel pool).  **12.9% of all GT wall clot is never reached**, and
**91% of the misses are topologically unreachable** (hops >= 40) rather than rejected by the
shear criterion -- so the deficiency is the connected growth, as suspected.  But the missed
nodes are *high* shear (5.5x lss vs the TP group's 3.5x) and *low* pressure: they sit at the
stenosis THROAT, not in the post-throat recirculation zone.  Physically the clot there is
transported, not locally generated, so the low-shear admission rule is right to reject it
locally.  Concentrated in p028 (58% of its GT missed), p012 (48%), p044 (31%), p041 (27%) --
which are among the worst out-of-fold vessels.

TWO REPAIRS WERE TRIED ON THIS AND BOTH FAILED -- do not re-derive them:
  * union with the advective reach field (`log_src_reach`): recovering 40% of the misses
    drops mask precision 0.918 -> 0.464, and the severity metric is F0.5-weighted.
  * physics conjunctions (reach AND low pressure / high vorticity / high advected Mat /
    residence time, 18 variants): best delta F0.5 = **0.0000**.
The 227 missed nodes are not separable from the ~8500 true negatives by any hand-written
local-physics threshold.  The GNN itself does separate them (pool wall AUC 0.9973), so the
information is present -- it lives in the learned combination, not in a rule.

Pool only -- no SEALED vessel is read.

    python scripts/diag_phys_mask_misses.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.config import BiochemConfig  # noqa: E402

LOCKED = REPO / "outputs/clot_ml/locked/clot_gnn_v4"
FIELDS = ("log_sr", "speed_nd", "p_nd", "vort", "u_n", "u_t", "width_nd", "sdf_nd",
          "log_tau", "att_reach", "log_src_reach")


def hops_to_seed(ei, n, seed, wall, adm, max_hops=40):
    """Graph distance to the nearest gate seed, walking ONLY through admissible wall nodes.

    Infinite (returned as ``max_hops``) exactly when the growth in `physics_mask` cannot
    reach the node -- which is the quantity this script exists to measure.
    """
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    reach = seed & wall
    dist = np.full(n, max_hops, dtype=np.int32)
    dist[reach] = 0
    frontier = reach.copy()
    for h in range(1, max_hops):
        nxt = ((A @ frontier.astype(np.int8)) > 0) & adm & ~reach
        if not nxt.any():
            break
        dist[nxt] = h
        reach |= nxt
        frontier = nxt
    return dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save", default="outputs/diag_phys_mask_misses.json")
    args = ap.parse_args()

    bio = BiochemConfig(phase="biochem")
    lss = float(bio.lss)
    man = json.loads((LOCKED / "manifest.json").read_text())
    pool = man["training_pool"]
    cache = attach_physics(load_cache(args.cache))
    pool = [a for a in pool if a in cache]

    agg: dict[str, list] = {g: [] for g in ("TP", "FN", "FP", "TN")}
    per_vessel = []
    for a in pool:
        S = cache[a]
        cols = [str(c) for c in S["cols"]]
        w = S["wall"].astype(bool)
        gt = (S["y"] > 0.5) & w
        m = S["phys_mask"].astype(bool) & w
        n = len(w)
        seed = (S["gate"] > 0) & w
        adm = (S["sr"] < lss * 2.0) & w
        dist = hops_to_seed(S["edge_index"], n, seed, w, adm)

        grp = {"TP": gt & m, "FN": gt & ~m, "FP": ~gt & m, "TN": ~gt & ~m & w}
        row = dict(v=a, n_wall=int(w.sum()), n_gt=int(gt.sum()), n_phys=int(m.sum()),
                   n_fn=int(grp["FN"].sum()),
                   fn_frac=float(grp["FN"].sum() / max(gt.sum(), 1)))
        for g, sel in grp.items():
            if not sel.any():
                continue
            d = dict(n=int(sel.sum()),
                     adm=float(adm[sel].mean()),           # would the criterion admit it?
                     unreach=float((dist[sel] >= 40).mean()),   # but can growth get there?
                     hops=float(np.median(dist[sel])),
                     sr_over_lss=float(np.median(S["sr"][sel] / lss)))
            for f in FIELDS:
                if f in cols:
                    d[f] = float(np.median(S["X"][sel, cols.index(f)]))
            agg[g].append(d)
            row[g] = d
        per_vessel.append(row)
        print("%-12s wall=%4d gt=%4d phys=%4d  MISSED=%4d (%.0f%% of GT)"
              % (a, row["n_wall"], row["n_gt"], row["n_phys"], row["n_fn"],
                 100 * row["fn_frac"]), flush=True)

    tot_gt = sum(r["n_gt"] for r in per_vessel)
    tot_fn = sum(r["n_fn"] for r in per_vessel)
    print("\nacross the pool: %d of %d GT wall nodes are never reached (%.1f%%)"
          % (tot_fn, tot_gt, 100 * tot_fn / max(tot_gt, 1)))

    keys = ["n", "adm", "unreach", "hops", "sr_over_lss"] + [f for f in FIELDS]
    print("\nmedian per group, pooled over vessels (weighted by node count)")
    print("%-6s %8s %7s %9s %7s %12s %s"
          % ("group", "nodes", "adm", "unreach", "hops", "sr/lss",
             " ".join("%10s" % f[:10] for f in FIELDS)))
    for g in ("TP", "FN", "FP", "TN"):
        rs = agg[g]
        if not rs:
            continue
        wts = np.array([r["n"] for r in rs], float)

        def wm(k):
            v = np.array([r.get(k, np.nan) for r in rs], float)
            ok = ~np.isnan(v)
            return float(np.average(v[ok], weights=wts[ok])) if ok.any() else float("nan")

        print("%-6s %8d %7.3f %9.3f %7.1f %12.3f %s"
              % (g, int(wts.sum()), wm("adm"), wm("unreach"), wm("hops"), wm("sr_over_lss"),
                 " ".join("%10.3f" % wm(f) for f in FIELDS)))

    (REPO / args.save).write_text(json.dumps(dict(per_vessel=per_vessel, groups=agg),
                                             default=float))
    print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
