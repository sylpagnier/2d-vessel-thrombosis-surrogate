"""Can the off-wall domain be fixed by a CUT, or is the ranking itself wrong?

Context -- two families already measured and dead:

* `diag_wound_global_stall.py` killed the global-stall story.  GT lumen speed on
  `wound_patient003` moves **1.002x** over the whole run and `sr` RISES 1.065x, so there is
  no vessel-scale stall to model; an oracle stagnation cut on the FINAL GT speed field adds
  66 true positives against **657** false ones.
* `diag_wound_lumen_shell.py` killed the blanket owner-shell family.  Even seeded on the GT
  solid set the best depth scores **0.5949** on the off domain, because the richest band is
  only 29% GT-positive.

What both leave standing is the observation that **97 of the 169 missed lumen nodes sit on
top of a solid node the model has already committed**, and 74 of 74 currently-hit ones do.
So the candidate set is not the problem -- the ORDERING inside it is.

This script scopes the question to that set: lumen nodes owned by a committed solid node,
within a few corner shells.  For each candidate ranking it reports the AUC inside that set
and the best prefix any cut could achieve.

    python scripts/diag_wound_offwall_ranking.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_scores, predict_temporal_v4_wound,
)
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import median_edge_length  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score: np.ndarray, y: np.ndarray) -> float:
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    ap.add_argument("--max-hop", type=int, default=6)
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    bundle = load_temporal_v4_wound(name=args.name)

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        n = int(data.num_nodes)
        solid = solid_mask(data)
        wnd = wound_mask(data)
        off = ~solid
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)

        S = build_sample(data, bio, flow="gt", variant="v4")
        out = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt", sample=S)
        pred = out["series"][T - 1]
        gnn = np.asarray(predict_scores(bundle["base"]["ens"], S), dtype=np.float64)

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        hlen = median_edge_length(pos, ei_np)
        dist, which = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[which]
        A = adjacency(ei_np, n)
        hop_s = hop_distance(solid, A, max_h=20)

        traj, _ = ode_trajectory(data, bio, flow="gt")
        mat_own = traj[-1][owner] / crit

        cand = off & (hop_s <= args.max_hop) & pred[owner]
        ygt = gt & cand
        print("=" * 100)
        print(f"{stem}  lumen={int(off.sum())}  lumen GT+={int((off & gt).sum())}  "
              f"shipped off={domain_score(pred, gt, ei, off, solid):.4f} "
              f"(TP {int((pred & gt & off).sum())} FP {int((pred & ~gt & off).sum())})")
        print(f"  candidate set (owner committed, hop<={args.max_hop}): {int(cand.sum())} nodes, "
              f"{int(ygt.sum())} GT+ ({ygt.sum() / max(cand.sum(), 1):.3f}) -- covers "
              f"{int(ygt.sum())}/{int((off & gt).sum())} of the lumen GT+")

        allc = pred.copy()
        allc[cand] = True
        print(f"  commit the WHOLE candidate set: off={domain_score(allc, gt, ei, off, solid):.4f} "
              f"(TP {int((allc & gt & off).sum())} FP {int((allc & ~gt & off).sum())})")
        orc = pred.copy()
        orc[ygt] = True
        print(f"  commit only its GT+ (ORACLE)  : off={domain_score(orc, gt, ei, off, solid):.4f}")

        y = gt[cand]
        cands = {
            "gnn score": gnn,
            "mat_owner": mat_own,
            "-dist/h": -dist / hlen,
            "-hop_from_solid": -hop_s.astype(np.float64),
            "mat_owner / (d/h)": mat_own / np.maximum(dist / hlen, 0.5),
            "gnn * mat_owner": gnn * np.log1p(mat_own),
            "gnn / (d/h)": gnn / np.maximum(dist / hlen, 0.5),
        }
        print(f"\n  {'ranking':22s} {'AUC':>6s} | best prefix: {'k':>5s} {'TP':>5s} "
              f"{'FP':>5s} {'prec':>6s} {'off score':>10s}")
        cidx = np.flatnonzero(cand)
        ks = sorted({int(round(x)) for x in np.geomspace(5, max(int(cand.sum()), 6), 24)})
        for name, f in cands.items():
            order = cidx[np.argsort(-f[cand], kind="stable")]
            best = (-1.0, 0, 0, 0)
            for k in ks:
                m = pred.copy()
                m[order[:k]] = True
                s = domain_score(m, gt, ei, off, solid)
                if s > best[0]:
                    best = (s, k, int((m & gt & off).sum()), int((m & ~gt & off).sum()))
            s, k, tp, fp = best
            print(f"  {name:22s} {auc(f[cand], y):6.4f} | {'':13s}{k:5d} {tp:5d} {fp:5d} "
                  f"{tp / max(tp + fp, 1):6.3f} {s:10.4f}")

        if wnd.any():
            hop_w = hop_distance(wnd, A, max_h=80)
            print(f"\n  clip the shipped prediction to <= k hops from the wound:")
            for k in (8, 32, 80):
                m = pred.copy()
                m[off & (hop_w > k)] = False
                print(f"      k={k:3d}  off={domain_score(m, gt, ei, off, solid):.4f}  "
                      f"TP {int((m & gt & off).sum()):4d} FP {int((m & ~gt & off).sum()):4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
