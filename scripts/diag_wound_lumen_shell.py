"""Where does wound-vessel lumen clot actually sit, and what does an owner-shell rule buy?

`diag_wound_global_stall.py` falsified the global-stall story: GT's own lumen speed on
`wound_patient003` moves 0.8652 -> 0.8667 over the whole run and `sr` RISES 1.065x, so
there is no vessel-scale stall to model, and an oracle stagnation cut on the final GT speed
field adds 66 true positives against 657 false ones.

What the same diagnostic did find is that **97 of the 169 missed lumen nodes sit on top of a
solid node the model has ALREADY committed**, and every currently-hit lumen node does too.
So the miss is in the wall->lumen extension rule, not in the flow.

This script measures that rule's geometry: how deep GT lumen clot goes above its owner, and
what committing k corner shells above the committed solid set costs in false positives -- at
the GT solid set (the ceiling) and at the model's own (the deployable number).

    python scripts/diag_wound_lumen_shell.py
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
from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import median_edge_length  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys = PhysicsConfig(phase="biochem")
    BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        n = int(data.num_nodes)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        wnd = wound_mask(data)
        off = ~solid
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)

        out = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt")
        pred = out["series"][T - 1]

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        h = median_edge_length(pos, ei_np)
        dist, which = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[which]
        A = adjacency(ei_np, n)
        hop_s = hop_distance(solid, A, max_h=20)

        print("=" * 100)
        print(f"{stem}  T={T}  lumen={int(off.sum())}  lumen GT+={int((off & gt).sum())}  "
              f"median edge={h:.4f}")

        print("\n  [1] GT lumen clot depth above the solid boundary")
        print(f"      {'hop':>4s} {'nodes':>7s} {'GT+':>6s} {'rate':>6s} "
              f"{'ownerGT':>8s} {'d/h p50':>8s}")
        for k in range(1, 9):
            band = off & (hop_s == k)
            nb = int(band.sum())
            if nb == 0:
                continue
            g = int((band & gt).sum())
            ogt = int((band & gt & gt[owner]).sum())
            print(f"      {k:4d} {nb:7d} {g:6d} {g / nb:6.3f} {ogt:8d} "
                  f"{np.median(dist[band & gt]) / h if g else float('nan'):8.3f}")

        print("\n  [2] OWNER-SHELL RULE: commit lumen nodes at hop<=k whose owner is committed")
        print(f"      {'src':>6s} {'k':>3s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
              f"{'prec':>6s} {'rec':>6s} {'off score':>10s}")
        for tag, src in (("GT", gt & solid), ("pred", pred & solid)):
            for k in (1, 2, 3, 4, 6, 8):
                add = off & (hop_s <= k) & src[owner]
                p2 = (pred & ~off) | ((pred | add) & off)
                tp = int((p2 & gt & off).sum())
                fp = int((p2 & ~gt & off).sum())
                fn = int((~p2 & gt & off).sum())
                sc = domain_score(p2, gt, ei, off, solid)
                print(f"      {tag:>6s} {k:3d} {tp:5d} {fp:5d} {fn:5d} "
                      f"{tp / max(tp + fp, 1):6.3f} {tp / max(tp + fn, 1):6.3f} {sc:10.4f}")

        print("\n  [3] CEILINGS on the off domain (current wall/solid set kept)")
        base = domain_score(pred, gt, ei, off, solid)
        no_fp = pred & ~(off & ~gt)
        all_tp = pred | (off & gt)
        perfect = (pred & ~off) | (gt & off)
        print(f"      as shipped                 {base:.4f}")
        print(f"      + drop every lumen FP      {domain_score(no_fp, gt, ei, off, solid):.4f}")
        print(f"      + add every lumen GT+      {domain_score(all_tp, gt, ei, off, solid):.4f}")
        print(f"      perfect lumen              {domain_score(perfect, gt, ei, off, solid):.4f}")

        if wnd.any():
            hop_w = hop_distance(wnd, A, max_h=60)
            fpn = pred & off & ~gt
            print(f"\n  [4] current lumen FALSE POSITIVES: {int(fpn.sum())}  "
                  f"hops-from-wound p10/p50/p90 "
                  f"{np.percentile(hop_w[fpn], 10):.0f}/{np.median(hop_w[fpn]):.0f}/"
                  f"{np.percentile(hop_w[fpn], 90):.0f}"
                  f"   hop-from-solid p50 {np.median(hop_s[fpn]):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
