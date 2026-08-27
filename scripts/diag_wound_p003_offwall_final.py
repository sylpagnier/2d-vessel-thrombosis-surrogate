"""Final-time off-wall / wound-lumen recall on wound_patient003.

The wall MOT gap (scripts/diag_wound_p003_wall_mot.py) is a CLOCK.  This script is the SET:
does the last frame miss lumen clot, and who owns those nodes?

    python scripts/diag_wound_p003_offwall_final.py
    python scripts/diag_wound_p003_offwall_final.py --stems wound_patient003
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

from src.clot_ml.evaluate import domain_score, f1  # noqa: E402
from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
DOM_ORDER = ("wall", "off", "wnd", "w_reg", "w_lum", "far", "full")


def prf(pred, gt, dom):
    p, g = pred & dom, gt & dom
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return tp, fp, fn, prec, rec, f1(p, g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)
    print(f"[i] artifact={args.name}  FINAL-time set only\n")

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = [0, T - 1]
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        wnd = wound_mask(data)
        region, lumen, far = wound_region_masks(data)
        domains = {
            "wall": wall, "off": off, "wnd": wnd,
            "w_reg": region, "w_lum": lumen, "far": far,
            "full": np.ones_like(wall),
        }

        out = predict_temporal_v4_wound(bundle, data, times, flow="gt")
        pred = out["series"][T - 1]
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5

        print("=" * 92)
        print(f"{stem}  T={T}  wall={int(wall.sum())}  wound={int(wnd.sum())}  "
              f"lumen={int(off.sum())}")
        print(f"  {'dom':8s} {'score':>7s} {'F1':>7s} {'prec':>6s} {'rec':>6s}  "
              f"{'TP':>5s} {'FP':>5s} {'FN':>5s}  {'GT+':>5s} {'pred+':>6s}")
        for name in DOM_ORDER:
            dom = domains[name]
            tp, fp, fn, prec, rec, f1v = prf(pred, gt, dom)
            sc = domain_score(pred, gt, ei, dom, solid)
            print(f"  {name:8s} {sc:7.4f} {f1v:7.4f} {prec:6.3f} {rec:6.3f}  "
                  f"{tp:5d} {fp:5d} {fn:5d}  {int((gt & dom).sum()):5d} "
                  f"{int((pred & dom).sum()):6d}")

        if stem != "wound_patient003":
            continue

        pos = data.x[:, :2].numpy().astype(np.float64)
        _, j = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[j]
        h = hop_distance(wnd, adjacency(data.edge_index.numpy(), int(data.num_nodes)),
                         max_h=40)
        miss = (~pred) & gt & off
        hit = pred & gt & off

        print("\n--- missed LUMEN GT clot by hops from the wound ---")
        print(f"   {'hops':10s} {'miss':>6s} {'GT+':>6s} {'hit':>6s}  rec")
        for a, b in ((0, 2), (3, 4), (5, 6), (7, 8), (9, 14), (15, 40)):
            band = (h >= a) & (h <= b) & off
            tot = int((band & gt).sum())
            if tot == 0:
                continue
            m = int((miss & band).sum())
            print(f"   {a:2d}-{b:<2d}      {m:6d} {tot:6d} {tot - m:6d}  "
                  f"{(tot - m) / tot:.2f}")

        print("\n--- who OWNS the missed lumen nodes ---")
        ow = owner[miss]
        wnd_owned = wnd[ow]
        wall_owned = wall[ow] & ~wnd[ow]
        print(f"   owner is wound         : {int(wnd_owned.sum()):3d}")
        print(f"   owner is healthy wall  : {int(wall_owned.sum()):3d}  "
              f"on {np.unique(ow[wall_owned]).size} distinct nodes")
        u = np.unique(ow[wall_owned])
        if u.size:
            f0 = t0_flow_fields(data, bio, hops=3, flow_source="gt")
            committed = pred[u]
            print(f"   those owners in the FINAL wall set : {int(committed.sum())}/{u.size}")
            print(f"   t=0 gate med {np.median(f0.gate[u]):.3f}  frac>0 "
                  f"{(f0.gate[u] > 0).mean():.2f}  sr med {np.median(f0.sr[u]):.1f} /s")

        print("\n--- lumen GT+ that IS hit: hops / owner ---")
        ow_h = owner[hit]
        print(f"   hit lumen GT+ : {int(hit.sum()):3d}   "
              f"wound-owned {int(wnd[ow_h].sum()):3d}  "
              f"wall-owned {int((wall[ow_h] & ~wnd[ow_h]).sum()):3d}")
        print(f"   far-lumen GT+ (beyond 8 hops): {int((gt & far).sum())}  "
              f"missed {int((miss & far).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
