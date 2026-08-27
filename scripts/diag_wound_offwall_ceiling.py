"""What each stage of the off-wall chain is actually worth, and one physics gate that is missing.

CEILING LADDER.  Every earlier diagnostic answered "does mechanism X help".  This one answers
"how much is left", which is the question that decides whether the >0.75 target on
`wound_patient003` is reachable at all with the current wall set.

THE GATE.  PHASE7 12.5 and the wound complement both commit an off-wall node only when its
owner's ``Mat`` clears ``crit / off_att`` -- a MAGNITUDE condition, 6.25x crit at the shipped
0.16.  `clot_gnn_v5`'s own off-wall readout does not apply it: it cuts the score field and
then inherits whatever wall nodes are committed.  On `wound_patient001/002` that produces
83 / 35 far-field lumen commitments in a domain with **zero** GT positives.  So the last
column here intersects the shipped off-wall set with the admissibility the physics already
asserts everywhere else.

    python scripts/diag_wound_offwall_ceiling.py
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
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
)
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask, wound_owned_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
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

        S = build_sample(data, bio, flow="gt", variant="v4")
        out = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt", sample=S)
        pred = out["series"][T - 1]

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        _, which = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[which]
        A = adjacency(ei_np, n)
        hop_s = hop_distance(solid, A, max_h=20)
        _, owned_off, _ = wound_owned_masks(data)

        w = bundle["wound"]
        traj, _ = ode_trajectory(data, bio, flow="gt",
                                 wound_rate=(float(w["g_pre"]), float(w["g_post"])))
        mat_own = traj.max(axis=0)[owner] / crit

        def sc(m):
            return domain_score(m, gt, ei, off, solid)

        def line(tag, m):
            print(f"    {tag:44s} {sc(m):7.4f}  TP {int((m & gt & off).sum()):4d} "
                  f"FP {int((m & ~gt & off).sum()):5d} FN {int((~m & gt & off).sum()):4d}")

        print("=" * 100)
        print(f"{stem}  lumen GT+={int((gt & off).sum())}  "
              f"wall {int((pred & gt & wall).sum())}/{int((gt & wall).sum())} TP, "
              f"{int((pred & ~gt & wall).sum())} FP")

        print("\n  [A] CEILING LADDER on the off domain")
        line("shipped", pred)
        line("+ drop every lumen FP", pred & ~(off & ~gt))
        cand = off & (hop_s <= 6) & pred[owner]
        line("+ oracle lumen on committed owners", pred | (cand & gt))
        line("  ... and drop every lumen FP", (pred | (cand & gt)) & ~(off & ~gt))
        candg = off & (hop_s <= 6) & gt[owner]
        line("+ oracle lumen on GT owners", pred | (candg & gt))
        line("perfect lumen", (pred & ~off) | (gt & off))

        print("\n  [B] MAGNITUDE GATE: intersect the shipped off-wall set with "
              "Mat_owner >= crit/att")
        print(f"      (wound-owned shell is exempt -- it has its own rate model; "
              f"{int(owned_off.sum())} nodes)")
        print(f"      max Mat_owner/crit over predicted lumen: "
              f"p10 {np.percentile(mat_own[pred & off], 10):.2f} "
              f"p50 {np.median(mat_own[pred & off]):.2f} "
              f"p90 {np.percentile(mat_own[pred & off], 90):.2f}")
        for att in (2.0, 1.4, 1.0, 0.7, 0.5, 0.32, 0.16):
            ok = (mat_own >= 1.0 / att) | owned_off
            m = (pred & ~off) | (pred & off & ok)
            print(f"      att={att:4.2f}  (Mat_owner >= {1/att:6.2f} crit)   "
                  f"off={sc(m):7.4f}  TP {int((m & gt & off).sum()):4d} "
                  f"FP {int((m & ~gt & off).sum()):4d}")

        if wnd.any():
            hop_w = hop_distance(wnd, A, max_h=80)
            fpn = pred & off & ~gt
            far_fp = fpn & (hop_w > 8)
            print(f"\n  [C] the far-field lumen FPs ({int(far_fp.sum())} of "
                  f"{int(fpn.sum())}): max Mat_owner/crit "
                  f"p10 {np.percentile(mat_own[far_fp], 10):.2f} "
                  f"p50 {np.median(mat_own[far_fp]):.2f} "
                  f"p90 {np.percentile(mat_own[far_fp], 90):.2f}"
                  if far_fp.any() else "\n  [C] no far-field lumen FPs")
            tpn = pred & off & gt
            if tpn.any():
                print(f"      the lumen TRUE positives:            max Mat_owner/crit "
                      f"p10 {np.percentile(mat_own[tpn], 10):.2f} "
                      f"p50 {np.median(mat_own[tpn]):.2f} "
                      f"p90 {np.percentile(mat_own[tpn], 90):.2f}")
            gtl = gt & off & ~pred
            if gtl.any():
                print(f"      the MISSED lumen GT+:                max Mat_owner/crit "
                      f"p10 {np.percentile(mat_own[gtl], 10):.2f} "
                      f"p50 {np.median(mat_own[gtl]):.2f} "
                      f"p90 {np.percentile(mat_own[gtl], 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
