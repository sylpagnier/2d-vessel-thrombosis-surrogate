"""Is the wound_patient003 off-wall miss reachable by a GLOBAL FLOW STALL?

The handoff hypothesis: the wound gels, the vessel's flow collapses everywhere, and the
ordinary stagnation admission (`spd < speed_thresh` and `sr < sr_max`, the
`grow_into_lumen` rule) then picks up the far-field lumen clot at `s ~ 0.52` that no local
shell rule can reach.

This script tests the hypothesis WITH THE ORACLE FIRST, which is the only order that can
falsify it cheaply.  If GT's own evolving velocity field does not separate the missed lumen
GT-positives from the lumen true negatives, then no flow model -- reduced-order, corrector,
or exact -- can, and the lever is somewhere else.

    python scripts/diag_wound_global_stall.py
    python scripts/diag_wound_global_stall.py --stems wound_patient003
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

from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d  # noqa: E402
from src.core_physics.physics_wall_model import M_TO_CM  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def gt_flow(data, ti, Dx, Dy, u_ref, d_bar):
    u = data.y[ti, :, 0].detach().cpu().numpy().astype(np.float64)
    v = data.y[ti, :, 1].detach().cpu().numpy().astype(np.float64)
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    return np.hypot(u, v), sr


def q(name, a, b):
    """Print the p10/median/p90 of one quantity on two node sets."""
    def s(x):
        if x.size == 0:
            return "        --            "
        return f"{np.percentile(x, 10):8.3f}{np.median(x):8.3f}{np.percentile(x, 90):8.3f}"
    print(f"   {name:26s} {s(a)}  | {s(b)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
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

        out = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt")
        pred = out["series"][T - 1]

        pos = node_positions(data)
        ei = data.edge_index.detach().cpu().numpy()
        Dx, Dy = build_mls_gradient(pos, ei, hops=3)
        u_ref = float(data.u_ref.reshape(-1)[0])
        d_bar = float(data.d_bar.reshape(-1)[0])

        spd0, sr0 = gt_flow(data, 0, Dx, Dy, u_ref, d_bar)
        spdF, srF = gt_flow(data, T - 1, Dx, Dy, u_ref, d_bar)

        A = adjacency(ei, n)
        hop_w = hop_distance(wnd, A, max_h=60) if wnd.any() else np.full(n, 999)

        miss = off & gt & ~pred
        hitp = off & gt & pred
        neg = off & ~gt

        print("=" * 104)
        print(f"{stem}  T={T}  lumen={int(off.sum())}  lumen GT+={int((off & gt).sum())}  "
              f"missed={int(miss.sum())}")

        # 1. Does the vessel stall globally in GT at all?
        bulk = off & (spd0 > np.median(spd0[off]))
        print(f"\n  [1] GLOBAL GT FLOW, t=0 -> t={T-1}")
        print(f"      lumen speed median      {np.median(spd0[off]):9.4f} -> "
              f"{np.median(spdF[off]):9.4f}   ({np.median(spdF[off])/max(np.median(spd0[off]),1e-12):.3f}x)")
        print(f"      fast-half speed median  {np.median(spd0[bulk]):9.4f} -> "
              f"{np.median(spdF[bulk]):9.4f}   ({np.median(spdF[bulk])/max(np.median(spd0[bulk]),1e-12):.3f}x)")
        print(f"      lumen sr median         {np.median(sr0[off]):9.2f} -> "
              f"{np.median(srF[off]):9.2f}   ({np.median(srF[off])/max(np.median(sr0[off]),1e-12):.3f}x)")

        # 2. Do the missed nodes look stagnant -- at t=0, and at final time?
        print(f"\n  [2] SEPARABILITY.  p10 / median / p90 over MISSED GT+ lumen | lumen TRUE NEG")
        q("speed t=0", spd0[miss], spd0[neg])
        q("speed final (ORACLE)", spdF[miss], spdF[neg])
        q("sr t=0", sr0[miss], sr0[neg])
        q("sr final (ORACLE)", srF[miss], srF[neg])
        q("speed ratio final/t0", (spdF / np.maximum(spd0, 1e-12))[miss],
          (spdF / np.maximum(spd0, 1e-12))[neg])

        # 3. What a stagnation cut would actually buy, at the oracle flow.
        print(f"\n  [3] ORACLE STAGNATION CUT on the FINAL GT speed field "
              f"(lumen only, added to the current prediction)")
        print(f"      {'thresh':>8s} {'newTP':>6s} {'newFP':>6s} {'rec':>6s} {'prec':>6s}")
        tp0, fp0 = int((pred & gt & off).sum()), int((pred & ~gt & off).sum())
        ngt = int((off & gt).sum())
        for th in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            adm = off & (spdF < th) & ~pred
            ntp, nfp = int((adm & gt).sum()), int((adm & ~gt).sum())
            rec = (tp0 + ntp) / max(ngt, 1)
            prec = (tp0 + ntp) / max(tp0 + ntp + fp0 + nfp, 1)
            print(f"      {th:8.2f} {ntp:6d} {nfp:6d} {rec:6.3f} {prec:6.3f}")

        # 4. Where do the missed nodes sit relative to the wound and to committed wall?
        if wnd.any():
            print(f"\n  [4] MISSED lumen by hops from the wound")
            for a, b in ((0, 4), (5, 8), (9, 14), (15, 24), (25, 60)):
                band = (hop_w >= a) & (hop_w <= b) & off
                tot = int((band & gt).sum())
                if tot:
                    print(f"      {a:2d}-{b:<2d}  GT+ {tot:4d}  missed {int((miss & band).sum()):4d}  "
                          f"speed0 med {np.median(spd0[band & gt]):.3f}  "
                          f"speedF med {np.median(spdF[band & gt]):.3f}")

        # 5. Owner structure of the missed nodes: is the WALL under them right?
        _, j = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[j]
        ow = np.unique(owner[miss])
        print(f"\n  [5] OWNERS of the missed lumen: {ow.size} distinct solid nodes; "
              f"{int(pred[ow].sum())} already predicted clot, {int(gt[ow].sum())} GT clot")
        print(f"      missed nodes whose owner IS already predicted: "
              f"{int((miss & pred[owner]).sum())}/{int(miss.sum())}")
        print(f"      hit    nodes whose owner IS already predicted: "
              f"{int((hitp & pred[owner]).sum())}/{int(hitp.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
