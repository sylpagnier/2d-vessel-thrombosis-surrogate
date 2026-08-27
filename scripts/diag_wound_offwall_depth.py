"""How DEEP into the lumen does wound GT clot go -- and does the shipped shell contain it?

Every off-wall arm in ``physics_lumen_model`` predicts inside ONE layer: ``first_corner_shell``
is the first species-carrying row, ~550-600 nodes.  That is a hard recall ceiling, and nothing
in the ODE, the gate, or a blockage can raise it.  Before tuning any rule, measure the ceiling.

    python scripts/diag_wound_offwall_depth.py
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

from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    first_corner_shell, median_edge_length,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        _, _, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

        h = median_edge_length(pos, ei_np)
        dist, _ = cKDTree(pos[solid]).query(pos)
        d = dist / h
        shell = first_corner_shell(pos, solid, ei_np)
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt")["series"][T - 1]

        g_off = gt & off
        print("=" * 96)
        print(f"{stem}  off GT+={int(g_off.sum())}  shell={int((shell & off).sum())}  "
              f"median edge={h:.4g}")
        print(f"  in first_corner_shell : {int((g_off & shell).sum())}/{int(g_off.sum())}"
              f"  = {(g_off & shell).sum() / max(g_off.sum(), 1):.3f}   <-- RECALL CEILING")
        print(f"  predicted off-wall    : {int((pred & off).sum())}  "
              f"of which in shell {int((pred & off & shell).sum())}")
        print(f"  {'depth [median edges]':24s} {'GT+':>6s} {'inShell':>8s} {'pred':>6s} "
              f"{'farGT+':>7s}")
        edges = [0.0, 1.35, 2.2, 3.0, 3.9, 4.8, 6.0, 8.0, 100.0]
        for a, b in zip(edges[:-1], edges[1:]):
            band = off & (d >= a) & (d < b)
            n_gt = int((band & gt).sum())
            if n_gt == 0 and int((band & pred).sum()) == 0:
                continue
            print(f"  {a:5.2f} - {b:<6.2f}          {n_gt:6d} "
                  f"{int((band & gt & shell).sum()):8d} {int((band & pred).sum()):6d} "
                  f"{int((band & gt & far).sum()):7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
