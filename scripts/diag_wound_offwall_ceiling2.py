"""What score is REACHABLE off-wall, given the one-shell architecture?

``diag_wound_offwall_depth.py`` measured the structural limit: ``wound_patient003``'s off-wall
GT clot is THREE layers deep (161 nodes in the first species row, 59 in the second, 11 in the
third, 12 in the empty bridge band), while ``wound_patient001/002`` are exactly one layer.
Every shipped off-wall arm predicts inside ``first_corner_shell`` only, so 003's recall
ceiling is 0.663 no matter what the ODE, the gate or any blockage does.

This script converts that ceiling into a SCORE, because the deploy metric is hop-relaxed and
recall 0.663 does not mean score 0.663.  It also tests whether iterating the topological shell
construction outward actually recovers the deeper layers -- if it does, the multi-layer rule
is available; if it does not, the pack's own connectivity cannot express 003's clot.

    python scripts/diag_wound_offwall_ceiling2.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import first_corner_shell  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def shell_layers(pos, solid, ei_np, n_layers=4):
    """Iterate the topological shell outward: layer k is the first row beyond layers < k."""
    layers = []
    acc = solid.copy()
    for _ in range(n_layers):
        s = first_corner_shell(pos, acc, ei_np) & ~acc
        if not s.any():
            break
        layers.append(s)
        acc = acc | s
    return layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)
        solid = solid_mask(data)
        off = ~solid
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt")["series"][T - 1]
        pw = pred & solid

        layers = shell_layers(pos, solid, ei_np)
        cum = np.zeros(len(solid), dtype=bool)
        cums = []
        for s in layers:
            cum = cum | s
            cums.append(cum.copy())

        print("=" * 100)
        print(f"{stem}  off GT+={int((gt & off).sum())}   layers found={len(layers)}")
        for k, s in enumerate(layers, 1):
            print(f"   layer {k}: {int((s & off).sum()):5d} nodes, "
                  f"GT+ inside {int((s & gt & off).sum()):4d}")
        print(f"  {'arm':34s} {'off':>7s} {'prec':>6s} {'rec':>6s} {'TP':>5s} {'FP':>5s}")

        arms = {"shipped v5w": pred & off, "PERFECT off-wall": gt & off}
        for k, c in enumerate(cums, 1):
            arms[f"oracle inside layers 1..{k}"] = gt & off & c
            arms[f"ALL of layers 1..{k}"] = off & c
        for tag, m in arms.items():
            tp = int((m & gt & off).sum())
            fp = int((m & ~gt & off).sum())
            fn = int((~m & gt & off).sum())
            sc = domain_score(m | pw, gt, ei, off, solid)
            print(f"  {tag:34s} {sc:7.4f} {tp / max(tp + fp, 1):6.3f} "
                  f"{tp / max(tp + fn, 1):6.3f} {tp:5d} {fp:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
