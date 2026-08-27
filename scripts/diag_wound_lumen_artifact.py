"""Confirm the ``wound.lumen`` artifact field through the REAL deploy dispatcher.

Two properties, both of which have to hold before the field can be promoted:

1. the shipped artifact (no ``lumen`` key) is bit-identical to an explicit ``shell`` -- so
   every number recorded before 2026-08-24 still means what it said;
2. ``recursive`` is exactly inert on 001/002 and only moves 003, which is the MODEL_REVIEW
   5b.5 gate, checked on the deploy path rather than on ``predict_wound_series`` in isolation.

    python scripts/diag_wound_lumen_artifact.py
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
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
)
from src.clot_ml.wound import solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
DOMS = ("wall", "off", "w_lum", "far", "full")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)
    print(f"[i] artifact={args.name}  wound.lumen present in manifest: "
          f"{'lumen' in bundle['wound']}")
    variants = {
        "shipped (key absent)": bundle,
        "explicit shell": dict(bundle, wound=dict(bundle["wound"], lumen="shell")),
        "recursive": dict(bundle, wound=dict(bundle["wound"], lumen="recursive")),
    }
    acc: dict[str, dict[str, list[float]]] = {}

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = [0, T - 1]
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        _, w_lum, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        doms = {"wall": wall, "off": off, "w_lum": w_lum, "far": far,
                "full": np.ones_like(wall)}

        S = build_sample(data, bio, flow="gt", variant="v4")
        masks = {}
        print("=" * 92)
        print(f"{stem}  T={T}")
        print("  " + f"{'variant':22s}" + "".join(f"{d:>9s}" for d in DOMS)
              + f"{'offTP':>7s}{'offFP':>7s}")
        for tag, b in variants.items():
            pred = predict_temporal_v4_wound(b, data, times, flow="gt",
                                             sample=S)["series"][T - 1]
            masks[tag] = pred
            line = f"  {tag:22s}"
            for d in DOMS:
                sc = domain_score(pred, gt, ei, doms[d], solid)
                acc.setdefault(tag, {}).setdefault(d, []).append(sc)
                line += f"{sc:9.4f}"
            print(line + f"{int((pred & gt & off).sum()):7d}"
                         f"{int((pred & ~gt & off).sum()):7d}")

        same = np.array_equal(masks["shipped (key absent)"], masks["explicit shell"])
        print(f"  default == explicit shell : {'IDENTICAL' if same else 'DIFFERS  <-- BUG'}")
        d_rec = int((masks["recursive"] != masks["explicit shell"]).sum())
        print(f"  recursive vs shell        : {d_rec} node(s) differ"
              + ("   (inert, as required)" if d_rec == 0 else ""))
        add = masks["recursive"] & ~masks["explicit shell"]
        rem = masks["explicit shell"] & ~masks["recursive"]
        if d_rec:
            print(f"     added {int(add.sum())} (GT+ {int((add & gt).sum())}), "
                  f"removed {int(rem.sum())}  <- removals must be 0 (strictly additive)")

    print("=" * 92)
    print("  " + f"{'MEAN':22s}" + "".join(f"{d:>9s}" for d in DOMS))
    for tag in variants:
        print(f"  {tag:22s}" + "".join(
            f"{np.nanmean(acc[tag][d]):9.4f}" for d in DOMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
