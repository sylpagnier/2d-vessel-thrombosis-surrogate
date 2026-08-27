"""Does ANY trainable vessel reach `wound_patient003`'s wall `Mat` magnitude?

The v6 plan is to learn the surface `Mat` field, because WOUND_PROGRESS 16 showed the ODE
orders 003's far-field candidates at chance while GT `Mat` separates them at AUC 0.996.  That
plan has one obvious way to fail: 003's wall `Mat` reaches p90 **27.8x crit** against 8.5x on
`wound_patient001/002`, and 14.1(3) records that 003 is the only vessel in the dataset showing
near-wall platelet activation.  If no legal training vessel reaches that magnitude, the model
would have to EXTRAPOLATE and the off-wall bar (6.25x crit) sits outside its training support.

Cheap to answer -- it only reads `data.y`, no features, no model.

    python scripts/diag_mat_magnitude_cohort.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.data_gen.lib.mesh_wls import solid_boundary_nodes  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    rows = []
    for p in sorted(PACKS.glob("*.pt")):
        if p.name.endswith(".prenormalfix"):
            continue
        stem = p.stem
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            names = d.y_channel_names.split(",")
            T = int(d.y.shape[0])
            m = mat_si_for_gelation_from_log1p(
                d.y[T - 1, :, names.index("Mat_log1p_nd")], bio).reshape(-1).numpy()
            solid = solid_boundary_nodes(d)
            s = m[solid] / crit
            rows.append((stem, T, int(solid.sum()), float(np.percentile(s, 50)),
                         float(np.percentile(s, 90)), float(np.percentile(s, 99)),
                         float((s >= 6.25).mean())))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {stem}: {type(exc).__name__} {exc}")
    rows.sort(key=lambda r: -r[4])
    print(f"{'stem':28s} {'T':>4s} {'solid':>6s} {'p50':>8s} {'p90':>8s} {'p99':>9s} "
          f"{'frac>=6.25x':>11s}")
    for r in rows:
        mark = "  <<<" if r[0].startswith("wound_") else ""
        print(f"{r[0]:28s} {r[1]:4d} {r[2]:6d} {r[3]:8.2f} {r[4]:8.2f} {r[5]:9.2f} "
              f"{r[6]:11.3f}{mark}")
    p90 = np.array([r[4] for r in rows])
    non_wound = np.array([r[4] for r in rows if not r[0].startswith("wound_")])
    print(f"\ncohort wall Mat/crit p90:  median {np.median(p90):.2f}  "
          f"max {p90.max():.2f}   non-wound max {non_wound.max():.2f}")
    tgt = [r[4] for r in rows if r[0] == "wound_patient003"]
    if tgt:
        n_above = int((non_wound >= tgt[0]).sum())
        print(f"wound_patient003 p90 = {tgt[0]:.2f}; non-wound vessels at or above it: "
              f"{n_above}  -> {'IN training support' if n_above >= 3 else 'EXTRAPOLATION RISK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
