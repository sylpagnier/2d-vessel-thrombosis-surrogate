"""Refit the gelation-wake kernel in `src/core_physics/gelation_wake.py` from GT.

The kernel is the measured relation between a wall node's SUPERPOSED gelled-neighbour load

    w_i  =  sum over gelled wall nodes j of  exp(-hops(i, j) / WAKE_LAMBDA_HOPS)

and GT `sr(t)/sr(0)` at that node, restricted to nodes that have NOT yet gelled themselves --
those are the ones whose gate the wake has to open.  A node that has gelled takes the
per-node step `GELLED_SR_RATIO` instead, which `scripts/diag_closed_loop_feasibility.py`
measures separately.

Pooling across wound AND no-wound vessels is the point: the constant has to be the same
object in both or it does not transfer.  Prints the table to paste into the module.

    python scripts/fit_gelation_wake_kernel.py
    python scripts/fit_gelation_wake_kernel.py --stems patient020 patient044
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.wound import prepare_vessel  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.gelation_wake import (  # noqa: E402
    WAKE_LAMBDA_HOPS, WAKE_MAX_HOPS, wall_wake_operator,
)
from src.core_physics.mls_gradient import build_mls_gradient  # noqa: E402
from src.core_physics.physics_wall_model import node_positions, shear_rate_2d  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
#: Wound vessels plus a spread of no-wound geometries.  SEALED 007/013/031/043 excluded.
DEFAULT_STEMS = ("wound_patient001", "wound_patient002", "wound_patient003",
                 "patient012", "patient016", "patient020", "patient028",
                 "patient032", "patient035", "patient041", "patient044")
BINS = np.array([0, .25, .5, 1, 2, 4, 6, 8, 1e9])


def profile(stem: str, bio, lam: float, max_hops: int, n_steps: int) -> dict[int, float]:
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    T = int(data.y.shape[0])
    V = prepare_vessel(data, bio, flow="gt")
    crit, wall = V["C"].crit, V["wall"]
    widx = np.flatnonzero(wall)
    pos, ei = node_positions(data), data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=3)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    names = data.y_channel_names.split(",")
    gel = mat_si_for_gelation_from_log1p(
        data.y[:, :, names.index("Mat_log1p_nd")], bio).reshape(T, -1).numpy() >= crit
    K = wall_wake_operator(data, wall, lam=lam, max_hops=max_hops)

    def sr_at(s: int) -> np.ndarray:
        u = data.y[s, :, 0].numpy().astype(np.float64)
        v = data.y[s, :, 1].numpy().astype(np.float64)
        return shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)

    sr0 = sr_at(0)
    acc: dict[int, list[float]] = {i: [] for i in range(len(BINS) - 1)}
    for s in range(1, T, max(T // n_steps, 1)):
        if not gel[s].any():
            continue
        r = sr_at(s) / np.maximum(sr0, 1e-9)
        w = np.zeros(len(wall))
        w[widx] = K @ gel[s][widx].astype(np.float64)
        keep = wall & ~gel[s]                       # the population the wake must move
        b = np.digitize(w, BINS) - 1
        for i in range(len(BINS) - 1):
            m = keep & (b == i)
            if m.sum() >= 3:
                acc[i].append(float(np.median(r[m])))
    return {i: float(np.median(v)) for i, v in acc.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(DEFAULT_STEMS))
    ap.add_argument("--lam", type=float, default=WAKE_LAMBDA_HOPS)
    ap.add_argument("--max-hops", type=int, default=WAKE_MAX_HOPS)
    ap.add_argument("--steps", type=int, default=24, help="time samples per vessel")
    ap.add_argument("--out", default="outputs/gelation_wake_kernel.json")
    args = ap.parse_args()

    bio = BiochemConfig(phase="biochem")
    rows = {}
    for stem in args.stems:
        if not (PACKS / f"{stem}.pt").exists():
            print(f"  {stem:22s} MISSING"); continue
        rows[stem] = profile(stem, bio, args.lam, args.max_hops, args.steps)
        print(f"  {stem:22s} " + " ".join(
            f"w<{BINS[i + 1]:g}:{rows[stem][i]:.3f}" for i in sorted(rows[stem])))

    print(f"\nPOOLED (lambda={args.lam}, max_hops={args.max_hops}) "
          f"-- amp on NOT-yet-gelled wall")
    pooled = {}
    for i in range(len(BINS) - 1):
        vals = [rows[s][i] for s in rows if i in rows[s]]
        if len(vals) >= 3:
            pooled[str(BINS[i + 1])] = float(np.median(vals))
            print(f"   w in [{BINS[i]:5g},{BINS[i + 1]:5g})  n_vessels {len(vals):3d}"
                  f"   amp {np.median(vals):.4f}   spread {min(vals):.3f}-{max(vals):.3f}")
    print("\n[i] bins with fewer than 3 vessels are NOT reported: in GT a node under that"
          "\n    much load has already gelled, so the not-yet-gelled population runs out."
          "\n    That is why WAKE_LOAD_AMP is clamped rather than extrapolated.")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(lam=args.lam, max_hops=args.max_hops,
                                   bins=BINS.tolist(), pooled=pooled, per_vessel=rows),
                              indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
