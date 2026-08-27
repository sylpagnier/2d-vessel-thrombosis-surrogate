"""Is the off-wall magnitude rule right and the ODE's ``Mat`` wrong, or is the rule wrong?

`diag_wound_offwall_ceiling.py` found that intersecting the shipped off-wall set with
``Mat_owner >= crit / 0.16`` takes `wound_patient001` from **0.4755 to 0.9708** and
`002` from **0.6736 to 0.9708**, and takes `003` from 0.5293 down to **0.3536**.

That split has two readings and they call for opposite work:

* the RULE is right and 003's ODE ``Mat`` is too small -- on that vessel the trajectory tops
  out near 2x crit where the rule needs 6.25x, so the gate degenerates into "healthy wall
  never grows lumen", which is true on 001/002 and false on 003;
* the RULE is wrong and the two healthy-flow vessels only look good because they have no
  far-field GT clot to lose.

GT ``Mat`` decides it.  If the same threshold applied to COMSOL's own surface field separates
003's lumen positives from its negatives, the lever is the ODE's magnitude.  If it does not,
the magnitude rule is a 001/002 artefact and must not ship.

    python scripts/diag_wound_mat_magnitude.py
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
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score, y):
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


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

        traj, _ = ode_trajectory(data, bio, flow="gt")
        ode_own = traj[-1][owner] / crit

        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()
        gt_own = gmat[-1][owner] / crit

        print("=" * 100)
        print(f"{stem}  lumen GT+={int((gt & off).sum())}")

        print("\n  [1] Mat/crit at the SOLID boundary, final time: ODE against GT")
        for tag, m in (("healthy wall", wall), ("wound", wnd)):
            if not m.any():
                continue
            print(f"      {tag:14s} ODE p50 {np.median(traj[-1][m] / crit):8.2f}  "
                  f"p90 {np.percentile(traj[-1][m] / crit, 90):9.2f}   |   "
                  f"GT p50 {np.median(gmat[-1][m] / crit):8.2f}  "
                  f"p90 {np.percentile(gmat[-1][m] / crit, 90):9.2f}")

        print("\n  [2] Mat_owner/crit over LUMEN nodes, GT+ against GT-")
        for tag, f in (("ODE", ode_own), ("GT ", gt_own)):
            a, b = f[off & gt], f[off & ~gt]
            print(f"      {tag}  GT+ p10/p50/p90 {np.percentile(a, 10):9.2f} "
                  f"{np.median(a):9.2f} {np.percentile(a, 90):9.2f}   |   "
                  f"GT- {np.percentile(b, 10):8.2f} {np.median(b):8.2f} "
                  f"{np.percentile(b, 90):8.2f}   AUC {auc(f[off], gt[off]):.4f}")

        print("\n  [3] THE RULE ON A CORRECT Mat: commit lumen where "
              "GT Mat_owner >= 1/att and hop<=k")
        print(f"      {'att':>5s} {'k':>3s} {'TP':>5s} {'FP':>5s} {'prec':>6s} "
              f"{'rec':>6s} {'off score':>10s}")
        for att in (0.16, 0.08, 0.04, 0.02):
            for k in (2, 4, 6):
                m = (pred & ~off) | (off & (hop_s <= k) & (gt_own >= 1.0 / att))
                tp = int((m & gt & off).sum())
                fp = int((m & ~gt & off).sum())
                print(f"      {att:5.2f} {k:3d} {tp:5d} {fp:5d} "
                      f"{tp / max(tp + fp, 1):6.3f} {tp / max(int((gt & off).sum()), 1):6.3f} "
                      f"{domain_score(m, gt, ei, off, solid):10.4f}")

        print("\n  [4] the same rule on the ODE's own Mat (deployable)")
        print(f"      {'att':>5s} {'k':>3s} {'TP':>5s} {'FP':>5s} {'prec':>6s} "
              f"{'rec':>6s} {'off score':>10s}")
        for att in (1.0, 0.5, 0.25, 0.16):
            for k in (2, 4, 6):
                m = (pred & ~off) | (off & (hop_s <= k) & (ode_own >= 1.0 / att))
                tp = int((m & gt & off).sum())
                fp = int((m & ~gt & off).sum())
                print(f"      {att:5.2f} {k:3d} {tp:5d} {fp:5d} "
                      f"{tp / max(tp + fp, 1):6.3f} {tp / max(int((gt & off).sum()), 1):6.3f} "
                      f"{domain_score(m, gt, ei, off, solid):10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
