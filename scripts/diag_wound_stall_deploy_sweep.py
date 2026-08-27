"""Deploy-path sweep: near-stall radius x lumen extension, scored on wall AND the full off domain.

Everything upstream of this script narrowed the problem to two coupled gates on
`wound_patient003`:

* **the wall gates 77 of the 169 missed lumen nodes** -- their owning solid node is not in
  the committed set at all, and no lumen rule can reach a node whose source is absent;
* **the lumen extension gates the other 92** -- their owner IS committed, and an ORACLE that
  commits exactly the GT-positive members of the owner shell still only reaches off
  **0.6731**, because the 49 existing false positives and the 77 orphans cap it there.

So the two have to move together, which is what this sweeps.  `hops` is the near-stall
radius (`src.core_physics.near_stall.STALL_HOPS`, shipped 1); `lumen` is what is committed
above the newly-ignited wall.

    python scripts/diag_wound_stall_deploy_sweep.py
    python scripts/diag_wound_stall_deploy_sweep.py --stems wound_patient003
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
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.near_stall import make_near_stall_blockage  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    deposition_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def stall_traj(data, bio, f, wall, hops):
    hook = make_rollout_hook(SHIPPED, bio, f.sr)
    gate = deposition_gate(data, f, wall=wall, wound_source=True)
    blk = (None if hops is None else
           make_near_stall_blockage(data, bio, f, wall=wall, hops=int(hops)))
    traj, _ = integrate_mat_trajectory(data, bio, gate, da_scale=SHIPPED_DA_SCALE,
                                       ap_closure=hook, blockage=blk)
    return np.asarray(traj)


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
        pred0 = out["series"][T - 1]

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        _, which = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[which]
        A = adjacency(ei_np, n)
        hop_s = hop_distance(solid, A, max_h=20)

        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        ung = wall & (np.asarray(f.gate) * wall <= 0)

        def row(tag, m):
            w = domain_score(m, gt, ei, wall, solid)
            o = domain_score(m, gt, ei, off, solid)
            print(f"  {tag:28s} {w:7.4f} {o:7.4f} | "
                  f"{int((m & gt & wall).sum()):4d} {int((m & ~gt & wall).sum()):4d} | "
                  f"{int((m & gt & off).sum()):4d} {int((m & ~gt & off).sum()):5d}")

        print("=" * 100)
        print(f"{stem}  T={T}  wall GT+={int((gt & wall).sum())}  "
              f"lumen GT+={int((gt & off).sum())}  ungated wall={int(ung.sum())}")
        print(f"  {'arm':28s} {'wall':>7s} {'off':>7s} | {'wTP':>4s} {'wFP':>4s} | "
              f"{'oTP':>4s} {'oFP':>5s}")
        row("shipped", pred0)

        for hops in (1, 2, 4, 6, 8):
            traj = stall_traj(data, bio, f, wall, hops)
            hot = traj[-1] >= crit
            m = pred0 | (ung & hot)
            row(f"stall h={hops}", m)
            # + first corner shell above every newly committed wall node
            newly = (ung & hot) & ~pred0
            if newly.any():
                shell = off & (hop_s <= 2) & newly[owner]
                row(f"stall h={hops} + shell1", m | shell)
                shell4 = off & (hop_s <= 4) & newly[owner]
                row(f"stall h={hops} + shell2", m | shell4)

        if wnd.any():
            hop_w = hop_distance(wnd, A, max_h=80)
            print(f"  [ref] far-field lumen FP of the shipped arm: "
                  f"{int((pred0 & ~gt & off & (hop_w > 8)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
