"""Is there a deploy-legal signal that separates 003's far-field lumen clot from its lumen?

Where the far-field gap actually sits, after the other arms are ruled out:

* the stall hypothesis is closed -- a TOTAL stall pins the gate at 1 and reaches 2.31x crit
  where the shell rule needs 6.25x, for zero off-wall TP (``diag_wound_gt_gate_ceiling.py``);
* the one-shell architecture caps 003's off-wall recall at 0.663 / score 0.8667
  (``diag_wound_offwall_ceiling2.py``), so the target is reachable but only just;
* ``lumen="recursive"`` is worth +0.0034 off on 003 and is exactly inert on 001/002
  (``diag_wound_lumen_modes.py``) -- real, safe, and far too small.

That leaves the far field: 147 of 003's 243 off-wall GT+, of which 116 sit in shell 1, owned
by healthy wall at the second clot station where the wall arm is already excellent (204 TP,
ZERO FP).  Committing the shell behind every committed wall node gets 84 of them but drags in
106 false positives, and does damage on 001/002 whose far field is 100% clot-free.

So the question is only whether anything RANKS the candidates.  Reported as AUC over the far
shell-1 candidates on 003 (higher is better) alongside what the same signal does to 001/002,
where the correct answer is to commit nothing at all.

    python scripts/diag_wound_far_separator.py
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

from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_scores, predict_temporal_v4_wound,
)
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    first_corner_shell, topological_owner,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(score, float))) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    bundle = load_temporal_v4_wound(name=args.name)

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        _, _, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

        S = build_sample(data, bio, flow="gt", variant="v4")
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt",
                                         sample=S)["series"][T - 1]
        sc_gnn = np.asarray(predict_scores(bundle["base"]["ens"], S), dtype=np.float64)
        wr = bundle["base"]["temporal"].get("wound_rate")
        traj, _ = ode_trajectory(data, bio, flow="gt",
                                 wound_rate=None if wr is None else tuple(wr))
        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()

        shell = first_corner_shell(pos, solid, ei_np)
        town = topological_owner(pos, solid, ei_np)
        has = town >= 0
        pw = pred & solid
        ow_c = np.zeros(len(wall), bool)
        ow_c[has] = pw[town[has]]

        # the candidate set the far-field rule would ever look at
        cand = shell & off & far & ow_c
        y = gt[cand]
        o = town[cand]

        sig = {
            "ODE Mat_owner": traj[-1][o],
            "GT Mat_owner (ORACLE)": gmat[-1][o],
            "GNN score (own node)": sc_gnn[cand],
            "GNN score (owner)": sc_gnn[o],
            "wall-committed frac h8": None,
        }
        from src.clot_ml.features import adjacency, hop_distance
        A = adjacency(ei_np, int(data.num_nodes))
        dens = np.asarray(A @ pw.astype(np.float64)).reshape(-1)
        for _ in range(3):
            dens = np.asarray(A @ dens).reshape(-1) / 4.0
        sig["wall-committed frac h8"] = dens[o]

        print("=" * 96)
        print(f"{stem}  far GT+={int((gt & far).sum())}  "
              f"far shell-1 owner-committed candidates={int(cand.sum())}  "
              f"of which GT+={int(y.sum())}")
        if cand.sum() == 0:
            continue
        print(f"  {'signal':26s} {'AUC':>7s}   {'medGT+':>10s} {'medGT-':>10s}")
        for tag, s in sig.items():
            s = np.asarray(s, float)
            gp = np.median(s[y]) if y.any() else float("nan")
            gn = np.median(s[~y]) if (~y).any() else float("nan")
            print(f"  {tag:26s} {auc(s, y):7.4f}   {gp:10.4g} {gn:10.4g}")
        print(f"  [ref] ODE Mat_owner / crit  p50={np.median(traj[-1][o]) / crit:.2f}  "
              f"GT p50={np.median(gmat[-1][o]) / crit:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
