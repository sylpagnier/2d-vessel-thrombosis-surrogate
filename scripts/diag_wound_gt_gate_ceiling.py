"""The strongest form of the handoff's flow->gate hypothesis, run as an ORACLE.

The handoff asked for a ``blockage`` callable that models the flow reduction the wound
causes, on the theory that a stalled vessel would drop ``sr`` at the distant clot station and
let the ordinary stagnation admission pick up the far-field lumen.  Two things are already
measured against that:

* there is no global stall to model -- GT lumen speed on `wound_patient003` moves **1.002x**
  across the whole run and `sr` RISES (``diag_wound_global_stall.py``);
* the deployable blockages that DO exist move the wall's ``Mat`` by ~10% where the off-wall
  magnitude rule needs 10x (``diag_wound_mat_arms.py``: wall ``Mat``/crit p90 1.73 shipped,
  1.81 near-stall h=2, 1.90 h=4, 1.73 wake, against GT's **19.45**).

This script removes the last excuse by handing the ODE a **zero-error flow model**: the gate
recomputed from COMSOL's own velocity at every timestep (`gt_flow_gate_series`).  If that
does not close the ``Mat`` magnitude gap, then the gap is not in the flow and no blockage --
reduced-order, corrector, or exact -- will close it.

Slow: MLS gradients at every stored time, minutes per vessel.

    python scripts/diag_wound_gt_gate_ceiling.py --stems wound_patient003
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
from src.clot_ml.wound import solid_mask, wound_mask, wound_rate_blockage  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    deposition_gate, gt_flow_gate_series, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def auc(score, y):
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=["wound_patient003"])
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
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt",
                                         sample=S)["series"][T - 1]

        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
        _, which = cKDTree(pos[solid]).query(pos)
        owner = np.flatnonzero(solid)[which]
        A = adjacency(ei_np, n)
        hop_s = hop_distance(solid, A, max_h=20)

        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()

        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        w = bundle["wound"]
        gp, gq = float(w["g_pre"]), float(w["g_post"])

        print(f"[i] {stem}: recomputing the GT gate at all {T} timesteps ...", flush=True)
        gser = gt_flow_gate_series(data, bio, hops=3, wall=wall)

        def oracle_blockage(mat, gate0, step):
            g = np.asarray(gate0, dtype=np.float64).copy()
            i = int(np.clip(step, 0, gser.shape[0] - 1))
            g[wall] = gser[i][wall]
            return g

        def saturated(level):
            """Every wall node gated at ``level`` for all time -- not a flow field.

            The gate enters the ODE only as the multiplier ``g``, and a total stall is
            exactly ``sr -> 0``, i.e. the stagnation branch pinned at 1.  So ``level=1``
            upper-bounds EVERY blockage that works by stalling the flow, and ``level=10``
            upper-bounds every blockage of any kind: no shear field can reach it
            (``gate_from_shear`` maxes near 2.5 on this cohort).
            """
            def blk(mat, gate0, step):
                g = np.asarray(gate0, dtype=np.float64).copy()
                g[wall] = float(level)
                return g
            return blk

        hook = make_rollout_hook(SHIPPED, bio, f.sr)
        gate = deposition_gate(data, f, wall=wall, wound_source=True)
        arms = {
            "frozen t=0 gate (shipped)": None,
            "GT-evolving gate (ORACLE)": oracle_blockage,
            "GT gate + fitted wound rate": wound_rate_blockage(
                data, bio, g_pre=gp, g_post=gq, inner=oracle_blockage),
            "wall gate == 1 (total stall)": saturated(1.0),
            "wall gate == 10 (unphysical)": saturated(10.0),
        }
        print("=" * 100)
        print(f"{stem}  lumen GT+={int((gt & off).sum())}  wall GT+={int((gt & wall).sum())}")
        print(f"  {'arm':30s} {'wallp90':>8s} {'wndp50':>8s} {'ign':>5s} {'AUC':>7s} | "
              f"{'k':>2s} {'TP':>4s} {'FP':>5s} {'off':>7s}")
        print(f"  {'GT Mat (reference)':30s} "
              f"{np.percentile(gmat[-1][wall] / crit, 90):8.2f} "
              f"{np.median(gmat[-1][wnd] / crit):8.2f} "
              f"{int((gmat[-1] >= crit).sum()):5d} "
              f"{auc((gmat[-1][owner] / crit)[off], gt[off]):7.4f} |")
        for tag, blk in arms.items():
            traj, _ = integrate_mat_trajectory(data, bio, gate, da_scale=SHIPPED_DA_SCALE,
                                               ap_closure=hook, blockage=blk)
            traj = np.asarray(traj)
            ow = traj[-1][owner] / crit
            best = max(((domain_score(
                (pred & ~off) | (off & (hop_s <= k) & (ow >= 1 / 0.16)),
                gt, ei, off, solid), k) for k in (2, 4, 6)))
            k = best[1]
            m = (pred & ~off) | (off & (hop_s <= k) & (ow >= 1 / 0.16))
            print(f"  {tag:30s} {np.percentile(traj[-1][wall] / crit, 90):8.2f} "
                  f"{np.median(traj[-1][wnd] / crit):8.2f} "
                  f"{int(((traj[-1] >= crit) & wall).sum()):5d} "
                  f"{auc(ow[off], gt[off]):7.4f} | {k:2d} "
                  f"{int((m & gt & off).sum()):4d} {int((m & ~gt & off).sum()):5d} "
                  f"{best[0]:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
