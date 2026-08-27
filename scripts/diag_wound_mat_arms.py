"""Which ODE arm gets ``Mat`` right enough for the off-wall magnitude rule to fire?

`diag_wound_mat_magnitude.py` located the bottleneck exactly.  The off-wall rule
``Mat_owner >= crit / 0.16`` is CORRECT -- applied to COMSOL's own ``Mat`` it scores
**0.9460 / 0.9411 / 0.7909** on the off domain of the three wound vessels at one shared
depth, every one of them past the 0.75 target.  Applied to the shipped ODE's ``Mat`` it
scores 0.5200 / 0.6911 / 0.4538, because the ODE under-produces:

    Mat/crit, final     ODE wall p90   GT wall p90   ODE wound p50   GT wound p50
    wound_patient001         1.19          1.90           1.35            9.04
    wound_patient002         0.73          1.02           1.40            8.70
    wound_patient003         1.73         19.45           2.31          103.84

and its ORDERING degrades with it: ``Mat_owner`` AUC over 003's lumen is 0.6702 against GT's
0.8445.

So this script asks which `blockage` arm -- the handoff's flow->gate coupling -- recovers the
magnitude and the ordering, and adds the one source term the ODE is still missing: the wound
runs at the complement's own FITTED two-regime rate rather than at the static prefactor 1
that WOUND_PROGRESS 14.6 installed.

    python scripts/diag_wound_mat_arms.py
"""
from __future__ import annotations

import argparse
import json
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
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.gelation_wake import make_gelation_wake_blockage  # noqa: E402
from src.core_physics.near_stall import make_near_stall_blockage  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    deposition_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
LOVO = REPO / "outputs/clot_ml/wound_rate/lovo.json"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score, y):
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def wound_rate(stem):
    """The complement's own fitted (G_pre, G_post), leave-one-vessel-out."""
    if not LOVO.exists():
        return 2.0, 10.0
    blob = json.loads(LOVO.read_text())
    f = (blob.get("folds") or {}).get(stem)
    if f:
        return float(f["g_pre"]), float(f["g_post"])
    fa = blob.get("fitted_all", {})
    return float(fa.get("g_pre", 2.0)), float(fa.get("g_post", 10.0))


def run_ode(data, bio, f, wall, *, blk=None, wound_pre=1.0, wound_post=None):
    """The surface ODE, optionally with a two-regime prefactor on the wound.

    ``wound_post`` switches the injured patch from the static ``srf2`` prefactor to the
    complement's fitted post-gelation rate once its own ``Mat`` crosses ``crit`` -- the same
    two-regime law `src/clot_ml/wound.py` integrates, applied inside the shared ODE so every
    downstream ``Mat`` consumer sees it.
    """
    crit = float(bio.viscosity_mat_crit)
    hook = make_rollout_hook(SHIPPED, bio, f.sr)
    gate = deposition_gate(data, f, wall=wall, wound_source=True, prefactor=wound_pre)
    wnd = wound_mask(data)
    if wound_post is not None and wnd.any():
        inner = blk

        def blk2(mat, gate0, step):
            g = gate0 if inner is None else np.asarray(inner(mat, gate0, step), np.float64)
            g = g.copy()
            hot = wnd & (np.asarray(mat) >= crit)
            g[wnd] = wound_pre
            g[hot] = float(wound_post)
            return g
        blk = blk2
    traj, _ = integrate_mat_trajectory(data, bio, gate, da_scale=SHIPPED_DA_SCALE,
                                       ap_closure=hook, blockage=blk)
    return np.asarray(traj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    ap.add_argument("--att", type=float, default=0.16)
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
        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        g_pre, g_post = wound_rate(stem)

        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()

        arms = {
            "shipped (prefactor 1)": dict(),
            "wound two-regime": dict(wound_pre=g_pre, wound_post=g_post),
            "stall h=2": dict(blk="stall2"),
            "stall h=2 + two-regime": dict(blk="stall2", wound_pre=g_pre, wound_post=g_post),
            "stall h=4 + two-regime": dict(blk="stall4", wound_pre=g_pre, wound_post=g_post),
            "wake + two-regime": dict(blk="wake", wound_pre=g_pre, wound_post=g_post),
        }

        print("=" * 108)
        print(f"{stem}  lumen GT+={int((gt & off).sum())}  "
              f"G_pre={g_pre:.2f} G_post={g_post:.2f}  shipped off="
              f"{domain_score(pred, gt, ei, off, solid):.4f}")
        print(f"  {'arm':26s} {'wallp90':>8s} {'wndp50':>8s} {'AUC':>7s} | "
              f"rule at att={args.att}: {'k':>2s} {'TP':>4s} {'FP':>5s} {'off':>7s} {'wall':>7s}")

        gow = gmat[-1][owner] / crit
        print(f"  {'GT Mat (reference)':26s} "
              f"{np.percentile(gmat[-1][wall] / crit, 90):8.2f} "
              f"{np.median(gmat[-1][wnd] / crit) if wnd.any() else float('nan'):8.2f} "
              f"{auc(gow[off], gt[off]):7.4f} |", end="")
        best = max(((domain_score((pred & ~off) | (off & (hop_s <= k) & (gow >= 1 / args.att)),
                                  gt, ei, off, solid), k) for k in (2, 4, 6)))
        k = best[1]
        m = (pred & ~off) | (off & (hop_s <= k) & (gow >= 1 / args.att))
        print(f"                  {k:2d} {int((m & gt & off).sum()):4d} "
              f"{int((m & ~gt & off).sum()):5d} {best[0]:7.4f} "
              f"{domain_score(m, gt, ei, wall, solid):7.4f}")

        for tag, kw in arms.items():
            blkname = kw.pop("blk", None)
            blk = None
            if blkname == "stall2":
                blk = make_near_stall_blockage(data, bio, f, wall=wall, hops=2)
            elif blkname == "stall4":
                blk = make_near_stall_blockage(data, bio, f, wall=wall, hops=4)
            elif blkname == "wake":
                blk = make_gelation_wake_blockage(data, bio, f, wall)
            traj = run_ode(data, bio, f, wall, blk=blk, **kw)
            ow = traj[-1][owner] / crit
            best = max(((domain_score(
                (pred & ~off) | (off & (hop_s <= k) & (ow >= 1 / args.att)),
                gt, ei, off, solid), k) for k in (2, 4, 6)))
            k = best[1]
            m = (pred & ~off) | (off & (hop_s <= k) & (ow >= 1 / args.att))
            print(f"  {tag:26s} {np.percentile(traj[-1][wall] / crit, 90):8.2f} "
                  f"{np.median(traj[-1][wnd] / crit) if wnd.any() else float('nan'):8.2f} "
                  f"{auc(ow[off], gt[off]):7.4f} |                  {k:2d} "
                  f"{int((m & gt & off).sum()):4d} {int((m & ~gt & off).sum()):5d} "
                  f"{best[0]:7.4f} {domain_score(m, gt, ei, wall, solid):7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
