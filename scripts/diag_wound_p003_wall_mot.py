"""Finish the diagnosis the previous session started on wound_patient003's WALL clock.

Headline to reproduce (from that session, on clot_gnn_v5w):

    MOT 0.5095  against  FINAL 0.8754   (cohort MOT-final gap is ~0.05)
    even FROZEN scores ~0.78 there

Frozen beating the learned head means the clock is late, not that the set is wrong.
This script scores wall MOT / FINAL / frozen / oracle-timing (same committed set, GT
onset) on all three wound vessels, then dumps GT vs ODE vs head onset on 003, split
near-wound vs far wall.

    python scripts/diag_wound_p003_wall_mot.py
    python scripts/diag_wound_p003_wall_mot.py --stems wound_patient003
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
from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
NEAR_HOPS = 25  # same radius as the wound trigger; 003's gelled neighbours sit at 12-14


def gt_at(data, ti, phys) -> np.ndarray:
    return gt_clot_phi_at_time(data, int(ti), phys).numpy() > 0.5


def first_true(series_bool: np.ndarray, T: int) -> np.ndarray:
    """series_bool [T, N] -> first index, or T if never."""
    any_ = series_bool.any(0)
    return np.where(any_, series_bool.argmax(0), T)


def score_wall(pred, gt, ei, wall, solid):
    return domain_score(pred, gt, ei, wall, solid)


def mot(series, gts, times, ei, wall, solid):
    vals = [score_wall(series[ti], gts[ti], ei, wall, solid) for ti in times]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle = load_temporal_v4_wound(name=args.name)
    th_w = bundle["base"]["temporal"]["time_th_wall"]
    print(f"[i] artifact={args.name}  time_th_wall={th_w}  "
          f"commit_final={th_w[1]}  wake_ode={bundle['base']['temporal'].get('wake_ode')}\n")

    rows = []
    detail = None
    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        times = sorted(set(list(range(0, T, args.every)) + [T - 1]))
        ei = torch.tensor(data.edge_index.detach().cpu().numpy())
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        wnd = wound_mask(data)
        gts = {ti: gt_at(data, ti, phys) for ti in times}

        out = predict_temporal_v4_wound(bundle, data, times, flow="gt")
        gm = out["mask"]
        series = out["series"]
        frozen = {ti: gm for ti in times}

        # Oracle timing on the SAME set: TPs follow GT onset; FPs stay off until the
        # last step (commit_final), so the ceiling isolates timing from the mask.
        oracle = {}
        seen = np.zeros_like(gm)
        for ti in times:
            seen = seen | (gm & gts[ti])
            oracle[ti] = gm.copy() if ti == times[-1] else seen.copy()

        fin = score_wall(series[times[-1]], gts[times[-1]], ei, wall, solid)
        m_head = mot(series, gts, times, ei, wall, solid)
        m_frz = mot(frozen, gts, times, ei, wall, solid)
        m_ora = mot(oracle, gts, times, ei, wall, solid)
        rows.append((stem, T, int(wall.sum()), fin, m_head, m_frz, m_ora))
        print(f"{stem:22s} T={T:3d}  wall n={int(wall.sum()):4d}  "
              f"FINAL {fin:.4f}  MOT {m_head:.4f}  frozen {m_frz:.4f}  "
              f"oracleT {m_ora:.4f}  MOT-gap {fin - m_head:.4f}")

        if stem == "wound_patient003":
            detail = (data, T, times, wall, solid, wnd, ei, out, gm, gts, bio)

    print()
    if len(rows) > 1:
        print(f"{'COHORT':22s}            "
              f"FINAL {np.mean([r[3] for r in rows]):.4f}  "
              f"MOT {np.mean([r[4] for r in rows]):.4f}  "
              f"frozen {np.mean([r[5] for r in rows]):.4f}  "
              f"oracleT {np.mean([r[6] for r in rows]):.4f}")
        print("[i] cohort MOT-final gap on the 23-vessel pool is ~0.05; frozen there is 0.79 "
              "and the head BEATS it (0.87).  If frozen > MOT here, the clock is inverted.\n")

    if detail is None:
        return 0

    data, T, times, wall, solid, wnd, ei, out, gm, gts, bio = detail
    crit = float(bio.viscosity_mat_crit)
    traj, _ = ode_trajectory(data, bio, flow="gt")
    oon = np.where((traj >= crit).any(0), (traj >= crit).argmax(0), T)
    head_on = out["onset"].astype(int)
    head_on = np.where(head_on >= 0, head_on, T)

    gt_full = np.stack([gt_at(data, ti, phys) for ti in range(T)])
    gt_on = first_true(gt_full, T)

    h = hop_distance(wnd, adjacency(data.edge_index.numpy(), int(data.num_nodes)), max_h=40)
    near = wall & (h <= NEAR_HOPS)
    far = wall & ~near

    print("--- 003 WALL onset (index, % of T) ---")
    print(f"   committed wall nodes: {int((gm & wall).sum())} of {int(wall.sum())}  "
          f"GT+ at final: {int((gts[times[-1]] & wall).sum())}")

    def dump(name, sel):
        n = int(sel.sum())
        if n == 0:
            print(f"   {name}: empty")
            return
        for tag, arr in (("GT  ", gt_on), ("ODE ", oon), ("HEAD", head_on)):
            a = arr[sel]
            print(f"   {name:14s} {tag}  n={n:4d}  "
                  f"med {np.median(a):6.1f} ({100 * np.median(a) / T:5.1f}% T)  "
                  f"p10 {np.percentile(a, 10):6.1f}  p90 {np.percentile(a, 90):6.1f}  "
                  f"ignited {(a < T).mean():.2f}")

    dump("all wall", wall)
    dump("wall AND SET", wall & gm)
    dump("SET AND GT+", wall & gm & gts[times[-1]])
    dump("near-wnd SET", near & gm)
    dump("far SET", far & gm)
    dump("GT+ missed", wall & ~gm & gts[times[-1]])

    # lateness of the head vs GT on the nodes both agree are clot at the end
    both = wall & gm & gts[times[-1]]
    delay = (head_on[both] - gt_on[both]).astype(float)
    ode_delay = (oon[both] - gt_on[both]).astype(float)
    print("\n--- lateness on SET AND GT+ wall (head - GT, ODE - GT), in steps ---")
    print(f"   HEAD delay  med {np.median(delay):.1f}  p10 {np.percentile(delay, 10):.1f}  "
          f"p90 {np.percentile(delay, 90):.1f}  frac late {(delay > 0).mean():.2f}")
    print(f"   ODE  delay  med {np.median(ode_delay):.1f}  p10 {np.percentile(ode_delay, 10):.1f}  "
          f"p90 {np.percentile(ode_delay, 90):.1f}  frac late {(ode_delay > 0).mean():.2f}")
    print(f"   HEAD vs ODE (head-oon) med {np.median(head_on[both] - oon[both]):.1f}  "
          f"-- >0 means the head is LATER than the ODE it is anchored on")

    # t=0 gate on SET AND GT+ vs missed
    from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: PLC0415
    f0 = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    print("\n--- t=0 gate / sr on those same groups ---")
    for name, sel in (("SET AND GT+", both), ("GT+ missed", wall & ~gm & gts[times[-1]]),
                      ("near-wnd SET", near & gm), ("far SET", far & gm)):
        if not sel.any():
            continue
        print(f"   {name:14s}  gate med {np.median(f0.gate[sel]):.3f}  "
              f"frac>0 {(f0.gate[sel] > 0).mean():.2f}  "
              f"sr med {np.median(f0.sr[sel]):.1f} /s  "
              f"GT onset med {np.median(gt_on[sel]):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
