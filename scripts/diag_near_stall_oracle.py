"""GT-oracle of the near-field stall kernel (step 1).

Feeds GT ``Mat >= crit`` as occupancy into :func:`make_near_stall_blockage` and asks
three things, in this order:

  1. Do ``wound_patient003``'s blind owners open?  (the prize)
  2. Do clot-free vessels pick up any extra wall gates?  (must be zero -- nothing gels)
  3. On 12 clot-carrying FIT vessels, of t=0-ungated wall, how many extra gates are
     true positives vs false positives at final GT clot?

This is a KERNEL ceiling, not a deploy number.  Occupancy is GT, flow is t=0.
If (1) fails, the stencil is wrong.  If (2) or (3) blows up, do not wire stall on.

    python scripts/diag_near_stall_oracle.py
    python scripts/diag_near_stall_oracle.py --hops 2 4
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

from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.near_stall import STALL_HOPS, make_near_stall_blockage  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, FIT  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
COHORT_N = 12


def _load(stem: str):
    p = PACKS / f"{stem}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def _gt_mat(data, bio):
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    return mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(
        int(data.y.shape[0]), -1).numpy()


def _blinds(data, bio, phys):
    """Healthy-wall owners of wound-region lumen GT clot whose t=0 gate is 0."""
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid, wnd = solid_mask(data), wound_mask(data)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    pos = data.x[:, :2].numpy().astype(np.float64)
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    _, lumen, _ = wound_region_masks(data)
    ow = np.unique(owner[(gt & lumen) & ~wnd[owner]])
    f0 = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    return ow[f0.gate[ow] == 0], f0, wall, solid


def _open_count(blk, mat, gate0, wall, sel=None):
    g = blk(mat, gate0, 0)
    opened = wall & (np.asarray(gate0) <= 0) & (np.asarray(g) > 0)
    if sel is not None:
        sel = np.asarray(sel)
        return int(opened[sel].sum()), int(sel.size), g
    return int(opened.sum()), int((wall & (np.asarray(gate0) <= 0)).sum()), g


def _mat_on(gtm_step, crit, keep):
    return np.where((gtm_step >= crit) & keep, 2.0 * crit, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", nargs="*", type=int, default=[2, STALL_HOPS])
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    hops_list = [int(h) for h in args.hops]
    print(f"[i] near-stall GT-oracle  hops={hops_list}  amp={0.1226}  "
          f"(GELLED_SR_RATIO)  occupancy = GT Mat >= crit on solid\n")

    # --- 003 blinds ----------------------------------------------------------
    data = _load("wound_patient003")
    if data is None:
        print("[ERR] wound_patient003 pack missing")
        return 1
    blinds, f0, wall, solid = _blinds(data, bio, phys)
    gtm = _gt_mat(data, bio)
    crit = float(bio.viscosity_mat_crit)
    T = gtm.shape[0]
    gate0 = f0.gate * wall
    wnd = wound_mask(data)
    t0_open = wall & (gate0 > 0)
    seeds = {
        "wound": wnd,
        "wound+t0": wnd | t0_open,
        "all_solid": solid,
    }
    print(f"wound_patient003  blinds={blinds.size}  t0 sr med "
          f"{float(np.median(f0.sr[blinds])):.1f} /s  lss={float(bio.lss)}")
    print("  occupancy: wound = gelled wound only (no GT-wall leak); "
          "wound+t0 = plus t=0-gated wall; all_solid = any gelled solid")
    print(f"  {'seed':10s}  {'hops':>4s}  {'step':>5s}  {'n_occ':>5s}  "
          f"{'blinds_open':>11s}  {'gate_med':>8s}")
    for seed_name, keep in seeds.items():
        for hops in hops_list:
            blk = make_near_stall_blockage(data, bio, f0, wall=wall, solid=solid, hops=hops)
            for step in (2, 5, 10, 20, 40, T - 1):
                mat = _mat_on(gtm[min(step, T - 1)], crit, keep)
                n_occ = int(((mat >= crit) & solid).sum())
                n_open, n_b, g = _open_count(blk, mat, gate0, wall, sel=blinds)
                print(f"  {seed_name:10s}  {hops:4d}  {step:5d}  {n_occ:5d}  "
                      f"{n_open:2d}/{n_b:<2d}         "
                      f"{float(np.median(g[blinds])):8.3f}")
    print()

    # --- clot-free -----------------------------------------------------------
    print("clot-free (must stay 0 extra gates -- GT occupancy is empty)")
    n_free = 0
    extra_free = {h: 0 for h in hops_list}
    for stem in CLOT_FREE:
        d = _load(stem)
        if d is None:
            continue
        n_free += 1
        wl = d.mask_wall.reshape(-1).bool().cpu().numpy()
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        g0 = f.gate * wl
        gtm_f = _gt_mat(d, bio)
        mat = np.where(gtm_f[-1] >= crit, 2.0 * crit, 0.0)
        for hops in hops_list:
            blk = make_near_stall_blockage(d, bio, f, wall=wl, hops=hops)
            n_open, _, _ = _open_count(blk, mat, g0, wl)
            extra_free[hops] += n_open
            if n_open:
                print(f"  [WARN] {stem} hops={hops} opened {n_open} previously-ungated wall nodes")
    for hops in hops_list:
        print(f"  hops={hops}  vessels={n_free}  extra ungated->gated wall nodes {extra_free[hops]}")
    print()

    # --- 12 FIT clot-carrying vessels ----------------------------------------
    stems = [s for s in FIT if _load(s) is not None][:COHORT_N]
    print(f"FIT clot-carrying (n={len(stems)}): extra gates on t=0-ungated wall, "
          f"vs final GT clot")
    print(f"  {'hops':>4s}  {'stem':22s}  {'TP':>5s} {'FP':>5s} {'FN':>5s}  "
          f"{'ungated':>7s}  {'GT+':>5s}")
    tot = {h: {"tp": 0, "fp": 0, "fn": 0, "u": 0, "gt": 0} for h in hops_list}
    for stem in stems:
        d = _load(stem)
        wl = d.mask_wall.reshape(-1).bool().cpu().numpy()
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        g0 = f.gate * wl
        ungated = wl & (g0 <= 0)
        Tloc = int(d.y.shape[0])
        gt_fin = gt_clot_phi_at_time(d, Tloc - 1, phys).numpy() > 0.5
        gtm_c = _gt_mat(d, bio)
        mat = np.where(gtm_c[-1] >= crit, 2.0 * crit, 0.0)
        for hops in hops_list:
            blk = make_near_stall_blockage(d, bio, f, wall=wl, hops=hops)
            _, _, g = _open_count(blk, mat, g0, wl)
            opened = ungated & (g > 0)
            tp = int((opened & gt_fin).sum())
            fp = int((opened & ~gt_fin).sum())
            fn = int((ungated & gt_fin & ~opened).sum())
            tot[hops]["tp"] += tp
            tot[hops]["fp"] += fp
            tot[hops]["fn"] += fn
            tot[hops]["u"] += int(ungated.sum())
            tot[hops]["gt"] += int((ungated & gt_fin).sum())
            print(f"  {hops:4d}  {stem:22s}  {tp:5d} {fp:5d} {fn:5d}  "
                  f"{int(ungated.sum()):7d}  {int((ungated & gt_fin).sum()):5d}")
    print()
    for hops in hops_list:
        t = tot[hops]
        print(f"  SUM hops={hops}  TP {t['tp']}  FP {t['fp']}  FN {t['fn']}  "
              f"ungated wall {t['u']}  of which GT+ {t['gt']}")
    print("\n[i] Snapshot oracle only.  FIT FN cannot march; FIT FP is the halo at "
          "final occupancy.  Verdict is in the tables, not a pass line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
