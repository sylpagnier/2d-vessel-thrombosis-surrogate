"""ODE + deploy oracle for the hop-1 B-march (the 003 deficiency).

Answers, in order, whether we can take a concrete step:

  1. Does a SELF-CONSISTENT stall ODE ignite the 003 blinds (and when)?
  2. Which stencil (hops=1 vs 2, mu1 vs median, dsrx on vs off) is the one that
     marches without painting a disk?
  3. Are those blinds already in the shipped GNN committed set ``gm``?
  4. If we OR stall-only wall ignitions into the shipped series, what happens to
     003 wall/lumen AND to 012 / clot-free FP?

Illegal as a model: GT t=0 flow.  Decisive as a ceiling on this operator.

    python scripts/diag_stall_deploy_oracle.py
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

from src.clot_ml.evaluate import f1  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    _committed_set_v4, build_sample, load_temporal_v4_wound, predict_scores,
    predict_temporal_v4_wound,
)
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.gelation_wake import GELLED_SR_RATIO  # noqa: E402
from src.core_physics.near_stall import make_near_stall_blockage  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    deposition_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = (
    "wound_patient003",
    "wound_patient001",
    "wound_patient002",
    "patient012",
    "patient017",
)


def _load(stem):
    p = PACKS / f"{stem}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def _gt_hot(data, phys):
    T = int(data.y.shape[0])
    return gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5, T


def _blinds(data, f0, phys):
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid, wnd = solid_mask(data), wound_mask(data)
    gt, _ = _gt_hot(data, phys)
    pos = data.x[:, :2].numpy().astype(np.float64)
    if not solid.any():
        return np.zeros(0, dtype=int), wall, solid, wnd
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    if wnd.any():
        _, lumen, _ = wound_region_masks(data)
        ow = np.unique(owner[(gt & lumen) & ~wnd[owner]])
    else:
        ow = np.unique(owner[gt & ~solid])
        ow = ow[wall[ow]]
    g0 = f0.gate * wall
    return ow[g0[ow] <= 0], wall, solid, wnd, owner


def _ode(data, bio, *, blk=None, wound_source=True):
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    hook = make_rollout_hook(SHIPPED, bio, f.sr)
    gate = deposition_gate(data, f, wall=wall, wound_source=wound_source)
    traj, t = integrate_mat_trajectory(
        data, bio, gate, da_scale=SHIPPED_DA_SCALE, ap_closure=hook, blockage=blk)
    return np.asarray(traj), np.asarray(t).reshape(-1), f, wall


def _first(traj, crit, sel):
    hot = traj[:, sel] >= crit
    T = traj.shape[0]
    return np.where(hot.any(0), hot.argmax(0), T)


def _prf(pred, gt, dom):
    p, g = pred & dom, gt & dom
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return tp, fp, fn, prec, rec, f1(p, g)


def _arms(data, bio, f, wall):
    return {
        "frozen": None,
        "h1_mu1_B": make_near_stall_blockage(
            data, bio, f, wall=wall, hops=1, scale_dsrx=False),
        "h1_mu1_both": make_near_stall_blockage(
            data, bio, f, wall=wall, hops=1, scale_dsrx=True),
        "h1_med_B": make_near_stall_blockage(
            data, bio, f, wall=wall, hops=1, scale_dsrx=False, sr_ratio=0.45),
        "h2_mu1_B": make_near_stall_blockage(
            data, bio, f, wall=wall, hops=2, scale_dsrx=False),
    }


def _run_vessel(stem, data, bio, phys, bundle, crit):
    print("=" * 96)
    print(stem)
    gt, T = _gt_hot(data, phys)
    traj0, tgrid, f0, wall = _ode(data, bio, blk=None)
    blinds, wall, solid, wnd, owner = _blinds(data, f0, phys)
    off = ~solid
    print(f"  T={T}  wall={int(wall.sum())}  wound={int(wnd.sum())}  "
          f"blinds={blinds.size}  GT+ wall {int((gt & wall).sum())}  "
          f"GT+ off {int((gt & off).sum())}")

    # --- ODE arms -----------------------------------------------------------
    print(f"  {'arm':14s}  {'ignW':>5s}  {'dIgn':>5s}  {'W-FP':>5s}  "
          f"{'bl_n':>4s}  {'bl_med_t':>8s}  {'extra_B_never':>14s}")
    odes = {"frozen": traj0}
    ign0 = (traj0[-1] >= crit) & wall
    results = {}
    for name, blk in _arms(data, bio, f0, wall).items():
        traj = traj0 if name == "frozen" else _ode(data, bio, blk=blk)[0]
        odes[name] = traj
        ign = (traj[-1] >= crit) & wall
        extra = ign & ~ign0
        fp = extra & ~gt
        never = extra & ~gt
        n_bl = int((traj[-1, blinds] >= crit).sum()) if blinds.size else 0
        t_bl = _first(traj, crit, blinds) if blinds.size else np.zeros(0, int)
        opened = t_bl < T
        med_t = float(np.median(t_bl[opened])) if opened.any() else float("nan")
        print(f"  {name:14s}  {int(ign.sum()):5d}  {int(extra.sum()):5d}  "
              f"{int(fp.sum()):5d}  {n_bl:2d}/{blinds.size:<2d}  "
              f"{med_t:8.1f}  {int(never.sum()):14d}")
        results[name] = dict(traj=traj, ign=ign, extra=extra, t_bl=t_bl, n_bl=n_bl)

    if blinds.size:
        print("  blinds first-crossing (T=never)  frozen vs h1_mu1_B vs h2_mu1_B")
        tF, t1, t2 = results["frozen"]["t_bl"], results["h1_mu1_B"]["t_bl"], results["h2_mu1_B"]["t_bl"]
        print(f"    {'i':>3s}  {'frz':>5s}  {'h1':>5s}  {'h2':>5s}")
        for i in range(blinds.size):
            print(f"    {i:3d}  {tF[i]:5d}  {t1[i]:5d}  {t2[i]:5d}")

    # --- shipped GNN set ----------------------------------------------------
    times = [0, T - 1]
    out = predict_temporal_v4_wound(bundle, data, times, flow="gt")
    pred = out["series"][T - 1]
    S = build_sample(data, bio, flow="gt", variant="v4")
    sc = predict_scores(bundle["base"]["ens"], S)
    temporal = bundle["base"]["temporal"]
    if bundle.get("wound", {}).get("readout") and wnd.any():
        temporal = dict(temporal, wound_spec=dict(bundle["wound"]["readout"]))
    gm = _committed_set_v4(S, sc, temporal)

    print("\n  shipped v5w FINAL vs GT")
    for name, dom in (("wall", wall), ("off", off),
                      ("w_lum", wound_region_masks(data)[1] if wnd.any() else np.zeros_like(wall))):
        if not dom.any() and name != "wall":
            continue
        tp, fp, fn, prec, rec, sc_ = _prf(pred, gt, dom)
        print(f"    {name:6s}  tp {tp:4d} fp {fp:4d} fn {fn:4d}  "
              f"prec {prec:.3f} rec {rec:.3f}  f1 {sc_:.3f}")

    if blinds.size:
        in_gm = gm[blinds]
        in_pred = pred[blinds]
        print(f"  blinds in gm {int(in_gm.sum())}/{blinds.size}  "
              f"in FINAL pred {int(in_pred.sum())}/{blinds.size}  "
              f"score med {float(np.median(sc[blinds])):.4f}  "
              f"score max {float(np.max(sc[blinds])):.4f}")
        # lumen owned by blinds
        lum_of = off & np.isin(owner, blinds) & gt
        print(f"  GT lumen owned by blinds: {int(lum_of.sum())}  "
              f"in gm {int((lum_of & gm).sum())}  "
              f"in pred {int((lum_of & pred).sum())}  "
              f"owner-blocked (in gm, owner not in gm) "
              f"{int((lum_of & gm & ~gm[owner]).sum())}")

    # missed off-wall: in gm but owner missing vs not in gm
    miss_off = off & gt & ~pred
    print(f"  missed off GT+: {int(miss_off.sum())}  in gm {int((miss_off & gm).sum())}  "
          f"owner-blocked {int((miss_off & gm & ~gm[owner]).sum())}  "
          f"owner in gm {int((miss_off & gm[owner]).sum())}")

    # --- union / OR of stall-only wall ignitions into shipped FINAL ---------
    print("\n  deploy-proxy FINAL (OR stall-only wall extra into shipped series, then owner)")
    print(f"  {'arm':14s}  {'W rec':>6s}  {'W fp':>5s}  {'off rec':>7s}  {'off fp':>6s}  "
          f"{'bl':>5s}  {'lum_bl':>6s}")
    for name in ("frozen", "h1_mu1_B", "h1_med_B", "h2_mu1_B"):
        extra_w = results[name]["extra"]  # stall ignitions not in frozen ODE
        # OR into shipped final, then owner precedence (same as series_masks)
        m = pred | extra_w
        m = m & (wall | m[owner])
        tw, fw, fnw, _, recw, _ = _prf(m, gt, wall)
        to, fo, fno, _, reco, _ = _prf(m, gt, off)
        n_bl = int(m[blinds].sum()) if blinds.size else 0
        lum_bl = 0
        if blinds.size:
            lum_of = off & np.isin(owner, blinds) & gt
            lum_bl = int((m & lum_of).sum())
        mark = ""
        if name == "frozen":
            mark = " (control: extra empty)"
        print(f"  {name:14s}  {recw:6.3f}  {fw:5d}  {reco:7.3f}  {fo:6d}  "
              f"{n_bl:2d}/{blinds.size:<2d}  {lum_bl:6d}{mark}")

    # gm union stall ignitions (commit_final style): who the GNN would have IF blinds enter gm
    print("\n  gm | stall_ign_wall  then owner-filter (would-be committed SET, no time head)")
    print(f"  {'arm':14s}  {'W rec':>6s}  {'W fp':>5s}  {'off rec':>7s}  {'off fp':>6s}")
    for name in ("frozen", "h1_mu1_B", "h2_mu1_B"):
        gm2 = gm | results[name]["ign"]
        m = gm2 & (wall | gm2[owner])
        _, fw, _, _, recw, _ = _prf(m, gt, wall)
        _, fo, _, _, reco, _ = _prf(m, gt, off)
        print(f"  {name:14s}  {recw:6.3f}  {fw:5d}  {reco:7.3f}  {fo:6d}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    print(f"[i] stall deploy oracle  artifact={args.name}  "
          f"GELLED_SR_RATIO={GELLED_SR_RATIO}  flow=gt\n")
    bundle = load_temporal_v4_wound(name=args.name)
    for stem in args.stems:
        d = _load(stem)
        if d is None:
            print(f"[miss] {stem}")
            continue
        _run_vessel(stem, d, bio, phys, bundle, crit)
        print()
    print("[i] Read: h1_mu1_B is the designed operator.  If blinds stay never on that")
    print("    arm, do not union.  If they open and clot-free extra_B_never stays 0,")
    print("    the concrete step is OR stall-only wall ignitions into the shipped series.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
