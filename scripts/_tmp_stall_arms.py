"""One-shot: stall arm extras on wound 001/002/003. Delete after."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
from src.core_physics.near_stall import make_near_stall_blockage
from src.core_physics.physics_wall_model import deposition_gate, integrate_mat_trajectory, t0_flow_fields
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
crit = float(bio.viscosity_mat_crit)
PACKS = Path("data/processed/graphs_biochem_anchors")


def load(stem):
    return torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)


def ode(data, blk=None):
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    hook = make_rollout_hook(SHIPPED, bio, f.sr)
    gate = deposition_gate(data, f, wall=wall, wound_source=True)
    traj, t = integrate_mat_trajectory(
        data, bio, gate, da_scale=SHIPPED_DA_SCALE, ap_closure=hook, blockage=blk)
    return np.asarray(traj), f, wall


def blinds(data, f0, wall):
    solid, wnd = solid_mask(data), wound_mask(data)
    T = int(data.y.shape[0])
    gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    pos = data.x[:, :2].numpy().astype(np.float64)
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    if wnd.any():
        _, lumen, _ = wound_region_masks(data)
        ow = np.unique(owner[(gt & lumen) & ~wnd[owner]])
    else:
        ow = np.unique(owner[gt & ~solid])
        ow = ow[wall[ow]]
    g0 = f0.gate * wall
    return ow[g0[ow] <= 0], gt, wnd


arms = [
    ("h1_seed0", dict(hops=1, seed_wound=False, scale_dsrx=False)),
    ("h1_seed1", dict(hops=1, seed_wound=True, scale_dsrx=False)),
    ("h2_seed0", dict(hops=2, seed_wound=False, scale_dsrx=False)),
    ("h2_seed1", dict(hops=2, seed_wound=True, scale_dsrx=False)),
]
for stem in ("wound_patient001", "wound_patient002", "wound_patient003"):
    d = load(stem)
    traj0, f0, wall = ode(d)
    ign0 = (traj0[-1] >= crit) & wall
    bl, gt, wnd = blinds(d, f0, wall)
    ung = wall & (np.asarray(f0.gate) * wall <= 0)
    print("====", stem, "T", traj0.shape[0], "blinds", bl.size,
          "GT+wall", int((gt & wall).sum()))
    print("  arm        extra ungTP ungFP ignW  bl")
    for name, kw in arms:
        blk = make_near_stall_blockage(d, bio, f0, wall=wall, **kw)
        traj, _, _ = ode(d, blk=blk)
        ign = (traj[-1] >= crit) & wall
        extra = ign & ~ign0
        eung = extra & ung
        nbl = int((traj[-1, bl] >= crit).sum()) if bl.size else 0
        print(f"  {name:10s} {int(extra.sum()):5d} {int((eung & gt).sum()):5d} "
              f"{int((eung & ~gt).sum()):5d} {int(ign.sum()):5d} {nbl:2d}/{bl.size}")
