import torch
import numpy as np
import sys
from pathlib import Path

# Add repo root to path
REPO = Path(__file__).resolve().parents[0]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.physics_wall_model import node_positions, t0_flow_fields, integrate_mat_trajectory, graded_gate
from src.core_physics.shear_redistribution import build_crosssection_operator, sdf_nd, make_blockage
from src.core_physics.thrombin_field import make_thrombin_solver, make_ap_boost
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.evaluation.clot_relaxed_metrics import compute_clot_relaxed_metrics, metrics_to_deploy_prefix, clot_score_from_deploy_dict
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index

def first_crossing(series_over_t, thresh):
    hot = series_over_t >= thresh
    idx = np.full(hot.shape[1], -1, dtype=int)
    for i in range(hot.shape[0]):
        newly = hot[i] & (idx == -1)
        idx[newly] = i
    return idx

d = torch.load("data/processed/graphs_biochem_anchors/patient044.pt", map_location="cpu", weights_only=False)
bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
wall = d.mask_wall.reshape(-1).bool().numpy()
pos = node_positions(d)
f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
g0 = graded_gate(f, bio, mode="hard") * wall
ts, _ = make_thrombin_solver(d, bio, pos, f.sr, wash_coef=0.0, wall=wall)
B = build_crosssection_operator(pos, sdf_nd(d), wall, radius_mult=0.30)
blk = make_blockage(f, bio, B, wall, every=5, feedback="wake", wake=8.0)
boost = make_ap_boost(ts, bio, gain=4.0, every=5)
traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=40.0, blockage=blk, ap_boost=boost)
crit = float(bio.viscosity_mat_crit)

idx = first_crossing(traj, crit)
pred = torch.tensor((((idx >= 0) & wall)).astype(np.float32))

t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
phi_gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
phi_gt = phi_gt * torch.tensor(wall.astype(np.float32))

mm = compute_clot_relaxed_metrics(pred, phi_gt, d.edge_index, wall_mask=torch.tensor(wall))
score = clot_score_from_deploy_dict(metrics_to_deploy_prefix(mm))

print(f"patient044 Deploy Score: {score:.4f}")
print(f"Metrics dict: {mm}")
print(f"GT count: {int(phi_gt.sum())}  |  Pred count: {int(pred.sum())}")
print(f"GT count: {int(phi_gt.sum())}  |  Pred count: {int(pred.sum())}")
