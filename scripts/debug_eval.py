import torch
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.locked import load_temporal_v4_wound, predict_temporal_v4_wound
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_v0
from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner
from scripts.go_mat_field_v6 import solid_shells

def score_vessel(bundle_v5, bundle_v0, stem, PACKS):
    data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    TIME_GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    times = TIME_GRID * float(data.t[-1].numpy())
    
    print("  Evaluating V5w...", flush=True)
    comp_v5 = predict_temporal_v4_wound(bundle_v5, data, times)
    mask_v5 = comp_v5["mask"]
    
    print("  Evaluating V0...", flush=True)
    comp_v0 = predict_clot_ml_v0(bundle_v0, data, times)
    mask_v0 = comp_v0["mask"]
    
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.config import PhysicsConfig
    phys = PhysicsConfig(phase="biochem")
    T = data.y.shape[0]
    gt_mask = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    
    mask_wound = data.mask_wound.reshape(-1).bool().numpy() if hasattr(data, "mask_wound") else np.zeros(data.num_nodes, dtype=bool)
    mask_wall = data.mask_wall.reshape(-1).bool().numpy()
    solid = mask_wall | mask_wound
    
    v5_wall_f1 = f1_score(gt_mask[solid], mask_v5[solid], zero_division=1.0)
    v0_wall_f1 = f1_score(gt_mask[solid], mask_v0[solid], zero_division=1.0)
    
    pos = data.x[:, :2].numpy().astype(np.float64)
    ei_np = data.edge_index.numpy()
    
    shell1 = first_corner_shell(pos, solid, ei_np)
    town = topological_owner(pos, solid, ei_np)
    shells, owner = solid_shells(dict(solid=solid, edge_index=ei_np, shell=shell1, pos=pos, town=town), 10)
    
    off = ~solid
    cand = shells[0] & off & (town >= 0)
    
    if cand.any():
        v5_off_f1 = f1_score(gt_mask[cand], mask_v5[cand], zero_division=1.0)
        v0_off_f1 = f1_score(gt_mask[cand], mask_v0[cand], zero_division=1.0)
    else:
        v5_off_f1 = 0.0
        v0_off_f1 = 0.0
        
    return {
        "v5_wall": v5_wall_f1,
        "v0_wall": v0_wall_f1,
        "v5_off": v5_off_f1,
        "v0_off": v0_off_f1,
        "off_gt": gt_mask[cand].sum()
    }

def main():
    print("Loading bundles...", flush=True)
    bundle_v5 = load_temporal_v4_wound("clot_gnn_v5w")
    bundle_v0 = load_v0_bundle("clot_ml_v0")
    
    PACKS = REPO / "data" / "processed" / "graphs_biochem_anchors"
    
    stem = "wound_patient003"
    print(f"Scoring {stem}...", flush=True)
    res = score_vessel(bundle_v5, bundle_v0, stem, PACKS)
    
    print(f"{'Vessel':<20} | {'V5w Wall F1':<12} | {'V0 Wall F1':<12} | {'V5w Off F1':<12} | {'V0 Off F1':<12} | {'Off GT+':<8}", flush=True)
    print("-" * 90, flush=True)
    print(f"{stem:<20} | {res['v5_wall']:<12.4f} | {res['v0_wall']:<12.4f} | {res['v5_off']:<12.4f} | {res['v0_off']:<12.4f} | {res['off_gt']:<8}", flush=True)

if __name__ == "__main__":
    main()
