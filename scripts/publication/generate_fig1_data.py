"""Generate Figure 1 data (RGP-DEQ vs FEM vs GT)."""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR
from scripts.publication.utils import get_pack_path
from src.clot_ml.v0 import solve_fem_into_pack

def main():
    print("[i] Generating Data for Figure 1: Flow Comparisons")
    
    records = []
    
    for stem in CONFIG.fig1_vessels:
        print(f"  -> Processing {stem}...")
        pack_path = get_pack_path(stem)
        
        # 1. Load pack with RGP-DEQ flow (cached in u0_pred, v0_pred)
        data = torch.load(pack_path, map_location="cpu", weights_only=False)
        data.graph_stem = stem
        
        # Ground Truth at t=0
        # Channels: 0:u, 1:v, 2:p, 3:mu, 4:shear (assuming standard encoding)
        u_gt = data.y[0, :, 0].numpy()
        v_gt = data.y[0, :, 1].numpy()
        mag_gt = np.sqrt(u_gt**2 + v_gt**2)
        
        if getattr(data, "u0_pred", None) is not None:
            u_rgp = data.u0_pred.numpy()
            v_rgp = data.v0_pred.numpy()
        else:
            print(f"  [WARN] {stem} has no u0_pred (RGP-DEQ output). Skipping RGP.")
            u_rgp = np.zeros_like(u_gt)
            v_rgp = np.zeros_like(v_gt)
            
        mag_rgp = np.sqrt(u_rgp**2 + v_rgp**2)
        err_rgp = np.abs(mag_rgp - mag_gt)
        
        # 2. Solve FEM
        print(f"     Solving FEM for {stem}...")
        data_fem = torch.load(pack_path, map_location="cpu", weights_only=False)
        data_fem.graph_stem = stem
        solve_fem_into_pack(data_fem)
        
        u_fem = data_fem.u0_pred.numpy()
        v_fem = data_fem.v0_pred.numpy()
        mag_fem = np.sqrt(u_fem**2 + v_fem**2)
        err_fem = np.abs(mag_fem - mag_gt)
        
        # Wall mask: pull from biochem sample if possible
        wall = None
        try:
            from src.config import BiochemConfig
            from src.clot_ml.locked import build_sample
            bio = BiochemConfig(phase="biochem")
            S = build_sample(data, bio, flow="pred", variant="v4")
            wall = np.asarray(S["wall"], dtype=bool)
        except Exception:
            pass

        out_dict = {
            "pos": data.x[:, 0:2].numpy(),
            "wall": wall,
            "u_gt": u_gt, "v_gt": v_gt,
            "u_rgp": u_rgp, "v_rgp": v_rgp,
            "u_fem": u_fem, "v_fem": v_fem,
        }
        torch.save(out_dict, DATA_DIR / f"fig1_{stem}_flow.pt")
        
        # Store metrics
        records.append({
            "vessel": stem,
            "mae_u_rgp": float(np.mean(np.abs(u_rgp - u_gt))),
            "mae_v_rgp": float(np.mean(np.abs(v_rgp - v_gt))),
            "mae_mag_rgp": float(np.mean(err_rgp)),
            "mae_u_fem": float(np.mean(np.abs(u_fem - u_gt))),
            "mae_v_fem": float(np.mean(np.abs(v_fem - v_gt))),
            "mae_mag_fem": float(np.mean(err_fem)),
        })
        
    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "fig1_metrics.csv", index=False)
    print(f"[OK] Saved Fig 1 metrics to {DATA_DIR / 'fig1_metrics.csv'}")

if __name__ == "__main__":
    main()
