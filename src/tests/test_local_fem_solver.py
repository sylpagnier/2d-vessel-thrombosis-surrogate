import torch
import numpy as np
import pytest
import os
from src.core_physics.local_fem_solver import solve_local_t0_flow
from src.config import PhysicsConfig

def test_local_fem_solver():
    pt_path = "C:/Users/pgssy/thrombus_ml_model/data/processed/graphs_biochem_anchors/patient001.pt"
    nas_path = "C:/Users/pgssy/thrombus_ml_model/data/raw/biochem_anchors/patient001.nas"
    
    if not os.path.exists(pt_path) or not os.path.exists(nas_path):
        pytest.skip("Data not found")
        
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    phys_cfg = PhysicsConfig()
    
    # Run solver
    u_pred = solve_local_t0_flow(nas_path, data, phys_cfg, max_iters=2)
    
    # u_pred should be [N, 2]
    assert u_pred.shape == (data.x.shape[0], 2)
    
    # Check if max velocity is somewhat reasonable
    max_v = np.max(np.linalg.norm(u_pred, axis=1))
    assert max_v > 0.0

if __name__ == "__main__":
    test_local_fem_solver()
    print("Test passed!")
