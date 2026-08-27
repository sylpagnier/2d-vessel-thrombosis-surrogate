import torch
import sys
from pathlib import Path

from src.differentiable_wall_model.temporal_models import TemporalDifferentiableWallModel, PredictorCorrectorGNN, FNOProxy
from src.config import PhysicsConfig

def main():
    data_path = Path("data/processed/graphs_biochem_anchors/patient020.pt")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    
    # 1. Baseline
    model_base = TemporalDifferentiableWallModel(temporal_corrector=None).to(device)
    model_base.eval()
    with torch.no_grad():
        out_base = model_base(data, flow_source="gt", device=device)
        
    # 2. FNOProxy
    fno = FNOProxy().to(device)
    model_fno = TemporalDifferentiableWallModel(temporal_corrector=fno).to(device)
    model_fno.eval()
    with torch.no_grad():
        out_fno = model_fno(data, flow_source="gt", device=device)
        
    # 3. PredictorCorrector
    pc = PredictorCorrectorGNN().to(device)
    model_pc = TemporalDifferentiableWallModel(temporal_corrector=pc).to(device)
    model_pc.eval()
    with torch.no_grad():
        out_pc = model_pc(data, flow_source="gt", device=device)
        
    print(f"Base mat_final mean: {out_base['mat_final'].mean().item():.6f}")
    print(f"FNO mat_final mean: {out_fno['mat_final'].mean().item():.6f}")
    print(f"PC mat_final mean: {out_pc['mat_final'].mean().item():.6f}")
    
    diff_fno = torch.abs(out_base['mat_final'] - out_fno['mat_final']).max().item()
    diff_pc = torch.abs(out_base['mat_final'] - out_pc['mat_final']).max().item()
    print(f"Max diff Base vs FNO: {diff_fno}")
    print(f"Max diff Base vs PC: {diff_pc}")
    
if __name__ == '__main__':
    main()
