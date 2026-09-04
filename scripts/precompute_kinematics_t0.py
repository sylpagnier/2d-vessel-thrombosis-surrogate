import torch
from src.utils.paths import anchor_packs_dir
import sys
from pathlib import Path

# Run directly (`python scripts/precompute_kinematics_t0.py`) needs the repo root importable.
# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.kinematics_inference import predict_kinematics_and_latent
from src.biochem_gnn.config import apply_deploy_env
from src.utils.kinematics_inference import resolve_kinematics_checkpoint
from src.utils.kinematics_inference import load_kinematics_predictor
import os

def main():
    anchors = [
        "comsol005", "comsol006", "comsol010", 
        "comsol023", "comsol002", "comsol020", "comsol034"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = resolve_kinematics_checkpoint()
    print(f"Loading kinematics predictor from {ckpt_path}")
    kine = load_kinematics_predictor(ckpt_path, device)
    kine.eval()
    
    apply_deploy_env()
    
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    
    for anchor in anchors:
        print(f"Precomputing {anchor}...")
        graph_path = str(anchor_packs_dir() / f"{anchor}.pt")
        data = torch.load(graph_path, weights_only=False).to(device)
        
        from pathlib import Path
        pred, z_kin = predict_kinematics_and_latent(
            kine, 
            data, 
            disk_cache_dir=Path(".cache/kinematics_t0"),
            disk_cache_key=f"{stem}/{anchor}"
        )
        print(f"Done {anchor}. Cache populated.")

if __name__ == "__main__":
    main()
