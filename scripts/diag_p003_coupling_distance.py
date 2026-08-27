"""Is 003's shear collapse hop-local (wake) or axial? GT-only, no GNN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.features import adjacency, hop_distance  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.gelation_wake import wake_amplitude, wall_wake_operator  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

STEM = "wound_patient003"


def main() -> int:
    data = torch.load(
        REPO / "data/processed/graphs_biochem_anchors" / f"{STEM}.pt",
        map_location="cpu", weights_only=False)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    T = int(data.y.shape[0])
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid, wnd = solid_mask(data), wound_mask(data)
    crit = float(bio.viscosity_mat_crit)
    mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
    gtm = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()
    hot = gtm >= crit
    gt_on = np.where(hot.any(0), hot.argmax(0), T)
    gt_clot = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    pos = data.x[:, :2].numpy().astype(np.float64)
    _, j = cKDTree(pos[solid]).query(pos)
    owner = np.flatnonzero(solid)[j]
    _, lumen, _ = wound_region_masks(data)
    ow = np.unique(owner[(gt_clot & lumen) & ~wnd[owner]])
    f0 = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    blind = ow[f0.gate[ow] == 0]
    early = wall & (gt_on <= 5)
    print(f"T={T}  wall={int(wall.sum())}  early-gelled(<=5)={int(early.sum())}  "
          f"blind owners={blind.size}")

    A = adjacency(data.edge_index.numpy(), int(data.num_nodes))
    seed = np.zeros(int(data.num_nodes), dtype=bool)
    seed[early] = True
    h_early = hop_distance(seed, A, max_h=80)
    h_wnd = hop_distance(wnd, A, max_h=80)
    print("blind -> nearest GT-gelled-by-step-5 hops: "
          f"min {h_early[blind].min():.0f}  med {np.median(h_early[blind]):.1f}  "
          f"p90 {np.percentile(h_early[blind], 90):.1f}")
    print("blind -> wound hops: "
          f"min {h_wnd[blind].min():.0f}  med {np.median(h_wnd[blind]):.1f}")
    print("early-gelled -> wound hops: "
          f"min {h_wnd[early].min():.0f}  med {np.median(h_wnd[early]):.1f}")

    X = pos[wall] - pos[wall].mean(0)
    _, evecs = np.linalg.eigh(X.T @ X)
    s = pos @ evecs[:, -1]
    s = (s - s.min()) / max(float(s.ptp()), 1e-9)
    print("axial s (0-1):")
    print(f"  wound        med {np.median(s[wnd]):.3f}")
    print(f"  early-gelled med {np.median(s[early]):.3f}  "
          f"p10 {np.percentile(s[early], 10):.3f}  p90 {np.percentile(s[early], 90):.3f}")
    print(f"  blind        med {np.median(s[blind]):.3f}  "
          f"p10 {np.percentile(s[blind], 10):.3f}  p90 {np.percentile(s[blind], 90):.3f}")
    print(f"  |s_blind - med(s_early)| med {np.median(np.abs(s[blind] - np.median(s[early]))):.3f}")

    from src.core_physics.physics_wall_model import gate_from_shear  # noqa: PLC0415

    need = float(bio.lss) / max(float(np.median(f0.sr[blind])), 1e-9)
    print(f"t=0 sr med at blinds {float(np.median(f0.sr[blind])):.1f} /s; "
          f"need amp < {need:.3f} to open the low-shear gate")
    print("GT-oracle wake load at blind owners (hop-kernel CEILING):")
    for label, src in (("healthy wall (shipped)", wall), ("solid wall|wound", solid)):
        Kw = wall_wake_operator(data, src)
        sidx = np.flatnonzero(src)
        print(f"  -- {label}  n_src={int(src.sum())}")
        for step in (2, 5, 10, 20, 40, T - 1):
            occ = hot[min(step, T - 1)]
            load = np.zeros(len(wall), dtype=np.float64)
            load[sidx] = Kw @ occ[sidx].astype(np.float64)
            amp = np.ones(len(wall), dtype=np.float64)
            amp[sidx] = wake_amplitude(load[sidx])
            g = gate_from_shear(f0.sr * amp, f0.dsrx * amp, bio)[blind]
            print(f"     s{step:3d}  load med/max {np.median(load[blind]):.3f}/"
                  f"{load[blind].max():.3f}  amp med {np.median(amp[blind]):.3f}  "
                  f"sr med {np.median(f0.sr[blind] * amp[blind]):.1f}  "
                  f"gate>0 {(g > 0).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
