"""Can a rate scalar close the 0.66 -> 0.75 gap on chemistry-driven ODE Mat?

``diag_chem_oracle_v6.py`` put GT chemistry through v6's replace+depth readout and landed
at **0.6624** on ``wound_patient003`` -- ordering 0.965, magnitude 6.79x crit against GT's
27.78x.  §16.3 already said the remaining failure is calibration, and ``da_scale_auto`` is
the documented lever (COMSOL's own two Damkohler terms sit at a 3.07x ratio).  This sweeps
that scalar, and a denser attenuation grid, on ONE vessel, so we know whether the next
build is a rate tweak or a new model.

    python scripts/diag_chem_oracle_cal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.diag_chem_oracle_v6 import DEPTH, _score_field  # noqa: E402
from scripts.diag_wound_ode_closure_cell import auc, gt_species  # noqa: E402
from scripts.go_mat_field_v6 import solid_shells  # noqa: E402
from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_rate_blockage, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient  # noqa: E402
from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    WASHOUT_LAMBDA, deposition_gate, gt_flow_gate_series, integrate_mat_trajectory,
    node_positions, shear_rate_2d, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEM = "wound_patient003"
DA_AUTO = (None, 80.0, 123.0, 200.0, 400.0)
ATTS_FINE = np.geomspace(0.08, 1.2, 16)


def main() -> int:
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    w = load_temporal_v4_wound(name="clot_gnn_v5w")["wound"]
    data = torch.load(PACKS / f"{STEM}.pt", map_location="cpu", weights_only=False)
    T = int(data.y.shape[0])
    ei_np = data.edge_index.detach().cpu().numpy()
    ei = torch.tensor(ei_np)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    solid = solid_mask(data)
    off = ~solid
    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)
    gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
    shell1 = first_corner_shell(pos, solid, ei_np)
    town = topological_owner(pos, solid, ei_np)
    shells, owner = solid_shells(
        dict(solid=solid, edge_index=ei_np, shell=shell1, pos=pos, town=town), DEPTH)
    seed = gt & solid
    _, _, far = wound_region_masks(data)
    cand = shells[0] & off & far & (town >= 0)
    y_far, o_far = gt[cand], town[cand]

    print(f"[i] {STEM}: GT gate + shear ...", flush=True)
    f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
    hook = make_rollout_hook(SHIPPED, bio, f.sr)
    gate = deposition_gate(data, f, wall=wall, wound_source=True)
    gser = gt_flow_gate_series(data, bio, hops=3, wall=wall)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    Dx, Dy = build_mls_gradient(node_positions(data), ei_np, hops=3)
    srt = np.zeros((T, int(data.num_nodes)))
    for ti in range(T):
        uu = data.y[ti, :, 0].numpy().astype(np.float64)
        vv = data.y[ti, :, 1].numpy().astype(np.float64)
        srt[ti] = shear_rate_2d(Dx @ uu, Dy @ uu, Dx @ vv, Dy @ vv) * (u_ref / d_bar)
    sp_gt = gt_species(data, bio)

    def evolving_gate(mat, gate0, step):
        g = np.asarray(gate0, dtype=np.float64).copy()
        g[wall] = gser[int(np.clip(step, 0, T - 1))][wall]
        return g

    def run(tag, species, da_auto):
        blk = wound_rate_blockage(data, bio, g_pre=float(w["g_pre"]),
                                  g_post=float(w["g_post"]), inner=evolving_gate)
        traj, _ = integrate_mat_trajectory(
            data, bio, gate, da_scale=SHIPPED_DA_SCALE, da_scale_auto=da_auto,
            ap_closure=hook, blockage=blk, species=species,
            washout=WASHOUT_LAMBDA, washout_sr=srt)
        fld = np.asarray(traj)[-1]
        sc = _score_field(fld, shells, owner, seed, gt, ei, off, solid, crit)
        # denser att at depth 1 and 3
        ow = np.where(owner >= 0, fld[np.maximum(owner, 0)], 0.0)
        fine = {}
        for att in ATTS_FINE:
            for d in (1, 3):
                m = seed.copy()
                for j in range(d):
                    m = m | (shells[j] & (ow >= crit / max(float(att) ** (j + 1), 1e-30)))
                fine[f"a{att:.3f}d{d}"] = domain_score(m, gt, ei, off, solid)
        best_k = max(fine, key=fine.get)
        p90 = float(np.percentile(fld[solid], 90) / crit)
        a_far = auc(fld[o_far], y_far)
        print(f"  {tag:28s} da_auto={str(da_auto):6s}  p90 {p90:6.2f}  "
              f"farAUC {a_far:.4f}  att0.23d3 {sc['att0.23d3']:.4f}  "
              f"BEST {best_k} {fine[best_k]:.4f}", flush=True)
        return fine[best_k]

    print(f"[i] wall {domain_score(seed, gt, ei, wall, solid):.4f}  "
          f"off GT+ {int((gt & off).sum())}  far GT+ {int(y_far.sum())}", flush=True)
    print("  -- GT chemistry + evolving gate + washout, sweep da_scale_auto --")
    best_chem = 0.0
    for da in DA_AUTO:
        best_chem = max(best_chem, run("GT-chem+gate+wash", sp_gt, da))
    print("  -- frozen chemistry (t=0 AP * closure), same gate+wash, same sweep --")
    best_frz = 0.0
    for da in DA_AUTO:
        best_frz = max(best_frz, run("frozen-chem+gate+wash", None, da))
    print(f"  BEST GT-chem {best_chem:.4f}   BEST frozen-chem {best_frz:.4f}   "
          f"target 0.75 is "
          f"{'MET by chemistry' if best_chem >= 0.75 else 'NOT MET'}; "
          f"frozen-only "
          f"{'MET' if best_frz >= 0.75 else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
