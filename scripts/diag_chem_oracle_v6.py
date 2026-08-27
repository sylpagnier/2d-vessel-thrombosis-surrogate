"""Does GT chemistry, scored through v6's REPLACE+DEPTH readout, clear 0.75 on 003?

§16.3 measured chemistry at +0.38 far-field AUC on the raw ODE, through the UNION readout
that §17.1 showed actively costs points.  Nobody has put the chemistry-driven ODE ``Mat``
through the readout v6 actually uses (replace the shipped off-wall verdict; walk shells 1-3
off the whole solid boundary).  That number decides whether a species surrogate is worth
building: if even perfect chemistry cannot drive the rule past 0.75, more wound simulations
are the answer, not a GNN.

Also ranks GT ``AP_owner`` / ``RP_owner`` as raw separators.  If AP itself orders the far
field, the surrogate is a ranking problem; if only the ODE's *response* to AP orders it, the
surrogate has to feed the ODE, not replace it.

    python scripts/diag_chem_oracle_v6.py
    python scripts/diag_chem_oracle_v6.py --stems wound_patient003 --no-gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.diag_wound_ode_closure_cell import auc, gt_species  # noqa: E402
from scripts.go_mat_field_v6 import solid_shells  # noqa: E402
from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_rate_blockage, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    WASHOUT_LAMBDA, deposition_gate, gt_flow_gate_series, integrate_mat_trajectory,
    node_positions, shear_rate_2d, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
ATTS = (0.16, 0.23)
DEPTH = 3


def _score_field(fld, shells, owner, seed, gt, ei, off, solid, crit):
    """Replace+depth score for a magnitude field.  Returns ``{arm: score}``."""
    out = {}
    fld = np.asarray(fld, dtype=np.float64)
    ow = fld[np.maximum(owner, 0)]
    ow = np.where(owner >= 0, ow, 0.0)
    for att in ATTS:
        for d in range(1, DEPTH + 1):
            m = seed.copy()
            for j in range(d):
                m = m | (shells[j] & (ow >= crit / max(float(att) ** (j + 1), 1e-30)))
            out[f"att{att:g}d{d}"] = domain_score(m, gt, ei, off, solid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the GT-evolving-gate arm (saves the per-timestep MLS pass)")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    w = load_temporal_v4_wound(name=args.name)["wound"]
    gp, gq = float(w["g_pre"]), float(w["g_post"])

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
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
        entry = dict(solid=solid, edge_index=ei_np, shell=shell1, pos=pos, town=town)
        shells, owner = solid_shells(entry, DEPTH)
        seed = gt & solid  # wall/wound kept; off-wall handed to the field (replace)
        # far-field candidates as in §16: shell-1, off, far from wound, owned by committed wall
        _, _, far = wound_region_masks(data)
        cand = shells[0] & off & far & (town >= 0)
        y_far = gt[cand]
        o_far = town[cand]

        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()
        sp_gt = gt_species(data, bio)
        rp_gt, ap_gt = sp_gt

        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f.sr)
        gate = deposition_gate(data, f, wall=wall, wound_source=True)

        srt = None
        gser = None
        if not args.no_gate:
            print(f"[i] {stem}: GT gate + shear at {T} timesteps ...", flush=True)
            gser = gt_flow_gate_series(data, bio, hops=3, wall=wall)
            from src.core_physics.mls_gradient import build_mls_gradient
            u_ref = float(data.u_ref.reshape(-1)[0])
            d_bar = float(data.d_bar.reshape(-1)[0])
            Dx, Dy = build_mls_gradient(node_positions(data), ei_np, hops=3)
            srt = np.zeros((T, int(data.num_nodes)))
            for ti in range(T):
                uu = data.y[ti, :, 0].numpy().astype(np.float64)
                vv = data.y[ti, :, 1].numpy().astype(np.float64)
                srt[ti] = shear_rate_2d(Dx @ uu, Dy @ uu, Dx @ vv, Dy @ vv) * (u_ref / d_bar)
        else:
            print(f"[i] {stem}: frozen-gate arms only", flush=True)
            srt = np.broadcast_to(f.sr, (T, int(data.num_nodes)))

        def evolving_gate(mat, gate0, step):
            g = np.asarray(gate0, dtype=np.float64).copy()
            g[wall] = gser[int(np.clip(step, 0, T - 1))][wall]
            return g

        arms = []

        def add(tag, fld, *, is_mat=True):
            p90 = float(np.percentile(np.asarray(fld)[solid], 90) / crit) if is_mat else float("nan")
            a_far = auc(np.asarray(fld)[o_far], y_far) if cand.any() else float("nan")
            sc = _score_field(fld, shells, owner, seed, gt, ei, off, solid, crit) if is_mat else {}
            arms.append((tag, p90, a_far, sc))

        def integrate(species, wash, gate_ev):
            inner = evolving_gate if (gate_ev and gser is not None) else None
            blk = wound_rate_blockage(data, bio, g_pre=gp, g_post=gq, inner=inner)
            traj, _ = integrate_mat_trajectory(
                data, bio, gate, da_scale=SHIPPED_DA_SCALE, ap_closure=hook, blockage=blk,
                species=species,
                washout=WASHOUT_LAMBDA if wash else 0.0,
                washout_sr=(srt if wash else None))
            return np.asarray(traj)[-1]

        print(f"[i] {stem}: integrating ODE arms ...", flush=True)
        add("ODE frozen (shipped)", integrate(None, False, False))
        add("ODE GT-chem", integrate(sp_gt, False, False))
        add("ODE GT-chem+wash", integrate(sp_gt, True, False))
        if gser is not None:
            add("ODE chem+gate+wash", integrate(sp_gt, True, True))
        add("GT Mat (ceiling)", gmat[-1])

        print("=" * 108)
        print(f"{stem}  off GT+={int((gt & off).sum())}  far cand={int(cand.sum())} "
              f"(GT+ {int(y_far.sum())})  wall {domain_score(seed, gt, ei, wall, solid):.4f}")
        # AP/RP as RAW separators -- not a deploy rule, a well-posedness check
        print(f"  ranking AUC on far candidates:  "
              f"AP_owner {auc(ap_gt[-1][o_far], y_far):.4f}  "
              f"RP_owner {auc(rp_gt[-1][o_far], y_far):.4f}  "
              f"GT Mat_owner {auc(gmat[-1][o_far], y_far):.4f}")
        hdr = "  ".join(f"{a:>10s}" for a in [f"att{x:g}d{d}" for x in ATTS for d in range(1, DEPTH + 1)])
        print(f"  {'arm':22s} {'p90x':>7s} {'farAUC':>7s}  {hdr}")
        for tag, p90, a_far, sc in arms:
            cols = "  ".join(f"{sc.get(k, float('nan')):10.4f}"
                             for x in ATTS for d in range(1, DEPTH + 1)
                             for k in [f"att{x:g}d{d}"])
            print(f"  {tag:22s} {p90:7.2f} {a_far:7.4f}  {cols}")
        # headline: best replace+depth score on the chemistry-driven ODE (not the GT Mat ceiling)
        chem_arms = [(t, s) for t, _, _, s in arms if t.startswith("ODE") and "frozen" not in t]
        if chem_arms:
            best = max((v, t, k) for t, s in chem_arms for k, v in s.items())
            print(f"  BEST chemistry-driven arm: {best[1]} {best[2]} -> {best[0]:.4f}   "
                  f"target 0.75 is {'MET' if best[0] >= 0.75 else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
