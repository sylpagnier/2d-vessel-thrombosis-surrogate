"""Does the wall-AP upwind renewal module recover the missing far-field AUC on wound_patient003?

CONTEXT (docs/WOUND_PROGRESS.md §18).

§18.2 established that GT chemistry + evolving gate + washout + da_scale_auto=123 scores
**0.8512** off-wall on ``wound_patient003`` at att=0.23, depth=3 — target met as an oracle.
The next step is a deploy-legal AP field: a wall-AP ODE with Damkohler sink + upwind graph
renewal (src/core_physics/wall_ap_renewal.py), rather than a species GNN (which would face
the same OOD gap as v6).

This script runs four families of arms on the three wound vessels:

  frozen          shipped baseline (static SHIPPED ApClosure, frozen t=0 gate)
  renewal         WallApRenewal(renewal_scale=1), optionally + da_scale_auto + gate + wash
  gt_chem         GT AP/RP oracle (ceiling, §18.2 = 0.8512 on 003 with all options on)
  no_closure      pure frozen AP0 with no ApClosure (lower bound, for comparison)

All arms use the SHIPPED wound-rate blockage (g_pre=1.98, g_post=14.28).

DECISION GATE (printed at end):
  renewal far-field AUC on 003 >= 0.90  →  upwind renewal captures the mechanism;
                                            next: promote with da_scale_auto=123
  renewal far-field AUC on 003  ~ 0.86  →  same as frozen → ApClosure / frozen ceiling;
                                            next: ClotGNN residual on the AP field (§18.3)

Usage:
    python scripts/diag_wall_ap_renewal.py
    python scripts/diag_wall_ap_renewal.py --stems wound_patient003 --no-gate
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

from scripts.diag_chem_oracle_v6 import ATTS, DEPTH, _score_field  # noqa: E402
from scripts.diag_wound_ode_closure_cell import auc, gt_species      # noqa: E402
from scripts.go_mat_field_v6 import solid_shells                      # noqa: E402
from src.clot_ml.evaluate import domain_score                         # noqa: E402
from src.clot_ml.locked import load_temporal_v4_wound                 # noqa: E402
from src.clot_ml.wound import solid_mask, wound_rate_blockage, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig                   # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient          # noqa: E402
from src.core_physics.physics_lumen_model import first_corner_shell, topological_owner  # noqa: E402
from src.core_physics.physics_wall_model import (                     # noqa: E402
    WASHOUT_LAMBDA, deposition_gate, gt_flow_gate_series,
    integrate_mat_trajectory, node_positions, shear_rate_2d,
    t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time        # noqa: E402
from src.core_physics.wall_ap_renewal import WallApRenewal, make_species_from_renewal  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")
# da_scale_auto values to sweep — None = same as da_scale (shipped), 123 = COMSOL's ratio
DA_AUTO_SWEEP = (None, 123.0)


def _integrate(data, bio, gate, hook, *, species, washout, washout_sr, blk, gate_ev_fn,
               da_auto):
    """Run integrate_mat_trajectory with the given species and options."""
    inner = gate_ev_fn   # None → frozen gate; callable → evolving gate oracle
    wound_blk = wound_rate_blockage(
        data, bio,
        g_pre=float(blk["g_pre"]), g_post=float(blk["g_post"]),
        inner=inner,
    )
    traj, _ = integrate_mat_trajectory(
        data, bio, gate,
        da_scale=SHIPPED_DA_SCALE,
        da_scale_auto=da_auto,
        ap_closure=hook if species is None else None,   # closure only for frozen arm
        species=species,
        blockage=wound_blk,
        washout=WASHOUT_LAMBDA if washout else 0.0,
        washout_sr=washout_sr if washout else None,
    )
    return np.asarray(traj)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the GT-evolving-gate oracle arm (saves the per-step MLS pass)")
    args = ap.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    w = load_temporal_v4_wound(name=args.name)["wound"]

    # ---- collect per-vessel results ---------------------------------------
    renewal_far_aucs = []   # renewal+da=123 arm, for the decision gate

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
        shells, owner = solid_shells(
            dict(solid=solid, edge_index=ei_np, shell=shell1, pos=pos, town=town), DEPTH)
        seed = gt & solid                     # keep wall/wound as-is; replace off-wall
        _, _, far = wound_region_masks(data)
        cand = shells[0] & off & far & (town >= 0)
        y_far, o_far = gt[cand], town[cand]

        # ---- t=0 flow fields (GT for oracle; later: pred for deploy) -------
        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f.sr)
        gate = deposition_gate(data, f, wall=wall, wound_source=True)

        # ---- GT species [T, N] in CGS -------------------------------------
        sp_gt = gt_species(data, bio)

        # ---- upwind-renewal AP [T, N] in CGS (deploy-legal) ---------------
        renewal = WallApRenewal(renewal_scale=1.0)
        sp_renewal = make_species_from_renewal(data, bio, f, renewal=renewal)

        # ---- evolving-gate oracle (expensive, skip with --no-gate) --------
        gate_ev_fn = None
        washout_sr = np.broadcast_to(f.sr, (T, int(data.num_nodes)))
        if not args.no_gate:
            print(f"[i] {stem}: computing GT gate series at {T} timesteps ...", flush=True)
            gser = gt_flow_gate_series(data, bio, hops=3, wall=wall)
            u_ref = float(data.u_ref.reshape(-1)[0])
            d_bar = float(data.d_bar.reshape(-1)[0])
            Dx, Dy = build_mls_gradient(node_positions(data), ei_np, hops=3)
            srt = np.zeros((T, int(data.num_nodes)))
            for ti in range(T):
                uu = data.y[ti, :, 0].numpy().astype(np.float64)
                vv = data.y[ti, :, 1].numpy().astype(np.float64)
                srt[ti] = shear_rate_2d(Dx @ uu, Dy @ uu, Dx @ vv, Dy @ vv) * (u_ref / d_bar)
            washout_sr = srt

            def _gate_ev_fn(mat, gate0, step):
                g = np.asarray(gate0, dtype=np.float64).copy()
                g[wall] = gser[int(np.clip(step, 0, T - 1))][wall]
                return g

            gate_ev_fn = _gate_ev_fn
        else:
            print(f"[i] {stem}: frozen-gate arms only", flush=True)

        # ---- define all arms -----------------------------------------------
        # Each entry: (tag, species, use_washout, use_gate_ev, da_auto)
        # species=None  → ap_closure=SHIPPED is used (frozen arm)
        # species=(.)   → ap_closure=None (dynamic field)
        arms_spec = [
            ("frozen (SHIPPED closure)", None, False, False, None),
        ]
        for da in DA_AUTO_SWEEP:
            da_str = str(int(da)) if da is not None else "None"
            arms_spec.append((f"renewal(rs=1) da={da_str}",
                               sp_renewal, False, False, da))
        if not args.no_gate:
            for da in DA_AUTO_SWEEP:
                da_str = str(int(da)) if da is not None else "None"
                arms_spec.append((f"renewal(rs=1)+gate+wash da={da_str}",
                                   sp_renewal, True, True, da))
        # GT-chemistry oracle arms (reference ceiling)
        arms_spec.append(("gt_chem (oracle)", sp_gt, False, False, None))
        if not args.no_gate:
            arms_spec.append(("gt_chem+gate+wash da=123 (§18.2)",
                               sp_gt, True, True, 123.0))

        print(f"[i] {stem}: integrating {len(arms_spec)} ODE arms ...", flush=True)

        results = []
        for tag, species, use_wash, use_gate_ev, da_auto in arms_spec:
            fld = _integrate(data, bio, gate, hook,
                             species=species,
                             washout=use_wash,
                             washout_sr=washout_sr,
                             blk=w,
                             gate_ev_fn=gate_ev_fn if use_gate_ev else None,
                             da_auto=da_auto)
            p90 = float(np.percentile(fld[solid], 90) / crit)
            a_far = auc(fld[o_far], y_far) if cand.any() else float("nan")
            sc = _score_field(fld, shells, owner, seed, gt, ei, off, solid, crit)
            results.append((tag, p90, a_far, sc))

            # track the key arm for the decision gate
            if "renewal" in tag and "gate" not in tag and da_auto == 123.0:
                renewal_far_aucs.append((stem, a_far))

        # ---- print table ---------------------------------------------------
        print("=" * 116)
        wall_sc = domain_score(seed, gt, ei, wall, solid)
        print(f"{stem}  wall_F1={wall_sc:.4f}  "
              f"off GT+={int((gt & off).sum())}  "
              f"far cand={int(cand.sum())} (GT+ {int(y_far.sum())})")
        if cand.any():
            print(f"  AP_owner AUC:  "
                  f"gt_final {auc(sp_gt[1][-1][o_far], y_far):.4f}  "
                  f"renewal_t0 {auc(sp_renewal[1][0][o_far], y_far):.4f}  "
                  f"renewal_final {auc(sp_renewal[1][-1][o_far], y_far):.4f}")
        hdr = "  ".join(f"{a:>10s}" for a in [f"att{x:g}d{d}"
                                               for x in ATTS for d in range(1, DEPTH + 1)])
        print(f"  {'arm':38s} {'p90x':>7s} {'farAUC':>7s}  {hdr}")
        for tag, p90, a_far, sc in results:
            cols = "  ".join(
                f"{sc.get(k, float('nan')):10.4f}"
                for x in ATTS for d in range(1, DEPTH + 1)
                for k in [f"att{x:g}d{d}"]
            )
            print(f"  {tag:38s} {p90:7.2f} {a_far:7.4f}  {cols}")

    # ---- decision gate -----------------------------------------------------
    print("\n" + "=" * 60)
    print("DECISION GATE (renewal(rs=1) da=123, far-field AUC on 003):")
    for stem, a in renewal_far_aucs:
        if "003" in stem:
            if np.isnan(a):
                verdict = "no far candidates — check domain masks"
            elif a >= 0.90:
                verdict = "MECHANISM CAPTURED — proceed to promotion + da_scale_auto=123"
            elif a >= 0.86:
                verdict = "MARGINAL — check per-vessel att0.23d3 scores"
            else:
                verdict = "MECHANISM NOT CAPTURED — next: ClotGNN residual on AP field (§18.3)"
            print(f"  {stem}: far AUC {a:.4f}  →  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
