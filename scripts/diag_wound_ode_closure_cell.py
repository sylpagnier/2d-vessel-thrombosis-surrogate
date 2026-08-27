"""The gate x chemistry x washout cross, scored on the quantity that decides the far field.

WHY THIS SCRIPT.  ``diag_wound_far_separator.py`` localised the whole off-wall gap to ONE
number.  On ``wound_patient003``'s far-field shell-1 candidates, GT ``Mat_owner`` separates
clot from lumen at **AUC 0.9961** (median 16x crit against 1.8x, straddling the rule's 6.25x
bar), and the shipped ODE's ``Mat_owner`` is at **chance, AUC 0.5048**, with the two medians
equal to three significant figures.  Feeding GT ``Mat`` into the otherwise-unchanged shipped
rule takes off-wall 0.4755/0.6736/0.5293 -> 0.9755/0.9755/0.7897, so the shell, the owner map
and the 0.16 constant are all fine and the ODE's ``Mat`` is the single broken component.

The ODE has lost its DYNAMIC RANGE, not its scale: its committed wall nodes span 1.01x where
GT spans 8.7x.  PHASE7 9.3 explains why -- with the gate and ``ap``/``rp`` frozen at t=0 the
source is constant in time, ``mas`` saturates, and every gated node integrates the same thing.
9.3 also measured that switching the removal term on ALONE makes the shipped model worse
(rho_corner 0.482 -> 0.084), because a constant source against a linear sink has one attractor
whose ordering is the ``1/sr`` null -- and 9.4 measured that the term only pays when flow AND
chemistry evolve (0.310 -> 0.464 on oracle inputs).

MODEL_REVIEW 2.2 calls that the only physics route in the repo with a measured mechanism, a
measured magnitude and an unmeasured deploy number, and 2.4 says to test it on the wound
vessels first.  This runs the cross with GT flow and GT chemistry as ORACLES -- not a deploy
path -- to find out whether the closed loop is worth building before anyone builds it.

    python scripts/diag_wound_ode_closure_cell.py --stems wound_patient003
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
)
from src.clot_ml.wound import solid_mask, wound_region_masks, wound_rate_blockage  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    first_corner_shell, topological_owner,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    M_TO_CM, PER_M3_TO_PER_CM3, WASHOUT_LAMBDA, deposition_gate, gt_flow_gate_series,
    integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(np.asarray(score, float))) + 1.0
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def gt_species(data, bio_cfg):
    """``(rp, ap)`` [T, N] in CGS from COMSOL -- the chemistry oracle."""
    names = data.y_channel_names.split(",")
    scales = bio_cfg.get_species_scales(device="cpu")
    rp = torch.expm1(data.y[:, :, names.index("RP_log1p_nd")].clamp(-10, 8)).numpy()
    ap = torch.expm1(data.y[:, :, names.index("AP_log1p_nd")].clamp(-10, 8)).numpy()
    return (rp * float(scales[0]) * PER_M3_TO_PER_CM3,
            ap * float(scales[1]) * PER_M3_TO_PER_CM3)


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stems", nargs="*", default=list(STEMS))
    ap_.add_argument("--name", default="clot_gnn_v5w")
    args = ap_.parse_args()

    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    bundle = load_temporal_v4_wound(name=args.name)
    w = bundle["wound"]

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        _, _, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

        S = build_sample(data, bio, flow="gt", variant="v4")
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt",
                                         sample=S)["series"][T - 1]
        pw = pred & solid
        shell = first_corner_shell(pos, solid, ei_np)
        town = topological_owner(pos, solid, ei_np)
        has = town >= 0
        ow_c = np.zeros(len(wall), bool)
        ow_c[has] = pw[town[has]]
        cand = shell & off & far & ow_c
        y_far = gt[cand]
        o_far = town[cand]

        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()

        f = t0_flow_fields(data, bio, hops=3, flow_source="gt")
        hook = make_rollout_hook(SHIPPED, bio, f.sr)
        gate = deposition_gate(data, f, wall=wall, wound_source=True)
        gp, gq = float(w["g_pre"]), float(w["g_post"])
        sp_gt = gt_species(data, bio)

        print(f"[i] {stem}: GT gate at all {T} timesteps ...", flush=True)
        gser = gt_flow_gate_series(data, bio, hops=3, wall=wall)
        u_ref = float(data.u_ref.reshape(-1)[0])
        d_bar = float(data.d_bar.reshape(-1)[0])
        srt = np.zeros((T, int(data.num_nodes)))
        from src.core_physics.mls_gradient import build_mls_gradient
        from src.core_physics.physics_wall_model import node_positions, shear_rate_2d
        Dx, Dy = build_mls_gradient(node_positions(data), ei_np, hops=3)
        for ti in range(T):
            uu = data.y[ti, :, 0].numpy().astype(np.float64)
            vv = data.y[ti, :, 1].numpy().astype(np.float64)
            srt[ti] = shear_rate_2d(Dx @ uu, Dy @ uu, Dx @ vv, Dy @ vv) * (u_ref / d_bar)

        def evolving_gate(mat, gate0, step):
            g = np.asarray(gate0, dtype=np.float64).copy()
            g[wall] = gser[int(np.clip(step, 0, T - 1))][wall]
            return g

        print("=" * 108)
        print(f"{stem}  far shell-1 candidates={int(cand.sum())} "
              f"(GT+ {int(y_far.sum())})   [GT Mat AUC {auc(gmat[-1][o_far], y_far):.4f}]")
        # The ordering and the CALIBRATION are separate failures and have to be reported
        # separately: the score is a threshold crossing at `crit`, not a ranking (PHASE7 7.2),
        # so `att*` is the best off score reachable if the bar were recalibrated to whatever
        # this arm's magnitude actually is.  A cell that orders well but scores badly is a
        # scale problem; one that orders badly is a physics problem.
        atts = np.geomspace(0.02, 4.0, 40)
        print(f"  {'gate':9s} {'chem':9s} {'wash':5s} | {'wallp90':>8s} {'p90/p50':>7s} "
              f"{'farAUC':>7s} {'off':>7s} {'offTP':>6s} {'offFP':>6s} | "
              f"{'off@att*':>8s} {'att*':>6s}")

        for g_ev, c_ev, wash in itertools.product((False, True), (False, True),
                                                  (False, True)):
            blk = evolving_gate if g_ev else None
            blk = wound_rate_blockage(data, bio, g_pre=gp, g_post=gq, inner=blk)
            traj, _ = integrate_mat_trajectory(
                data, bio, gate, da_scale=SHIPPED_DA_SCALE, ap_closure=hook, blockage=blk,
                species=sp_gt if c_ev else None,
                washout=WASHOUT_LAMBDA if wash else 0.0,
                washout_sr=(srt if wash else None))
            m = np.asarray(traj)[-1]
            ow = np.zeros(len(wall))
            ow[has] = m[town[has]]
            pm = shell & off & (0.16 * ow >= crit)
            sc = domain_score(pm | pw, gt, ei, off, solid)
            live = m[pw & wall]
            rng = (np.percentile(live, 90) / max(np.percentile(live, 50), 1e-30)
                   if live.size else float("nan"))
            best = max((domain_score(
                (shell & off & (a * ow >= crit)) | pw, gt, ei, off, solid), a)
                for a in atts)
            print(f"  {'GT-evolv' if g_ev else 'frozen':9s} "
                  f"{'GT' if c_ev else 'frozen':9s} {'on' if wash else 'off':5s} | "
                  f"{np.percentile(m[wall], 90) / crit:8.2f} {rng:7.2f} "
                  f"{auc(m[o_far], y_far):7.4f} {sc:7.4f} "
                  f"{int((pm & gt & off).sum()):6d} {int((pm & ~gt & off).sum()):6d} | "
                  f"{best[0]:8.4f} {best[1]:6.3f}")
        gl = gmat[-1][pw & wall]
        g_ow = np.zeros(len(wall))
        g_ow[has] = gmat[-1][town[has]]
        g_best = max((domain_score(
            (shell & off & (a * g_ow >= crit)) | pw, gt, ei, off, solid), a) for a in atts)
        print(f"  {'GT Mat (reference)':25s} | "
              f"{np.percentile(gmat[-1][wall], 90) / crit:8.2f} "
              f"{np.percentile(gl, 90) / max(np.percentile(gl, 50), 1e-30):7.2f} "
              f"{auc(gmat[-1][o_far], y_far):7.4f} "
              f"{domain_score((shell & off & (0.16 * g_ow >= crit)) | pw, gt, ei, off, solid):7.4f}"
              f"{'':14s} | {g_best[0]:8.4f} {g_best[1]:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
