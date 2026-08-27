"""Which off-wall RULE fits all three wound vessels at once?

The stall hypothesis is closed (``diag_wound_gt_gate_ceiling.py``): a total stall pins the
gate at 1 and the wall ``Mat`` p90 reaches 2.31x crit where the shell rule needs 6.25x, for
zero off-wall true positives.  So the lever is not the ODE's magnitude.

What the three vessels actually ask for is opposite things, which is the real finding:

    001   off 0.4755   prec 0.466  rec 1.000     87 FP, all of them far-field
    002   off 0.6736   prec 0.661  rec 1.000     39 FP, all of them far-field
    003   off 0.5293   prec 0.602  rec 0.305    169 FN, 102 of them far-field

001/002 are precision-limited in the far field and 003 is recall-limited there.  The
discriminator has to be something the far field itself carries.  001/002 have 94/58 wall GT+
concentrated at the wound; 003 has 254 spread over a second clot station at s=0.52 -- and the
shipped wall arm already finds it (recall 0.803, ZERO false positives).  So test the rule
that keys off committed WALL tissue rather than off ``Mat`` magnitude or off flow.

    python scripts/diag_wound_offwall_rules.py
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

from src.clot_ml.evaluate import domain_score  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound,
)
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.clot_ml.wound import solid_mask, wound_mask, wound_region_masks  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import mat_si_for_gelation_from_log1p  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    first_corner_shell, topological_owner,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--name", default="clot_gnn_v5w")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    bundle = load_temporal_v4_wound(name=args.name)
    rows: dict[str, list[float]] = {}

    for stem in args.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        T = int(data.y.shape[0])
        ei_np = data.edge_index.detach().cpu().numpy()
        ei = torch.tensor(ei_np)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        solid = solid_mask(data)
        off = ~solid
        wnd = wound_mask(data)
        _, _, far = wound_region_masks(data)
        gt = gt_clot_phi_at_time(data, T - 1, phys).numpy() > 0.5
        pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64)

        S = build_sample(data, bio, flow="gt", variant="v4")
        pred = predict_temporal_v4_wound(bundle, data, [0, T - 1], flow="gt",
                                         sample=S)["series"][T - 1]
        wr = bundle["base"]["temporal"].get("wound_rate")
        traj, _ = ode_trajectory(data, bio, flow="gt",
                                 wound_rate=None if wr is None else tuple(wr))

        shell = first_corner_shell(pos, solid, ei_np)
        town = topological_owner(pos, solid, ei_np)
        has = town >= 0
        pw = pred & solid                       # the shipped wall/wound set
        owner_committed = np.zeros(len(wall), dtype=bool)
        owner_committed[has] = pw[town[has]]
        owner_mat = np.zeros(len(wall))
        owner_mat[has] = traj[-1][town[has]]

        # The same shipped magnitude rule, fed GT `Mat` instead of the ODE's.  This is not a
        # deploy path -- it isolates WHICH component is broken, because the rule, the shell,
        # the owner map and the 0.16 constant are all held fixed and only `Mat` changes.
        mi = data.y_channel_names.split(",").index("Mat_log1p_nd")
        gmat = mat_si_for_gelation_from_log1p(data.y[:, :, mi], bio).reshape(T, -1).numpy()
        owner_gmat = np.zeros(len(wall))
        owner_gmat[has] = gmat[-1][town[has]]

        cand = {
            "shipped v5w": pred & off,
            "shell & owner committed": shell & off & owner_committed,
            "shell & 0.16*Mat_own>=crit": shell & off & (0.16 * owner_mat >= crit),
            "shell & 0.16*GTMat>=crit (ORACLE)": shell & off & (0.16 * owner_gmat >= crit),
            "shipped OR owner-committed": (pred & off) | (shell & off & owner_committed),
            "shipped AND owner-committed": pred & off & owner_committed,
        }
        print("=" * 104)
        print(f"{stem}  T={T}  lumen={int(off.sum())}  GT+ off={int((gt & off).sum())}  "
              f"far GT+={int((gt & far).sum())}  shell={int((shell & off).sum())}")
        print(f"  {'rule':34s} {'off':>7s} {'prec':>6s} {'rec':>6s} "
              f"{'TP':>5s} {'FP':>5s} {'FN':>5s} | {'farTP':>5s} {'farFP':>5s}")
        for tag, m in cand.items():
            tp = int((m & gt & off).sum())
            fp = int((m & ~gt & off).sum())
            fn = int((~m & gt & off).sum())
            sc = domain_score(m | (pred & solid), gt, ei, off, solid)
            rows.setdefault(tag, []).append(sc)
            print(f"  {tag:34s} {sc:7.4f} {tp / max(tp + fp, 1):6.3f} "
                  f"{tp / max(tp + fn, 1):6.3f} {tp:5d} {fp:5d} {fn:5d} | "
                  f"{int((m & gt & far).sum()):5d} {int((m & ~gt & far).sum()):5d}")

    print("=" * 104)
    print(f"  {'rule':34s} {'mean':>7s}   per-vessel")
    for tag, v in rows.items():
        print(f"  {tag:34s} {np.mean(v):7.4f}   " + "  ".join(f"{x:.4f}" for x in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
