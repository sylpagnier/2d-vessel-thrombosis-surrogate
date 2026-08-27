"""Can the corrector reproduce the flow collapse the closed loop depends on? (C3 gate)

WHY THIS RUNS BEFORE ANY C3 TRAINING.  MODEL_REVIEW 2.4 proposes closing the loop
`Mat -> mu_eff -> corrector -> sr -> gate`, and the whole case rests on WOUND_PROGRESS 3.3's
observation that the wound's shear collapses once its own clot gels.  Measured from GT `y`:

    wound_patient001   sr 148 -> 18.6 between t=14 and t=28, gate 0% -> 91%
    wound_patient003   sr 128 -> 16.1 by t=25,               gate 0% -> 85%

and on 001 that window is exactly where `Mat` crosses `crit` (0.6x at t=14, 1.8x at t=28).
The mechanism is real and enormous -- an order of magnitude above any noise floor in this
project, which is why 2.4 says to test it on n=3.

**But the loop can only work if the corrector can actually produce that collapse.**  This
script hands the corrector an ORACLE: GT `Mat` at each stored time, converted to a viscosity
bump exactly as `corrector_blockage` does, and asks what shear it predicts at the wound.  It
is a ceiling, not a deploy path.

  * if predicted `sr` collapses like GT's, the loop is buildable and C3 is worth its days;
  * if it does not, C3 is dead on arrival -- no amount of ODE work reaches a gate that the
    flow model will not open, and the honest next step is D3 (mesh) or a corrector retrain.

    python scripts/diag_closed_loop_feasibility.py
    python scripts/diag_closed_loop_feasibility.py --stems wound_patient003
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

from src.clot_ml.wound import wound_mask  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)

PACKS = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
STEMS = ("wound_patient001", "wound_patient002", "wound_patient003")


def gt_shear(data, Dx, Dy, scale, ti):
    u = data.y[ti, :, 0].double().numpy()
    v = data.y[ti, :, 1].double().numpy()
    return shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*", default=list(STEMS))
    ap.add_argument("--n-times", type=int, default=6)
    ap.add_argument("--ckpt", default="outputs/local_corrector/local_corrector.pt",
                    help="the shipped LocalKinematicCorrector checkpoint")
    ap.add_argument("--delta-mu", type=float, default=0.68,
                    help="viscosity bump at committed nodes, SI; the CorrectorArm default "
                         "and the measured GT median at committed wall nodes")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    lss = float(bio.lss)

    try:
        from src.core_physics.coupled_shear_gnn import load_local_corrector
        from src.inference.corrector_coupling import couple_flow_with_corrector  # noqa: F401
        corrector = load_local_corrector(REPO / args.ckpt, torch.device("cpu"))
    except Exception as e:  # noqa: BLE001
        print("[i] no shipped corrector loader (%s)." % e)
        print("    Falling back to the GT-Mat -> gate half of the chain, which still answers")
        print("    the weaker question: does the OCCLUSION alone predict the gate opening?")
        corrector = None

    for stem in args.stems:
        p = PACKS / f"{stem}.pt"
        if not p.exists():
            print("[miss] %s" % stem)
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        ei = d.edge_index.detach().cpu().numpy()
        pos = node_positions(d)
        Dx, Dy = build_mls_gradient(pos, ei, hops=3)
        scale = float(d.u_ref.reshape(-1)[0]) / float(d.d_bar.reshape(-1)[0])
        w = wound_mask(d)
        T = int(d.y.shape[0])
        names = d.y_channel_names.split(",")
        mat = np.expm1(d.y[:, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]

        print("=" * 92)
        print("%s   T=%d   wound=%d nodes   lss=%.0f" % (stem, T, int(w.sum()), lss))
        print("  %-7s %11s %11s %11s %11s %11s"
              % ("t", "Mat/crit", "GT sr", "GT gate%", "occluded", "pred sr"))
        for ti in times:
            sr_gt = gt_shear(d, Dx, Dy, scale, ti)
            occ = mat[ti] >= crit
            pred = float("nan")
            if corrector is not None:
                dmu = torch.tensor((occ * args.delta_mu).astype(np.float32))
                u0 = d.y[0, :, 0].float()
                v0 = d.y[0, :, 1].float()
                try:
                    uu, vv, _ = couple_flow_with_corrector(
                        d, u0, v0, dmu, corrector=corrector, phys_cfg=phys,
                        device=torch.device("cpu"))
                    un = uu.double().numpy()
                    vn = vv.double().numpy()
                    pred = float(np.median(
                        (shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * scale)[w]))
                except Exception as e:  # noqa: BLE001
                    print("     corrector failed at t=%d: %s" % (ti, e))
                    corrector = None
            print("  %-7d %11.2f %11.1f %10.0f%% %11d %11s"
                  % (ti, np.median(mat[ti][w]) / crit, np.median(sr_gt[w]),
                     100 * np.mean(sr_gt[w] < lss), int(occ.sum()),
                     "%.1f" % pred if pred == pred else "-"))
        print("  READ: `pred sr` is what the corrector produces from ORACLE occlusion.  If it")
        print("        does not fall with `GT sr`, the closed loop cannot open the gate and")
        print("        C3 is not buildable on this corrector (MODEL_REVIEW 2.4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
