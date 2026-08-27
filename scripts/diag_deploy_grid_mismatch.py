"""The shipped temporal head has `n_times=11`; the viz replays it at every timestep (201).

`clot_gnn_v4`'s manifest carries `n_times: 11` and `burden_gate: 0`.  The zero gate means
`offwall_by_learned_lag` ALWAYS fires, and on its `times=None` branch it adds the predicted
lag in WHOLE GRID STEPS.  A lag fitted as "+4 of 11" is 36% of the run; replayed on a
201-step grid the same integer is 2%.  `lag_features`' `t_adv`/`t_own` columns are grid-step
counts too, so the regression's INPUTS shift range with the grid as well.

What this does and does not touch: `time_th_wall`/`time_th_off` both carry
`commit_final=True`, so the final-time mask equals the committed set by construction and the
headline SEALED numbers are grid-independent (verified below -- they agree to four
decimals).  What moves is the SCHEDULE, i.e. the animation the viz actually shows.

    python scripts/diag_deploy_grid_mismatch.py --vessel patient042
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.locked import load_default, predict_default_series  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel", default="patient042")
    ap.add_argument("--grids", default="11,0", help="comma list; 0 = every simulated step")
    args = ap.parse_args()
    assert args.vessel not in FINAL_HALF, "FINAL_HALF is SEALED -- docs/SEALED_SPLIT.md"

    phys = PhysicsConfig(phase="biochem")
    bundle, kind = load_default()
    assert kind == "temporal_v4", f"expected clot_gnn_v4 shipped, got {kind}"
    tp = bundle["temporal"]
    print("shipped head n_times=%s  burden_gate=%s  time_th_wall=%s  time_th_off=%s\n"
          % (tp["n_times"], tp["burden_gate"], tp["time_th_wall"], tp["time_th_off"]))

    d = torch.load(PACKS / f"{args.vessel}.pt", map_location="cpu", weights_only=False)
    wall = d.mask_wall.reshape(-1).bool().numpy()
    tt = d.t.reshape(-1).numpy()
    T = len(tt)
    gtf = gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu")).numpy() > 0.5
    sv = SeverityScorer(d.edge_index.numpy(), gtf, len(wall), DEFAULT)
    off = ~wall

    for g in (int(x) for x in args.grids.split(",")):
        times = (list(range(T)) if g <= 0 else
                 [int(round(x)) for x in np.linspace(0, T - 1, g)])
        res = predict_default_series(bundle, kind, d, times, flow="gt")
        fin, onset = res["series"][times[-1]], res["onset"]
        on_off = np.array([tt[o] / tt[-1] for o, m in zip(onset, fin & off) if m and o >= 0])
        on_w = np.array([tt[o] / tt[-1] for o, m in zip(onset, fin & wall) if m and o >= 0])
        first_gt = float("nan")
        for ti in times:
            g_ti = gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu")).numpy() > 0.5
            if (g_ti & off).sum() > 0:
                first_gt = tt[ti] / tt[-1]
                break
        print("--- grid = %d points ---" % len(times))
        print("   off-wall committed n=%d   median MODEL onset %.0f%% of run   "
              "(wall median %.0f%%)" % (int((fin & off).sum()),
                                        np.median(on_off) * 100, np.median(on_w) * 100))
        print("   GT off-wall first appears at %.0f%% of run" % (first_gt * 100))
        print("   FINAL wall %.4f   off %.4f\n"
              % (sv.score(fin & wall, wall), sv.score(fin & off, off)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
