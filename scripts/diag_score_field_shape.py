"""ROOT CAUSE of the SEALED wall shortfall: separate RANKING from CUT PLACEMENT.

A threshold readout can fail two ways, and they need opposite fixes:

  * the network cannot ORDER the nodes            -> AUC collapses; no cut helps
  * the network orders them fine but the CUT      -> AUC normal, but oracle-minus-shipped
    lands in the wrong place                         is large

The pool's own numbers say why cross-validation could not see the second one coming.  On all
19 training vessels only **2.7%** of wall nodes sit in the band the cut passes through
(0.30 < p < 0.90); the cut therefore slices a near-empty region, and the gap between the
shipped readout and a per-vessel oracle cut is **+0.006**.  The cut's exact position is both
unconstrained and nearly harmless -- which is the same fact `scripts/eval_readout_bagged.py`
sees from the other side as a flat objective with a bootstrap IQR of 0.14 on the add gate.

Everything here runs through the SHIPPED path for every vessel -- same weights, same
`build_sample`, same metric -- because the earlier version of this script compared pool
vessels scored from the `outputs/phase9_scores/*.npz` fold models against held-out vessels
scored from the shipped members, and that mismatch produced a spurious 20x effect.  Pool rows
are IN-SAMPLE and are here as the calibration-set baseline, not as a generalization estimate.

VIZ_HALF only -- see docs/SEALED_SPLIT.md.

    python scripts/diag_score_field_shape.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import GRID  # noqa: E402
from src.clot_ml.locked import build_sample, load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
LOCKED = REPO / "outputs/clot_ml/locked/clot_gnn_v4"
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def auc(score, y):
    """Rank-based separability -- immune to any monotone recalibration, so it isolates
    whether the network ORDERS the nodes correctly from where the cut happens to sit."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="patient042,patient001")
    ap.add_argument("--band", default="0.30,0.90", help="the band the cut passes through")
    ap.add_argument("--save", default="outputs/diag_score_field_shape.json")
    args = ap.parse_args()

    b_lo, b_hi = (float(x) for x in args.band.split(","))
    man = json.loads((LOCKED / "manifest.json").read_text())
    pool = man["training_pool"]
    held = [a for a in args.held.split(",") if a]
    for a in held:
        assert a not in FINAL_HALF, "FINAL_HALF is SEALED -- docs/SEALED_SPLIT.md"

    ens = load_ensemble(name="clot_gnn_v4")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    rows = []
    print("%-12s %6s %7s %6s %8s %8s %8s %8s"
          % ("vessel", "", "stat", "band%", "AUC", "oracle", "shipped", "cut-gap"))
    for a in pool + held:
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        S = build_sample(d, bio, phys, flow="gt", variant="v4")
        sc = predict_scores(ens, S)
        w = S["wall"].astype(bool)
        T = len(d.t.reshape(-1))
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu")).numpy() > 0.5
        sv = SeverityScorer(S["edge_index"], gt, len(w), DEFAULT)

        band = float(((sc[w] > b_lo) & (sc[w] < b_hi)).mean())
        oracle = max(sv.score(w & (sc >= t), w) for t in GRID)
        # the shipped committed set, wall domain, from the promoted readout
        from src.clot_ml.locked import _committed_set_v4  # noqa: PLC0415
        import pickle  # noqa: PLC0415
        with (LOCKED / man["temporal_file"]).open("rb") as fh:
            temporal = pickle.load(fh)
        shipped = sv.score(_committed_set_v4(S, sc, temporal) & w, w)

        rows.append(dict(v=a, held=a in held, stat=float(sc[w].mean()), band=band,
                         auc=auc(sc[w], gt[w]), oracle=float(oracle),
                         shipped=float(shipped), gap=float(oracle - shipped)))
        r = rows[-1]
        print("%-12s %6s %7.3f %6.1f %8.4f %8.4f %8.4f %+8.4f"
              % (a, "*HELD" if r["held"] else "", r["stat"], 100 * r["band"], r["auc"],
                 r["oracle"], r["shipped"], r["gap"]), flush=True)

    P = [r for r in rows if not r["held"]]
    H = [r for r in rows if r["held"]]
    for nm, rs in (("POOL (in-sample)", P), ("HELD OUT", H)):
        print("\n%-18s band%% %5.1f   AUC %.4f   oracle %.4f   shipped %.4f   cut-gap %+.4f"
              % (nm, 100 * np.mean([r["band"] for r in rs]), np.mean([r["auc"] for r in rs]),
                 np.mean([r["oracle"] for r in rs]), np.mean([r["shipped"] for r in rs]),
                 np.mean([r["gap"] for r in rs])))

    bd = np.array([r["band"] for r in P])
    au = np.array([r["auc"] for r in P])
    print("\nagainst the calibration set's own spread:")
    for r in H:
        print("  %-12s band%% %5.1f (pool %4.1f, max %4.1f, z=%+.2f)   AUC %.4f "
              "(pool %.4f, min %.4f, z=%+.2f)"
              % (r["v"], 100 * r["band"], 100 * bd.mean(), 100 * bd.max(),
                 (r["band"] - bd.mean()) / bd.std(ddof=1), r["auc"], au.mean(), au.min(),
                 (r["auc"] - au.mean()) / au.std(ddof=1)))

    (REPO / args.save).write_text(json.dumps(rows, default=float))
    print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
