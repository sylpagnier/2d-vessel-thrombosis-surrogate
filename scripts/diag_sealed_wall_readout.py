"""Is a low wall score the NETWORK failing to separate, or the READOUT landing wrong?

The distinction decides what to do about it, and the two SEALED-VIZ vessels answer it
differently -- which is why a single cohort number hid it:

    patient001   oracle cut 0.9799, shipped readout 0.7406   -> the READOUT
    patient042   oracle cut 0.7677, shipped readout 0.7179   -> the NETWORK

For every vessel, at the FINAL time on the WALL domain, this reports the raw ensemble score
distribution, the statistic the shipped adaptive cut leans on, and the score under three
readouts: the shipped `resid_adapt`, the same cuts with the adaptive slope off (`b=0`), and
a per-vessel ORACLE cut -- the ceiling the network's own ranking allows.

The pool rows are IN-SAMPLE (the shipped ensemble trained on all 19) and are here for the
contrast, not as an estimate: the gap between the shipped readout and the oracle is 0.006 on
the pool and 0.145 off it, which is the whole point of the script.

VIZ_HALF only -- see docs/SEALED_SPLIT.md.

    python scripts/diag_sealed_wall_readout.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import GRID, apply_adapt, readout_resid  # noqa: E402
from src.clot_ml.locked import build_sample, load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
LOCKED = REPO / "outputs/clot_ml/locked/clot_gnn_v4"
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def wall_of(S):
    return S["wall"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="patient042,patient001")
    ap.add_argument("--save", default="outputs/diag_sealed_wall_readout.json")
    args = ap.parse_args()

    man = json.loads((LOCKED / "manifest.json").read_text())
    pool = man["training_pool"]
    held = [a for a in args.held.split(",") if a]
    for a in held:
        assert a not in FINAL_HALF, "FINAL_HALF is SEALED -- docs/SEALED_SPLIT.md"

    with (LOCKED / man["temporal_file"]).open("rb") as fh:
        temporal = pickle.load(fh)
    spec = temporal["wall_spec"]
    assert spec["kind"] == "resid_adapt", "this script reads the resid_adapt wall spec"
    th, b, med = tuple(spec["th"]), spec["b"], spec["med"]
    print("shipped wall spec: kind=%s th=%s b=%s med=%.4f\n" % (spec["kind"], th, b, med))

    ens = load_ensemble(name="clot_gnn_v4")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    rows = []
    for a in pool + held:
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        S = build_sample(d, bio, phys, flow="gt", variant="v4")
        sc = predict_scores(ens, S)
        w = S["wall"].astype(bool)
        T = len(d.t.reshape(-1))
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu")).numpy() > 0.5
        sv = SeverityScorer(S["edge_index"], gt, len(w), DEFAULT)

        stat = float(sc[w].mean())
        m_ship = apply_adapt(S, sc, "resid", th, wall_of, b, med) & w
        m_b0 = readout_resid(S, sc, th) & w
        s_or, t_or = max((sv.score(w & (sc >= t), w), float(t)) for t in GRID)
        rows.append(dict(v=a, held=a in held, stat=stat, shift=b * (stat - med),
                         n_gt=int((gt & w).sum()), n_ship=int(m_ship.sum()),
                         n_b0=int(m_b0.sum()), ship=sv.score(m_ship, w),
                         b0=sv.score(m_b0, w), oracle=s_or, t_oracle=t_or,
                         physfrac=float((S["phys_mask"] & w).sum() / w.sum()),
                         q90=float(np.quantile(sc[w], 0.9))))
        r = rows[-1]
        print("%-12s%s stat=%.3f shift=%+.3f physfrac=%.2f | n_gt=%4d n_ship=%4d n_b0=%4d "
              "| ship=%.4f b0=%.4f oracle=%.4f@%.2f"
              % (a, "*HELD" if r["held"] else "     ", r["stat"], r["shift"], r["physfrac"],
                 r["n_gt"], r["n_ship"], r["n_b0"], r["ship"], r["b0"], r["oracle"],
                 r["t_oracle"]), flush=True)

    P = [r for r in rows if not r["held"]]
    H = [r for r in rows if r["held"]]

    def avg(rs, k):
        return float(np.mean([r[k] for r in rs]))

    print("\n%-20s %7s %8s %7s %7s %8s %s" % ("", "stat", "shift", "ship", "b0", "oracle",
                                              "n_ship/n_gt"))
    for nm, rs in (("POOL (in-sample)", P), ("HELD OUT", H)):
        print("%-20s %7.3f %+8.3f %7.4f %7.4f %8.4f   %.2f"
              % (nm, avg(rs, "stat"), avg(rs, "shift"), avg(rs, "ship"), avg(rs, "b0"),
                 avg(rs, "oracle"), np.mean([r["n_ship"] / max(r["n_gt"], 1) for r in rs])))
    print("\nreadout headroom (oracle - shipped):  pool %+.3f   held out %+.3f"
          % (avg(P, "oracle") - avg(P, "ship"), avg(H, "oracle") - avg(H, "ship")))
    print("adapt statistic range on the pool: [%.3f, %.3f]   held out: %s"
          % (min(r["stat"] for r in P), max(r["stat"] for r in P),
             ", ".join("%s %.3f" % (r["v"][-3:], r["stat"]) for r in H)))

    (REPO / args.save).write_text(json.dumps(rows, default=float))
    print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
