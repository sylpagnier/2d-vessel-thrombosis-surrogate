"""FEM-vs-GT t=0 flow audit over the whole pack corpus.

The `fem` arm is the deploy-legal flow source for `DeployClot`.  `flow="pred"` needed a
fitted `dsrx` gain because the RGP-DEQ surrogate is amplitude-deficient at the wall
(docs/DEPLOY_FLOW_PLAN.md 1d).  `fem` currently takes hops=3 and gain=1.0 on the argument
that a converged Carreau solve is on COMSOL's own scale.  This measures whether that holds,
per vessel, in the four quantities the clot gate actually consumes:

    u/v      rel L2 against COMSOL's t=0 field
    sr       wall shear rate -- correlation and median ratio
    dsrx     wall d(sr)/dx  -- correlation and median-|.| ratio  (the gate's dominant input)
    gate     union Jaccard against the GT gate, and fire-rate ratio

    python scripts/diag_fem_flow_audit.py --out outputs/deployclot/fem_flow_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.v0 import solve_fem_into_pack  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _rel_l2(a, b) -> float:
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(b)
    return float(d / n) if n > 0 else float("nan")


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-30 or b.std() < 1e-30:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _jac(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def audit_one(stem: str) -> dict:
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    bio = BiochemConfig(phase="biochem")
    wall = d.mask_wall.reshape(-1).bool().numpy()
    t0 = time.time()
    solve_fem_into_pack(d)
    t_solve = time.time() - t0

    g = t0_flow_fields(d, bio, hops=3, flow_source="gt")
    f = t0_flow_fields(d, bio, hops=3, flow_source="fem")

    def med_ratio(a, b, m):
        bb = float(np.median(np.abs(b[m])))
        return float(np.median(np.abs(a[m])) / bb) if bb > 0 else float("nan")

    row = dict(
        stem=stem, n=int(d.x.shape[0]), n_wall=int(wall.sum()), solve_s=round(t_solve, 2),
        rel_l2_u=_rel_l2(f.u, g.u), rel_l2_v=_rel_l2(f.v, g.v),
        rel_l2_uv=_rel_l2(np.stack([f.u, f.v]), np.stack([g.u, g.v])),
        sr_corr_wall=_corr(f.sr[wall], g.sr[wall]),
        sr_ratio_wall=med_ratio(f.sr, g.sr, wall),
        dsrx_corr_wall=_corr(f.dsrx[wall], g.dsrx[wall]),
        dsrx_ratio_wall=med_ratio(f.dsrx, g.dsrx, wall),
        gate_jac=_jac(f.gate[wall] > 0, g.gate[wall] > 0),
        gate_low_jac=_jac(f.gate_low[wall] > 0, g.gate_low[wall] > 0),
        gate_sep_jac=_jac(f.gate_sep[wall] > 0, g.gate_sep[wall] > 0),
        gate_fire_gt=float((g.gate[wall] > 0).mean()),
        gate_fire_fem=float((f.gate[wall] > 0).mean()),
    )
    row["gate_fire_ratio"] = (row["gate_fire_fem"] / row["gate_fire_gt"]
                              if row["gate_fire_gt"] > 0 else float("nan"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/deployclot/fem_flow_audit.json")
    ap.add_argument("--stems", nargs="*", default=None)
    args = ap.parse_args()

    stems = args.stems or sorted(p.stem for p in PACKS.glob("*.pt"))
    rows = []
    for s in stems:
        try:
            r = audit_one(s)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {s}: {e}", flush=True)
            continue
        rows.append(r)
        print("[ok  ] %-18s relL2 %.3f  sr r=%.3f x%.2f  dsrx r=%+.3f x%.2f  "
              "gateJ %.3f  fire x%.2f  (%.0fs)"
              % (s, r["rel_l2_uv"], r["sr_corr_wall"], r["sr_ratio_wall"],
                 r["dsrx_corr_wall"], r["dsrx_ratio_wall"], r["gate_jac"],
                 r["gate_fire_ratio"], r["solve_s"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def agg(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return (float(np.median(v)), float(np.percentile(v, 10)),
                float(np.percentile(v, 90))) if v else (float("nan"),) * 3

    print("\nCOHORT (median [p10, p90]), n=%d" % len(rows))
    for k in ("rel_l2_uv", "sr_corr_wall", "sr_ratio_wall", "dsrx_corr_wall",
              "dsrx_ratio_wall", "gate_jac", "gate_sep_jac", "gate_low_jac",
              "gate_fire_ratio"):
        m, lo, hi = agg(k)
        print("  %-16s %8.4f  [%8.4f, %8.4f]" % (k, m, lo, hi))
    print("[save] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
