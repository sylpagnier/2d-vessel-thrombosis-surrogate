"""Can the local FEM solve be made accurate on the five vessels where it is not?

docs/DEPLOYCLOT.md 3b: the deploy penalty is not spread evenly.  On the 22 vessels where the
solve reproduces COMSOL to rel L2 <= 0.03 the paired wall loss is -0.0160, inside the noise
floor; on the other five it is -0.0821.  So the whole cost of deploying on solved flow is
carried by `patient012`, `041`, `042`, `045` and `046`, and 1a already localised the error on
045/046 -- zero at the wall, confined to one downstream recirculation window, on the two
highest-peak-velocity vessels in the corpus.

That is the signature of STABILISATION, not of the physics: `solve_local_t0_flow` adds SUPG
crosswind diffusion at `art_visc = 0.70`, which is a numerical smear whose whole purpose is to
damp exactly the kind of sharp shear layer a recirculation carries.  A smear that is harmless
on a smooth Poiseuille-like vessel is not harmless on a jet.

This sweeps the stabilisation on the affected vessels and on two controls, and asks whether
any setting improves the outliers WITHOUT moving the vessels that are already right.  A change
that helps 045 and costs 020 is not an improvement, it is a re-allocation.

    python scripts/diag_fem_stabilisation_sweep.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.v0 import _resolve_anchor_mesh  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.local_fem_solver import solve_local_t0_flow  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"

#: the five vessels carrying the deploy penalty, plus two the solver already gets right --
#: the controls are the point, since a stabilisation change is global.
OUTLIERS = ("patient045", "patient046", "patient012", "patient041", "patient042")
CONTROLS = ("patient020", "patient044")

#: (art_visc, stab).  0.70/"iso" is what ships.
ARMS = ((0.70, "iso"), (0.35, "iso"), (0.15, "iso"), (0.0, "iso"))


def _rel_l2(a, b) -> float:
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n > 0 else float("nan")


def _jac(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def run(stem: str, art_visc: float, stab: str) -> dict:
    logging.disable(logging.INFO)
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    bio = BiochemConfig(phase="biochem")
    wall = d.mask_wall.reshape(-1).bool().numpy()

    y = d.y
    u_gt = y[0, :, 0:2].detach().cpu().numpy()
    u_gt = u_gt if np.isfinite(u_gt).all() and np.abs(u_gt).max() > 0 else None

    t0 = time.time()
    u_dim = solve_local_t0_flow(_resolve_anchor_mesh(d), d, PhysicsConfig(), max_iters=300,
                                tol=1e-9, u_gt_inlet_nd=u_gt, art_visc=art_visc, stab=stab,
                                verbose=False)
    if isinstance(u_dim, torch.Tensor):
        u_dim = u_dim.numpy()
    took = time.time() - t0
    nd = u_dim / float(d.u_ref.reshape(-1)[0])
    d.u0_pred = torch.tensor(nd[:, 0], dtype=torch.float32)
    d.v0_pred = torch.tensor(nd[:, 1], dtype=torch.float32)

    g = t0_flow_fields(d, bio, hops=3, flow_source="gt")
    f = t0_flow_fields(d, bio, hops=3, flow_source="fem")
    return dict(stem=stem, art_visc=art_visc, stab=stab, solve_s=round(took, 1),
                rel_l2=_rel_l2(np.stack([f.u, f.v]), np.stack([g.u, g.v])),
                sr_corr=float(np.corrcoef(f.sr[wall], g.sr[wall])[0, 1]),
                gate_jac=_jac(f.gate[wall] > 0, g.gate[wall] > 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/deployclot/fem_stabilisation_sweep.json")
    ap.add_argument("--stems", nargs="*", default=None)
    args = ap.parse_args()

    stems = args.stems or list(OUTLIERS) + list(CONTROLS)
    rows = []
    for stem in stems:
        tag = "OUTLIER" if stem in OUTLIERS else "control"
        for av, st in ARMS:
            try:
                r = run(stem, av, st)
            except Exception as e:  # noqa: BLE001
                print(f"[ERR ] {stem} art_visc={av} {st}: {e}", flush=True)
                continue
            r["role"] = tag
            rows.append(r)
            print(f"[{tag:7s}] {stem:12s} art_visc={av:<5.2f} {st:4s}  relL2 {r['rel_l2']:.4f}  "
                  f"sr r {r['sr_corr']:.4f}  gateJ {r['gate_jac']:.4f}  ({r['solve_s']:.0f}s)",
                  flush=True)

    print()
    print(f"{'vessel':14s} {'role':8s} " + " ".join(f"{f'av={a:g}':>10s}" for a, _ in ARMS))
    for stem in stems:
        cells = []
        for av, st in ARMS:
            m = [r for r in rows if r["stem"] == stem and r["art_visc"] == av]
            cells.append(f"{m[0]['rel_l2']:10.4f}" if m else f"{'--':>10s}")
        role = "OUTLIER" if stem in OUTLIERS else "control"
        print(f"{stem:14s} {role:8s} " + " ".join(cells))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
