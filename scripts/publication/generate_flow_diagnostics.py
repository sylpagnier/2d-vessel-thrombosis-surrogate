"""Per-vessel flow-quality diagnostics: the candidate health checks for panel (a).

The question the panel answers is *which statistic tells you a flow surrogate will break the
clot readout*.  `eval_clot_ml_0.py` emits the OUTCOME (the score under each flow) but none of
the CANDIDATE PREDICTORS, so this script computes them, on the same packs, for the same cohort.

Each diagnostic is measured on WALL nodes, because that is where the deposition gate is
evaluated and therefore the only place its agreement matters:

  rel_l2        velocity rel-L2 of the surrogate against GT.  The conventional metric, and the
                one the paper argues is uninformative.
  gate_jaccard  Jaccard overlap of the FIRING SET {gate > 0} between GT and surrogate.
  fire_ratio    how many wall nodes the surrogate's gate fires on, over GT's.  1.0 is matched;
                0.0 means it fires nowhere.
  empty_gate    1.0 when the surrogate's wall gate fires on NOTHING -- the discontinuity, since
                the readout seeds from `(gate > 0) & wall` and an empty seed zeroes the
                downstream channels outright.
  dsrx_corr     Pearson correlation of the shear-gradient field, GT vs surrogate, on the wall.

The gate is recomputed here exactly as `src/clot_ml/features.py` builds it -- same `lss`/`sgt`
constants, same `_flow_hops` stencil per flow source, same `dsrx_gain` amplitude correction --
so the numbers are the consumer's own, not a re-derivation that might differ in a detail.

Usage:
    python scripts/publication/generate_flow_diagnostics.py            # cohort
    python scripts/publication/generate_flow_diagnostics.py --stems comsol010 comsol005
"""
from __future__ import annotations
from src.utils.units import M_TO_CM
from src.utils.paths import anchor_packs_dir

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402

PACKS = anchor_packs_dir()

def _cohort_stems() -> list[str]:
    skip = set(SEALED) | set(CLOT_FREE)
    out = [a for a in list(FIT) + list(DEV)
           if a not in skip and (PACKS / f"{a}.pt").exists()]
    for s in ("wound_comsol001", "wound_comsol002", "wound_comsol003"):
        if (PACKS / f"{s}.pt").exists() and s not in out:
            out.append(s)
    return out


def _fields(data, flow: str, bio) -> dict:
    """Shear, shear-gradient and gate at t=0 under one flow source.

    The gate itself comes from `src/clot_ml/preflight.py`, which is also what the shipped
    deploy path calls -- one definition, so the diagnostic and the pre-flight check can never
    disagree about what "the gate fires" means.
    """
    from src.clot_ml.preflight import wall_gate_firing

    if flow == "gt":
        u = data.y[0, :, 0].reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.y[0, :, 1].reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)

    gate, _wall = wall_gate_firing(data, flow, bio)

    # `dsrx` is reported on its own (it is one of the candidate diagnostics), so recompute the
    # gradient here rather than widening preflight's return for a diagnostic-only quantity.
    from src.clot_ml.temporal import _flow_hops
    from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d
    from src.core_physics.physics_wall_model import dsrx_gain

    ei = data.edge_index.detach().cpu().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    Dx, Dy = build_mls_gradient(node_positions(data), ei, hops=_flow_hops(flow))
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = ((Dx @ sr) / (d_bar * M_TO_CM)) * dsrx_gain(flow)
    return {"u": u, "v": v, "sr": sr, "dsrx": dsrx, "gate": gate}


def _diagnostics(gt: dict, pr: dict, wall: np.ndarray) -> dict:
    fire_gt = (gt["gate"] > 0) & wall
    fire_pr = (pr["gate"] > 0) & wall
    inter = int((fire_gt & fire_pr).sum())
    union = int((fire_gt | fire_pr).sum())
    n_gt, n_pr = int(fire_gt.sum()), int(fire_pr.sum())

    num = np.linalg.norm(np.stack([pr["u"] - gt["u"], pr["v"] - gt["v"]]))
    den = np.linalg.norm(np.stack([gt["u"], gt["v"]]))

    a, b = gt["dsrx"][wall], pr["dsrx"][wall]
    ok = np.isfinite(a) & np.isfinite(b)
    dsrx_corr = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                 if ok.sum() >= 3 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan"))

    return {
        "rel_l2": float(num / den) if den > 0 else float("nan"),
        "gate_jaccard": float(inter / union) if union else float("nan"),
        "fire_ratio": float(n_pr / n_gt) if n_gt else float("nan"),
        "empty_gate": 1.0 if n_pr == 0 else 0.0,
        "dsrx_corr": dsrx_corr,
        "n_fire_gt": n_gt,
        "n_fire_pred": n_pr,
        "n_wall": int(wall.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--flow", default="pred", choices=("pred", "fem"),
                    help="surrogate flow source to diagnose against GT")
    ap.add_argument("--out", default=str(REPO / "outputs/runs/flow_diagnostics.json"))
    a = ap.parse_args()

    stems = a.stems or _cohort_stems()
    bio = BiochemConfig(phase="biochem")
    rows, failed = [], []

    print(f"[flow-diag] {len(stems)} vessels, surrogate flow = {a.flow}\n")
    for stem in stems:
        try:
            data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
            if getattr(data, "graph_stem", None) is None:
                data.graph_stem = stem
            if a.flow == "fem":
                from src.clot_ml.v0 import solve_fem_into_pack
                solve_fem_into_pack(data)
            elif getattr(data, "u0_pred", None) is None:
                failed.append({"stem": stem, "error": "no u0_pred on pack"})
                print(f"  {stem:<20} SKIP (no u0_pred)")
                continue
            wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
            d = _diagnostics(_fields(data, "gt", bio), _fields(data, a.flow, bio), wall)
            d["stem"] = stem
            rows.append(d)
            print(f"  {stem:<20} relL2={d['rel_l2']:.4f}  gateJ={d['gate_jaccard']:.3f}  "
                  f"fire={d['fire_ratio']:.2f}  empty={int(d['empty_gate'])}  "
                  f"dsrx_r={d['dsrx_corr']:+.3f}", flush=True)
        except Exception as exc:
            failed.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {stem:<20} FAILED  {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        print("\n[flow-diag] nothing computed")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n_empty = int(sum(r["empty_gate"] for r in rows))
    print(f"\n  {len(rows)} vessels;  wall gate EMPTY on {n_empty}")
    print(f"  median gate Jaccard {np.nanmedian([r['gate_jaccard'] for r in rows]):.3f}"
          f"   median rel-L2 {np.nanmedian([r['rel_l2'] for r in rows]):.4f}")
    if failed:
        print(f"  {len(failed)} failed/skipped")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
