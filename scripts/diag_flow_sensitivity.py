"""How wrong can the t=0 flow be before the clot readout breaks?

    python scripts/diag_flow_sensitivity.py patient010 patient005

Blends ground-truth and predicted velocity, `u(a) = (1-a)*u_gt + a*u_pred`, and scores the
wall/off readout at each `a`.  `a=0` must reproduce the GT score exactly (a self-check on the
whole harness); `a=1` is the deployed surrogate.  The shape of the curve between them is the
question: a straight line means the readout degrades in proportion to flow error, a cliff
means there is a tolerance to find and design against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.eval_clot_ml_0 import PACKS, _times  # noqa: E402
from scripts.eval_wound_complement import gt_series, score_domains  # noqa: E402
from src.clot_ml.data import eval_domains  # noqa: E402
from src.clot_ml.locked import (  # noqa: E402
    build_sample, load_temporal_v4_wound, predict_temporal_v4_wound)
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0, solve_fem_into_pack  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402


def score_at(bundle_base, bundle_v0, data, times, bio, phys, flow):
    S = build_sample(data, bio, flow=flow, variant="v4")
    ei = torch.tensor(np.asarray(S["edge_index"]))
    gts = gt_series(data, phys, times)
    v0 = predict_clot_ml_0(bundle_v0, data, times, flow=flow, sample=S)
    last = times[-1]
    wall, off = eval_domains(S)          # domains come from the SAMPLE, as in eval_clot_ml_0
    d = score_domains(v0["series"][last], gts[last], ei, np.asarray(wall, bool),
                      dict(wall=np.asarray(wall, bool), off=np.asarray(off, bool)))
    return float(d.get("wall", float("nan"))), float(d.get("off", float("nan")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--alphas", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    ap.add_argument("--source", default="pred", choices=("pred", "fem"))
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--v0", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default="outputs/runs/flow_sensitivity.json")
    a = ap.parse_args()

    alphas = [float(x) for x in a.alphas.split(",")]
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    bundle_base = load_temporal_v4_wound(a.baseline)
    bundle_v0 = load_v0_bundle(a.v0)

    rows = []
    for stem in a.stems:
        data = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
        data.graph_stem = stem
        times = _times(data, a.every)
        y = data.y[0] if data.y.dim() == 3 else data.y
        u_gt, v_gt = y[:, 0].clone(), y[:, 1].clone()
        if a.source == "fem":
            solve_fem_into_pack(data)
        u_pr, v_pr = data.u0_pred.clone(), data.v0_pred.clone()
        rel = float(torch.linalg.vector_norm(torch.stack([u_pr - u_gt, v_pr - v_gt]))
                    / torch.linalg.vector_norm(torch.stack([u_gt, v_gt])))
        print(f"[{stem}] source={a.source}  rel-L2(pred,gt) = {rel:.4f}", flush=True)
        for al in alphas:
            data.u0_pred = (1 - al) * u_gt + al * u_pr
            data.v0_pred = (1 - al) * v_gt + al * v_pr
            w, o = score_at(bundle_base, bundle_v0, data, times, bio, phys, "pred")
            rows.append(dict(stem=stem, source=a.source, alpha=al, rel_at_alpha=al * rel,
                             wall=w, off=o))
            print(f"   a={al:<5} rel {al*rel:6.4f}   wall {w:.3f}   off {o:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[save] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
