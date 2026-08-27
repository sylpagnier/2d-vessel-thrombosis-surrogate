"""Does the viz score the same way the CV does?  (It does not -- but it costs ~0.002.)

`scripts/gen_v4_temporal_data.py` scores with `clot_guiding` -- the flat deploy score,
`0.5*dilation_IoU + 0.5*relaxed_F0.5`, with no burden grace, and it restricts to a domain by
zeroing `pred`/`gt` outside it and then dilating over the WHOLE graph.  The CV
(`scripts/eval_strict*.py`) scores with `SeverityScorer(DEFAULT)`, which has the tau/rho
graces and clips both dilated envelopes to the domain.

Two different metric families on the same masks, so the SEALED numbers in the artifact were
not, strictly, comparable to the strict-CV numbers they were printed next to.  This measures
the difference and reports the four variants side by side, isolating the graces from the
domain clipping.  MEASURED: ~0.002 at the final time -- a real inconsistency worth fixing,
but it explains none of the SEALED gap.  Do not go looking for the answer here again.

VIZ_HALF only -- see docs/SEALED_SPLIT.md.  Looking is allowed; selecting on it is not.

    python scripts/diag_sealed_metric_gap.py --vessels patient042,patient001 --n-times 11
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
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.locked import load_default, predict_default_series  # noqa: E402
from src.clot_ml.severity_metric import (  # noqa: E402
    DEFAULT, LEGACY, SeverityScorer, dilation_operator,
)
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

PACKS = REPO / "data/processed/graphs_biochem_anchors"
FINAL_HALF = {"patient007", "patient013", "patient031", "patient043"}


def viz_score(pred_hot, gt_hot, ei_t, wall, domain_f):
    """`gen_v4_temporal_data.py::domain_score`, verbatim, so this measures what SHIPPED."""
    pred_d = torch.tensor(pred_hot.astype(np.float32)) * domain_f
    gt_d = torch.tensor(gt_hot.astype(np.float32)) * domain_f
    m = compute_clot_relaxed_metrics(pred_d, gt_d, ei_t, wall_mask=torch.tensor(wall))
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def guiding_domain_clipped(pred, gt, D, dom, beta=0.5):
    """The viz's flat guiding score with the dilations clipped to the domain, as
    `severity_components` does.  Isolates the domain leak from the grace terms."""
    g, p = gt & dom, pred & dom
    n_g, n_p = int(g.sum()), int(p.sum())
    if n_g == 0:
        return float("nan")
    if n_p == 0:
        return 0.0
    gd = ((D @ g.astype(np.int8)) > 0) & dom
    pd_ = ((D @ p.astype(np.int8)) > 0) & dom
    prec, rec = int((p & gd).sum()) / n_p, int((g & pd_).sum()) / n_g
    b2 = beta * beta
    f = 0.0 if (b2 * prec + rec) <= 0 else (1 + b2) * prec * rec / (b2 * prec + rec)
    iou = int((pd_ & gd).sum()) / max(int((pd_ | gd).sum()), 1)
    return 0.5 * iou + 0.5 * f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessels", default="patient042,patient001")
    ap.add_argument("--n-times", type=int, default=11,
                    help="grid density; 0 uses every simulated timestep, as the viz does")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    phys = PhysicsConfig(phase="biochem")
    bundle, kind = load_default()
    assert kind == "temporal_v4", f"expected clot_gnn_v4 shipped, got {kind}"

    rows = []
    for a in args.vessels.split(","):
        assert a not in FINAL_HALF, "FINAL_HALF is SEALED -- docs/SEALED_SPLIT.md"
        t0 = time.time()
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n, ei_t, T = len(wall), d.edge_index, len(d.t.reshape(-1))
        times = (list(range(T)) if args.n_times <= 0 else
                 [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)])
        series = predict_default_series(bundle, kind, d, times, flow="gt")["series"]
        D = dilation_operator(ei_t.numpy(), n, 2)
        wall_f = torch.tensor(wall.astype(np.float32))
        off_f = torch.tensor((~wall).astype(np.float32))

        per_t = {"wall": [], "off": []}
        for ti in times:
            pred = series[int(ti)]
            gt = gt_clot_phi_at_time(d, int(ti), phys,
                                     device=torch.device("cpu")).numpy() > 0.5
            sc_d = SeverityScorer(ei_t.numpy(), gt, n, DEFAULT)
            sc_l = SeverityScorer(ei_t.numpy(), gt, n, LEGACY)
            for dom_name, dom, dom_f in (("wall", wall, wall_f), ("off", ~wall, off_f)):
                per_t[dom_name].append(dict(
                    ti=int(ti), n_gt=int((gt & dom).sum()), n_pred=int((pred & dom).sum()),
                    viz=viz_score(pred, gt, ei_t, wall, dom_f),
                    viz_clipped=guiding_domain_clipped(pred, gt, D, dom),
                    sev_legacy=sc_l.score(pred & dom, dom),
                    sev_default=sc_d.score(pred & dom, dom)))
        rows.append(dict(vessel=a, T=T, n_times=len(times), per_t=per_t))

        print("\n=== %s  (T=%d, grid=%d, %.0fs) ===" % (a, T, len(times), time.time() - t0))
        print("  FINAL   n_gt  n_pred |    viz  viz_clip  sev_leg  sev_DEF(=CV metric)")
        for nm, r in (("wall", per_t["wall"][-1]), ("off ", per_t["off"][-1])):
            print("  %s   %5d %6d | %.4f   %.4f   %.4f   %.4f"
                  % (nm, r["n_gt"], r["n_pred"], r["viz"], r["viz_clipped"],
                     r["sev_legacy"], r["sev_default"]))
        for k in ("viz", "viz_clipped", "sev_legacy", "sev_default"):
            print("  mean-over-time[%-12s]  wall=%.4f  off=%.4f"
                  % (k, np.nanmean([r[k] for r in per_t["wall"]]),
                     np.nanmean([r[k] for r in per_t["off"]])))

    if args.save:
        (REPO / args.save).write_text(json.dumps(rows, default=float))
        print("\nwrote", REPO / args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
