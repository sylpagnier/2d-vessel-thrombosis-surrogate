"""The matched A/B counterfactual: one vessel outline, simulated with and without a wound.

docs/WOUND_PROGRESS.md 7 named this the single most useful missing simulation --

    "There is no paired A/B.  `wound_comsol001` is *not* the same vessel as `comsol001` ...
     Nothing here isolates the wound's effect on a fixed geometry.  Re-running one existing
     cohort `.nas` with and without the `sel1` selection is the single most useful next
     simulation."

It exists now: `wound_comsol005` and `comsol048` share a vessel outline to a median
wall-node distance of 0.0000 nd (remeshed, so the node sets differ; every one of
`comsol048`'s nodes registers onto `wound_comsol005` within 0.22% of the domain span, and
exactly 58 of them land on the 58 wound nodes).

WHY IT IS WORTH ITS OWN SCRIPT.  Every other number in this project scores a model against
one vessel's labels.  This one scores the DIFFERENCE: with geometry, inflow, mesh family and
physics all held fixed, whatever changes between the two runs is the injury and nothing else.
So it asks the question a per-vessel score cannot -- does the model reproduce the EFFECT of
the wound, or does it merely score two vessels acceptably for unrelated reasons?

    D_gt   = clot(wound) AND NOT clot(no wound)     on the shared node set, matched HORIZON
    D_pred = the same, from the model

and the readout is how well `D_pred` recovers `D_gt`.  A model that predicted each vessel
perfectly *except* that it attributed the extra clot to the wrong place would score well
per-vessel and near zero here.

HORIZON MATCHING IS NOT OPTIONAL.  The wound run stops at 11975 s and the no-wound run at
30000 s, and "final Mat" is a horizon quantity (WOUND_PROGRESS 7).  Both sides are read at
the last time index the WOUND run reaches, matched by simulated seconds, not by index.

    python scripts/eval_wound_ab_pair.py --model clot_ml_0 --flow fem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.biochem_gnn.wall_cohort_constants import WOUND_AB_PAIR  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.v0 import load_v0_bundle, predict_clot_ml_0, solve_fem_into_pack  # noqa: E402
from src.clot_ml.wound import solid_mask  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def _load(stem: str, flow: str):
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    if flow == "fem":
        solve_fem_into_pack(d)
    return d


def _gt_mask(d, ti: int) -> np.ndarray:
    phys = PhysicsConfig(phase="biochem")
    return (gt_clot_phi_at_time(d, int(ti), phys, device=torch.device("cpu"))
            .reshape(-1).numpy() > 0.5)


def _agree(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Precision / recall / F1 / IoU of one boolean field against another."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    ok = prec == prec and rec == rec and (prec + rec) > 0
    f1 = (2 * prec * rec / (prec + rec)) if ok else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return dict(n_gt=int(gt.sum()), n_pred=int(pred.sum()), tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1, iou=iou)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--flow", default="fem", choices=["gt", "pred", "fem"])
    ap.add_argument("--every", type=int, default=4, help="time-grid stride for the rollout")
    ap.add_argument("--wound", default=WOUND_AB_PAIR[0])
    ap.add_argument("--nowound", default=WOUND_AB_PAIR[1])
    ap.add_argument("--out", default="outputs/deployclot/ab_pair.json")
    args = ap.parse_args()

    print(f"[i] A/B pair: {args.wound} (wound) vs {args.nowound} (no wound), flow={args.flow}",
          flush=True)
    A, B = _load(args.wound, args.flow), _load(args.nowound, args.flow)
    tA, tB = np.asarray(A.t).reshape(-1), np.asarray(B.t).reshape(-1)
    iA = int(len(tA) - 1)
    iB = int(np.argmin(np.abs(tB - tA[iA])))
    print(f"[i] matched horizon: wound idx {iA} (t={tA[iA]:.0f}s) <-> "
          f"no-wound idx {iB} (t={tB[iB]:.0f}s)", flush=True)

    # Register B onto A.  B is the sparser mesh, so B's node set is the common frame and the
    # map is B -> A; going the other way would leave A's extra nodes unmatched.
    pA = np.asarray(A.x)[:, :2].astype(np.float64)
    pB = np.asarray(B.x)[:, :2].astype(np.float64)
    dist, nn = cKDTree(pA).query(pB)
    span = float(np.linalg.norm(pA.max(0) - pA.min(0)))
    reg = dict(median=float(np.median(dist)), p99=float(np.percentile(dist, 99)),
               max=float(dist.max()), span=span, frac_of_span=float(dist.max() / span))
    print("[i] registration B->A: median %.2e, max %.2e (%.3f%% of span)"
          % (reg["median"], reg["max"], 100 * reg["frac_of_span"]), flush=True)

    bundle = load_v0_bundle(args.model)
    outs = {}
    for tag, d, last in (("wound", A, iA), ("nowound", B, iB)):
        T = int(d.y.shape[0])
        grid = sorted({*range(0, last + 1, max(args.every, 1)), last})
        print(f"[i] rollout {tag} ({T} frames, {len(grid)} scored) ...", flush=True)
        r = predict_clot_ml_0(bundle, d, grid, flow=args.flow)
        outs[tag] = dict(pred=np.asarray(r["series"][last], bool), gt=_gt_mask(d, last),
                         solid=solid_mask(d), T=T, last=last)

    # ---- per-vessel severity, for context ------------------------------------------------
    per_vessel = {}
    for tag, d in (("wound", A), ("nowound", B)):
        o = outs[tag]
        sc = SeverityScorer(np.asarray(d.edge_index), o["gt"], len(o["gt"]), DEFAULT)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        per_vessel[tag] = dict(
            stem=(args.wound if tag == "wound" else args.nowound),
            wall=float(sc.score(o["pred"] & wall, wall, empty_gt="nan")),
            off=float(sc.score(o["pred"] & ~o["solid"], ~o["solid"], empty_gt="nan")),
            gt_burden=int(o["gt"].sum()), pred_burden=int(o["pred"].sum()))

    # ---- the counterfactual, on the shared frame -----------------------------------------
    gtA, gtB = outs["wound"]["gt"][nn], outs["nowound"]["gt"]
    prA, prB = outs["wound"]["pred"][nn], outs["nowound"]["pred"]
    d_gt = gtA & ~gtB          # clot the injury CREATED
    d_pr = prA & ~prB
    r_gt = gtB & ~gtA          # clot the injury removed -- flow feedback, expected small
    r_pr = prB & ~prA

    solidB = outs["nowound"]["solid"]
    woundB = A.mask_wound.reshape(-1).bool().numpy()[nn]

    res = dict(
        pair=dict(wound=args.wound, nowound=args.nowound, flow=args.flow,
                  model=args.model, horizon_s=float(tA[iA]),
                  idx=dict(wound=iA, nowound=iB), registration=reg,
                  n_shared_nodes=int(len(pB)), n_wound_nodes=int(woundB.sum())),
        per_vessel=per_vessel,
        burden=dict(
            gt=dict(wound=int(gtA.sum()), nowound=int(gtB.sum()),
                    delta=int(gtA.sum()) - int(gtB.sum())),
            pred=dict(wound=int(prA.sum()), nowound=int(prB.sum()),
                      delta=int(prA.sum()) - int(prB.sum()))),
        created=dict(all=_agree(d_pr, d_gt),
                     on_wound=_agree(d_pr & woundB, d_gt & woundB),
                     on_solid_off_wound=_agree(d_pr & solidB & ~woundB,
                                               d_gt & solidB & ~woundB),
                     in_lumen=_agree(d_pr & ~solidB, d_gt & ~solidB)),
        removed=_agree(r_pr, r_gt),
    )

    print()
    print("PER-VESSEL (severity, matched horizon)")
    for tag in ("wound", "nowound"):
        v = per_vessel[tag]
        print("  %-9s %-18s wall %.4f  off %.4f   GT burden %5d  pred %5d"
              % (tag, v["stem"], v["wall"], v["off"], v["gt_burden"], v["pred_burden"]))
    print()
    print("COUNTERFACTUAL -- clot the injury CREATED, on the shared node set")
    b = res["burden"]
    print("  burden  GT   wound %5d  no-wound %5d  delta %+5d"
          % (b["gt"]["wound"], b["gt"]["nowound"], b["gt"]["delta"]))
    print("          pred wound %5d  no-wound %5d  delta %+5d"
          % (b["pred"]["wound"], b["pred"]["nowound"], b["pred"]["delta"]))
    print("  %-22s %6s %6s %9s %8s %7s %7s"
          % ("region", "n_gt", "n_pred", "precision", "recall", "F1", "IoU"))
    for k in ("all", "on_wound", "on_solid_off_wound", "in_lumen"):
        a = res["created"][k]
        print("  %-22s %6d %6d %9.4f %8.4f %7.4f %7.4f"
              % (k, a["n_gt"], a["n_pred"], a["precision"], a["recall"], a["f1"], a["iou"]))
    a = res["removed"]
    print("  %-22s %6d %6d %9.4f %8.4f %7.4f %7.4f"
          % ("(clot removed)", a["n_gt"], a["n_pred"], a["precision"], a["recall"],
             a["f1"], a["iou"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
