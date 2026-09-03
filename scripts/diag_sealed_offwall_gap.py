"""Why is off-wall generalisation 0.618 on SEALED against 0.835 in cross-validation?

THIS SPENDS NOTHING FURTHER.  The SEALED read was taken on 2026-09-03 and those four vessels
are burnt.  Re-reading the predictions already made is not a second read; what would be a
second read is *tuning* on them, and nothing here does that -- the oracle cut computed below
is a DIAGNOSTIC of how much of the gap is readout, and is never fed back into the artifact.

THE QUESTION.  Off-wall lands 0.217 below its cross-validated value, three times the +/-0.074
noise floor.  There are two candidate causes and they point at completely different work:

    readout        the score field RANKS the off-wall clot correctly on these vessels, but the
                   cohort-fitted cut sits in the wrong place for them.  Fixable, and cheaply --
                   this is the `readout_gap` that MODEL_REVIEW_2026-08-22 8f already closed
                   once for the training cohort with the C0 constraint.
    representation the field does not rank it.  No cut recovers that, and the next build is a
                   different model rather than a different threshold.

The separator is the PER-VESSEL ORACLE CUT: the best score this ranking admits on each vessel.
If oracle >> deployed, the ranking is fine and the cut is wrong.  If oracle ~ deployed, the
ranking is the problem.

    python scripts/diag_sealed_offwall_gap.py
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

from src.clot_ml.data import eval_domains  # noqa: E402
from src.clot_ml.locked import build_sample, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.v0 import load_v0_bundle, solve_fem_into_pack  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import SEALED  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GRID = np.round(np.linspace(0.02, 0.98, 49), 4)


def probe(stem: str, bundle, flow: str) -> dict:
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    d.graph_stem = stem
    if flow == "fem":
        solve_fem_into_pack(d)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    S = build_sample(d, bio, flow=flow, variant="v4")
    T = int(d.y.shape[0])
    gt = (gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu"))
          .reshape(-1).numpy() > 0.5)

    ens = bundle["base"]["base"]["ens"] if "base" in bundle["base"] else bundle["base"]["ens"]
    sc = predict_scores(ens, S)

    wall, off = eval_domains(S)
    vs = SeverityScorer(S["edge_index"], gt, len(wall), DEFAULT)
    out = dict(stem=stem, n_off=int(off.sum()), n_off_gt=int((gt & off).sum()),
               n_wall_gt=int((gt & wall).sum()))
    for name, dom in (("wall", wall), ("off", off)):
        if not (gt & dom).any():
            out[f"{name}_oracle"] = float("nan")
            out[f"{name}_oracle_cut"] = float("nan")
            out[f"{name}_auc"] = float("nan")
            continue
        best, bt = -1.0, float("nan")
        for t in GRID:
            v = vs.score(dom & (sc >= t), dom, empty_gt="nan")
            if v == v and v > best:
                best, bt = float(v), float(t)
        out[f"{name}_oracle"] = best
        out[f"{name}_oracle_cut"] = bt
        y, x = gt[dom], sc[dom]
        if y.any() and (~y).any():
            from sklearn.metrics import roc_auc_score
            out[f"{name}_auc"] = float(roc_auc_score(y, x))
        else:
            out[f"{name}_auc"] = float("nan")
    # where the deployed field actually sits, for the off-wall domain
    out["off_score_p50"] = float(np.median(sc[off]))
    out["off_score_p99"] = float(np.percentile(sc[off], 99))
    out["off_gt_score_p50"] = (float(np.median(sc[off & gt])) if (off & gt).any()
                               else float("nan"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DeployClot_0")
    ap.add_argument("--flow", default="fem")
    ap.add_argument("--out", default="outputs/deployclot/sealed_offwall_gap.json")
    args = ap.parse_args()

    bundle = load_v0_bundle(args.model)
    deployed = {r["stem"]: r for r in json.loads(
        (REPO / "outputs/deployclot/eval_sealed.json").read_text())}

    rows = [probe(s, bundle, args.flow) for s in SEALED if (PACKS / f"{s}.pt").exists()]
    print()
    print("SEALED, off-wall: what the ranking admits against what the shipped cut takes")
    print(f"{'vessel':14s} {'off GT':>7s} {'deployed':>9s} {'oracle':>8s} {'gap':>8s} "
          f"{'AUC':>7s} {'oracle cut':>11s} {'GT p50':>8s} {'all p99':>8s}")
    for r in rows:
        dep = deployed.get(r["stem"], {}).get("v0_fin_off", float("nan"))
        gap = (r["off_oracle"] - dep) if (r["off_oracle"] == r["off_oracle"]
                                          and dep == dep) else float("nan")
        print(f"{r['stem']:14s} {r['n_off_gt']:7d} {dep:9.4f} {r['off_oracle']:8.4f} "
              f"{gap:+8.4f} {r['off_auc']:7.4f} {r['off_oracle_cut']:11.2f} "
              f"{r['off_gt_score_p50']:8.4f} {r['off_score_p99']:8.4f}")

    dep = [deployed.get(r["stem"], {}).get("v0_fin_off", float("nan")) for r in rows]
    orc = [r["off_oracle"] for r in rows]
    auc = [r["off_auc"] for r in rows]
    md, mo, ma = (float(np.nanmean(dep)), float(np.nanmean(orc)), float(np.nanmean(auc)))
    print(f"{'MEAN':14s} {'':7s} {md:9.4f} {mo:8.4f} {mo - md:+8.4f} {ma:7.4f}")
    print()
    print(f"deployed off-wall            {md:.4f}")
    print(f"per-vessel oracle cut        {mo:.4f}   <- what this RANKING admits")
    print(f"readout gap on SEALED        {mo - md:+.4f}")
    print(f"off-wall ranking AUC         {ma:.4f}")
    print()
    print("Cross-validated off-wall was 0.8351 and its readout gap 0.045 "
          "(MODEL_REVIEW_2026-08-22 8f).")
    print("If the SEALED gap is much larger, the ranking transfers and the CUT does not.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(model=args.model, flow=args.flow, per_vessel=rows,
             deployed_mean=md, oracle_mean=mo, readout_gap=mo - md, auc_mean=ma,
             note="diagnostic only; the oracle cut is never fed back into the artifact"),
        indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
