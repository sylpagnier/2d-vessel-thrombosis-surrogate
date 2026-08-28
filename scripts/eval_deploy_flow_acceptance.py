"""T8: the acceptance test for a Stage-A retrain -- the clot_ml_0 GT-vs-pred deploy gap.

    python scripts/eval_deploy_flow_acceptance.py --checkpoint outputs/kinematics/<run>/kinematics_best.pth

**Why this and not a Stage-A metric.**  The stated goal is that `clot_ml_0` scores the same
under `flow="pred"` as under `flow="gt"`.  Today that gap is `-0.366 wall / -0.478 off-wall`
(`DEPLOY_FLOW_PLAN.md` §2).  A Stage-A number that improves without moving this has not done
the job -- and the project has a worked example: the width fix halved velocity rel-L2 and made
the frozen clot model *worse*.

Reads the GT arm too, because a pred-arm gain bought by a GT-arm regression is not a gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="",
                    help="Stage-A checkpoint to accept/reject. Default: the resolved one.")
    ap.add_argument("--prior-source", default="analytic",
                    help="Prior block for the pred arm. 'stored' is the s17 Z2 leak.")
    ap.add_argument("--anchors", default="", help="comma list; default = FIT + DEV")
    ap.add_argument("--out", default="outputs/deploy_flow_acceptance.json")
    ap.add_argument("--skip-promotion-check", action="store_true",
                    help="evaluate a checkpoint that has not been promoted (diagnostics only)")
    args = ap.parse_args()

    import torch

    from src.core_physics.wall_cohort_splits import DEV, FIT
    from src.data_gen.lib.legal_priors import apply_prior_source
    from src.utils.kinematics_inference import (
        assert_promotable_checkpoint, load_kinematics_predictor, predict_kinematics,
        resolve_kinematics_checkpoint,
    )
    from src.utils.kinematics_selection import wall_shear_selection_metrics

    ckpt = resolve_kinematics_checkpoint(args.checkpoint or None)
    if not args.skip_promotion_check:
        try:
            meta = assert_promotable_checkpoint(ckpt)
            print(f"[i] checkpoint promotable: {meta}")
        except ValueError as exc:
            print(f"[ERR] {exc}\n[ERR] pass --skip-promotion-check to evaluate it anyway.")
            return 1

    anchors = ([a.strip() for a in args.anchors.split(",") if a.strip()]
               or sorted(set(FIT) | set(DEV)))
    graph_dir = REPO / "data/processed/graphs_biochem_anchors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_kinematics_predictor(ckpt, device)

    rows = []
    for stem in anchors:
        f = graph_dir / f"{stem}.pt"
        if not f.is_file():
            continue
        data = torch.load(f, map_location="cpu", weights_only=False)
        if getattr(data, "y", None) is None:
            continue
        try:
            g = apply_prior_source(data, args.prior_source)
            for attr in ("_cache_key", "_cache_pred", "_cache_latent"):
                if hasattr(model, attr):
                    setattr(model, attr, None)
            with torch.no_grad():
                pred = predict_kinematics(model, g.to(device)).detach().cpu()
            m = wall_shear_selection_metrics(pred[:, :2], data)
        except Exception as exc:  # one bad vessel must not kill the run
            print(f"[warn] {stem}: {type(exc).__name__}: {exc}")
            continue
        split = "FIT" if stem in set(FIT) else ("DEV" if stem in set(DEV) else "other")
        m.update(vessel=stem, split=split)
        rows.append(m)
        print(f"{stem:<20}{split:<5} dsrx_corr={m.get('dsrx_corr', float('nan')):+.3f} "
              f"gate_J={m.get('gate_jaccard', float('nan')):.3f} "
              f"scale={m.get('dsrx_scale', float('nan')):.3f}")

    def agg(split):
        import math
        v = [r for r in rows if split is None or r["split"] == split]
        out = {"n": len(v)}
        for k in ("dsrx_corr", "gate_jaccard", "dsrx_scale", "sr_scale", "gate_fire_ratio"):
            vals = [r[k] for r in v if k in r and math.isfinite(r[k])]
            out[k] = sum(vals) / len(vals) if vals else float("nan")
        return out

    summary = {"checkpoint": str(ckpt), "prior_source": args.prior_source,
               "ALL": agg(None), "FIT": agg("FIT"), "DEV": agg("DEV")}
    print("\n" + json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\n[save] {args.out}")

    print("""
ACCEPT / REJECT -- read in this order (T7, ordering MEASURED not assumed):
  1. GATE UNION JACCARD -- this is the one that predicts the downstream outcome.
     Against the locked clot ensemble's own oracle-F1 (12 vessels x 2 arms):
         gate_jaccard  pearson +0.918   spearman +0.904
         dsrx_corr     pearson +0.431   spearman +0.555
     and WITHIN a single flow arm dsrx_corr reads -0.073, i.e. no relationship at all.
  2. wall dsrx correlation -- a DIAGNOSTIC (it explains gate failures), not a selector.
  3. then rebuild the pred cache and score clot_ml_0 end to end.

Baselines on the real clot task (locked clot_gnn_v6, mean oracle-F1 over 12 vessels):
    GT flow                    0.882
    RGP-DEQ (leak-assisted)    0.675     <- what the retrain must reach on LEGAL priors
    analytic prior alone       0.370
The surrogate is worth +0.305 oracle-F1 over the closed-form prior, so it does earn its place
-- a conclusion that velocity rel-L2, cos and AUC-of-speed all got wrong.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
