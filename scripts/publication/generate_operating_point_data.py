"""Precision-recall of the committed SET, threshold-free, with the shipped cut marked.

PUBLICATION_NOTES 7.8 item 6.  Every set number in the paper is quoted at one operating
point, and a reviewer's first question about a single cut on an imbalanced problem
(~150 positives against ~15k nodes) is "why there?".  A PR curve answers it without
re-running anything: the out-of-fold scores and the final-time labels are both already on
disk, so this is a read, not an experiment.

WHAT IS PLOTTED, precisely.

  score   `oofs[arm][a]` -- the base readout's per-node probability for vessel `a`, taken
          from the fold that held `a` out.  Genuinely out-of-fold, the same array
          `candidate_mask` thresholds.
  label   `cache[a]["y"] > 0.5` -- the FINAL-time ground-truth clot set, which is exactly
          what `eval_strict_temporal.tune_set` tunes the committed set against.
  domain  wall / off-wall, split by the same `wall_domain` / `off_domain` the deploy metric
          uses, because the two are different problems with different base rates and a
          pooled curve would hide that.

AUC-PR is reported against the base rate (the prevalence), not against 0.5: on a 1%-positive
problem a random classifier scores 0.01, so "AUC-PR 0.42" means nothing until the floor is
printed beside it.  Both are emitted.

The curve is computed by POOLING nodes across vessels, which is the right thing for the
question asked ("where should the cut go?") but not a per-vessel generalization statement --
Table 4 is that.  A per-vessel AUC-PR spread is emitted alongside so the pooled number cannot
be mistaken for a tight one.

    python scripts/publication/generate_operating_point_data.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.publication.config import CONFIG, DATA_DIR, REPO_ROOT
from scripts.publication.utils import metric_identity
from src.clot_ml.mirror_branch import eval_gt


def pr_curve(score: np.ndarray, label: np.ndarray) -> dict:
    """Precision/recall at every distinct score, plus average precision.

    Average precision is the step-wise sum ``sum_k (R_k - R_{k-1}) * P_k`` -- the estimator
    that does NOT interpolate between operating points, so it cannot report a precision the
    classifier never actually achieves.
    """
    order = np.argsort(-score, kind="mergesort")
    y = label[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = int(label.sum())
    if n_pos == 0:
        return {}
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    # Thin to at most 2000 points for a plottable payload; keep the exact endpoints.
    idx = np.unique(np.linspace(0, len(recall) - 1, min(2000, len(recall))).astype(int))
    return {
        "recall": recall[idx].tolist(),
        "precision": precision[idx].tolist(),
        "threshold": score[order][idx].tolist(),
        "average_precision": ap,
        "base_rate": float(n_pos / len(label)),
        "n_nodes": int(len(label)),
        "n_positive": n_pos,
    }


def operating_point(label: np.ndarray, mask: np.ndarray, have: np.ndarray) -> dict:
    """Precision/recall actually achieved by a committed set, however it was chosen.

    ``have`` restricts the count to nodes where a committed set EXISTS.  The shipped external
    masks cover 19 of the 27 vessels; scoring the other 8 as "committed nothing" turned every
    one of their positives into a false negative and pushed the marked recall well below what
    the shipped cut achieves.  A marker that misreports the operating point is worse than no
    marker -- the whole point of the panel is to locate it on the curve.
    """
    label, mask = label[have], mask[have]
    tp = int((mask & label).sum())
    fp = int((mask & ~label).sum())
    fn = int((~mask & label).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "precision": prec, "recall": rec,
        "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
        "n_committed": int(mask.sum()), "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    from scripts.eval_strict import load_scores
    from src.clot_ml.data import attach_physics, load_cache, off_domain, wall_domain
    from src.core_physics.wall_cohort_splits import CLOT_FREE

    arm = CONFIG.oof_arms[0]
    cache = attach_physics(load_cache(CONFIG.oof_cache))
    pool_, folds_, sc_ = load_scores(arm.split(","))
    pool = [a for a in pool_ if a in cache and a not in CLOT_FREE]
    fold_of = {a: k for k, held in folds_.items() for a in held}
    print(f"[i] arm={arm} cache={CONFIG.oof_cache}  {len(pool)} clot-carrying vessels")

    # The committed set the shipped path actually uses, where it is supplied externally.
    ext = {}
    try:
        z = np.load(REPO_ROOT / CONFIG.oof_set_masks)
        ext = {a: z[a].astype(bool) for a in z.files}
        print(f"[i] shipped committed set available for {len(ext)} vessels")
    except (OSError, ValueError) as exc:
        print(f"[!] no external set masks ({exc}); the operating point will be omitted")

    doms = {"wall": wall_domain, "off": off_domain}
    pooled = {k: {"score": [], "label": [], "committed": [], "have": []} for k in doms}
    per_vessel: list[dict] = []

    for a in pool:
        S = cache[a]
        s = np.asarray(sc_[(fold_of[a], a)], dtype=np.float64).reshape(-1)
        y = eval_gt(a, np.asarray(S["y"]).reshape(-1), S["pos"])
        m = ext.get(a)
        row = {"vessel": a, "fold": int(fold_of[a])}
        for key, dom_of in doms.items():
            d = np.asarray(dom_of(S)).reshape(-1).astype(bool)
            if d.sum() == 0 or y[d].sum() == 0:
                row[key] = None            # no positives in this domain: recall undefined
                continue
            pooled[key]["score"].append(s[d])
            pooled[key]["label"].append(y[d])
            pooled[key]["committed"].append(
                (m[d] if m is not None else np.zeros(int(d.sum()), bool)))
            pooled[key]["have"].append(np.full(int(d.sum()), m is not None, bool))
            c = pr_curve(s[d], y[d])
            row[key] = {"average_precision": c["average_precision"],
                        "base_rate": c["base_rate"], "n_positive": c["n_positive"]}
        per_vessel.append(row)

    out: dict = {
        "source": {"arm": arm, "cache": CONFIG.oof_cache, "profile": CONFIG.profile,
                   "set_masks": str(CONFIG.oof_set_masks), "n_vessels": len(pool)},
        "metric": metric_identity(),
        "note": ("Pooled over nodes across vessels: this answers 'where should the cut go', "
                 "not 'how well does it generalize per vessel' -- Table 4 is that. AUC-PR is "
                 "meaningful only against the base rate printed beside it."),
        "domains": {},
        "per_vessel": per_vessel,
    }

    for key in doms:
        if not pooled[key]["score"]:
            continue
        s = np.concatenate(pooled[key]["score"])
        y = np.concatenate(pooled[key]["label"])
        c = pr_curve(s, y)
        aps = [r[key]["average_precision"] for r in per_vessel
               if r.get(key) and r[key].get("average_precision") is not None]
        c["per_vessel_ap"] = {
            "n": len(aps),
            "median": float(np.median(aps)) if aps else None,
            "min": float(np.min(aps)) if aps else None,
            "max": float(np.max(aps)) if aps else None,
        }
        cm = np.concatenate(pooled[key]["committed"])
        hv = np.concatenate(pooled[key]["have"])
        if hv.any():
            c["shipped_operating_point"] = operating_point(y, cm, hv)
            c["shipped_operating_point"]["n_vessels_with_committed_set"] = len(
                [a for a in pool if a in ext])
        out["domains"][key] = c
        op = c.get("shipped_operating_point")
        print(f"  {key:<4} AUC-PR {c['average_precision']:.4f}  base rate "
              f"{c['base_rate']:.4f}  ({c['n_positive']}/{c['n_nodes']} positive)"
              + (f"  | shipped cut P={op['precision']:.3f} R={op['recall']:.3f}"
                 if op else ""))

    dst = DATA_DIR / "operating_point.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
