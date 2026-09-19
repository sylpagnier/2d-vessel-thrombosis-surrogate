"""The leakage gate for a coupling arm: can ONE coupled feature recover the wall label on its own?

STORY 2.1f's lesson, as a runnable check.  The GT-driven oracle lowered shear exactly where the
answer was, so `log_mat_phys` alone recovered the wall label at F1 0.96 -- a "coupling gain"
that was the model reading a leaked label.  A clean arm lifts that single-feature score only
modestly over the uncoupled one.  Run it FIRST on any new coupling arm.

    score   best F1 over thresholds (and AUC) of `log_mat_phys` against the wall label, pooled
            over the wall nodes of every vessel that has wall clot.  One feature, no training,
            nothing to overfit -- so a large lift can only come from what the feature encodes.

    python scripts/publication/diag_coupling_leakage.py --caches fem_open=v5_fem_open v5_fem_wake v5_fem_femcpl
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()


def _pooled(cache: str, feature: str, stems: list[str] | None):
    d = REPO / "outputs" / f"clot_ml_cache_{cache}"
    xs, ys, used = [], [], []
    for p in sorted(d.glob("*.npz")):
        if stems is not None and p.stem not in stems:
            continue
        z = np.load(p, allow_pickle=True)
        wall = z["wall"].astype(bool)
        y = z["y"].reshape(-1)[wall] > 0.5
        if not y.any():
            continue
        cols = [str(c) for c in z["cols"]]
        xs.append(z["X"][wall, cols.index(feature)].astype(np.float64))
        ys.append(y)
        used.append(p.stem)
    return np.concatenate(xs), np.concatenate(ys), used


def _best_f1_auc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-x, kind="stable")
    xs, ys = x[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.r_[np.flatnonzero(np.diff(xs) != 0), xs.size - 1]   # cut only between ties
    P = float(ys.sum())
    f1 = 2 * tp[last] / (last + 1 + P)
    tpr = np.r_[0.0, tp[last] / P]
    fpr = np.r_[0.0, fp[last] / max(float((~ys).sum()), 1.0)]
    return float(f1.max()), float(np.sum(np.diff(fpr) * 0.5 * (tpr[1:] + tpr[:-1])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True,
                    help="cache names (outputs/clot_ml_cache_<name>), optionally label=name")
    ap.add_argument("--feature", default="log_mat_phys")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    specs = [c.split("=", 1) if "=" in c else (c, c) for c in args.caches]
    # every arm is scored on the SAME vessels -- the intersection of what each cache holds
    common = None
    for _, name in specs:
        _, _, used = _pooled(name, args.feature, None)
        common = set(used) if common is None else common & set(used)
    common = sorted(common)
    res = {}
    for label, name in specs:
        x, y, used = _pooled(name, args.feature, common)
        f1, auc = _best_f1_auc(x, y)
        res[label] = dict(cache=name, single_feature_best_f1=round(f1, 4),
                          single_feature_auc=round(auc, 4), n=len(used))
        print(f"{label:<16} F1 {f1:.4f}  AUC {auc:.4f}  n={len(used)}")
    if args.out:
        (REPO / args.out).write_text(json.dumps(dict(feature=args.feature, vessels=common,
                                                     result=res), indent=2), encoding="utf-8")
        print(f"[i] wrote {REPO / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
