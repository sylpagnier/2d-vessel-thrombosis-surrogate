"""Do the 8 clot-free vessels carry any usable signal, or a free 1.0000?

WHY THIS EXISTS.  The 2026-08-22 cohort decision admitted `wall_cohort_splits.CLOT_FREE` to
training and to scoring on the argument that empty-GT vessels are real evidence about FALSE
POSITIVES (MODEL_REVIEW_2026-08-22 8b).  `scripts/sweep_ml_clean_protocol.py` had already
recorded the opposite worry in one line -- *"patient017 has zero GT clot, so
`empty_gt_fp_tol` awards a free 1.0000"* -- and the first smoke run reproduced it: all four
clot-free vessels scored 1.0000 under both the model AND the zero-parameter physics backbone,
with **zero** nodes committed.

A vessel that every arm gets perfectly right measures nothing.  Before Phase B spends a
rebuild on 8 extra vessels, this asks whether that is a property of the vessels or of the
under-trained smoke model:

* **phys_mask commitment** -- does the zero-parameter backbone fire anywhere on them?  If it
  commits nothing on all 8, they cannot separate the physics from anything.
* **score headroom** -- how far below the readout cut does the model's out-of-fold score
  actually reach?  A vessel whose max score is 0.02 under a 0.83 cut is saturated; one whose
  max is 0.80 is one calibration slip away from a false positive and is worth training on.
* **the gate margin** -- `sr/lss` and `dsrx/sgt` at their closest approach, which is the
  physics-side version of the same question and needs no model at all.

    python scripts/diag_clot_free_headroom.py --cache smoke --tag smoke
    python scripts/diag_clot_free_headroom.py --cache v5 --tag v5a,v5b,v5c
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--tag", default="", help="phase9_scores tags for the out-of-fold field")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    free = [a for a in sorted(cache) if a in CLOT_FREE]
    carrying = [a for a in sorted(cache) if a not in CLOT_FREE]
    if not free:
        print("[ERR] no clot-free vessels in cache %r -- rebuild it "
              "(scripts/build_clot_ml_cache.py)" % args.cache)
        return 2

    oof = {}
    if args.tag:
        from eval_strict import load_scores
        pool, folds, sc = load_scores(args.tag.split(","))
        fold_of = {a: k for k, held in folds.items() for a in held}
        oof = {a: sc[(fold_of[a], a)] for a in pool if a in cache}

    print("PHYSICS BACKBONE on the clot-free vessels (it has nothing fitted, so a node it "
          "commits\nhere is a false positive no readout can be blamed for)\n")
    # Gate margins, and the branch indicators themselves so the margins can be checked.
    # The law is `1[sr < lss] + 1[d(sr,x) < sgt]` with **sgt = -750** (negative), and the
    # cached column is `dsrx / |sgt|` -- so the separation branch fires at **-1**, not +1, and
    # the closest approach is the MINIMUM of that column.  Getting this backwards reads a
    # comfortable margin as a firing gate; `gate_sep` is printed alongside as the check.
    # Both are over WALL nodes, which is where the deposition law is evaluated.
    print("%-12s %7s | %7s %7s | %10s %11s | %8s %8s | %8s"
          % ("vessel", "nodes", "ph wall", "ph off", "min sr/lss", "min dsrx/sgt",
             "n low", "n sep", "max oof"))
    tot_ph = 0
    for a in free:
        S = cache[a]
        w = S["wall"]
        ph = S["phys_mask"]
        nw, no = int((ph & w).sum()), int((ph & ~w).sum())
        tot_ph += nw + no
        cols = [str(c) for c in S["cols"]]
        srl = S["X"][:, cols.index("sr_over_lss")][w]
        dsx = S["X"][:, cols.index("dsrx_over_sgt")][w]
        n_low = int((S["X"][:, cols.index("gate_low")][w] > 0).sum())
        n_sep = int((S["X"][:, cols.index("gate_sep")][w] > 0).sum())
        mx = ("%8.4f" % float(np.max(oof[a]))) if a in oof else "       -"
        print("%-12s %7d | %7d %7d | %10.3f %11.3f | %8d %8d | %s"
              % (a, len(w), nw, no, float(srl.min()), float(dsx.min()),
                 n_low, n_sep, mx))
    print("%-12s %7s | %7d %7s |   the low-shear branch fires at sr/lss < 1.000, the "
          "separation branch at dsrx/sgt < -1.000" % ("TOTAL", "", tot_ph, ""))

    if oof:
        print("\nOUT-OF-FOLD SCORE FIELD -- how close does the model come to committing?\n")
        print("%-12s %8s %8s %8s %8s   %s"
              % ("vessel", "max", "p99.9", "p99", "mean", "group"))
        for grp, sub in (("clot-free", free), ("carrying", carrying)):
            for a in sub:
                if a not in oof:
                    continue
                v = np.asarray(oof[a], np.float64)
                print("%-12s %8.4f %8.4f %8.4f %8.4f   %s"
                      % (a, v.max(), np.quantile(v, 0.999), np.quantile(v, 0.99),
                         v.mean(), grp))
        fm = [float(np.max(oof[a])) for a in free if a in oof]
        cm = [float(np.max(oof[a])) for a in carrying if a in oof]
        if fm and cm:
            print("\n  max score, clot-free  median %.4f  (range %.4f-%.4f)"
                  % (float(np.median(fm)), min(fm), max(fm)))
            print("  max score, carrying   median %.4f  (range %.4f-%.4f)"
                  % (float(np.median(cm)), min(cm), max(cm)))
            print("\n  READ THIS AS: if the clot-free maxima sit far below every cut the "
                  "readout\n  selects, those vessels contribute no gradient to the cut and "
                  "score a free 1.0000.\n  If they reach into the cut's range they are "
                  "carrying real false-positive pressure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
