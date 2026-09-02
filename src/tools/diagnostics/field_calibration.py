"""Is the score field comparable ACROSS vessels?  The mechanism behind the readout gap.

WHY THIS EXISTS.  Phase B measured the readout costing **0.193 off-wall** against a per-vessel
oracle, on a field whose ranking is intact (out-of-fold AUC 0.989) -- so the loss is not in
what the model knows about a node, it is in the fact that a score means something different
on each vessel.  `patient032`'s off-wall GT sits at a median score of 0.1156 while
`patient014`, which has no off-wall GT at all, has a tail reaching 0.8551.  One cohort cut
cannot serve both, and no readout family already built gets within 0.15 of the oracle
(MODEL_REVIEW_2026-08-22 8f.2).

This reports the statistics a C0 arm has to move, so an arm can be judged on its MECHANISM
and not only on a cohort mean that the +/-0.074 floor may swallow:

* **GT-median spread** -- where a cut would have to sit to catch each vessel's clot, and how
  far apart those places are.  This is the quantity a distributional constraint targets.
* **The separation margin** -- per vessel, the gap between its GT median and the *noise* tail
  of the vessels that have no GT in that domain.  Negative anywhere means no single cut can
  win on both.
* **Implied-burden error** -- at the deploy cut, how far the committed count is from the true
  one, reported as median/p90/max because 2026-08-22 measured a term that fixed the median
  and left the tail untouched (11.6% -> 5.7% median, 28.3% -> 32.2% p90).

    python scripts/diag_field_calibration.py --tags v5a,v5b,v5c --cache v5
    python scripts/diag_field_calibration.py --tags c0sq --cache v5 --compare v5a
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap

import argparse
import sys
from pathlib import Path

import numpy as np


from src.clot_ml.data import attach_physics, load_cache, off_domain, wall_domain  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402


def oof_field(tags: list[str], cache: dict) -> dict:
    from eval_strict import load_scores

    pool, folds, sc = load_scores(tags)
    fold_of = {a: k for k, held in folds.items() for a in held}
    return {a: sc[(fold_of[a], a)] for a in pool if a in cache}


def stats(F: dict, cache: dict, dom_of, cut: float) -> dict:
    """Per-vessel calibration statistics for one domain."""
    gt_med, noise_tail, burden_err, rows = {}, {}, {}, {}
    for a in sorted(F):
        if a in CLOT_FREE:
            continue
        S = cache[a]
        gt = S["y"] > 0.5
        d = dom_of(S)
        v = np.asarray(F[a], np.float64)
        if not d.any():
            continue
        if (d & gt).any():
            gt_med[a] = float(np.median(v[d & gt]))
            k_true = int((d & gt).sum())
            k_pred = int((v[d] >= cut).sum())
            burden_err[a] = abs(k_pred - k_true) / max(k_true, 1)
            rows[a] = (gt_med[a], k_true, k_pred)
        else:
            noise_tail[a] = float(np.quantile(v[d], 0.999))
    return dict(gt_med=gt_med, noise_tail=noise_tail, burden_err=burden_err, rows=rows)


def report(name: str, st: dict) -> dict:
    g = np.array(list(st["gt_med"].values()))
    n = np.array(list(st["noise_tail"].values()))
    b = np.array(list(st["burden_err"].values()))
    out = dict(sd=float(g.std()) if g.size else float("nan"),
               lo=float(g.min()) if g.size else float("nan"),
               hi=float(g.max()) if g.size else float("nan"),
               margin=float(g.min() - n.max()) if (g.size and n.size) else float("nan"),
               b_med=float(np.median(b)) if b.size else float("nan"),
               b_p90=float(np.percentile(b, 90)) if b.size else float("nan"),
               b_max=float(b.max()) if b.size else float("nan"))
    print("  %-10s GT-median across vessels: sd %.4f  range %.4f-%.4f (n=%d)"
          % (name, out["sd"], out["lo"], out["hi"], g.size))
    if n.size:
        print("  %-10s separation margin (worst GT median - worst no-GT tail): %+.4f"
              % ("", out["margin"]))
    print("  %-10s implied-burden error at the cut: median %.1f%%  p90 %.1f%%  max %.1f%%"
          % ("", 100 * out["b_med"], 100 * out["b_p90"], 100 * out["b_max"]))
    return out


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--compare", default="", help="baseline tags, same format")
    ap.add_argument("--wall-cut", type=float, default=0.86)
    ap.add_argument("--off-cut", type=float, default=0.62,
                    help="representative deploy cuts; the statistics are reported AT a cut, "
                         "so quote the cut alongside them")
    args = ap.parse_args(argv)

    cache = attach_physics(load_cache(args.cache))
    arms = [("candidate", args.tags)] + ([("baseline", args.compare)] if args.compare else [])
    got = {}
    for label, tags in arms:
        F = oof_field(tags.split(","), cache)
        print("\n=== %s (%s) ===" % (label, tags))
        for dname, dom_of, cut in (("wall", wall_domain, args.wall_cut),
                                   ("off", off_domain, args.off_cut)):
            print(" %s domain, cut %.2f" % (dname.upper(), cut))
            got[(label, dname)] = report("", stats(F, cache, dom_of, cut))

    if args.compare:
        print("\nCANDIDATE - BASELINE  (negative sd = tighter field = what C0 is for)")
        for dname in ("wall", "off"):
            c, b = got[("candidate", dname)], got[("baseline", dname)]
            print("  %-4s  sd %+.4f | margin %+.4f | burden med %+.1f%% p90 %+.1f%% max %+.1f%%"
                  % (dname, c["sd"] - b["sd"], c["margin"] - b["margin"],
                     100 * (c["b_med"] - b["b_med"]), 100 * (c["b_p90"] - b["b_p90"]),
                     100 * (c["b_max"] - b["b_max"])))
        print("\n  A mechanism claim needs the sd or the burden TAIL to move, on all three "
              "configurations.\n  MODEL_REVIEW 8f.4: one configuration is not evidence, "
              "however many statistics it yields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
