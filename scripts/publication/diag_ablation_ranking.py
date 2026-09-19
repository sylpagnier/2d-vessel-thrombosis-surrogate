"""Score every ablation arm THRESHOLD-FREE, to separate the model from the readout.

WHY THIS EXISTS.  The ladder in `generate_ablation_data.py` reports post-readout severity
scores: a per-fold threshold search picks cuts, then the metric is applied.  Two properties of
that pipeline can make an arm win for reasons that have nothing to do with what it learned.

  1. **The metric grants grace.**  `severity` uses an absolute miss grace `tau_abs = 15` capped
     at `rho = 0.25` of the true burden, plus a precision grace.  Off-wall burdens here run
     4-126 nodes, so on most vessels the cap binds and `recall_eff = TP / (0.75 * n_gt)` --
     committing about three quarters of the true nodes already reads recall 1.0.  An arm with a
     badly calibrated but broadly ordered field can buy score by committing MORE.
  2. **The cut is fitted.**  The threshold search will find that trade if it exists, so a
     difference between arms can be a difference in how cuttable their fields happen to be,
     not in how well they rank clot.

Average precision over the raw out-of-fold field has neither property: no threshold, no grace,
no family choice.  It answers the only question the ablation is really asking -- *does this
arm's field know where the clot is* -- and it is the number to trust when the two disagree.

    python scripts/publication/diag_ablation_ranking.py
    python scripts/publication/diag_ablation_ranking.py --tag-prefix ablfem --cache v5_fem
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score

from src.utils.paths import get_project_root

REPO = get_project_root()

from scripts.eval_strict import load_scores  # noqa: E402
from scripts.run_ablation_ladder import ORDER, TAG_PREFIX  # noqa: E402
from scripts.publication.generate_ablation_data import tags_present  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache, off_domain, wall_domain  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402
from src.clot_ml.mirror_branch import eval_gt  # noqa: E402


def arm_ranking(cache, tags) -> dict:
    """Pooled and per-vessel average precision of an arm's out-of-fold field."""
    pool, folds, sc = load_scores(tags)
    pool = [a for a in pool if a in cache]
    fold_of = {a: k for k, held in folds.items() for a in held}
    carrying = [a for a in pool if a not in CLOT_FREE]

    out = {}
    for dom_name, dom_of in (("wall", wall_domain), ("off", off_domain)):
        ys, ss, per = [], [], []
        n_pos = n_tot = 0
        for a in carrying:
            S = cache[a]
            d = dom_of(S)
            y = eval_gt(a, S["y"], S["pos"])[d]
            s = sc[(fold_of[a], a)][d]
            if y.size == 0:
                continue
            n_pos += int(y.sum())
            n_tot += int(y.size)
            ys.append(y)
            ss.append(s)
            # a vessel with no positives in this domain has no AP -- excluded exactly the way
            # the recall-bearing means exclude it, so the two tables cover the same vessels
            if y.any() and not y.all():
                per.append(float(average_precision_score(y, s)))
        Y, Sc = np.concatenate(ys), np.concatenate(ss)
        out[dom_name] = dict(
            ap_pooled=float(average_precision_score(Y, Sc)) if Y.any() else float("nan"),
            base_rate=float(Y.mean()),
            ap_vessel_median=float(np.median(per)) if per else float("nan"),
            ap_vessel_mean=float(np.mean(per)) if per else float("nan"),
            n_vessels=len(per), n_nodes=n_tot, n_positive=n_pos)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v5_split")
    ap.add_argument("--tag-prefix", default=TAG_PREFIX)
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pfx = args.tag_prefix
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or [
        a for a in ORDER if tags_present(a, pfx)]
    arms = [a for a in arms if tags_present(a, pfx)]
    if not arms:
        raise SystemExit("no ladder score files with prefix %r" % pfx)

    cache = attach_physics(load_cache(args.cache))
    rows = {}
    for a in arms:
        tags = tags_present(a, pfx)
        rows[a] = dict(tags=tags, seeds=3 * len(tags), **arm_ranking(cache, tags))
        print("  ranked %-9s (%d seeds)" % (a, rows[a]["seeds"]), flush=True)

    ref = "A4" if "A4" in rows else arms[0]
    print("\nTHRESHOLD-FREE RANKING  cache=%s  prefix=%s  ref=%s" % (args.cache, pfx, ref))
    print("average precision of the raw out-of-fold field; no cut, no grace, no readout\n")
    hdr = "%-9s %2s | %8s %8s | %8s %8s | %8s %8s" % (
        "arm", "sd", "wallAP", "d vs ref", "offAP", "d vs ref", "wallMed", "offMed")
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        r = rows[a]
        dw = r["wall"]["ap_pooled"] - rows[ref]["wall"]["ap_pooled"]
        do = r["off"]["ap_pooled"] - rows[ref]["off"]["ap_pooled"]
        print("%-9s %2d | %8.4f %+8.4f | %8.4f %+8.4f | %8.4f %8.4f"
              % (a, r["seeds"], r["wall"]["ap_pooled"], dw,
                 r["off"]["ap_pooled"], do,
                 r["wall"]["ap_vessel_median"], r["off"]["ap_vessel_median"]))
    b = rows[ref]
    print("\nbase rate: wall %.4f (%d/%d nodes), off %.4f (%d/%d)"
          % (b["wall"]["base_rate"], b["wall"]["n_positive"], b["wall"]["n_nodes"],
             b["off"]["base_rate"], b["off"]["n_positive"], b["off"]["n_nodes"]))
    print("\nIf an arm wins the severity table but loses here, the severity table was measuring\n"
          "the READOUT, not the model -- see this file's docstring for the mechanism.")

    out = REPO / (args.out or "outputs/ablation/ablation_ranking%s.json"
                  % ("" if pfx == TAG_PREFIX else "_" + pfx))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(cache=args.cache, tag_prefix=pfx, ref=ref, arms=rows),
                              indent=2), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
