"""Physics alone vs learning alone vs both -- the paper's central comparison, paired.

WHY THIS AND NOT THE ABLATION LADDER.  The ladder answers "what does each feature group add",
which turned out to be beyond the power of n=27 for every step above the first
(`docs/PHYSICS_ABLATION_PLAN.md` 8). This asks the coarser question the paper actually needs,
where the effects are large enough to resolve:

    is the physics backbone needed, is the learned model needed, or is either sufficient alone?

Three predictors, one protocol, the same 27 clot-carrying vessels, every score held out:

    phys    the zero-parameter physics backbone's own occlusion mask (`S["phys_mask"]`).
            No learning, no threshold, nothing fitted -- so it also has no selection noise,
            which is worth stating rather than glossing: it is advantaged in that one respect
            and disadvantaged in having no ability to adapt.
    plain   `A1_pure` -- a GNN with geometry and flow but no physics in the features, no
            physics residual base, no physics rollout seed, read out under the `plain` family
            so `phys_mask` cannot enter the threshold either.
    full    `A4` -- the shipped conditioning, same readout family, so the only difference
            from `plain` is the physics.

**Everything is PAIRED over vessels.** Unpaired means on 27 vessels hide effects this size:
`phys` and `plain` differ by +0.05 at the wall and that difference is NOT significant, while
`full` beats both and is. Reporting the three means without the paired intervals is what
produced a wrong claim on 2026-09-08.

    python scripts/publication/diag_physics_vs_learned.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()

from scripts.eval_strict import FAMILIES, GRID, BoundScorer, load_scores  # noqa: E402
from src.clot_ml.data import (  # noqa: E402
    attach_physics, load_cache, off_domain, wall_domain,
)
from src.clot_ml.severity_metric import DEFAULT  # noqa: E402
from src.clot_ml.mirror_branch import eval_gt  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE, SEALED  # noqa: E402


def strict_masks(cache, tags, family):
    """Per-vessel HELD-OUT predicted mask under `eval_strict`'s nested protocol.

    The readout family is pinned rather than chosen, because `resid` reads the physics
    occlusion mask -- letting the tuner pick it would put physics back into the "no physics"
    arm through the threshold.
    """
    pool, folds, sc = load_scores(tags)
    pool = [a for a in pool if a in cache]
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}
    vs = {a: BoundScorer(cache[a]["edge_index"], eval_gt(a, cache[a]["y"], cache[a]["pos"]),
                         len(cache[a]["wall"]), DEFAULT,
                         "score" if a in CLOT_FREE else "nan") for a in pool}
    out = {}
    for _k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        tune, apply_ = FAMILIES[family]
        th = tune(cache, vs, sel, {a: oof[a] for a in sel}, GRID)
        for a in held:
            out[a] = apply_(cache[a], oof[a], th)
    return out, vs


def physics_score_field(S):
    """A CONTINUOUS physics estimate per node, so the backbone can be given a fitted cut.

    `phys` above is the backbone's binary occlusion mask: a parameter-free rule with no
    threshold.  Comparing it against models whose cut is tuned out-of-fold is not like-for-like,
    and off-wall it is badly unfair -- audited 2026-09-08, the mask commits *nothing* off-wall on
    7 of 27 vessels and its burden spans 0.03x to 9.3x of GT, against 0.97x at the wall.  Some of
    the apparent "learning beats physics off-wall" gap is therefore "fitted beats unfitted", and
    that bias runs in favour of our own conclusion, which is the direction to distrust.

    This builds the physics side's own continuous field so it can go through the SAME tuner:

        wall      `log_mat_phys`     -- log1p(Mat/crit) from the integrated deposition ODE
        off-wall  `log_mat_off_est`  -- the owner's Mat times the flow-computed attenuation,
                                        which is the backbone's own off-wall estimate
                                        (`mat_phys` itself is identically zero off the wall)

    Squashed by `x / (1 + x)`, which is **strictly monotone and parameter-free**.  A monotone map
    cannot change which masks are reachable by thresholding -- only which grid value selects
    them -- so this fixes the scale for `GRID` without fitting anything or changing the answer.
    """
    cols = [str(c) for c in S["cols"]]
    x = np.zeros(len(S["wall"]), dtype=float)
    w, o = wall_domain(S), off_domain(S)
    X = S["X"]
    x[w] = X[w, cols.index("log_mat_phys")]
    x[o] = X[o, cols.index("log_mat_off_est")]
    x = np.maximum(x, 0.0)
    return x / (1.0 + x)


def phys_tuned_masks(cache, pool, folds, vs):
    """The physics field, thresholded by the SAME nested tuner the learned arms use."""
    sc = {a: physics_score_field(cache[a]) for a in pool}
    tune, apply_ = FAMILIES["plain"]
    out = {}
    for _k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        th = tune(cache, vs, sel, {a: sc[a] for a in sel}, GRID)
        for a in held:
            out[a] = apply_(cache[a], sc[a], th)
    return out


def paired(rows, a_key, b_key, dom, n=20000, seed=0):
    """Bootstrap of (b - a) over vessels where both are defined."""
    rng = np.random.default_rng(seed)
    d = np.array([rows[v][dom][b_key] - rows[v][dom][a_key] for v in rows
                  if rows[v][dom][a_key] == rows[v][dom][a_key]
                  and rows[v][dom][b_key] == rows[v][dom][b_key]])
    if len(d) == 0:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_le0=float("nan"), n=0)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return dict(delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)),
                p_le0=float((bs <= 0).mean()), n=int(len(d)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v5_split")
    ap.add_argument("--plain-arm", default="abl_A1_pure",
                    help="the physics-INFORMED architecture with physics conditioning removed")
    ap.add_argument("--naive-arm", default="abl_A_naive",
                    help="the from-scratch control; included automatically when its score "
                         "file exists, skipped silently when it does not")
    ap.add_argument("--full-arm", default="abl_A4")
    ap.add_argument("--family", default="plain", choices=["plain", "resid"])
    ap.add_argument("--out", default="outputs/ablation/physics_vs_learned.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    tags = lambda stem: [t for t in (stem, stem + "_seedB")
                         if (REPO / f"outputs/phase9_scores/{t}.npz").is_file()]

    m_plain, vs = strict_masks(cache, tags(args.plain_arm), args.family)
    m_full, _ = strict_masks(cache, tags(args.full_arm), args.family)

    # the physics side, given the same fitted cut as the learned arms -- same folds, same tuner
    _pool, _folds, _ = load_scores(tags(args.full_arm))
    _pool = [a for a in _pool if a in cache]
    m_ptun = phys_tuned_masks(cache, _pool, _folds, vs)

    # The from-scratch control, when it has been trained.  `plain` is NOT naive -- it is this
    # project's architecture with the physics columns zeroed, and that architecture already
    # encodes physics knowledge (anisotropic messages, the `Mat` auxiliary target, the
    # metric-shaped loss, C0, the wall/shell/owner decomposition).  `naive` bounds how much of
    # the reported physics contribution is really "our architecture" rather than "the physics".
    m_naive = None
    if tags(args.naive_arm):
        m_naive, _ = strict_masks(cache, tags(args.naive_arm), args.family)

    carrying = [a for a in sorted(cache) if a not in SEALED and a not in CLOT_FREE]
    rows = {}
    for a in carrying:
        S = cache[a]
        r = {}
        for dname, dom_of in (("wall", wall_domain), ("off", off_domain)):
            d = dom_of(S)
            r[dname] = dict(
                phys=vs[a].score(S["phys_mask"] & d, d),
                phys_tuned=vs[a].score(m_ptun[a] & d, d),
                plain=vs[a].score(m_plain[a] & d, d),
                full=vs[a].score(m_full[a] & d, d))
            if m_naive is not None:
                r[dname]["naive"] = vs[a].score(m_naive[a] & d, d)
        rows[a] = r

    keys = (("naive",) if m_naive is not None else ()) + (
        "phys", "phys_tuned", "plain", "full")
    means = {dom: {k: float(np.nanmean([rows[a][dom][k] for a in rows]))
                   for k in keys} for dom in ("wall", "off")}
    tests = {dom: {
        "phys_minus_plain": paired(rows, "plain", "phys", dom),
        "full_minus_plain": paired(rows, "plain", "full", dom),
        "full_minus_phys": paired(rows, "phys", "full", dom),
        # the fair head-to-head: both sides thresholded by the same out-of-fold tuner
        "phystuned_minus_plain": paired(rows, "plain", "phys_tuned", dom),
        "full_minus_phystuned": paired(rows, "phys_tuned", "full", dom),
        "phystuned_minus_phys": paired(rows, "phys", "phys_tuned", dom),
        **({"plain_minus_naive": paired(rows, "naive", "plain", dom),
            "full_minus_naive": paired(rows, "naive", "full", dom),
            "phystuned_minus_naive": paired(rows, "naive", "phys_tuned", dom)}
           if m_naive is not None else {}),
    } for dom in ("wall", "off")}

    print("PHYSICS vs LEARNING vs BOTH -- %d clot-carrying vessels, held out, "
          "severity, readout family=%s\n" % (len(rows), args.family))
    for dom in ("wall", "off"):
        print("%s domain" % dom.upper())
        for k in keys:
            print("  %-11s %.4f" % (k, means[dom][k]))
        for name, t in tests[dom].items():
            print("  %-24s %+.4f  CI [%+.4f, %+.4f]  P(<=0)=%.3f  n=%d"
                  % (name, t["delta"], t["lo"], t["hi"], t["p_le0"], t["n"]))
        print()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        cache=args.cache, family=args.family, plain_arm=args.plain_arm,
        full_arm=args.full_arm, n_vessels=len(rows), means=means, tests=tests,
        per_vessel=rows), indent=2, default=float), encoding="utf-8")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
