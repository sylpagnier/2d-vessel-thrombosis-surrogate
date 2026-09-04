"""Choose the mask that maximises the EXPECTED severity score, instead of thresholding it.

Every readout this project has used answers "which nodes score above a cut".  That is a
per-node question, and the metric is not per-node: it is
`0.5*dilation_IoU + 0.5*F_0.5(precision_eff, recall_eff)`, computed over a whole domain,
with a burden-dependent grace.  Whether the 40th-ranked node is worth committing depends on
how many are already committed and on how confident the rest are -- which a fixed cut cannot
express, and which is exactly why `scripts/diag_readout_ceiling.py` finds +0.042 wall /
+0.120 off-wall sitting between a cohort cut and a per-vessel oracle cut.

This asks the decision-theoretic question instead.  Treat the model's probabilities `p` as a
distribution over the unknown truth; for each prefix of the score-ranked node list, compute
the **expected** severity score of committing that prefix, using `soft_severity` with `p` in
the place of GT; commit the prefix that maximises it.  The stopping point is then a property
of this vessel's own confidence profile, needs no label, and adapts the budget automatically.

`src/clot_ml/calibration.py`'s rules all failed because they locate a cut from the *shape* of
a unitless score.  This does not locate a cut at all -- it evaluates the objective.

Two cohort-level corrections are offered, both fitted in-fold, because `p` is known to be
miscalibrated (`scripts/eval_reg_readout.py`: the regression head's physical anchor lands far
from `crit`, so the classifier's probabilities are not calibrated either):

    gamma   sharpen/flatten the probabilities, ``p -> p**gamma``, before taking expectations
    kscale  multiply the chosen prefix length, ``k -> kscale * k``

    python scripts/eval_expected_score_readout.py --tags v5a,v5b,v5c --cache v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.utils.paths import anchor_packs_dir, get_project_root

REPO = get_project_root()
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import (  # noqa: E402
    FAMILIES, GRID, apply_adapt, load_scores, tune_adapt,
)
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.calibration import apply_rule as cal_apply  # noqa: E402
from src.clot_ml.calibration import rule_grid as cal_grid  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import (  # noqa: E402
    DEFAULT, LEGACY, SeverityScorer, soft_severity,
)
from src.clot_ml.softmetric import dilation_operator, soft_dilate, to_torch_sparse  # noqa: E402

PACKS = anchor_packs_dir()
GAMMA = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
KSCALE = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
N_PREFIX = 40          # log-spaced prefix lengths evaluated per vessel/domain


#: Label-free per-vessel cut rules (`src/clot_ml/calibration.py`).  Each replaces the cohort
#: CONSTANT with a quantity computed from this vessel's own score distribution, so a vessel
#: whose field is uniformly shifted gets a correspondingly shifted cut for free.  They were
#: written on 2026-08-26 and, until now, never run against the shipped readout -- only a unit
#: test imported them.  `absolute` is omitted here because it IS `cohort_cut`, the control.
CAL_RULES = ("quantile", "rel_max", "phys_anchored", "gap")


def norm_rank(sc, d, phys):
    """Empirical CDF of the score WITHIN the domain: the fully scale-free field.

    Every vessel's normalised field is uniform on [0, 1], so a cohort cut becomes a
    committed FRACTION.  Burden runs 11-313 nodes across this cohort and domain size runs
    with it, so a fraction is not a constant count -- but it is the strongest assumption
    here, and it is the one to beat.
    """
    v = np.asarray(sc, dtype=np.float64)
    out = np.zeros_like(v)
    idx = np.flatnonzero(d)
    if idx.size == 0:
        return out
    order = idx[np.argsort(v[idx])]
    out[order] = np.linspace(0.0, 1.0, idx.size)
    return out


def norm_relmax(sc, d, phys):
    """``score / max(score)`` inside the domain -- keeps the field's SHAPE, drops its scale.

    Weaker than `norm_rank`: a vessel with one sharp peak stays sharp, so the burden signal
    in the field's shape survives the normalisation.
    """
    v = np.asarray(sc, dtype=np.float64)
    idx = np.flatnonzero(d)
    if idx.size == 0:
        return np.zeros_like(v)
    m = float(v[idx].max())
    return v / m if m > 0 else np.zeros_like(v)


def norm_physq(sc, d, phys):
    """Rank CDF RE-CENTRED so the physics mask's own size sits at 0.5.

    THE IDEA.  `docs/DEPLOYCLOT.md` 14 localised the off-wall deficit to cut PLACEMENT: the
    ranking transfers (AUC 0.9964 on vessels never seen) while the cohort constant does not,
    because each vessel's score SCALE is its own.  `PHASE9_ML` 4 killed budget rules that
    predicted a COUNT from the physics mask.  This uses the same quantity as a LOCATION
    instead: the backbone commits `n_p` nodes in this domain with zero free parameters, so
    the quantile `1 - n_p/n_d` is a deploy-legal guess at where the boundary sits, and the
    cohort then fits one scalar saying how far off the backbone systematically is.

    A cohort cut of exactly 0.5 commits exactly the physics count; above 0.5 is stricter
    than the backbone, below is looser, and the mapping is monotone so the RANKING -- the
    part that transfers -- is untouched.
    """
    v = np.asarray(sc, dtype=np.float64)
    idx = np.flatnonzero(d)
    if idx.size == 0:
        return np.zeros_like(v)
    u = norm_rank(v, d, phys)
    n_p = int((np.asarray(phys, dtype=bool) & d).sum())
    q = 1.0 - float(np.clip(n_p, 1, idx.size - 1)) / float(idx.size)
    q = float(np.clip(q, 1e-3, 1.0 - 1e-3))
    out = np.zeros_like(v)
    lo = u <= q
    out[lo & d] = 0.5 * u[lo & d] / q
    hi = (~lo) & d
    out[hi] = 0.5 + 0.5 * (u[hi] - q) / (1.0 - q)
    return out


NORMS = {"rank": norm_rank, "relmax": norm_relmax, "physq": norm_physq}


def oracle_cut_score(vs_a, sc_a, d):
    """Best score any single cut reaches on THIS vessel -- the ceiling, never an arm."""
    best = float("nan")
    for t in GRID:
        x = vs_a.score(d & (sc_a >= t), d)
        if x == x and (best != best or x > best):
            best = float(x)
    return best


def expected_curve(sc, dom, D_t, dev, gamma):
    """-> (ks, expected score at each prefix length) for one vessel/domain."""
    d = torch.tensor(np.asarray(dom, np.float32), device=dev)
    p_raw = np.clip(np.asarray(sc, np.float64), 1e-6, 1 - 1e-6) ** gamma
    p = torch.tensor(p_raw.astype(np.float32), device=dev)
    gt_dil = soft_dilate(p * d, D_t).detach()
    idx = np.argsort(-np.asarray(sc)[np.asarray(dom, bool)])
    order = np.flatnonzero(np.asarray(dom, bool))[idx]
    n = len(order)
    if n < 4:
        return np.array([0]), np.array([0.0])
    ks = np.unique(np.clip(np.geomspace(1, n, N_PREFIX).astype(int), 1, n))
    vals = []
    for k in ks:
        m = np.zeros(len(sc), np.float32)
        m[order[:k]] = 1.0
        v = soft_severity(torch.tensor(m, device=dev), p, D_t, d, gt_dil, DEFAULT)
        vals.append(float(v) if v is not None else -1e9)
    return ks, np.asarray(vals)


def rank01(x, d):
    out = np.zeros_like(np.asarray(x, np.float64))
    v = np.asarray(x, np.float64)[d]
    if v.size == 0:
        return out
    out[d] = np.argsort(np.argsort(v)).astype(np.float64) / max(v.size - 1, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save", default="")
    ap.add_argument("--save-masks", default="",
                    help="npz of the nested-pick committed mask per vessel, for "
                         "scripts/eval_strict_temporal.py --set-masks")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    # the regression head, if the tags carry it -- a MEASURABLY better off-wall field
    # (scripts/eval_reg_readout.py: 0.6006 against the classifier's 0.5105 on the same
    # weights) that fusion could not use as a THRESHOLD field.  Here it is used only to
    # ORDER the nodes; the prefix length still comes from the expected-score objective, so
    # the rank-flattening that killed `fuse_rank` does not apply.
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True)
          for t in args.tags.split(",")]
    has_reg = all(any(k.startswith("reg|") for k in z.files) for z in zs)
    reg = {}
    if has_reg:
        fo0 = {a: k for k, held in folds.items() for a in held}
        for a in pool:
            reg[a] = np.mean([z["reg|%d|%s" % (fo0[a], a)] for z in zs], axis=0)
    classes = classes_for(pool, PACKS)
    fo = {a: k for k, held in folds.items() for a in held}
    sc = {a: sc_all[(fo[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    # The SAME masks under the `guiding` score -- `SeverityScorer(..., LEGACY)` reproduces
    # `evaluate.domain_score` exactly (verified to 0.00e+00 over 13 vessel-domains), and
    # `guiding` is what `species_continuous_clout_score_mode()` returns by default, so it is
    # the metric `eval_clot_ml_0.py`, the SEALED read and every wound number are already on.
    # Reporting both is what stops a severity number being quoted against a guiding one
    # (docs/DEPLOYCLOT.md 22).  SELECTION is unchanged -- it stays on `vs` -- so this run is
    # bit-identical in what it picks; only the reporting is wider.
    vsg = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                             len(cache[a]["wall"]), LEGACY) for a in pool}
    Dt = {a: to_torch_sparse(dilation_operator(cache[a]["edge_index"],
                                               len(cache[a]["wall"]), 2), dev) for a in pool}
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}

    # precompute the expected-score curves once per (vessel, domain, gamma)
    print("[i] building expected-score curves ...", flush=True)
    curves = {}
    for a in pool:
        for dk, d_of in doms.items():
            for g in GAMMA:
                curves[(a, dk, g)] = expected_curve(sc[a], d_of(cache[a]), Dt[a], dev, g)
    print("[i] done", flush=True)

    def order_field(a, d, how):
        if how == "cls" or not has_reg:
            return sc[a]
        if how == "reg":
            return reg[a]
        return 0.5 * (rank01(sc[a], d) + rank01(reg[a], d))       # "both"

    def mask_for(a, dk, d_of, g, ks_scale, how="cls"):
        ks, vals = curves[(a, dk, g)]
        if len(ks) < 2:
            return np.zeros(len(sc[a]), bool)
        k = int(np.clip(round(ks[int(np.argmax(vals))] * ks_scale), 1, ks[-1]))
        d = d_of(cache[a])
        f = order_field(a, d, how)
        order = np.flatnonzero(d)[np.argsort(-f[d])]
        m = np.zeros(len(sc[a]), bool)
        m[order[:k]] = True
        return m

    ARMS = (["cohort_cut", "expected", "expected_tuned", "expected_reg", "expected_both",
             "resid", "resid_adapt"]
            + ["cal_%s" % r for r in CAL_RULES]
            + ["resid_%s" % n for n in NORMS]
            + ["nested_pick", "oracle_cut"])
    rows = {r: {a: {} for a in pool} for r in ARMS}
    rowsg = {r: {a: {} for a in pool} for r in ARMS}
    masks = {a: np.zeros(len(sc[a]), bool) for a in pool}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        for dk, d_of in doms.items():
            # control: one cohort cut
            top, t_cut = -1e9, float(GRID[0])
            for t in GRID:
                v = [vs[a].score(d_of(cache[a]) & (sc[a] >= t), d_of(cache[a])) for a in sel]
                v = [x for x in v if x == x]
                if v and np.mean(v) > top:
                    top, t_cut = float(np.mean(v)), float(t)
            # expected-score readout, gamma and kscale fitted on the selection vessels
            bests = {}
            for how in ("cls", "reg", "both"):
                b = None
                for g in GAMMA:
                    for kscl in KSCALE:
                        v = []
                        for a in sel:
                            x = vs[a].score(mask_for(a, dk, d_of, g, kscl, how),
                                            d_of(cache[a]))
                            if x == x:
                                v.append(x)
                        q = float(np.mean(v)) if v else -1e9
                        if b is None or q > b[0]:
                            b = (q, g, kscl)
                bests[how] = b
            best = bests["cls"]
            _, g_b, k_b = best
            # the physics-conditioned readout, and its adaptive perturbation
            sub = {a: sc[a] for a in sel}
            th_r = FAMILIES["resid"][0](cache, vs, sel, sub, GRID)
            b_r, med_r = tune_adapt(cache, vs, sel, sub, "resid", th_r, d_of)

            def q_of(fn):
                v = [x for x in (fn(a) for a in sel) if x == x]
                return float(np.mean(v)) if v else -1e9

            cands = {
                "cohort_cut": (q_of(lambda a: vs[a].score(
                    d_of(cache[a]) & (sc[a] >= t_cut), d_of(cache[a]))),
                    lambda a: d_of(cache[a]) & (sc[a] >= t_cut)),
                "expected_tuned": (best[0], lambda a: mask_for(a, dk, d_of, g_b, k_b)),
                "expected_reg": (bests["reg"][0], lambda a: mask_for(
                    a, dk, d_of, bests["reg"][1], bests["reg"][2], "reg")),
                "expected_both": (bests["both"][0], lambda a: mask_for(
                    a, dk, d_of, bests["both"][1], bests["both"][2], "both")),
                "resid": (q_of(lambda a: vs[a].score(
                    FAMILIES["resid"][1](cache[a], sc[a], th_r) & d_of(cache[a]),
                    d_of(cache[a]))),
                    lambda a: FAMILIES["resid"][1](cache[a], sc[a], th_r) & d_of(cache[a])),
                "resid_adapt": (q_of(lambda a: vs[a].score(
                    apply_adapt(cache[a], sc[a], "resid", th_r, d_of, b_r, med_r)
                    & d_of(cache[a]), d_of(cache[a]))),
                    lambda a: apply_adapt(cache[a], sc[a], "resid", th_r, d_of, b_r, med_r)
                    & d_of(cache[a])),
            }

            # --- label-free per-vessel cut rules, one cohort scalar each, fitted in fold
            for rname in CAL_RULES:
                bq, bp = -1e9, float(cal_grid(rname)[0])
                for pv in cal_grid(rname):
                    v = [x for x in (vs[a].score(
                        cal_apply(rname, sc[a], d_of(cache[a]), cache[a]["phys_mask"], pv),
                        d_of(cache[a])) for a in sel) if x == x]
                    q = float(np.mean(v)) if v else -1e9
                    if q > bq:
                        bq, bp = q, float(pv)
                cands["cal_%s" % rname] = (bq, (lambda rn, pv: lambda a: cal_apply(
                    rn, sc[a], d_of(cache[a]), cache[a]["phys_mask"], pv))(rname, bp))

            # --- the SHIPPED resid readout, run on a per-vessel NORMALISED field.  The
            # readout is unchanged; only the field it cuts is made scale-free, so this
            # isolates "the cut is in the wrong place" from "the rule is the wrong shape".
            for nname, nfn in NORMS.items():
                nsc = {a: nfn(sc[a], d_of(cache[a]), cache[a]["phys_mask"]) for a in pool}
                th_n = FAMILIES["resid"][0](cache, vs, sel, {a: nsc[a] for a in sel}, GRID)
                cands["resid_%s" % nname] = (
                    q_of((lambda ns, th: lambda a: vs[a].score(
                        FAMILIES["resid"][1](cache[a], ns[a], th) & d_of(cache[a]),
                        d_of(cache[a])))(nsc, th_n)),
                    (lambda ns, th: lambda a: FAMILIES["resid"][1](cache[a], ns[a], th)
                     & d_of(cache[a]))(nsc, th_n))

            pick = max(cands, key=lambda r: cands[r][0])
            for a in held:
                d = d_of(cache[a])
                rows["cohort_cut"][a][dk] = vs[a].score(cands["cohort_cut"][1](a), d)
                rows["expected"][a][dk] = vs[a].score(mask_for(a, dk, d_of, 1.0, 1.0), d)
                rows["expected_tuned"][a][dk] = vs[a].score(
                    cands["expected_tuned"][1](a), d)
                rows["expected_reg"][a][dk] = vs[a].score(cands["expected_reg"][1](a), d)
                rows["expected_both"][a][dk] = vs[a].score(cands["expected_both"][1](a), d)
                rows["resid"][a][dk] = vs[a].score(cands["resid"][1](a), d)
                rows["resid_adapt"][a][dk] = vs[a].score(cands["resid_adapt"][1](a), d)
                for rname in CAL_RULES:
                    rows["cal_%s" % rname][a][dk] = vs[a].score(
                        cands["cal_%s" % rname][1](a), d)
                for nname in NORMS:
                    rows["resid_%s" % nname][a][dk] = vs[a].score(
                        cands["resid_%s" % nname][1](a), d)
                # the per-vessel ceiling: what the BEST single cut on this vessel reaches.
                # Reported, never selected on -- it reads the held-out vessel's own label.
                rows["oracle_cut"][a][dk] = oracle_cut_score(vs[a], sc[a], d)
                for r in ARMS:
                    if r in ("nested_pick", "oracle_cut", "expected"):
                        continue
                    rowsg[r][a][dk] = vsg[a].score(cands[r][1](a), d)
                rowsg["expected"][a][dk] = vsg[a].score(
                    mask_for(a, dk, d_of, 1.0, 1.0), d)
                rowsg["oracle_cut"][a][dk] = max(
                    (vsg[a].score(d & (sc[a] >= t), d) for t in GRID),
                    key=lambda x: (x == x, x))
                m_pick = cands[pick][1](a)
                rows["nested_pick"][a][dk] = vs[a].score(m_pick, d)
                rowsg["nested_pick"][a][dk] = vsg[a].score(m_pick, d)
                masks[a] |= m_pick
            print("  fold %d %-4s cut=%.2f gamma=%.2f kscale=%.2f  pick=%s"
                  % (k, dk, t_cut, g_b, k_b, pick), flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL TIME POINT, strictly nested (tags=%s)" % args.tags)
    print("`guiding` is the DEFAULT deploy score and the one every other evaluation in this "
          "repo\nreports; `severity` is Deploy Score v2, more forgiving, and is what the "
          "arms below were\nSELECTED on.  Quote guiding.  Do not mix them.\n")
    print("%-18s | %-19s | %-19s" % ("", "GUIDING (deploy)", "severity (v2)"))
    print("%-18s | %9s %9s | %9s %9s | %9s %9s"
          % ("arm", "wall", "off", "wall", "off", "P wall", "P off"))
    for r in ARMS:
        R, G = rows[r], rowsg[r]
        mg = lambda D, k, P=pool: np.nanmean([D[a][k] for a in P if k in D[a]])  # noqa: E731
        print("%-18s | %9.4f %9.4f | %9.4f %9.4f | %9.4f %9.4f"
              % (r, mg(G, "wall"), mg(G, "off"), mg(R, "wall"), mg(R, "off"),
                 mg(G, "wall", prio), mg(G, "off", prio)))
    if args.save:
        Path(args.save).write_text(json.dumps(
            {"severity": rows, "guiding": rowsg}, indent=2, default=float))
        print("\nwrote %s" % args.save)
    if args.save_masks:
        np.savez_compressed(args.save_masks, **{a: masks[a] for a in pool})
        print("wrote %s" % args.save_masks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
