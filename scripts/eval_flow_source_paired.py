"""What the deploy-legal flow costs, paired over vessels, at the CV level.

The two cross-validation arms differ in exactly one thing -- the t=0 velocity field the
features are built from -- and they are scored on the same vessels with the same labels, so
the right statistic is a PAIRED one over vessels, not two cohort means quoted side by side.
`scripts/eval_significance.py` already does paired bootstrap, but it takes one cache for both
arms; here each arm has to be read against its own cache, because `phys_mask`, `gate`, `sr`
and the v4 transport channels are all functions of the flow.  Labels (`y`), `wall` and
`solid` are not, which is what makes the pairing legitimate.

Reported per domain, at the final time point, under the severity metric, with the readout
chosen per arm on the out-of-fold scores of the vessels OUTSIDE the held-out fold -- the same
strictly-nested protocol `eval_strict.py` uses, so neither arm is flattered.

    python scripts/eval_flow_source_paired.py --a dc_gt_c0 --a-cache v5 \\
                                              --b dc_fem_c0 --b-cache v5_fem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_strict import FAMILIES, GRID, BoundScorer, load_scores  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache, off_domain, wall_domain  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT  # noqa: E402
from src.core_physics.wall_cohort_splits import CLOT_FREE  # noqa: E402


def arm_scores(tag: str, cache_name: str) -> dict[str, dict[str, float]]:
    """anchor -> {wall, off}, strictly nested: readout chosen outside the held-out fold."""
    cache = attach_physics(load_cache(cache_name))
    pool, folds, sc = load_scores([tag])
    pool = [a for a in pool if a in cache]
    folds = {k: [a for a in held if a in pool] for k, held in folds.items()}
    fold_of = {a: k for k, held in folds.items() for a in held}
    vs = {a: BoundScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5, len(cache[a]["wall"]),
                         DEFAULT, "score" if a in CLOT_FREE else "nan") for a in pool}
    oof = {a: sc[(fold_of[a], a)] for a in pool}

    out: dict[str, dict[str, float]] = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        sel_sc = {a: oof[a] for a in sel}
        # one family for both domains, chosen on the whole selection set -- eval_strict.py's
        # measured-best default; per-domain and inner-CV choice both lose at this n.
        best = None
        for fam, (tune, apply_) in FAMILIES.items():
            th = tune(cache, vs, sel, sel_sc, GRID)
            vals = []
            for a in sel:
                S = cache[a]
                pr = apply_(S, sel_sc[a], th)
                for d in (wall_domain(S), off_domain(S)):
                    v = vs[a].sel(pr & d, d)
                    if v == v:
                        vals.append(v)
            q = float(np.mean(vals)) if vals else -1e9
            if best is None or q > best[0]:
                best = (q, fam, th)
        _, fam, th = best
        apply_ = FAMILIES[fam][1]
        for a in held:
            S = cache[a]
            pr = apply_(S, oof[a], th)
            w, o = wall_domain(S), off_domain(S)
            out[a] = dict(wall=vs[a].score(pr & w, w), off=vs[a].score(pr & o, o),
                          family=fam)
    return out


def paired(a: dict, b: dict, key: str, n_boot: int = 4000, seed: int = 0):
    """Mean paired difference b - a over vessels scoring in both arms, with a bootstrap CI."""
    stems = [s for s in sorted(set(a) & set(b))
             if s not in CLOT_FREE and a[s][key] == a[s][key] and b[s][key] == b[s][key]]
    if not stems:
        return None
    d = np.array([b[s][key] - a[s][key] for s in stems], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap P for "no difference"
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(n=len(stems), stems=stems,
                mean_a=float(np.mean([a[s][key] for s in stems])),
                mean_b=float(np.mean([b[s][key] for s in stems])),
                delta=float(d.mean()), ci=[float(lo), float(hi)], p=float(min(p, 1.0)),
                per_vessel={s: float(b[s][key] - a[s][key]) for s in stems})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline tag (e.g. the GT-flow arm)")
    ap.add_argument("--a-cache", required=True)
    ap.add_argument("--b", required=True, help="comparison tag (e.g. the FEM-flow arm)")
    ap.add_argument("--b-cache", required=True)
    ap.add_argument("--out", default="outputs/deployclot/flow_source_paired.json")
    args = ap.parse_args()

    print(f"[i] A = {args.a} on cache {args.a_cache}", flush=True)
    A = arm_scores(args.a, args.a_cache)
    print(f"[i] B = {args.b} on cache {args.b_cache}", flush=True)
    B = arm_scores(args.b, args.b_cache)

    res = {}
    print()
    print(f"{'domain':8s} {'A mean':>9s} {'B mean':>9s} {'B - A':>9s} "
          f"{'95% CI':>20s} {'P':>7s}  n")
    for key in ("wall", "off"):
        r = paired(A, B, key)
        if r is None:
            continue
        res[key] = r
        print(f"{key:8s} {r['mean_a']:9.4f} {r['mean_b']:9.4f} {r['delta']:+9.4f} "
              f"[{r['ci'][0]:+7.4f},{r['ci'][1]:+7.4f}] {r['p']:7.3f}  {r['n']}")

    print()
    print("per-vessel B - A (wall / off)")
    for s in sorted(set(A) & set(B)):
        if s in CLOT_FREE:
            continue
        dw = B[s]["wall"] - A[s]["wall"]
        do = B[s]["off"] - A[s]["off"]
        print(f"  {s:14s} {dw:+8.4f}  {do:+8.4f}" if do == do
              else f"  {s:14s} {dw:+8.4f}       --")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        dict(a=args.a, a_cache=args.a_cache, b=args.b, b_cache=args.b_cache,
             paired=res,
             per_vessel_a={k: {kk: vv for kk, vv in v.items() if kk != "family"}
                           for k, v in A.items()},
             per_vessel_b={k: {kk: vv for kk, vv in v.items() if kk != "family"}
                           for k, v in B.items()}), indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
