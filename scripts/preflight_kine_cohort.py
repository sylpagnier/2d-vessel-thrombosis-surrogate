"""Check a freshly generated / transferred kinematics cohort BEFORE committing to a train.

    python scripts/preflight_kine_cohort.py --src transfer/carreau

Every check here corresponds to a bug that has already cost this project a training run.  It is
cheap, it is CPU-only, and it is the last gate before GPU time.  Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--expect-p1", action="store_true",
                    help="the cohort is pre-elevation P1 (the normal case for fresh synthetic)")
    args = ap.parse_args()

    import numpy as np
    import torch

    from src.data_gen.lib.p1_corner_graph import identify_midside_nodes

    files = sorted(Path(args.src).glob("*.pt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[ERR] no .pt under {args.src}")
        return 1

    rows, results = [], []

    def check(name, status, detail):
        results.append((name, status, detail))

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = int(d.num_nodes)
        mid, _ = identify_midside_nodes(d)
        x = d.x
        w = x[:, 15]
        r = (w * 0.5).clamp(min=1e-6)
        y = getattr(d, "y", None)
        prior_rel = float("nan")
        if y is not None and y.dim() == 2 and y.shape[1] >= 2:
            yv = y[:, 0:2]
            prior_rel = float((x[:, 11:13] - yv).norm() / yv.norm().clamp(min=1e-30))
        rows.append(dict(
            stem=f.stem, n=n, mid=float(mid.float().mean()),
            sten=float(r.median() / r.min()),
            d1=float(x[:, 16].abs().max()), d2=float(x[:, 17].abs().max()),
            wn0=float((x[:, 4:6].norm(dim=1) < 1e-8).float().mean()),
            nt=float(x[:, 6:10].abs().max()),
            prior_rel=prior_rel,
            has_y=y is not None,
            uref=float(d.u_ref.reshape(-1)[0]) if hasattr(d, "u_ref") else float("nan"),
        ))

    n_f = len(rows)
    g = lambda k: np.array([r[k] for r in rows], dtype=float)

    # 1. topology
    midfrac = g("mid")
    if args.expect_p1:
        check("topology is P1 (pre-elevation)", OK if midfrac.max() < 0.05 else FAIL,
              f"max mid-side fraction {midfrac.max():.3f} (expect ~0)")
    else:
        check("topology is P2 (deploy order)", OK if midfrac.min() > 0.6 else FAIL,
              f"min mid-side fraction {midfrac.min():.3f} (deploy is 0.746)")

    # 2. the prior leak -- the single most expensive bug in this project's history
    pr = g("prior_rel")
    pr = pr[np.isfinite(pr)]
    if pr.size:
        leaked = int((pr < 0.05).sum())
        check("prior block is NOT the CFD solution", OK if leaked == 0 else FAIL,
              f"{leaked}/{pr.size} packs have rel-L2 < 0.05 vs their own labels "
              f"(min {pr.min():.4f}) -- s17 Z2 leak")
    else:
        check("prior block is NOT the CFD solution", WARN, "no labels to compare against")

    # 3. width derivatives inside the range the encoder was trained on
    d2 = g("d2")
    check("width_d2 within training range", OK if np.median(d2) < 500 else WARN,
          f"median {np.median(d2):.1f}, max {d2.max():.1f} (training p95 73.8; "
          f"1e4+ means a stale WLS operator, B13)")

    # 4. wall normals / node types populated (B14 -- these were identically zero for a year)
    check("wall_normal populated", OK if g("wn0").max() < 0.05 else FAIL,
          f"max zero-normal fraction {g('wn0').max():.3f}")
    check("node_type populated", OK if g("nt").min() > 0 else FAIL,
          f"min |node_type| max {g('nt').min():.3f} (0 means the channel is dead)")

    # 5. labels present
    check("labels present", OK if all(r["has_y"] for r in rows) else FAIL,
          f"{sum(r['has_y'] for r in rows)}/{n_f} packs carry y")

    # 6. the shape tail -- the regime the deploy cohort fails in
    st = g("sten")
    frac2 = float((st >= 2.0).mean())
    check("severe-stenosis coverage", OK if frac2 >= 0.10 else WARN,
          f"{100 * frac2:.0f}% of vessels at ratio >= 2.0 (deployment: 14%); "
          f"median {np.median(st):.2f}, max {st.max():.2f}")

    # 7. BC range overlap with deployment
    ur = g("uref")
    ur = ur[np.isfinite(ur)]
    if ur.size:
        check("u_ref overlaps deployment", OK if 0.05 < np.median(ur) < 0.25 else WARN,
              f"median {np.median(ur):.4f} (deployment 0.076-0.154)")

    print(f"\nPREFLIGHT  {args.src}   n={n_f}\n" + "=" * 78)
    for name, status, detail in results:
        mark = {OK: " ok ", WARN: "WARN", FAIL: "FAIL"}[status]
        print(f"  [{mark}] {name:<38} {detail}")
    print("=" * 78)
    print(f"  nodes      median {np.median(g('n')):.0f}   range {g('n').min():.0f}-{g('n').max():.0f}")
    print(f"  stenosis   median {np.median(st):.2f}   p95 {np.percentile(st, 95):.2f}")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"\n  {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        print("  DO NOT TRAIN on this cohort until the failures are resolved.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
