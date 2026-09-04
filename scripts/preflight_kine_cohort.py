"""Check a freshly generated / transferred kinematics cohort BEFORE committing to a train.

    python scripts/preflight_kine_cohort.py --src transfer/carreau

Every check here corresponds to a bug that has already cost this project a training run.  It is
cheap, it is CPU-only, and it is the last gate before GPU time.  Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wall-shear-sample", type=int, default=40,
                    help="vessels to measure the wall-shear regime on (0 disables). One MLS build each, so sampled, not exhaustive.")
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

    import json as _json

    from src.config import VesselConfig as _VCfg

    _mesh_dir = _VCfg(phase="kinematics").mesh_input_dir

    def meta_of(pack_path):
        jf = _mesh_dir / f"{pack_path.stem}.json"
        try:
            return _json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            return None

    def elevate_for_regime(d):
        """A P1 cohort is elevated at load, so read its regime AFTER elevation --
        that interpolation is a large part of what this check exists to catch."""
        from src.data_gen.lib.p2_elevation import elevate_to_p2
        try:
            return elevate_to_p2(d, keep_wls=False)
        except Exception:
            return d

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
        ei = d.edge_index
        _el = (x[ei[1], 0:2] - x[ei[0], 0:2]).norm(dim=1)
        _el = _el[_el > 0]
        rows.append(dict(
            stem=f.stem, n=n, mid=float(mid.float().mean()),
            h_nd=float(_el.median()) if _el.numel() else float("nan"),
            sten=float(r.median() / r.min()),
            d1=float(x[:, 16].abs().max()), d2=float(x[:, 17].abs().max()),
            wn0=float((x[:, 4:6].norm(dim=1) < 1e-8).float().mean()),
            nt=float(x[:, 6:10].abs().max()),
            prior_rel=prior_rel,
            has_y=y is not None,
            # An unsolved vessel still gets a pack: `mesh_to_graph` writes an all-zero `y` and
            # `is_anchor=False` when COMSOL produced no .npz.  `has_y` is True for those.
            solved=(
                bool(getattr(d, "is_anchor", torch.tensor([False])).reshape(-1)[0])
                and y is not None and float(y[:, 0:2].abs().max()) > 0.0
            ),
            uref=float(d.u_ref.reshape(-1)[0]) if hasattr(d, "u_ref") else float("nan"),
            has_inlet_bc=(
                torch.is_tensor(getattr(d, "u_inlet_bc", None))
                and getattr(d, "u_inlet_bc").shape[0] == n
                and float(getattr(d, "u_inlet_bc").abs().max()) > 0
            ),
            glevel=int(getattr(d, "geometry_level", torch.tensor([-1])).reshape(-1)[0])
            if hasattr(d, "geometry_level") else -1,
            reshaped=bool((meta_of(f) or {}).get("reshaped_from")),
            resh=(meta_of(f) or {}).get("reshaped_from") or {},
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

    # 3. width derivatives -- and the clamp bounds this cohort implies
    from src.config import WIDTH_D1_MAX, WIDTH_D2_MAX

    d1a, d2 = g("d1"), g("d2")
    new_d1, new_d2 = float(np.percentile(d1a, 95)), float(np.percentile(d2, 95))
    check("width_d2 operator is sane", OK if np.median(d2) < 1e3 else FAIL,
          f"median {np.median(d2):.1f}, max {d2.max():.1f} (1e4+ means a stale WLS operator, B13)")
    # The clamps are a property of the TRAINING CORPUS.  Carrying stale ones forward silently
    # truncates real signal: the shipped 4.14 / 73.8 came from a 40-vessel corpus with no severe
    # stenosis and would have clamped 44% / 34% of this cohort.
    over_d1 = float((d1a > WIDTH_D1_MAX).mean())
    over_d2 = float((d2 > WIDTH_D2_MAX).mean())
    stale = over_d1 > 0.10 or over_d2 > 0.10
    check("clamp bounds match this cohort", WARN if stale else OK,
          f"config {WIDTH_D1_MAX:.2f}/{WIDTH_D2_MAX:.1f} clamps "
          f"{100 * over_d1:.0f}%/{100 * over_d2:.0f}% of vessels; "
          f"this cohort's p95 = {new_d1:.2f}/{new_d2:.1f}")

    # 4. wall normals / node types populated (B14 -- these were identically zero for a year)
    check("wall_normal populated", OK if g("wn0").max() < 0.05 else FAIL,
          f"max zero-normal fraction {g('wn0').max():.3f}")
    check("node_type populated", OK if g("nt").min() > 0 else FAIL,
          f"min |node_type| max {g('nt').min():.3f} (0 means the channel is dead)")

    # 5. labels present -- and actually SOLVED
    check("labels present", OK if all(r["has_y"] for r in rows) else FAIL,
          f"{sum(r['has_y'] for r in rows)}/{n_f} packs carry y")

    # A failed COMSOL solve is not an error here: `mesh_to_graph` writes the pack anyway with an
    # all-zero `y` and `is_anchor=False`, and training uses those as unsupervised physics-only
    # graphs.  It was invisible, though -- the 2026-08-28 cohort shipped 39/250 (15.6%) unsolved
    # and every check passed, because `has_y` is True for a zero tensor.
    solved = np.array([r["solved"] for r in rows], dtype=bool)
    frac_uns = float((~solved).mean())
    check("COMSOL solve rate", OK if frac_uns <= 0.05 else WARN,
          f"{int(solved.sum())}/{n_f} solved; {int((~solved).sum())} unsolved "
          f"({100 * frac_uns:.1f}%) carry zero labels and can only contribute PDE terms")

    # 6. the shape tail -- the regime the deploy cohort fails in.
    # Counted over SOLVED vessels only.  Solve failure is not uniform: it rises monotonically
    # with stenosis (2.9% below 1.5, 40.6% above 3.0), so an unfiltered count overstates exactly
    # the tail this cohort is generated to add.
    st = g("sten")
    frac2 = float((st[solved] >= 2.0).mean()) if solved.any() else 0.0
    frac2_all = float((st >= 2.0).mean())
    # 26% is measured over all 53 biochem anchor packs (p50 1.36, p75 2.41, p90 4.59, max 20.5).
    # The 14% this used to quote came from a smaller sample and understated the deploy tail.
    check("severe-stenosis coverage (solved)", OK if frac2 >= 0.10 else WARN,
          f"{100 * frac2:.0f}% of SOLVED vessels at ratio >= 2.0 (deployment: 26%; "
          f"{100 * frac2_all:.0f}% before dropping unsolved); "
          f"median {np.median(st):.2f}, max {st.max():.2f}")
    resh = [r["resh"] for r in rows if r.get("reshaped")]
    if resh:
        # A substitution is EASIER than what it replaced, by design -- an equally extreme
        # re-draw fails for the same reason the original did (38 re-drawn, 36 still unsolved,
        # 2026-08-29).  Report how much severity that cost, because it is a real cost: it is
        # the severe tail this cohort exists to provide.
        was = np.array([float(x.get("severity_was", np.nan)) for x in resh])
        now = np.array([float(x.get("severity_now", np.nan)) for x in resh])
        ok_m = np.isfinite(was) & np.isfinite(now) & (was > 0)
        if ok_m.any():
            keep = now[ok_m] / was[ok_m]
            still_severe = int((now[ok_m] >= 2.0).sum())
            check("geometry substitutions", OK,
                  f"{len(resh)}/{n_f} vessels re-drawn easier after COMSOL could not solve the "
                  f"original (same level and class); severity kept "
                  f"{100 * np.median(keep):.0f}% (p10 {100 * np.percentile(keep, 10):.0f}%), "
                  f"{still_severe}/{int(ok_m.sum())} still at ratio >= 2.0")
        else:
            check("geometry substitutions", OK,
                  f"{len(resh)}/{n_f} vessels re-drawn easier after COMSOL could not solve the "
                  f"original (same level and class)")

    if not solved.all():
        lost = [(r["stem"], r["sten"]) for r in rows if not r["solved"]]
        lost.sort(key=lambda t: -t[1])
        print("  unsolved vessels (worst stenosis first, open these in COMSOL):")
        for stem, sr in lost[:15]:
            print(f"    {stem:14s} stenosis ratio {sr:.2f}")
        if len(lost) > 15:
            print(f"    ... and {len(lost) - 15} more")

    # 6b. resolution against deployment, in the units the model consumes.
    # The biochem anchor pipeline is fixed, so this is the corpus's job to match.  Positions are
    # stored as `x / d_bar`, which makes edge length comparable across vessels of any physical
    # size.  P2 elevation halves every edge exactly (a mid-side node per edge), so a P1 cohort's
    # deploy-equivalent spacing is `h_nd / 2`.
    # Deploy reference, measured over 53 biochem anchor packs: p10 0.0195, med 0.0245, p90 0.0339.
    DEPLOY_H_MED, DEPLOY_H_LO, DEPLOY_H_HI = 0.0245, 0.0195, 0.0339
    hn = g("h_nd")
    hn = hn[np.isfinite(hn)]
    if hn.size:
        h_p2 = np.median(hn) / (2.0 if args.expect_p1 else 1.0)
        ratio = h_p2 / DEPLOY_H_MED
        per = hn / (2.0 if args.expect_p1 else 1.0)
        inband = float(((per >= DEPLOY_H_LO) & (per <= DEPLOY_H_HI)).mean())
        # +-15% of the deploy median; the corpus sat at 1.17x before `mesh_lc` was set from this.
        check("resolution matches deployment", OK if 0.85 <= ratio <= 1.15 else WARN,
              f"h_nd {h_p2:.4f} at P2 vs deployment {DEPLOY_H_MED:.4f} ({ratio:.2f}x); "
              f"{100 * inband:.0f}% of vessels inside deploy's p10-p90 band "
              f"[{DEPLOY_H_LO:.4f}, {DEPLOY_H_HI:.4f}]")
    else:
        check("resolution matches deployment", WARN, "no usable edges to measure")

    # 6c. the width derivative channels against deployment.  These are model INPUTS, and the
    # clamp in `ginodeq` is applied to both corpora, so a corpus that never reaches deploy's
    # range trains the model on an unsaturated channel it will find saturated at deploy.
    # Deploy reference (53 biochem anchor packs, per-vessel max |channel|):
    #   width_d1  p50  10.54  p95    112.1     width_d2  p50  2109  p95  121295
    DEPLOY_D1_MED, DEPLOY_D2_MED = 10.54, 2109.4
    d1m, d2m = float(np.median(d1a)), float(np.median(d2))
    r1, r2 = d1m / DEPLOY_D1_MED, d2m / DEPLOY_D2_MED
    check("width channels reach deployment's range",
          OK if (0.25 <= r1 <= 4.0 and 0.25 <= r2 <= 4.0) else WARN,
          f"median per-vessel max: d1 {d1m:.2f} vs deploy {DEPLOY_D1_MED:.2f} ({r1:.2f}x), "
          f"d2 {d2m:.1f} vs deploy {DEPLOY_D2_MED:.0f} ({r2:.3f}x); "
          f"config clamps {100 * over_d1:.0f}%/{100 * over_d2:.0f}% here vs 38%/70% at deploy")

    # 7a. the inlet BC the analytic prior is anchored on (B4).  Without it
    # `inlet_anchored_umax_nd` silently falls back to fixed module constants -- the prior is
    # then 28% mis-scaled and nothing reports it.
    n_bc = sum(1 for r in rows if r["has_inlet_bc"])
    check("inlet BC present (analytic prior anchor)", OK if n_bc == n_f else FAIL,
          f"{n_bc}/{n_f} packs carry a usable u_inlet_bc")

    # 7b. geometry_level drives the curriculum and the level-stratified split.  A missing value
    # degrades to -1 ("unknown"), which silently drops the vessel out of the stratification.
    gl = g("glevel")
    n_unknown = int((gl < 0).sum())
    check("geometry_level present", OK if n_unknown == 0 else WARN,
          f"{n_unknown}/{n_f} packs read -1 (unknown); levels seen "
          f"{sorted(set(int(v) for v in gl if v >= 0))}")

    # 8. BC range overlap with deployment
    ur = g("uref")
    ur = ur[np.isfinite(ur)]
    if ur.size:
        check("u_ref overlaps deployment", OK if 0.05 < np.median(ur) < 0.25 else WARN,
              f"median {np.median(ur):.4f} (deployment 0.076-0.154)")

    # 9. THE CONSUMER'S OWN STATISTICS -- the check every other one here is a proxy for.
    #
    # Checks 1-8 are all producer-side: mesh order, operator sanity, element size, stenosis
    # ratio, BC range, solve rate.  They passed the 2026-08-28 cohort at 0 FAIL while its labels
    # did not contain what `clot_ml` reads.  The deposition gate is `(sr < lss) | (dsrx < sgt)`,
    # and at the FIT median **91.5% of firing wall nodes fire through the `dsrx` branch alone**;
    # in that cohort the branch fired on no wall node at all in more than half the vessels.
    # No mesh statistic can see that.
    if args.wall_shear_sample:
        from src.utils.wall_shear_regime import (
            compare_to_reference, load_reference, summarise, wall_shear_regime,
        )

        pick = files
        if len(pick) > args.wall_shear_sample:
            step = max(1, len(pick) // args.wall_shear_sample)
            pick = pick[::step][: args.wall_shear_sample]
        regime_rows = []
        for f in pick:
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
                regime_rows.append(
                    wall_shear_regime(elevate_for_regime(d) if args.expect_p1 else d))
            except Exception:
                regime_rows.append(None)
        regime_rows = [r for r in regime_rows if r]
        ref = load_reference()
        if not regime_rows:
            check("wall-shear regime matches deployment", WARN, "no vessel could be measured")
        elif ref is None:
            check("wall-shear regime matches deployment", WARN,
                  "no reference band -- run scripts/derive_deploy_wall_shear_band.py")
        else:
            verdicts = compare_to_reference(summarise(regime_rows), ref)
            worst = FAIL if any(v == FAIL for _, v, _ in verdicts) else (
                WARN if any(v == WARN for _, v, _ in verdicts) else OK)
            head = next((d for k, _, d in verdicts if k == "wall_sep_only"), "")
            check(f"wall-shear regime matches deployment (n={len(regime_rows)})", worst,
                  f"sep-only {head}")
            for k, v, detail in verdicts:
                if k != "wall_sep_only":
                    check("   " + k, v, detail)

    print(f"\nPREFLIGHT  {args.src}   n={n_f}\n" + "=" * 78)
    for name, status, detail in results:
        mark = {OK: " ok ", WARN: "WARN", FAIL: "FAIL"}[status]
        print(f"  [{mark}] {name:<38} {detail}")
    print("=" * 78)
    print(f"  nodes      median {np.median(g('n')):.0f}   range {g('n').min():.0f}-{g('n').max():.0f}")
    print(f"  stenosis   median {np.median(st):.2f}   p95 {np.percentile(st, 95):.2f}")
    if hn.size:
        print(f"  h_nd       median {np.median(hn):.4f} (P1)   "
              f"{np.median(hn) / 2:.4f} at P2   <- deployment {DEPLOY_H_MED:.4f}")
    print(f"  width_d1   p95 {new_d1:8.2f}   width_d2   p95 {new_d2:8.1f}"
          f"   <- src/config.py WIDTH_D1_MAX / WIDTH_D2_MAX")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"\n  {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        print("  DO NOT TRAIN on this cohort until the failures are resolved.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
