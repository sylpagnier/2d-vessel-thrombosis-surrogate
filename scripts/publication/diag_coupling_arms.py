"""Paired comparison of a coupled CV arm against the uncoupled one, judged against the seed null.

E1g's `coupling_selfdriven.json` was assembled by hand from `eval_strict` outputs; this is that
arithmetic written down so the next coupling arm (E1h, the real FEM re-solve) goes through the
identical instrument rather than a re-derivation of it.

    pairs      `cpl_<arm><sfx>` against `cpl_open<sfx>` for every suffix both exist under
               ("", "_s2", "_s3" = seed offsets 0, 1, 2).  Same vessels, same folds, same seed.
    delta      mean over vessels of (coupled - uncoupled), per domain, per pair; NaN rows (no
               off-wall GT) are dropped pairwise.
    seed null  the RANGE of the uncoupled arm's cohort means across ALL its seeds -- the
               definition STORY 2.1g quotes (wall 0.0119, off 0.1390).  An effect inside it is
               not established, whatever its sign.
    p_signflip paired sign-flip permutation over vessels, two-sided, on the pooled per-vessel
               delta.  Vessels are not independent across pairs, so this pools per-vessel
               deltas averaged over pairs first.
    collapse   a pair whose coupled wall mean sits more than 5 nulls below its partner -- the
               fold-level readout failure seen on k16 and wake at seed offset 0.  Reported,
               and excluded from `delta_clean`, never silently.

    delta_healthy    **THE NUMBER TO QUOTE.**  Mean of the arm's HEALTHY runs' cohort means minus
               the mean of the uncoupled arm's HEALTHY runs, with "healthy" decided by the same
               cut-signature rule on both sides (`degenerate`).  `se` uses the pooled within-arm
               run SD over the whole ladder (`run_noise`); `within_2se` is the verdict.
    delta_unpaired   diagnostic: the same with the uncoupled runs NOT screened.
    delta_loo_open   `delta_unpaired` recomputed with each uncoupled seed left out in turn, as
               [min, max].  If that interval spans zero, or is wider than the effect, the result
               depends on which baseline run you happened to pair against.

WHY THE PAIRED DELTA AND ITS p ARE NOT QUOTABLE (retraction, 2026-09-13).  Pairing by seed offset
shares the initial weights and nothing else: coupled and uncoupled features send training down
different paths from the first step, so a pair carries BOTH runs' noise.  Worse, every coupled
arm is paired with the SAME three uncoupled runs, and `cpl_open_s3` is a weak run (off-wall 0.690
against 0.816/0.829, from two degenerate fold cuts).  Across six coupling schedules the pair-3
off-wall delta was +0.114 with a spread of only 0.038 between arms -- the signature of a shared bad
reference, not of coupling -- while pairs 1-2 averaged -0.006.  The sign-flip p treated that shared
run as independent evidence and gave p=0.008.  `p_signflip` is kept for diagnostics only.

    python scripts/publication/diag_coupling_arms.py --arm femcpl \
        --out outputs/deployclot/coupling_fem_resolve.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()
SUFFIXES = ("", "_s2", "_s3")


def _rows(tag: str) -> dict | None:
    p = REPO / "outputs" / "deployclot" / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8"))["rows"] if p.is_file() else None


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _mean(rows: dict, dom: str) -> float:
    return float(np.mean([r[dom] for r in rows.values() if _ok(r[dom])]))


def _signflip(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    if d.size == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    return float((np.abs((flips * d).mean(axis=1)) >= obs - 1e-15).mean())


def _chosen(tag: str) -> dict | None:
    p = REPO / "outputs" / "deployclot" / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")).get("chosen") if p.is_file() else None


def degenerate(tag: str) -> bool:
    """A run whose fold readouts collapsed: >= 2 of its folds chose a 3rd `resid` cut <= 0.1.

    Read off the CUTS, not the score, so it applies identically to coupled and uncoupled runs.
    It flags exactly the five coupled runs whose wall score collapsed (femfinal, fem64_s3,
    femn20, wake, c16 -- all 0.66-0.74 wall) AND `cpl_open_s3` (wall 0.938 but off-wall 0.690),
    and no healthy run.  The old wall-score rule could not see `open_s3`, so bad coupled runs were
    dropped while the bad baseline run was kept -- an asymmetry that favoured coupling.
    """
    ch = _chosen(tag) or {}
    # only a `resid` fold has a 3rd cut; a fold that picked `plain` (2 cuts) cannot collapse this way
    return sum(1 for v in ch.values()
               if v["wall"][0] == "resid" and float(v["wall"][1][2]) <= 0.1) >= 2


#: per-vessel row key for each metric: BATC (the severity score) or the node-exact strict F1
#: `eval_strict` records beside it for the same final-time prediction.
METRIC_KEYS = {"batc": {"wall": "wall", "off": "off"},
               "strict_f1": {"wall": "wall_f1", "off": "off_f1"}}


#: every arm in the coupling ladder, for the pooled run-to-run noise estimate
LADDER = ("open", "wake", "femfinal", "fem64", "femn20", "femcpl", "femn5", "fem1")


def run_noise(dom: str, metric: str = "batc") -> float:
    """Pooled within-arm SD of HEALTHY runs' cohort means, over every arm in `LADDER`.

    Three seeds per arm is too few to estimate an arm's own spread, but the spread of a healthy
    run around its arm's mean is the same training/readout noise in every arm, so it pools.
    """
    key = METRIC_KEYS[metric][dom]
    ss, dof = 0.0, 0
    for a in LADDER:
        m = [_mean(_rows(f"cpl_{a}{s}"), key) for s in SUFFIXES
             if _rows(f"cpl_{a}{s}") is not None and not degenerate(f"cpl_{a}{s}")]
        if len(m) > 1:
            ss += float(np.sum((np.asarray(m) - np.mean(m)) ** 2))
            dof += len(m) - 1
    return float(np.sqrt(ss / dof)) if dof else float("nan")


def compare(arm: str, open_: str = "open", metric: str = "batc") -> dict:
    """The comparison for one arm; raises when no seed pair exists.  Shared with the figure.

    ``metric`` picks the per-vessel score ("batc" or "strict_f1"); the healthy-run screen
    (`degenerate`, read off the fold cuts) is the same for both, so the two are directly comparable.
    """
    opens = {s: _rows(f"cpl_{open_}{s}") for s in SUFFIXES}
    arms = {s: _rows(f"cpl_{arm}{s}") for s in SUFFIXES}
    pairs = [s for s in SUFFIXES if opens[s] is not None and arms[s] is not None]
    if not pairs:
        raise SystemExit(f"[ERR] no seed suffix has both cpl_{open_}* and cpl_{arm}*")

    bad_open = {s: degenerate(f"cpl_{open_}{s}") for s in SUFFIXES if opens[s] is not None}
    bad_arm = {s: degenerate(f"cpl_{arm}{s}") for s in pairs}
    # a PAIR is unusable if EITHER side collapsed -- symmetric by construction
    collapsed = {s: bad_arm[s] or bad_open.get(s, False) for s in pairs}

    res = {}
    for dom_name in ("wall", "off"):
        dom = METRIC_KEYS[metric][dom_name]
        healthy_open = [_mean(opens[s], dom) for s in bad_open if not bad_open[s]]
        healthy_arm = [_mean(arms[s], dom) for s in pairs if not bad_arm[s]]
        sd = run_noise(dom_name, metric)
        d_h = (float(np.mean(healthy_arm) - np.mean(healthy_open))
               if healthy_arm and healthy_open else float("nan"))
        se = (sd * math.sqrt(1 / len(healthy_arm) + 1 / len(healthy_open))
              if healthy_arm and healthy_open else float("nan"))
        open_means = [_mean(r, dom) for r in opens.values() if r is not None]
        null = float(max(open_means) - min(open_means)) if len(open_means) > 1 else float("nan")
        per_pair, per_vessel = {}, {}
        for s in pairs:
            O, A = opens[s], arms[s]
            common = sorted(k for k in O if k in A and _ok(O[k][dom]) and _ok(A[k][dom]))
            d = np.array([A[k][dom] - O[k][dom] for k in common])
            per_pair[s or "_s1"] = dict(
                delta=round(float(d.mean()), 4), n=len(common),
                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                max_vessel_move=round(float(np.abs(d).max()), 4), collapsed=bool(collapsed[s]))
            if collapsed[s]:
                continue
            for k, x in zip(common, d):
                per_vessel.setdefault(k, []).append(float(x))
        clean = [v["delta"] for v in per_pair.values() if not v["collapsed"]]
        pooled = np.array([np.mean(v) for v in per_vessel.values()])
        delta = float(np.mean([v["delta"] for v in per_pair.values()]))
        arm_means = [_mean(arms[s], dom) for s in pairs if not collapsed[s]]
        unpaired = (float(np.mean(arm_means) - np.mean(open_means)) if arm_means
                    else float("nan"))
        loo = ([float(np.mean(arm_means) - np.mean([m for j, m in enumerate(open_means)
                                                    if j != i]))
                for i in range(len(open_means))] if arm_means and len(open_means) > 1 else [])
        res[dom_name] = dict(
            delta_healthy=round(d_h, 4),
            se=round(se, 4),
            run_noise_sd=round(sd, 4),
            within_2se=bool(abs(d_h) <= 2 * se) if se == se else None,
            n_healthy_arm=len(healthy_arm), n_healthy_open=len(healthy_open),
            degenerate_open=[s or "_s1" for s, b in bad_open.items() if b],
            degenerate_arm=[s or "_s1" for s, b in bad_arm.items() if b],
            delta_unpaired=round(unpaired, 4),
            delta_loo_open=[round(min(loo), 4), round(max(loo), 4)] if loo else None,
            arm_healthy_run_means=[round(x, 4) for x in arm_means],
            delta=round(delta, 4),
            delta_clean=round(float(np.mean(clean)), 4) if clean else None,
            seed_null=round(null, 4),
            exceeds_seed_null=bool(clean and abs(float(np.mean(clean))) > null),
            p_signflip=round(_signflip(pooled), 4),
            p_signflip_note="diagnostic only: pairs share baseline runs, see module docstring",
            pairs=per_pair,
            open_seed_means=[round(x, 4) for x in open_means],
        )

    return dict(arm=arm, open=open_, metric=metric, pairs=[s or "_s1" for s in pairs],
                seed_null_definition="range of the uncoupled arm's cohort means across its seeds",
                result=res)


def _paired(A: dict, B: dict, dom: str) -> tuple[float, int, int, float]:
    """(mean B-A, n, B wins, max |move|) over vessels scored in both."""
    ks = [k for k in A if k in B and _ok(A[k][dom]) and _ok(B[k][dom])]
    d = np.array([B[k][dom] - A[k][dom] for k in ks])
    return float(d.mean()), len(ks), int((d > 1e-9).sum()), float(np.abs(d).max())


def _avg_rows(*runs: dict) -> dict:
    """Per-vessel mean over seed runs, per domain (NaN where no run scores the vessel)."""
    doms = ("wall", "off", "full")
    return {k: {d: (float(np.mean([r[k][d] for r in runs if _ok(r[k].get(d))]))
                    if any(_ok(r[k].get(d)) for r in runs) else float("nan")) for d in doms}
            for k in runs[0] if all(k in r for r in runs)}


def rebuild_oracle_summaries() -> None:
    """Re-derive the numbers in the three E1-era summaries that were assembled by hand.

    `coupling_trained.json` (STORY 2.1c), `coupling_stride.json` (2.1e) and
    `coupling_selfdriven.json` (2.1g) were written by hand from `eval_strict` outputs.  This is that
    arithmetic, checked on 2026-09-16 to reproduce every stored number from the runs as they stood,
    so re-scoring the runs (e.g. under a label change) updates the summaries instead of leaving them
    stale.  Only the numeric result blocks are replaced; the prose fields are kept as written.
    """
    def R(tag: str) -> dict:
        rows = _rows(tag)
        if rows is None:
            hint = (" -- score it with `eval_strict.py --labels comsol --save "
                    f"outputs/deployclot/{tag}.json`" if tag.endswith("_comsol_labels") else "")
            raise SystemExit(f"[ERR] missing outputs/deployclot/{tag}.json{hint}")
        return rows

    r4 = lambda x: round(float(x), 4)
    # LABELS.  The GT-oracle arms (closed, k16) feed COMSOL's own clot into the features, so they
    # are scored against COMSOL's labels as given (`eval_strict --labels comsol`), and the uncoupled
    # side of THOSE comparisons must be too -- `cpl_open*_comsol_labels`.  The self-driven (wake)
    # comparison is deploy-legal on both sides and keeps the default flow-branch labels.
    Oc = [R("cpl_open_comsol_labels"), R("cpl_open_s2_comsol_labels")]
    O, C = Oc, [R("cpl_closed"), R("cpl_closed_s2")]
    K, W = [R("cpl_c16"), R("cpl_c16_s2"), R("cpl_c16_s3")], [R("cpl_wake"), R("cpl_wake_s2")]
    Ow = [R("cpl_open"), R("cpl_open_s2")]
    O3, W3 = R("cpl_open_s3"), R("cpl_wake_s3")
    base = REPO / "outputs" / "deployclot"

    trained = json.loads((base / "coupling_trained.json").read_text(encoding="utf-8"))
    for dom in ("wall", "off", "full"):
        s0, s1 = _paired(O[0], C[0], dom), _paired(O[1], C[1], dom)
        avg = _paired(_avg_rows(*O), _avg_rows(*C), dom)
        mo, mc = [_mean(x, dom) for x in O], [_mean(x, dom) for x in C]
        trained["result"][dom] = dict(
            seed0=dict(delta=r4(s0[0]), n=s0[1], closed_wins=s0[2]),
            seed1=dict(delta=r4(s1[0]), n=s1[1], closed_wins=s1[2]),
            seed_averaged=dict(delta=r4(np.mean([mc[0] - mo[0], mc[1] - mo[1]])), n=avg[1],
                               closed_wins=avg[2]),
            seed_noise_open=r4(mo[1] - mo[0]), seed_noise_closed=r4(mc[1] - mc[0]),
            mean_open=r4(np.mean(mo)), mean_closed=r4(np.mean(mc)))
    (base / "coupling_trained.json").write_text(json.dumps(trained, indent=2), encoding="utf-8")

    stride = json.loads((base / "coupling_stride.json").read_text(encoding="utf-8"))
    self_ = json.loads((base / "coupling_selfdriven.json").read_text(encoding="utf-8"))
    for dom in ("wall", "off"):
        mo = [_mean(x, dom) for x in O]
        every = float(np.mean([_mean(x, dom) for x in C]) - np.mean(mo))
        km = [_mean(x, dom) for x in K]
        healthy = float(np.mean(km[1:]) - np.mean(mo))   # k16 seed 0 is the fold-collapse run
        kh = _paired(_avg_rows(*O), _avg_rows(*K[1:]), dom)
        stride["result"][dom].update(
            delta_every_step=r4(every), delta_k16_all3=r4(np.mean(km) - np.mean(mo)),
            delta_k16_healthy=r4(healthy), frac_of_ceiling_healthy=round(healthy / every, 3),
            seed_spread_open=r4(abs(mo[1] - mo[0])),
            seed_spread_closed=r4(abs(_mean(C[1], dom) - _mean(C[0], dom))),
            seed_spread_k16_all3=r4(max(km) - min(km)), seed_spread_k16_healthy=r4(abs(km[2] - km[1])),
            n=kh[1], k16_wins=kh[2])

        # self-driven (wake) vs uncoupled: both deploy-legal, flow-branch labels on both sides
        mw = [_mean(x, dom) for x in Ow]
        wm = [_mean(x, dom) for x in W]
        wa = _paired(_avg_rows(*Ow), _avg_rows(*W), dom)
        self_["result"][dom].update(
            mean_uncoupled=r4(np.mean(mw)), mean_selfcoupled=r4(np.mean(wm)),
            delta=r4(np.mean(wm) - np.mean(mw)), delta_seed0=r4(wm[0] - mw[0]),
            delta_seed1=r4(wm[1] - mw[1]), seed_noise_open=r4(abs(mw[1] - mw[0])),
            seed_noise_wake=r4(abs(wm[1] - wm[0])),
            exceeds_seed_noise=bool(abs(np.mean(wm) - np.mean(mw)) > abs(wm[1] - wm[0])),
            n=wa[1], wins=wa[2])
        h = _paired(Ow[1], W[1], dom)
        self_["result_healthy_seed"][dom].update(
            uncoupled=r4(mw[1]), self_coupled=r4(wm[1]), delta=r4(h[0]), n=h[1], wins=h[2],
            max_vessel_move=r4(h[3]), open_seed_noise=r4(abs(mw[1] - mw[0])))
        p1, p2 = _paired(Ow[1], W[1], dom)[0], _paired(O3, W3, dom)[0]
        om = mw + [_mean(O3, dom)]
        null = max(om) - min(om)
        self_["result_two_clean_pairs"][dom].update(
            delta_pair1=r4(p1), delta_pair2=r4(p2), delta_mean=r4((p1 + p2) / 2),
            open_seed_null=r4(null), exceeds_null=bool(abs((p1 + p2) / 2) > null), n=h[1])
    (base / "coupling_stride.json").write_text(json.dumps(stride, indent=2), encoding="utf-8")
    (base / "coupling_selfdriven.json").write_text(json.dumps(self_, indent=2), encoding="utf-8")
    print("[i] rebuilt coupling_trained / coupling_stride / coupling_selfdriven")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-oracle-summaries", action="store_true",
                    help="re-derive coupling_trained/stride/selfdriven.json from the cpl_* runs")
    ap.add_argument("--arm", help="coupled tag stem, e.g. femcpl -> cpl_femcpl*")
    ap.add_argument("--open", default="open", help="uncoupled tag stem")
    ap.add_argument("--metric", default="batc", choices=sorted(METRIC_KEYS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.rebuild_oracle_summaries:
        rebuild_oracle_summaries()
        return 0
    if not args.arm:
        ap.error("--arm is required unless --rebuild-oracle-summaries is given")

    out = compare(args.arm, args.open, args.metric)
    pairs, res = out["pairs"], out["result"]
    print(f"=== cpl_{args.arm} vs cpl_{args.open}  ({len(pairs)} seed pairs) ===")
    for dom, r in res.items():
        pp = "  ".join(f"{k}:{v['delta']:+.4f}{'(COLLAPSE)' if v['collapsed'] else ''}"
                       for k, v in r["pairs"].items())
        print(f"{dom:<5} HEALTHY {r['delta_healthy']:+.4f} +/- {r['se']:.4f} (SE)  "
              f"within 2SE {r['within_2se']}  runs {r['n_healthy_arm']}v{r['n_healthy_open']}"
              f"  degenerate arm {r['degenerate_arm']} open {r['degenerate_open']}"
              f"  | diagnostics: unpaired-all {r['delta_unpaired']:+.4f}  paired {r['delta_clean']}"
              f"  p {r['p_signflip']:.4f}  {pp}")
    if args.out:
        (REPO / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[i] wrote {REPO / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
