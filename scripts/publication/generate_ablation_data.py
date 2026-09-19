"""Score the physics/architecture ablation ladder and write the paper's table.

Reads the per-arm out-of-fold score files `scripts/run_ablation_ladder.py` produced and
grades every one of them through the SAME strictly-nested protocol the shipped numbers come
from -- `scripts/eval_significance.nested_rows`, which is `scripts/eval_strict.py`'s readout
selection with nothing re-implemented here.  That reuse is the point: the plan's honest-risks
section (risk 2) says an arm scored through a different pipeline is not comparable to the
shipped model, and the cheapest way to guarantee one pipeline is to have one copy of it.

TWO READOUTS PER ARM, and the second one matters.

  * `auto` -- the shipped protocol: `plain` and `resid` are both offered and the selection
    set picks.  This is the number directly comparable to every published score.
  * `plain` -- `resid` forced off.  The `resid` family thresholds physics-positive and
    physics-negative nodes separately, i.e. it READS THE PHYSICS BACKBONE'S OCCLUSION MASK.
    Under `auto`, a geometry-only arm can therefore still be read out through physics, and
    the ladder would understate what the physics conditioning buys.  `plain` closes that
    door at the readout the way `A1_pure` closes it at the architecture.

Every delta is a PAIRED bootstrap over vessels against the reference arm (`--ref`, default
A4 = the shipped conditioning), because arms share folds and vessels and an unpaired
interval on n=27 is far wider than the effect.

    python scripts/publication/generate_ablation_data.py
    python scripts/publication/generate_ablation_data.py --arms A0,A1,A4 --ref A1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()

from scripts.eval_significance import boot, nested_rows  # noqa: E402
from scripts.run_ablation_ladder import ORDER, TAG_PREFIX, tag_for  # noqa: E402
from src.clot_ml.ablation import ARM_NOTES, ARMS, describe_arm  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402

SCORES = REPO / "outputs/phase9_scores"
OUT = REPO / "outputs/ablation"

#: The archived production CV run at the identical configuration, PER CACHE.  `A4` re-runs it
#: inside the ladder, so the two are a free reproducibility check: they differ only in the fact
#: that one was launched months apart from the other, and should agree to seed noise.  If they
#: do not, something in the pipeline moved and no ablation delta is safe to quote.
#:
#: Keyed by cache because the ladder is run on more than one generation and the reference must
#: be the arm trained on the SAME features -- `dc_v5_split` against a `v5_fem` ladder would
#: compare two flow sources and call the difference an ablation.  Both families train at the
#: identical recipe (`shape_w=2.0`, `clot_free_w=0.25`), which is what makes them comparable
#: to their own ladder at all; asserted rather than assumed by reading each file's `cfg`.
SHIPPED_REFERENCE_FOR: dict[str, tuple[str, ...]] = {
    "v5_split": ("dc_v5_split", "dc_v5_split_seedB"),
    "v5_fem": ("dc_fem_cfw025", "dc_fem_seedB"),
}


def tags_present(arm: str, prefix: str = TAG_PREFIX) -> list[str]:
    """Both rounds of an arm if both ran, otherwise whichever did."""
    return [t for t in (tag_for(arm, 1, prefix), tag_for(arm, 2, prefix))
            if (SCORES / f"{t}.npz").exists()]


def _mean(rows: dict, pool: list[str], i: int) -> float:
    v = [rows[a][i] for a in pool if a in rows]
    return float(np.nanmean(v)) if v else float("nan")


def _arm_of(tags: list[str]) -> str:
    """What the score file itself says it is -- never what its filename suggests."""
    arms = set()
    for t in tags:
        z = np.load(SCORES / f"{t}.npz", allow_pickle=True)
        arms.add(str(z["arm"][0]) if "arm" in z.files else "?")
    return "+".join(sorted(arms))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v5_split")
    ap.add_argument("--tag-prefix", default=TAG_PREFIX,
                    help="which ladder to read; must match the prefix it was run under")
    ap.add_argument("--arms", default="", help="default: every arm with a score file")
    ap.add_argument("--ref", default="A4", help="arm every delta is measured against")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--out", default="",
                    help="default: outputs/ablation/ablation_report[_<prefix>].json -- one "
                         "file per ladder, so a second cache's run cannot overwrite the first")
    args = ap.parse_args()

    pfx = args.tag_prefix
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or [
        a for a in ORDER if tags_present(a, pfx)]
    arms = [a for a in arms if tags_present(a, pfx)]
    if not arms:
        raise SystemExit("no ablation score files found in %s -- run "
                         "scripts/run_ablation_ladder.py first" % SCORES)
    if args.ref not in arms:
        raise SystemExit("reference arm %r has no score file" % args.ref)

    cache = attach_physics(load_cache(args.cache))

    # the archived shipped run, when it is on disk: a reproducibility check on A4
    extra = {}
    ship = [t for t in SHIPPED_REFERENCE_FOR.get(args.cache, ())
            if (SCORES / f"{t}.npz").exists()]
    if ship:
        extra["SHIPPED"] = ship

    results: dict[str, dict] = {}
    pool_ref: list[str] = []
    for name, tags in [(a, tags_present(a, pfx)) for a in arms] + list(extra.items()):
        carrying, rows_auto = nested_rows(cache, tags)
        _, rows_plain = nested_rows(cache, tags, family="plain")
        pool_ref = pool_ref or carrying
        results[name] = dict(
            tags=tags, n_tags=len(tags), seeds=3 * len(tags),
            arm_recorded=_arm_of(tags),
            note=ARM_NOTES.get(name, ""),
            keeps="+".join(ARMS[name]) if name in ARMS else "",
            auto=dict(wall=_mean(rows_auto, carrying, 0), off=_mean(rows_auto, carrying, 1)),
            plain=dict(wall=_mean(rows_plain, carrying, 0),
                       off=_mean(rows_plain, carrying, 1)),
            _rows_auto={a: list(map(float, rows_auto[a])) for a in rows_auto},
            _rows_plain={a: list(map(float, rows_plain[a])) for a in rows_plain},
        )
        print("  scored %-9s tags=%s  wall %.4f  off %.4f"
              % (name, ",".join(tags), results[name]["auto"]["wall"],
                 results[name]["auto"]["off"]), flush=True)

    # --- paired deltas against the reference -------------------------------------------
    ref = results[args.ref]
    for name, r in results.items():
        r["delta"] = {}
        if name == args.ref:
            continue
        for readout in ("auto", "plain"):
            A = {a: tuple(ref["_rows_%s" % readout][a]) for a in ref["_rows_%s" % readout]}
            B = {a: tuple(r["_rows_%s" % readout][a]) for a in r["_rows_%s" % readout]}
            common = [a for a in pool_ref if a in A and a in B]
            d = {}
            for i, dom in ((0, "wall"), (1, "off")):
                mean, lo, hi, p_le0, n = boot(common, A, B, i, n=args.boot)
                # `boot` reports B - A, i.e. ARM MINUS REFERENCE.  Negative is the expected
                # sign for an ablation: it is what removing the physics costs.
                d[dom] = dict(delta=mean, lo=lo, hi=hi, p_arm_not_worse=p_le0, n=n)
            r["delta"][readout] = d

    payload = dict(
        cache=args.cache, tag_prefix=pfx, ref=args.ref, pool=pool_ref,
        n_carrying=len(pool_ref),
        arms={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
              for k, v in results.items()},
        per_vessel={k: dict(auto=v["_rows_auto"], plain=v["_rows_plain"])
                    for k, v in results.items()},
        describe={a: describe_arm(a) for a in arms if a in ARMS},
    )
    out = Path(args.out) if args.out else (
        OUT / ("ablation_report.json" if pfx == TAG_PREFIX
               else "ablation_report_%s.json" % pfx))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- the table -----------------------------------------------------------------------
    print("\nABLATION LADDER  cache=%s  n=%d clot-carrying vessels  ref=%s"
          % (args.cache, len(pool_ref), args.ref))
    print("strictly-nested out-of-fold, severity metric; delta is arm MINUS %s, "
          "paired bootstrap over vessels\n" % args.ref)
    hdr = ("%-9s %2s | %7s %7s | %16s %16s | %7s %7s"
           % ("arm", "sd", "wall", "off", "d wall [95% CI]", "d off  [95% CI]",
              "wall_p", "off_p"))
    print(hdr)
    print("-" * len(hdr))
    for name in list(arms) + list(extra):
        r = results[name]
        d = r["delta"].get("auto")
        def cell(dom):
            if not d:
                return "%16s" % "-- reference --"
            x = d[dom]
            return "%+.3f [%+.3f,%+.3f]" % (x["delta"], x["lo"], x["hi"])
        print("%-9s %2d | %7.4f %7.4f | %16s %16s | %7s %7s"
              % (name, r["seeds"], r["auto"]["wall"], r["auto"]["off"],
                 cell("wall"), cell("off"),
                 ("%.3f" % d["wall"]["p_arm_not_worse"]) if d else "",
                 ("%.3f" % d["off"]["p_arm_not_worse"]) if d else ""))

    print("\nreadout `plain` (the resid family, which reads the physics occlusion mask, "
          "forced off)\n")
    print("%-9s | %7s %7s | %16s %16s"
          % ("arm", "wall", "off", "d wall [95% CI]", "d off  [95% CI]"))
    for name in list(arms) + list(extra):
        r = results[name]
        d = r["delta"].get("plain")
        f = lambda dom: ("%+.3f [%+.3f,%+.3f]" % (d[dom]["delta"], d[dom]["lo"], d[dom]["hi"])
                         if d else "%16s" % "-- reference --")
        print("%-9s | %7.4f %7.4f | %16s %16s"
              % (name, r["plain"]["wall"], r["plain"]["off"], f("wall"), f("off")))

    print("\nwrote %s" % out)
    print("\nREAD THE HEADLINE AS: A4 - A1 under `auto`, and A4 - A1_pure for the genuinely "
          "generic\nmesh-GNN baseline.  A delta inside the pipeline's own noise floor "
          "(wall +/-0.005, off-wall\n+/-0.045 at three seeds) is not a result -- see "
          "scripts/eval_significance.py --floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
