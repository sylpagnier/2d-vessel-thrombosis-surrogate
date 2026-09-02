"""Recalibrate the geometry classifier against the REPAIRED `width_nd` (roadmap item A2).

WHY.  `width_nd` is produced by sphere-marching along `wall_normal`, and `wall_normal` was
identically zero at every wall node on every pack until 2026-08-22 (MODEL_REVIEW 6.5).  The
channel was therefore degenerate, `geometry_class.py`'s own docstring recorded the symptom --
*"unusable on 9 of 34 vessels"* -- and predicted the cure: *"fix the channel and the abstain
goes away."*  It did.  **The abstain is gone (0 of 45 unusable) and the two cuts came out of
it in opposite states**, which is the finding:

    aneurysm    designated 2.11 / 2.22 / 2.67   nearest cohort other 1.61 (patient013)
                SEPARATION +0.506 -- any cut in (1.61, 2.11) works; `BULGE_ANEURYSM = 2.0`
                sits inside it with margin +/-0.25.  KEEP.

    stenosis    designated 0.52 / 0.53 / 0.58   nearest cohort other 0.51 (patient012)
                SEPARATION -0.071 -- `patient012`, a baseline, is NARROWER than all three
                designated stenoses.  Unsmoothed it is the same ordering (012 p2/med 0.510
                against 041's 0.538).  **NO CUT SEPARATES THEM.**

So this run does not produce a new stenosis threshold, and that is the correct outcome rather
than a failure of the run: the classes overlap, and refitting a cut across overlapping classes
is coin-flipping, not calibration (PHASE10 13.3, `clot-cohort-noise-floor`).  What A2 changed
instead was the two things that WERE wrong:

  * `classify` now takes `USER_DESIGNATED` as authoritative, so the three labelled stenoses
    stop silently reclassifying to `baseline`;
  * `width_stats`'s along-wall smoothing includes the node itself.  Neighbours-only divided by
    `max(cnt, 1)`, so a selected node with no selected neighbour smoothed to exactly 0 and set
    the 2nd percentile -- `patient008`'s `narrowing = 0.0000` was 12 such nodes against a raw
    wall width that never goes below 0.61.  Fixed: 0.0000 -> 0.8533.

WHAT THIS SCRIPT IS FOR.  It prints the full ordered distribution of both statistics with the
step between consecutive vessels, so a cut is judged against the GAP it sits in rather than
against three points; it reports the separation each class achieves; it excludes the wound
packs from the calibration (a wound is a cavity in the wall and reads as pathology on its own
merits -- letting them in closes both gaps for the wrong reason) while showing them as a
check; and `--explain` dissects any vessel whose statistic looks anomalous.

    python scripts/diag_geometry_class_recal.py
    python scripts/diag_geometry_class_recal.py --explain patient008,patient041,patient012
"""
from __future__ import annotations

from src.tools.diagnostics._common import bootstrap, biochem_packs_dir

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from src.clot_ml.geometry_class import (  # noqa: E402
    BULGE_ANEURYSM, NARROWING_STENOSIS, USER_DESIGNATED, classify, disagreements,
    measured_class, width_stats,
)
from src.core_physics.wall_cohort_splits import CLOT_FREE, DEV, FIT, SEALED  # noqa: E402
from src.data_gen.lib.mesh_wls import solid_boundary_nodes  # noqa: E402

PACKS = biochem_packs_dir()


def all_stems() -> list[str]:
    stems = list(FIT) + list(DEV) + list(SEALED) + list(CLOT_FREE)
    stems += sorted(p.stem for p in PACKS.glob("wound_patient*.pt"))
    seen, out = set(), []
    for s in stems:
        if s not in seen and (PACKS / f"{s}.pt").exists():
            seen.add(s)
            out.append(s)
    return out


def gap_report(name: str, vals: dict, designated: set, *, low_is_pathology: bool):
    """Print the ordered statistic and the largest gap separating the designated set."""
    order = sorted(vals, key=lambda a: vals[a], reverse=not low_is_pathology)
    print("\n%s -- ordered, pathology first (* = user-designated)" % name.upper())
    prev = None
    for a in order:
        v = vals[a]
        mark = " *" if a in designated else "  "
        gap = "" if prev is None else "   (step %+.4f)" % (v - prev)
        print("   %-18s %8.4f%s%s" % (a, v, mark, gap))
        prev = v

    des = [vals[a] for a in designated if a in vals]
    oth = [vals[a] for a in vals if a not in designated]
    if not des or not oth:
        return
    if low_is_pathology:
        worst_des, best_oth = max(des), min(oth)
        sep = best_oth - worst_des
    else:
        worst_des, best_oth = min(des), max(oth)
        sep = worst_des - best_oth
    print("\n   designated span %.4f - %.4f | nearest other %.4f | SEPARATION %+.4f"
          % (min(des), max(des), best_oth, sep))
    if sep <= 0:
        print("   >>> NO CUT SEPARATES THEM.  A threshold here is a coin flip, not a "
              "measurement.")
    else:
        mid = 0.5 * (worst_des + best_oth)
        print("   >>> a cut anywhere in (%.4f, %.4f) separates them; midpoint %.4f, "
              "margin +/-%.4f" % (min(worst_des, best_oth), max(worst_des, best_oth), mid,
                                  0.5 * sep))


def explain(stem: str) -> None:
    """Why is this vessel's statistic what it is?  Enough detail to call it real or broken."""
    d = torch.load(PACKS / f"{stem}.pt", map_location="cpu", weights_only=False)
    ch = {c: i for i, c in enumerate(d.x_channel_names.split(","))}
    w = d.x[:, ch["width_nd"]].detach().cpu().numpy().astype(np.float64)
    wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
    solid = solid_boundary_nodes(d)
    print("\n=== %s ===" % stem)
    print("  nodes %d  wall %d  solid %d" % (len(w), int(wall.sum()), int(solid.sum())))
    for name, m in (("wall", wall), ("solid", solid), ("all", np.ones_like(wall))):
        v = w[m]
        print("  width_nd[%-5s] n=%6d  min %8.4f  p2 %8.4f  med %8.4f  p98 %8.4f  "
              "max %8.4f  zeros %d"
              % (name, len(v), v.min(), np.percentile(v, 2), np.median(v),
                 np.percentile(v, 98), v.max(), int((v <= 0).sum())))
    s = width_stats(d)
    print("  width_stats -> %s" % {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in s.items()})
    n_zero_wall = int((w[wall] <= 0).sum())
    if n_zero_wall:
        print("  !! %d wall nodes carry width_nd <= 0.  `width_stats` selects `w > 0`, so "
              "they are\n     excluded from the median but the p2 of what remains can still "
              "sit on the\n     sphere-march's floor -- that is a CHANNEL failure at those "
              "nodes, not a narrowing." % n_zero_wall)


def main(argv: list[str] | None = None) -> int:
    bootstrap()
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", default="", help="comma-separated stems to dissect")
    args = ap.parse_args(argv)

    if args.explain:
        for s in args.explain.split(","):
            explain(s.strip())
        return 0

    stems = all_stems()
    rows = {}
    for a in stems:
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        rows[a] = width_stats(d)

    print("%-18s %8s %10s %8s %-10s %-10s" %
          ("pack", "bulge", "narrowing", "usable", "classify", "designated"))
    n_unusable = 0
    for a in stems:
        s = rows[a]
        n_unusable += 0 if s.get("usable") else 1
        print("%-18s %8.4f %10.4f %8s %-10s %-10s"
              % (a, s.get("bulge", float("nan")), s.get("narrowing", float("nan")),
                 s.get("usable"), classify(s, a), USER_DESIGNATED.get(a, "")))
    print("\nunusable width_nd: %d of %d" % (n_unusable, len(stems)))

    # The cuts are calibrated on the COHORT.  Wound packs are a different population -- the
    # injured segment is a cavity in the wall, so it reads as a bulge or a narrowing on its
    # own merits -- and letting them into the calibration set makes both gaps look closed for
    # a reason that has nothing to do with the cohort's anatomy.  They are reported below
    # instead, where a measured class that fires on them is a check rather than a confound.
    wound = [a for a in stems if a.startswith("wound_")]
    usable = {a: rows[a] for a in stems
              if rows[a].get("usable") and not a.startswith("wound_")}
    an = {a for a, c in USER_DESIGNATED.items() if c == "aneurysm"} & set(usable)
    st = {a for a, c in USER_DESIGNATED.items() if c == "stenosis"} & set(usable)
    gap_report("bulge (aneurysm)", {a: usable[a]["bulge"] for a in usable}, an,
               low_is_pathology=False)
    gap_report("narrowing (stenosis)", {a: usable[a]["narrowing"] for a in usable}, st,
               low_is_pathology=True)

    if wound:
        print("\nWOUND PACKS -- excluded from the calibration above, shown as a check")
        for a in wound:
            r = rows[a]
            print("   %-18s bulge %8.4f  narrowing %8.4f  -> measured %s"
                  % (a, r.get("bulge", float("nan")), r.get("narrowing", float("nan")),
                     measured_class(r)))

    dis = disagreements(usable)
    print("\nDESIGNATION vs MEASUREMENT on the %d labelled vessels" % len(
        [a for a in usable if a in USER_DESIGNATED]))
    if not dis:
        print("   all reproduced by the statistics alone")
    else:
        for a, (want, got) in sorted(dis.items()):
            print("   %-18s designated %-10s measured %-10s   (bulge %.4f narrowing %.4f)"
                  % (a, want, got, usable[a]["bulge"], usable[a]["narrowing"]))
        print("   `classify` returns the DESIGNATION for these; only `measured_class` "
              "disagrees.")

    print("\nCURRENT CUTS: BULGE_ANEURYSM = %.2f, NARROWING_STENOSIS = %.2f"
          % (BULGE_ANEURYSM, NARROWING_STENOSIS))
    print("A cut is only worth moving if the gap it lands in is wide compared with the "
          "spread\nWITHIN each group.  Read the step column above before touching either "
          "constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
