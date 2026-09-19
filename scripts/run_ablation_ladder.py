"""Run the frozen ablation ladder (docs/PHYSICS_ABLATION_PLAN.md) end to end, resumably.

Every arm is one `scripts/run_phase9_cv.py` invocation at the SHIPPED recipe -- same cache,
same 5 geometry-stratified folds, same `shape_w`/`clot_free_w` -- differing only in what
`--arm` holds constant or switches off.  So an arm's out-of-fold score is directly comparable
to the shipped model's, which is the one thing the plan's honest-risks section insists on
(risk 2: "ablation arms must be scored through the *same* pipeline").

WHY A DRIVER AND NOT A SHELL LOOP.  Three reasons, all of them about not wasting the GPU:

  * **resumable.**  An arm whose score file already exists is skipped, so a crash costs the
    arm it was in, not the run.  `--force` re-runs anyway.
  * **ordered by what the paper needs.**  Round 1 is 3 seeds on every arm, which already
    ranks the ladder; round 2 adds the independent 3-seed replicate that the off-wall null
    (+/-0.045 at three seeds, DEPLOYCLOT/`go_deployclot_split.sh`) actually requires.  A
    kill at any point leaves a complete, balanced round rather than half a ladder.
  * **one place that knows the tag scheme,** which the report then reads back.

    python scripts/run_ablation_ladder.py                     # everything, ~15 h
    python scripts/run_ablation_ladder.py --round 1           # 3 seeds per arm, ~7 h
    python scripts/run_ablation_ladder.py --arms A1,A4        # just these
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.utils.paths import get_project_root

REPO = get_project_root()

from src.clot_ml.ablation import ARMS, describe_arm  # noqa: E402

SCORES = REPO / "outputs/phase9_scores"
LOGS = REPO / "outputs/logs/ablation"

#: The ladder, in the order the paper needs it.  A4 first: it is the reference every other
#: arm is a delta against, and it doubles as a reproducibility check on the archived
#: `dc_v5_split` run (identical config, so the two should agree to seed noise).
ORDER: tuple[str, ...] = (
    "A4", "A1", "A0", "A0_pure", "A0_fresh", "A_naive", "A1p", "A2", "A3", "A1_pure",
    "B_iso", "B_nomp", "B_mlp", "B_nobase", "B_noseed", "B_r1", "B_bce",
)

#: the shipped objective deltas the DeployClot split family trains under
#: (`scripts/go_deployclot_split.sh` BLOCK 4).  Anything not listed is `recipe()`'s default.
SHIPPED_OVERRIDES: tuple[str, ...] = ("--shape-w", "2.0", "--clot-free-w", "0.25")


#: Default score-file prefix.  It does NOT name the cache, and it must not be reused across
#: caches: a `v5_fem` replicate written under the same tags would silently overwrite the
#: `v5_split` ladder and the report would average two generations into one row.  Pass
#: `--tag-prefix ablfem` (or anything distinct) for a second cache.
TAG_PREFIX = "abl"


def tag_for(arm: str, rnd: int, prefix: str = TAG_PREFIX) -> str:
    return "%s_%s" % (prefix, arm) if rnd == 1 else "%s_%s_seedB" % (prefix, arm)


def run_arm(arm: str, rnd: int, *, cache: str, folds: int, seeds: int,
            force: bool, dry: bool, prefix: str = TAG_PREFIX) -> str:
    tag = tag_for(arm, rnd, prefix)
    out = SCORES / f"{tag}.npz"
    if out.exists() and not force:
        return "skip"
    cmd = [sys.executable, "scripts/run_phase9_cv.py",
           "--tag", tag, "--cache", cache, "--arm", arm,
           "--folds", str(folds), "--seeds", str(seeds),
           # ROUND 2 IS AN INDEPENDENT REPLICATE, not a longer run: same cache, same folds,
           # disjoint seeds.  That is what makes the pair readable against the pipeline's own
           # noise floor rather than against nothing.
           "--seed-offset", str(0 if rnd == 1 else seeds),
           *SHIPPED_OVERRIDES]
    if dry:
        print("  would run: %s" % " ".join(cmd), flush=True)
        return "dry"
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{tag}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0:
        print("  !! %s FAILED rc=%d after %.0fs -- see %s" % (tag, rc, dt, log), flush=True)
        return "fail"
    print("  ok %s (%.0f min)" % (tag, dt / 60.0), flush=True)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="v5_split",
                    help="the shipped DeployClot split cache; the ablation must read the "
                         "same features the published model was trained on")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of the ladder (default: all, in ORDER)")
    ap.add_argument("--round", type=int, default=0, choices=[0, 1, 2],
                    help="1 = seeds 0-2, 2 = the independent 3-seed replicate, 0 = both")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag-prefix", default=TAG_PREFIX,
                    help="score-file prefix.  MUST differ per cache -- see TAG_PREFIX.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ORDER)
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit("unknown arm(s): %s" % ", ".join(bad))
    rounds = [1, 2] if args.round == 0 else [args.round]

    print("ABLATION LADDER  cache=%s prefix=%s folds=%d seeds=%d rounds=%s\n"
          % (args.cache, args.tag_prefix, args.folds, args.seeds, rounds), flush=True)
    for a in arms:
        print("  %s" % describe_arm(a), flush=True)
    print(flush=True)

    t0, tally = time.time(), {}
    for rnd in rounds:
        print("--- round %d (%s) ---" % (rnd, "seeds 0-%d" % (args.seeds - 1) if rnd == 1
                                         else "seeds %d-%d" % (args.seeds, 2 * args.seeds - 1)),
              flush=True)
        for a in arms:
            st = run_arm(a, rnd, cache=args.cache, folds=args.folds, seeds=args.seeds,
                         force=args.force, dry=args.dry_run, prefix=args.tag_prefix)
            tally[(a, rnd)] = st
    fails = [k for k, v in tally.items() if v == "fail"]
    print("\nladder done in %.1f h  (%d ok, %d skipped, %d failed)"
          % ((time.time() - t0) / 3600.0,
             sum(v == "ok" for v in tally.values()),
             sum(v == "skip" for v in tally.values()), len(fails)), flush=True)
    if fails:
        print("  FAILED: %s" % ", ".join("%s/r%d" % k for k in fails), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
