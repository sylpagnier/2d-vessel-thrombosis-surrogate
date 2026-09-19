"""Pair the open-loop and oracle-closed-loop `clot_ml_0` runs into one artifact.

WHY.  PUBLICATION_NOTES.md 1 states the oracle closed-loop result -- ground-truth clot occupancy
fed back into the flow at every step, the measured post-gelation shear collapse applied under it,
the deposition gate re-evaluated with the consumer's own operator -- as an upper bound on what any
flow corrector could contribute.  The numbers there were computed by hand from two separate
`eval_clot_ml_0.py` runs and never saved as a diffable artifact, so nothing in
docs/publication/STORY.md 2.1 can carry a claim id for them.  This script runs both arms and
writes the paired deltas `verify_claims.py`'s `paired()` resolver already knows how to read.

    python scripts/eval_closed_loop_oracle.py --cohort --out outputs/deployclot/closed_loop_oracle.json

TWO SUBPROCESSES, AND THAT IS LOAD-BEARING -- NOT A STYLE CHOICE.  The toggle
`CLOT_ML_ORACLE_BLOCKAGE` is captured at MODULE IMPORT TIME
(`src/clot_ml/features.py:59`, a module-level constant).  Setting `os.environ` in-process after
the first arm has already imported the module leaves the second arm reading the stale value, and
the two arms come out BIT-IDENTICAL -- which looks exactly like the "closing the loop changes
nothing" result this script exists to measure.  That is the same failure mode the 2026-09-04 env
cleanup caused and that took until 2026-09-07 to notice, for the same reason: silent, and
identical rather than wrong.  Each arm therefore gets a fresh interpreter with the environment
set BEFORE import, exactly as the documented reproduce command does it, and
`--assert-differ` (on by default) fails the run if the two arms match bit for bit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root

REPO = get_project_root()


def _paired_stats(open_vals, closed_vals, n_boot: int = 10000, seed: int = 0) -> dict:
    """Delta, paired bootstrap 95% CI, and a paired sign-flip permutation p.

    Same shape as the RGP-DEQ arm's `crossfit5_vs_shipped.json`, so `paired()` in
    verify_claims.py reads it with no change.
    """
    o = np.asarray(open_vals, dtype=float)
    c = np.asarray(closed_vals, dtype=float)
    keep = ~(np.isnan(o) | np.isnan(c))
    o, c = o[keep], c[keep]
    n = int(len(o))
    if n == 0:
        return dict(delta=None, ci=[None, None], p=None, n=0)
    diffs = c - o
    delta = float(np.mean(diffs))
    if n < 2:
        return dict(delta=round(delta, 4), ci=[None, None], p=None, n=n,
                    mean_open=round(float(o.mean()), 4), mean_closed=round(float(c.mean()), 4))
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
    perm = (signs * diffs).mean(axis=1)
    return dict(
        delta=round(delta, 4),
        ci=[round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4)],
        p=round(float(np.mean(np.abs(perm) >= abs(delta))), 4),
        n=n,
        mean_open=round(float(o.mean()), 4),
        mean_closed=round(float(c.mean()), 4),
    )


def _run_arm(blockage: bool, cohort: bool, stems, every: int, flow: str, out: Path) -> list[dict]:
    """One arm, in its own interpreter, with the environment set before import."""
    cmd = [sys.executable, "scripts/eval_clot_ml_0.py",
           "--every", str(every), "--flow", flow, "--out", str(out)]
    if cohort:
        cmd.append("--cohort")
    elif stems:
        cmd += ["--stems", *stems]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    if blockage:
        env["CLOT_ML_ORACLE_BLOCKAGE"] = "1"
    else:
        env.pop("CLOT_ML_ORACLE_BLOCKAGE", None)
    label = "closed (oracle blockage ON)" if blockage else "open"
    print(f"\n[i] === arm: {label} ===", flush=True)
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[ERR] {label} arm exited {r.returncode}")
    if not out.is_file():
        raise SystemExit(f"[ERR] {label} arm wrote no output at {out}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="store_true",
                    help="all FIT+DEV clot-carrying packs plus the wound vessels")
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--out", default="outputs/deployclot/closed_loop_oracle.json")
    ap.add_argument("--keep-arms", action="store_true",
                    help="keep the two per-arm row files beside --out")
    ap.add_argument("--from-arms", nargs=2, metavar=("OPEN", "CLOSED"), default=None,
                    help="skip both runs and pair two EXISTING row files. This is how a "
                         "chunked or resumed run is finished: score the cohort in batches "
                         "(each batch a fresh interpreter, so memory does not accumulate "
                         "across vessels), concatenate each arm's rows, then pair them here.")
    ap.add_argument("--chunk", type=int, default=0,
                    help="run the cohort in batches of N vessels, each in its own pair of "
                         "interpreters. 0 (default) runs every vessel in one process, which "
                         "is faster but has been observed to die silently partway through a "
                         "full FEM cohort -- use 4-6 for an unattended run.")
    ap.add_argument("--allow-identical", action="store_true",
                    help="do NOT fail when the arms are bit-identical. Only for a deliberate "
                         "null test -- identical arms normally mean the env toggle is dead.")
    args = ap.parse_args()

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_arms or args.chunk:
        open_p = outp.with_name(outp.stem + "_arm_open.json")
        closed_p = outp.with_name(outp.stem + "_arm_closed.json")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="closedloop_"))
        open_p, closed_p = tmp / "open.json", tmp / "closed.json"

    if args.from_arms:
        open_rows = json.loads((REPO / args.from_arms[0]).read_text(encoding="utf-8"))
        closed_rows = json.loads((REPO / args.from_arms[1]).read_text(encoding="utf-8"))
        print(f"[i] pairing existing arms: {len(open_rows)} open / {len(closed_rows)} closed")
    elif args.chunk:
        # Each batch is a fresh pair of interpreters. Rows accumulate on disk, so a batch that
        # dies costs only that batch -- rerun with the same --out and the finished ones are
        # already written. Vessel-level memory does not carry across batches.
        import importlib
        mod = importlib.import_module("scripts.eval_clot_ml_0")
        all_stems = mod._cohort_stems() if args.cohort else (args.stems or list(mod.DEFAULT_STEMS))
        open_rows, closed_rows = [], []
        if open_p.is_file():
            open_rows = json.loads(open_p.read_text(encoding="utf-8"))
        if closed_p.is_file():
            closed_rows = json.loads(closed_p.read_text(encoding="utf-8"))
        done = {r["stem"] for r in open_rows} & {r["stem"] for r in closed_rows}
        todo = [s for s in all_stems if s not in done]
        if done:
            print(f"[i] resuming: {len(done)} vessels already paired, {len(todo)} to go")
        for i in range(0, len(todo), args.chunk):
            batch = todo[i:i + args.chunk]
            print(f"\n[i] ### batch {i // args.chunk + 1}: {' '.join(batch)}", flush=True)
            bo = open_p.with_name(open_p.stem + "_batch.json")
            bc = closed_p.with_name(closed_p.stem + "_batch.json")
            open_rows += _run_arm(False, False, batch, args.every, args.flow, bo)
            closed_rows += _run_arm(True, False, batch, args.every, args.flow, bc)
            open_p.write_text(json.dumps(open_rows, indent=2), encoding="utf-8")
            closed_p.write_text(json.dumps(closed_rows, indent=2), encoding="utf-8")
            bo.unlink(missing_ok=True)
            bc.unlink(missing_ok=True)
    else:
        open_rows = _run_arm(False, args.cohort, args.stems, args.every, args.flow, open_p)
        closed_rows = _run_arm(True, args.cohort, args.stems, args.every, args.flow, closed_p)

    # THE GUARD.  Identical arms mean the toggle never arrived -- see the module docstring.
    if open_rows == closed_rows and not args.allow_identical:
        raise SystemExit(
            "[ERR] the two arms are BIT-IDENTICAL.\n"
            "      That normally means CLOT_ML_ORACLE_BLOCKAGE never reached the closed arm,\n"
            "      not that closing the loop changes nothing. Check\n"
            "      src/clot_ml/features.py:59 still reads the environment, and that\n"
            "      src/utils/_env_allowlist.json still carries the name.\n"
            "      Pass --allow-identical only if you have confirmed the toggle IS live.")

    by_open = {r["stem"]: r for r in open_rows}
    by_closed = {r["stem"]: r for r in closed_rows}
    stems = sorted(set(by_open) & set(by_closed))
    nonwound = [s for s in stems if not by_open[s].get("wound")]
    wound = [s for s in stems if by_open[s].get("wound")]

    paired, per_vessel = {}, {}

    def add(key: str, group: list[str], field: str):
        if not group:
            return
        o = [by_open[s].get(field) for s in group]
        c = [by_closed[s].get(field) for s in group]
        paired[key] = _paired_stats(o, c)
        per_vessel[key] = [
            dict(stem=s, open=by_open[s].get(field), closed=by_closed[s].get(field),
                 delta=(None if by_open[s].get(field) is None or by_closed[s].get(field) is None
                        else round(float(by_closed[s][field] - by_open[s][field]), 4)))
            for s in group]

    add("nonwound_wall", nonwound, "v0_fin_wall")
    add("nonwound_off", nonwound, "v0_fin_off")
    add("wound_w_reg", wound, "v0_fin_w_reg")
    add("wound_w_lum", wound, "v0_fin_w_lum")
    add("wound_wall", wound, "v0_fin_wall")

    out = dict(
        protocol=("oracle closed loop: GT clot occupancy fed back into the flow at every step, "
                  "the measured post-gelation shear collapse applied under it, the deposition "
                  "gate re-evaluated with the consumer's own operator. An upper bound on any "
                  "flow corrector -- no model error, no localisation error. Toggle is "
                  "CLOT_ML_ORACLE_BLOCKAGE=1, captured at import, so each arm runs in its own "
                  "interpreter (see module docstring)."),
        metric=("evaluate.domain_score == BATC_0 (unadjusted). NOT the BATC setting the CV "
                "tables report -- the two run 0.19-0.22 apart off-wall. Deltas here are "
                "within-metric and like-for-like; absolute levels are not comparable to a "
                "BATC table."),
        flow=args.flow, every=args.every,
        n_nonwound=len(nonwound), n_wound=len(wound),
        nonwound_stems=nonwound, wound_stems=wound,
        noise_floor=dict(wall=0.024, off=0.074,
                         note="config spread of one arm on this cohort; per-vessel spread is "
                              "larger (median 0.042 wall / 0.112 off)"),
        paired=paired,
        per_vessel=per_vessel,
    )
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== oracle closed loop, paired (closed - open) ===", flush=True)
    print(f"{'domain':<16}{'delta':>9}{'95% CI':>20}{'p':>8}{'n':>4}")
    for k, v in paired.items():
        ci = "--" if v["ci"][0] is None else f"[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
        p = "--" if v["p"] is None else f"{v['p']:.4f}"
        d = "--" if v["delta"] is None else f"{v['delta']:+.4f}"
        print(f"{k:<16}{d:>9}{ci:>20}{p:>8}{v['n']:>4}")
    print(f"\n[i] wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
