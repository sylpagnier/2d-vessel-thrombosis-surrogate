"""Does the pre-flight check actually catch the failures it claims to?

The claim in `src/clot_ml/preflight.py` is that a check using only GROUND-TRUTH-FREE statistics
(the gate's own firing set) identifies flow fields that would produce a vacuous prediction.  This
script tests that claim on the cohort, as a detector:

    detection    of the vessels whose wall gate is empty, how many does it FAIL?
    false alarm  of the vessels whose flow is fit to deploy on, how many does it FAIL?

FEM flow is the shipped path and is the negative class; the learned surrogate is the positive
class -- it empties the gate on 5 of 33 cohort vessels.  A check that fires on all 5 and on none
of the 33 FEM vessels is doing its job.

Run the diagnostics first (they carry the per-vessel firing counts):

    python scripts/publication/generate_flow_diagnostics.py --flow pred
    python scripts/publication/generate_flow_diagnostics.py --flow fem --out outputs/runs/flow_diagnostics_fem.json

Usage:
    python scripts/validate_preflight.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.preflight import (  # noqa: E402
    FAIL, FIRE_FRAC_MAX, FIRE_FRAC_MIN, PASS, WARN, PreflightResult,
)


def _verdict_from_counts(n_fire: int, n_wall: int) -> PreflightResult:
    """Replay `preflight_check`'s decision from stored counts, without re-solving any flow."""
    if n_wall == 0:
        return PreflightResult(FAIL, 0, 0, float("nan"), ["no wall nodes"])
    if n_fire == 0:
        return PreflightResult(FAIL, n_wall, 0, 0.0, ["empty wall gate"])
    frac = n_fire / n_wall
    if frac < FIRE_FRAC_MIN:
        return PreflightResult(WARN, n_wall, n_fire, frac, ["under-firing"])
    if frac > FIRE_FRAC_MAX:
        return PreflightResult(WARN, n_wall, n_fire, frac, ["over-firing"])
    return PreflightResult(PASS, n_wall, n_fire, frac, [])


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"[preflight-val] missing {path}; see this script's docstring")
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _report(name: str, rows: list[dict]) -> dict:
    verdicts = {}
    for r in rows:
        v = _verdict_from_counts(int(r["n_fire_pred"]), int(r["n_wall"]))
        verdicts[r["stem"]] = v
    n = len(verdicts)
    fails = [s for s, v in verdicts.items() if v.verdict == FAIL]
    warns = [s for s, v in verdicts.items() if v.verdict == WARN]
    fracs = np.array([v.fire_frac for v in verdicts.values() if np.isfinite(v.fire_frac)])

    print(f"\n=== {name}  (n={n}) ===")
    print(f"  FAIL {len(fails):>2}   WARN {len(warns):>2}   PASS {n - len(fails) - len(warns):>2}")
    print(f"  fire_frac  min {fracs.min():.4f}   median {np.median(fracs):.4f}   "
          f"max {fracs.max():.4f}")
    if fails:
        print(f"  FAIL: {', '.join(sorted(fails))}")
    if warns:
        print(f"  WARN: {', '.join(sorted(warns))}")
    return {"n": n, "fail": fails, "warn": warns,
            "verdicts": {s: v.verdict for s, v in verdicts.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=str(REPO / "outputs/runs/flow_diagnostics.json"))
    ap.add_argument("--fem", default=str(REPO / "outputs/runs/flow_diagnostics_fem.json"))
    ap.add_argument("--out", default=str(REPO / "outputs/runs/preflight_validation.json"))
    a = ap.parse_args()

    pred_rows, fem_rows = _load(Path(a.pred)), _load(Path(a.fem))
    pred = _report("learned surrogate flow (RGP-DEQ)", pred_rows)
    fem = _report("local FEM flow (the shipped path)", fem_rows)

    # Detection: the check must FAIL exactly the vessels whose gate is genuinely empty.
    truly_empty = {r["stem"] for r in pred_rows if int(r["n_fire_pred"]) == 0}
    caught = truly_empty & set(pred["fail"])
    missed = truly_empty - set(pred["fail"])
    false_alarms = set(fem["fail"])

    print("\n=== detector performance ===")
    print(f"  empty-gate vessels under the surrogate : {len(truly_empty)}")
    print(f"  caught by the check                    : {len(caught)}/{len(truly_empty)}")
    print(f"  missed                                 : {len(missed)}"
          + (f"  ({', '.join(sorted(missed))})" if missed else ""))
    print(f"  FALSE ALARMS on the shipped FEM path   : {len(false_alarms)}/{fem['n']}"
          + (f"  ({', '.join(sorted(false_alarms))})" if false_alarms else ""))

    ok = not missed and not false_alarms
    print("\n  " + ("PASS -- catches every vacuous case, and never refuses a good one."
                    if ok else
                    "REVIEW -- see missed / false alarms above."))

    payload = {"pred": pred, "fem": fem,
               "detected": sorted(caught), "missed": sorted(missed),
               "false_alarms": sorted(false_alarms),
               "thresholds": {"fire_frac_min": FIRE_FRAC_MIN,
                              "fire_frac_max": FIRE_FRAC_MAX},
               "ok": ok}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
