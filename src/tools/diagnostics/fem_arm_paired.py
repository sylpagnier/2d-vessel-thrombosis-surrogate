"""Paired comparison between two arms of the local-FEM accuracy table.

`local_fem_accuracy` reports a median per arm.  Comparing two arms by differencing their
medians is not a paired test and it silently changes the vessel set: `gate_sep_jaccard` is
NaN wherever the separation gate never fires, and a hotter arm fires on vessels the cooler
one skips, so its median is pulled down by vessels the other arm never scored.  Report the
median of the per-vessel DELTA over the vessels both arms scored, with the n.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

METRICS = (
    "gate_jaccard",
    "gate_sep_jaccard",
    "gate_low_jaccard",
    "dsrx_corr",
    "dsrx_ratio",
    "fire_ratio",
    "sr_corr",
)


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def compare(rows: list[dict], arm_a: str, arm_b: str) -> dict[str, Any]:
    """Per-metric paired delta (arm_a - arm_b) over vessels both arms scored."""
    out: dict[str, Any] = {"arm_a": arm_a, "arm_b": arm_b, "n_rows": len(rows), "metrics": {}}
    for metric in METRICS:
        a_all, b_all, deltas, wins = [], [], [], 0
        for row in rows:
            a = _finite(row.get(f"{arm_a}_{metric}"))
            b = _finite(row.get(f"{arm_b}_{metric}"))
            if a is not None:
                a_all.append(a)
            if b is not None:
                b_all.append(b)
            if a is not None and b is not None:
                deltas.append(a - b)
                wins += int(a > b)
        out["metrics"][metric] = {
            "n_a": len(a_all),
            "n_b": len(b_all),
            "n_paired": len(deltas),
            "median_a": _median(a_all),
            "median_b": _median(b_all),
            "unpaired_median_delta": _median(a_all) - _median(b_all),
            "paired_median_delta": _median(deltas),
            "n_a_better": wins,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="outputs/diag_fem_arm_table.json")
    ap.add_argument("--arm-a", default="fem_h3_g1")
    ap.add_argument("--arm-b", default="fem_h6_g3")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.table).read_text(encoding="utf-8"))
    res = compare(rows, args.arm_a, args.arm_b)

    print(f"paired {args.arm_a} - {args.arm_b}   ({res['n_rows']} vessels in table)")
    head = f"{'metric':<20}{'med A':>9}{'med B':>9}{'unpaired':>10}{'PAIRED':>10}{'n':>5}{'A>B':>6}"
    print(head)
    print("-" * len(head))
    for metric, m in res["metrics"].items():
        print(f"{metric:<20}{m['median_a']:>9.3f}{m['median_b']:>9.3f}"
              f"{m['unpaired_median_delta']:>+10.3f}{m['paired_median_delta']:>+10.3f}"
              f"{m['n_paired']:>5}{m['n_a_better']:>6}")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
