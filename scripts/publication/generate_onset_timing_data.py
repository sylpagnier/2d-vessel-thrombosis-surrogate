"""Onset-timing diagnostic: does the model commit a node to clot early or late,
relative to ground truth -- not just "is the score low", but "is it out of phase"?

For every node that eventually clots in ground truth AND is eventually committed
by the model (both within the sampled OOF horizon), the signed lag is
    lag = t_pred_onset - t_gt_onset      (timesteps; +ve = model is LATE, -ve = EARLY)

Pooled across every vessel in the strict OOF archive -- not just the fig3/4/6
cohort -- for a real sample size. Nodes the model never commits (pure misses) or
commits without a matching GT clot (pure false alarms) are reported separately
as coverage, not folded into the timing histogram, since they have no lag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR  # noqa: E402
from scripts.publication.oof_data import (  # noqa: E402
    build_vessel_figure_data, ensure_oof_series, load_oof_archive,
)


def _onset_lags(payload: dict) -> dict:
    times = np.asarray(payload["times"], dtype=np.int64)
    n_t = len(times)
    n_nodes = payload["wall"].shape[0]
    gt_stack = np.zeros((n_t, n_nodes), dtype=bool)
    pred_stack = np.zeros((n_t, n_nodes), dtype=bool)
    for i, t in enumerate(times):
        fd = payload["frames"][int(t)]
        gt_stack[i] = fd["gt_mask"]
        pred_stack[i] = fd["pred_mask"]

    def first_true_idx(stack: np.ndarray) -> np.ndarray:
        # index of first True along axis 0, or -1 if never true for that node
        any_true = stack.any(axis=0)
        idx = np.where(any_true, stack.argmax(axis=0), -1)
        return idx

    gt_idx = first_true_idx(gt_stack)
    pred_idx = first_true_idx(pred_stack)

    gt_ever = gt_idx >= 0
    pred_ever = pred_idx >= 0
    both = gt_ever & pred_ever
    miss_only = gt_ever & ~pred_ever          # GT clots, model never commits: pure FN
    false_only = pred_ever & ~gt_ever         # model commits, GT never clots: pure FP

    lag_steps = (times[pred_idx[both]] - times[gt_idx[both]]).astype(np.int64) if both.any() \
        else np.array([], dtype=np.int64)

    return dict(
        lag_steps=lag_steps,
        n_both=int(both.sum()),
        n_miss_only=int(miss_only.sum()),
        n_false_only=int(false_only.sum()),
        n_gt_ever=int(gt_ever.sum()),
    )


def main() -> None:
    print("[i] Generating onset-timing data (predicts early vs. late)")
    oof_path = ensure_oof_series(CONFIG)
    archive = load_oof_archive(oof_path)

    rows = []
    all_lags = []
    for stem, oof in sorted(archive.vessels.items()):
        print(f"  -> {stem} [fold {oof.fold}] ...")
        payload = build_vessel_figure_data(stem, oof)
        r = _onset_lags(payload)
        if r["n_both"]:
            all_lags.append(r["lag_steps"])
        rows.append(dict(
            vessel=stem, fold=oof.fold, n_gt_clot_nodes=r["n_gt_ever"],
            n_matched=r["n_both"], n_missed_entirely=r["n_miss_only"],
            n_false_alarm_only=r["n_false_only"],
            median_lag=float(np.median(r["lag_steps"])) if r["n_both"] else float("nan"),
        ))

    if not all_lags:
        raise SystemExit("[ERR] no matched onset pairs found across the OOF archive")

    lags = np.concatenate(all_lags)
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(DATA_DIR / "onset_timing_by_vessel.csv", index=False)

    df_lags = pd.DataFrame({"lag_steps": lags})
    df_lags.to_csv(DATA_DIR / "onset_timing_lags.csv", index=False)

    pct_early = 100.0 * float((lags < 0).mean())
    pct_ontime = 100.0 * float((lags == 0).mean())
    pct_late = 100.0 * float((lags > 0).mean())
    print(f"[OK] {len(lags)} matched onset pairs across {len(rows)} vessels")
    print(f"     median lag {np.median(lags):+.1f} steps  "
          f"({pct_early:.1f}% early, {pct_ontime:.1f}% on-time, {pct_late:.1f}% late)")
    print(f"[OK] Saved {DATA_DIR / 'onset_timing_lags.csv'}")
    print(f"[OK] Saved {DATA_DIR / 'onset_timing_by_vessel.csv'}")


if __name__ == "__main__":
    main()
