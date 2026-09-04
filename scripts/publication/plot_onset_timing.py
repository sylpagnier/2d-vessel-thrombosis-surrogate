"""Onset-timing figure: does the model predict a clot node's onset early or late?

(a) pooled histogram of signed onset lag (timesteps) across every matched node
    in the strict OOF archive (+ve = model is late, -ve = early).
(b) per-vessel median lag, so the pooled number isn't hiding one outlier vessel.

Reads outputs/publication/data/onset_timing_lags.csv and
onset_timing_by_vessel.csv (generate_onset_timing_data.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.utils import setup_matplotlib_style  # noqa: E402


def main() -> None:
    print("[i] Plotting onset-timing (early vs. late)")
    setup_matplotlib_style()

    lags_path = DATA_DIR / "onset_timing_lags.csv"
    by_vessel_path = DATA_DIR / "onset_timing_by_vessel.csv"
    if not lags_path.is_file() or not by_vessel_path.is_file():
        print("[WARN] missing onset-timing data; run generate_onset_timing_data.py first")
        return

    lags = pd.read_csv(lags_path)["lag_steps"].to_numpy()
    by_vessel = pd.read_csv(by_vessel_path).dropna(subset=["median_lag"]).sort_values("median_lag")

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                          gridspec_kw={"width_ratios": [1.15, 1]})

    lo, hi = int(lags.min()), int(lags.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    ax_hist.hist(lags, bins=bins, color=CONFIG.color_model, edgecolor="white", linewidth=0.4)
    ax_hist.axvline(0, color="0.2", lw=1.2, ls="--")
    med = float(np.median(lags))
    pct_early = 100.0 * float((lags < 0).mean())
    pct_ontime = 100.0 * float((lags == 0).mean())
    pct_late = 100.0 * float((lags > 0).mean())
    ax_hist.axvline(med, color=CONFIG.color_fem, lw=1.4)
    ax_hist.text(0.02, 0.96, f"early {pct_early:.0f}%  ·  on-time {pct_ontime:.0f}%  ·  "
                 f"late {pct_late:.0f}%\nmedian lag {med:+.0f} steps  (n = {len(lags):,} nodes, "
                 f"{by_vessel.shape[0]} vessels)",
                 transform=ax_hist.transAxes, va="top", ha="left",
                 fontsize=CONFIG.font_size - 3)
    ax_hist.set_xlabel("onset lag  (timesteps; + = model late, - = model early)")
    ax_hist.set_ylabel("node count")
    ax_hist.set_title("(a) pooled onset-timing error", loc="left", fontsize=CONFIG.font_size)
    ax_hist.grid(axis="y", alpha=0.25)

    colors = [CONFIG.color_fem if v <= 0 else CONFIG.color_model for v in by_vessel["median_lag"]]
    y = np.arange(len(by_vessel))
    ax_bar.barh(y, by_vessel["median_lag"], color=colors)
    ax_bar.plot(by_vessel["median_lag"], y, "o", color="0.25", ms=2.5, zorder=5)
    ax_bar.axvline(0, color="0.2", lw=1.0)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(by_vessel["vessel"], fontsize=CONFIG.font_size - 4)
    ax_bar.set_xlabel("median onset lag (steps)")
    ax_bar.set_title("(b) per-vessel median (early = green, late = orange)",
                     loc="left", fontsize=CONFIG.font_size - 1)
    ax_bar.grid(axis="x", alpha=0.25)

    fig.suptitle("Onset timing: does the model commit a node early or late?",
                fontsize=CONFIG.font_size + 2, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = FIG_DIR / f"onset_timing.{CONFIG.fig_format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
