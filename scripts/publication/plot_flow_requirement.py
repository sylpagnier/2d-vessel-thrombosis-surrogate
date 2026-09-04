"""Plot Figure 9: what a t=0 flow surrogate has to get right.

Two panels:
  (a) correlation of each candidate diagnostic with the measured wall-score drop, one bar per
      diagnostic, annotated with n.  The point of the panel is the CONTRAST: velocity rel-L2
      sits near zero while the gate statistics do not.
  (b) the GT->surrogate blend curve per vessel -- wall score against the velocity error actually
      present at that blend.  The point is the SHAPE: a cliff, not a slope.

Reads `outputs/publication/data/fig9_flow_requirement.json`
(`generate_fig9_data.py`).  Panels are drawn independently, so a partial dataset still yields
the panel it can support rather than nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.utils import setup_matplotlib_style  # noqa: E402


def _panel_a(ax, panel: dict) -> None:
    corrs = panel["correlations"]
    order = sorted(corrs.items(), key=lambda kv: kv[1]["r"])
    labels = [c["label"] for _k, c in order]
    rs = [c["r"] for _k, c in order]
    # Colour by whether the diagnostic is informative at all, not by sign: a near-zero bar is
    # the finding, so it should not read as merely "a small negative".
    colors = [CONFIG.color_model if abs(r) >= 0.25 else "#b0b0b0" for r in rs]

    y = np.arange(len(rs))
    ax.barh(y, rs, color=colors, height=0.62)
    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("correlation with wall-score drop")
    lim = max(0.75, max(abs(r) for r in rs) * 1.25)
    ax.set_xlim(-lim, lim)
    for yi, r in zip(y, rs):
        ax.text(r + (0.03 if r >= 0 else -0.03), yi, f"{r:+.3f}",
                va="center", ha="left" if r >= 0 else "right", fontsize=CONFIG.font_size - 2)
    n = panel.get("n_vessels")
    ax.set_title(f"(a) which statistic predicts failure  (n = {n} vessels)", loc="left")
    ax.grid(axis="x", alpha=0.25)


def _panel_b(ax, panel: dict) -> None:
    curves = panel["curves"]
    cmap = plt.get_cmap("tab10")
    for i, (stem, rows) in enumerate(sorted(curves.items())):
        x = [float(r.get("rel_at_alpha", r.get("alpha", 0))) for r in rows]
        y = [float(r.get("wall", np.nan)) for r in rows]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.3, color=cmap(i % 10), label=stem)
    ax.set_xlabel("velocity rel-L2 error present at this blend")
    ax.set_ylabel("wall clot score")
    ax.set_title("(b) the readout does not degrade — it falls off", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=CONFIG.font_size - 3)


def main() -> int:
    setup_matplotlib_style()
    src = DATA_DIR / "flow_requirement.json"
    if not src.is_file():
        print(f"[flow-req] missing {src}; run generate_fig9_data.py first")
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))

    # `panel_a` can be present-but-empty when the eval rows carried no diagnostic columns;
    # treat that as absent rather than letting the bar plot fail on an empty sequence.
    def _plottable(key: str) -> bool:
        p = payload.get(key)
        if not p:
            return False
        return bool(p.get("correlations")) if key == "panel_a" else bool(p.get("curves"))

    have = [k for k in ("panel_a", "panel_b") if _plottable(k)]
    if not have:
        print("[flow-req] neither panel has data yet:")
        for m in payload.get("missing", []):
            print("  - " + m)
        return 1

    fig, axes = plt.subplots(1, len(have), figsize=(6.6 * len(have), 3.6))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, have):
        (_panel_a if key == "panel_a" else _panel_b)(ax, payload[key])
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"flow_requirement.{CONFIG.fig_format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"[flow-req] wrote {out}" + ("" if len(have) == 2 else f"  (only {have[0]})"))
    for m in payload.get("missing", []):
        print("  - outstanding: " + m.splitlines()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
