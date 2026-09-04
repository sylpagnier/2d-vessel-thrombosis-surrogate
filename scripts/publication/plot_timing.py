"""Table 3 / cost figure: where the deploy wall-clock goes, against the COMSOL reference.

Two panels:
  (a) stacked per-vessel cost (FEM t=0 / feature build / rollout), sorted by total, so the
      reader sees both the median and the spread rather than one headline number.
  (b) the log-scale comparison against a COMSOL solve.

Reads `outputs/publication/data/timing.json` (`generate_timing_data.py`).

BOUNDARY, repeated here because it is easy to lose between script and caption: these numbers
cover pack -> FEM t=0 -> features -> rollout.  Geometry construction and meshing are upstream
and excluded; the COMSOL reference covers geometry -> mesh -> solve.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np


from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.utils import setup_matplotlib_style  # noqa: E402

STAGES = (("fem_s", "FEM t=0", "#4c72b0"),
          ("sample_s", "features", "#dd8452"),
          ("rollout_s", "rollout", "#55a868"))


def main() -> int:
    setup_matplotlib_style()
    src = DATA_DIR / "timing.json"
    if not src.is_file():
        print(f"[timing] missing {src}; run generate_timing_data.py first")
        return 1
    p = json.loads(src.read_text(encoding="utf-8"))
    rows = sorted(p["per_vessel"], key=lambda r: r["deploy_s"])
    if not rows:
        print("[timing] no vessels timed")
        return 1

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                   gridspec_kw={"width_ratios": [2.3, 1]})

    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, label, color in STAGES:
        vals = np.array([r.get(key, 0.0) for r in rows], float)
        ax0.bar(x, vals, bottom=bottom, label=label, color=color, width=0.8)
        bottom += vals
    med = p["summary"]["deploy_s"]["median"]
    ax0.axhline(med, color="0.25", ls="--", lw=1.0)
    ax0.text(len(rows) - 0.5, med, f"  median {med:.0f} s", va="bottom", ha="right",
             fontsize=CONFIG.font_size - 2)

    # Median time burden per stage, as a % of that vessel's own total — not the
    # % of the median total, so one slow/fast vessel can't skew the split.
    pct_by_stage = {}
    for key, label, _color in STAGES:
        pct = np.array([100.0 * r.get(key, 0.0) / r["deploy_s"] for r in rows
                        if r.get("deploy_s")], float)
        pct_by_stage[label] = float(np.median(pct)) if pct.size else 0.0
    burden_line = "median share:  " + "  ·  ".join(
        f"{label} {pct_by_stage[label]:.0f}%" for _key, label, _color in STAGES)
    ax0.set_xticks(x)
    ax0.set_xticklabels([r["stem"].replace("comsol", "p").replace("wound_p", "w")
                         for r in rows], rotation=90, fontsize=CONFIG.font_size - 4)
    ax0.set_ylabel("wall-clock (s)")
    ax0.set_title(f"(a) deploy cost per vessel  (n = {len(rows)})\n{burden_line}",
                  loc="left", fontsize=CONFIG.font_size)
    ax0.legend(frameon=False, fontsize=CONFIG.font_size - 2)
    ax0.grid(axis="y", alpha=0.25)

    comsol_s = float(p.get("comsol_reference_hours", 48.0)) * 3600.0
    ax1.bar([0, 1], [comsol_s, med], color=["#c44e52", CONFIG.color_model], width=0.55)
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"COMSOL\n({p.get('comsol_reference_hours', 48):.0f} h)",
                         f"surrogate\n({med / 60:.1f} min)"])
    ax1.set_ylabel("wall-clock (s, log)")
    speedup = p.get("speedup_vs_comsol_median") or (comsol_s / med)
    ax1.set_title(f"(b) {speedup:,.0f}$\\times$ faster", loc="left")
    ax1.grid(axis="y", alpha=0.25, which="both")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"timing_cost.{CONFIG.fig_format}"
    fig.savefig(out)
    plt.close(fig)

    dev = p.get("env", {}).get("device", "?")
    print(f"[timing] wrote {out}")
    print(f"  median {med:.1f} s ({med / 60:.2f} min) on {dev};  speedup {speedup:,.0f}x")
    print("  boundary: meshing/geometry excluded on the surrogate side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
