"""Figure 0.4: what the tool is FOR -- dense pathology-strength sweeps no COMSOL budget allows.

Sweep one pathology axis finely on one vessel and read off how the predicted clot responds.

**Strength** is the pathology's size where it is largest: the percentage REDUCTION (stenosis) or
INCREASE (aneurysm) of the vessel width at its centre, relative to the inlet width. It is measured
back from each point's own generator geometry (`research_sweep_geometry.centre_width_change`) and
plotted as measured, not as the requested knob -- the two agree to within 0.5 percentage points.
The earlier version of this figure plotted "aneurysm factor", a wall-offset multiplier where 1.0
meant the width TRIPLED, on an axis a reader would take as a fraction.

Vessel: straight, 12 mm, Re 450, 8.3 h, pathology centred with the trained taper (stenosis
half-depth width 0.12 of length, aneurysm 0.20, as in the COMSOL cohort). Data:
`configs/research_sweeps/01_stenosis_strength.json` (38 points: 2.5% steps, 1.25% midpoints at the
transitions) and `02_aneurysm_strength.json` (42 points: 10% steps, 2.5% across 40-100%, 0.625%
across the 45-47.5% onset), run with `scripts/run_research_sweep.py`.

One row: total clot mass (clotted nodes, % of the vessel -- the app's `vessel_clot_pct`).

> **CAVEATS FOR THE CAPTION.** There is no COMSOL ground truth at any swept point: these are the
> surrogate's own predictions, and the evidence they can be trusted is the held-out cohort, not this
> figure. The shaded region lies beyond the most severe pathology in the training cohort (77%
> narrowing, comsol041/042/044; 99% widening, comsol040/047) and is extrapolation. Points are
> plotted, never a fitted curve; the faint line only guides the eye between neighbours.

    python scripts/publication/plot_applications.py
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.research_sweep_utils import load_sweep_summary
from src.data_gen.lib.customer_geometry_import import (
    TRAINED_ANEURYSM_WIDENING,
    TRAINED_STENOSIS_NARROWING,
)
from src.evaluation.research_sweep_config import load_sweep_config, resolve_sweep_path
from src.evaluation.research_sweep_geometry import arm_geometry_cache_spec, centre_width_change

#: (column, ylabel, colour, marker).  `vessel_clot_pct` is the app's own clot-mass measure:
#: clotted nodes as a percentage of the vessel (src/evaluation/research_parameters.py).
#: Peak occlusion was dropped on review (2026-09-16): it is the depth of the single deepest clot node
#: in whole wall hops, so one node 3 layers in at an aneurysm apex read as a 3 pp "occlusion" spike.
#: Wall clot was dropped with it; total clot mass is the one response the figure carries.
METRICS = [
    ("vessel_clot_pct_final", "total clot mass\n(% of vessel)", "#7b3294", "D"),
]
SWEEPS = [
    ("01_stenosis_strength", "stenosis: width reduction at centre (%)", "(a) stenosis",
     TRAINED_STENOSIS_NARROWING, -1.0),
    ("02_aneurysm_strength", "aneurysm: width increase at centre (%)", "(b) aneurysm",
     TRAINED_ANEURYSM_WIDENING, +1.0),
]


def sweep_table(sweep_id: str, sign: float) -> list[dict]:
    """One row per arm: nominal and MEASURED strength (%) plus the plotted metrics."""
    cfg = load_sweep_config(resolve_sweep_path(sweep_id))
    by_name = {a["name"]: a for a in cfg["arms"]}
    summary = load_sweep_summary(sweep_id)
    out = []
    for arm in summary["arms"]:
        spec_arm = by_name[arm["name"]]
        nominal = 100.0 * float(arm["axis_value"])
        spec = arm_geometry_cache_spec(spec_arm, cfg["control"])
        measured = 0.0 if nominal == 0 else 100.0 * sign * centre_width_change(spec)
        row = dict(name=arm["name"], strength_nominal_pct=round(nominal, 3),
                   strength_measured_pct=round(measured, 3))
        row.update({col: arm.get(col) for col, *_ in METRICS})
        out.append(row)
    return sorted(out, key=lambda r: r["strength_measured_pct"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tables = {sid: sweep_table(sid, sign) for sid, _, _, _, sign in SWEEPS}
    for sid, rows in tables.items():
        worst = max(abs(r["strength_measured_pct"] - r["strength_nominal_pct"]) for r in rows)
        if worst > 0.5:
            raise SystemExit(f"[ERR] {sid}: measured strength is {worst:.2f} pp off nominal")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "applications.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")

    plt.style.use(CONFIG.style_name)
    plt.rcParams.update({"font.size": CONFIG.font_size})
    fig = plt.figure(figsize=(9.4, 2.9))
    gs = fig.add_gridspec(len(METRICS), 2, hspace=0.14, wspace=0.22)
    for c, (sid, xlabel, title, trained_max, _sign) in enumerate(SWEEPS):
        rows = tables[sid]
        x = np.array([r["strength_measured_pct"] for r in rows])
        first = None
        for r_i, (col, ylabel, colour, marker) in enumerate(METRICS):
            ax = fig.add_subplot(gs[r_i, c], sharex=first)
            first = first or ax
            y = np.array([np.nan if r[col] is None else float(r[col]) for r in rows])
            ax.axvspan(100 * trained_max, x.max() * 1.03, color="#9aa3a6", alpha=0.16, lw=0,
                       zorder=0, label="beyond trained range" if r_i == 0 else None)
            ax.plot(x, y, "-", color=colour, lw=0.9, alpha=0.35, zorder=2)
            ax.plot(x, y, marker, color=colour, ms=4.6, zorder=3)
            if c == 0:
                ax.set_ylabel(ylabel, color=colour, fontsize=8.4)
            top = np.nanmax(y) if np.isfinite(y).any() else 1.0
            ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
            ax.set_xlim(-x.max() * 0.02, x.max() * 1.03)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7.5)
            if r_i == 0:
                ax.set_title(title, fontsize=10, loc="left")
                ax.legend(fontsize=7, frameon=False, loc="upper left")
            if r_i < len(METRICS) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(xlabel)

    stem = args.out or str(FIG_DIR / "applications")
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{stem}.{ext}", dpi=CONFIG.dpi, bbox_inches="tight")
    print(f"[i] wrote {stem}.{CONFIG.fig_format} / .png and {DATA_DIR / 'applications.json'}")
    for sid, rows in tables.items():
        print(f"  {sid}: {len(rows)} points, strength {rows[0]['strength_measured_pct']:.1f}-"
              f"{rows[-1]['strength_measured_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
