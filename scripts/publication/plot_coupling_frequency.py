"""Figure: how much clot->flow coupling is needed -- score against FEM re-solves per vessel.

The question after STORY 2.1i: the real FEM re-solve at stride 16 gave wall +0.0057 (null) and
off-wall +0.0517 (consistent sign, inside the across-seed null).  Is that gain bought by the
number of re-solves, or does one well-placed solve capture it?  So the arms sweep coupling
frequency, all GT-free, all on the same 36 vessels / 3 folds / arm A4 / 3 seed pairs:

    uncoupled   0 solves, the reference
    femfinal    1 solve at the end: coupled FLOW FEATURES only, the ODE never re-gated
    fem@1+N     re-solve once N new clot nodes have committed since the last solve
    fem@K       re-solve at most every K steps (once anything new has committed)
    fem@1       re-solve on every new clot == every step (an unchanged clot gives an
                unchanged flow, so the two are the same arm)

**The drawing decision.** x is the MEASURED mean number of FEM re-solves per vessel, read from
each cache (`n_flow_solves`), because the arms' nominal knobs (strides vs clot counts) are not
on one scale and cost is what the manuscript has to trade against.  y is each HEALTHY run's
cohort mean minus the mean of the HEALTHY uncoupled runs (grey dots at x=0), with the arm mean
+/- 2 SE and a +/- 2 SD band of pooled run-to-run noise.

**Two rows, two metrics.** Row 1 is BATC, the manuscript's score. Row 2 is strict F1 (node-exact,
no tolerance) of the SAME final-time predictions, recorded by `eval_strict` beside BATC, with the
same healthy-run screen and its own pooled run noise. Coupling is judged null only if neither metric
moves beyond its noise: BATC forgives near misses, strict F1 does not, so an effect hiding in
boundary placement would show in row 2 first.

**NOT paired by seed**, and this was a retraction (2026-09-13): the first version plotted
seed-paired deltas, and every arm's pair 3 was measured against the same degenerate uncoupled
run (`cpl_open_s3`), which manufactured a consistent +0.11 off-wall "gain" in every schedule.
Degenerate runs are now screened by one cut-signature rule applied to both sides
(`diag_coupling_arms.degenerate`).

Input: `outputs/deployclot/cpl_<stem>{,_s2,_s3}.json` (eval_strict) and
`outputs/clot_ml_cache_fem_<stem>/*.npz` (solve counts).

    python scripts/publication/plot_coupling_frequency.py
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.diag_coupling_arms import compare
from src.utils.paths import get_project_root

REPO = get_project_root()

#: (tag stem, label, CLOT_ML_ORACLE_BLOCKAGE) in rough order of coupling effort
ARMS = [
    ("femfinal", "final solve only", "femfinal"),
    ("fem64", "every 64 steps", "fem@64"),
    ("femn20", "every 20 new clot nodes", "fem@1+20"),
    ("femcpl", "every 16 steps", "fem@16"),
    ("femn5", "every 5 new clot nodes", "fem@1+5"),
    ("fem1", "every new clot node", "fem@1"),
]
DOMAINS = [("wall", "wall"), ("off", "off-wall")]
METRICS = [("batc", "BATC"), ("strict_f1", "strict F1")]


def _solves(stem: str) -> tuple[float, float, int]:
    """Mean re-solves and mean build seconds per vessel from the arm's v3 cache.

    `femcpl` was trained before caches recorded their solve count, so its count comes from a
    rebuild of the identical arm into `clot_ml_cache_fem_femcpl_count` (the trained cache is
    left untouched).
    """
    base = REPO / "outputs" / f"clot_ml_cache_fem_{stem}"
    count = base.with_name(base.name + "_count")
    files = sorted((count if count.is_dir() else base).glob("*.npz"))
    s, t = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "n_flow_solves" in z.files:
            s.append(float(z["n_flow_solves"]))
            t.append(float(z["build_seconds"]) if "build_seconds" in z.files else np.nan)
    return (float(np.mean(s)) if s else float("nan"),
            float(np.nanmean(t)) if t and np.isfinite(t).any() else float("nan"), len(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for stem, label, env in ARMS:
        try:
            cmp = compare(stem)
            cmp_f1 = compare(stem, metric="strict_f1")
        except SystemExit:
            print(f"[skip] {stem}: no scored seed pairs yet")
            continue
        solves, secs, n = _solves(stem)
        if not np.isfinite(solves):
            print(f"[skip] {stem}: cache carries no n_flow_solves")
            continue
        rows.append(dict(key="ABCDEFGH"[len(rows)], stem=stem, label=label, env=env,
                         solves=round(solves, 2),
                         build_seconds=None if secs != secs else round(secs, 1),
                         n_cache=n, result=cmp["result"], result_strict_f1=cmp_f1["result"]))
    if not rows:
        raise SystemExit("[ERR] no arm is ready to plot")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "coupling_frequency.json").write_text(json.dumps(rows, indent=2),
                                                      encoding="utf-8")

    plt.style.use(CONFIG.style_name)
    plt.rcParams.update({"font.size": CONFIG.font_size})
    rows.sort(key=lambda r: r["solves"])
    for i, r in enumerate(rows):
        r["key"] = "ABCDEFGH"[i]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), sharex=True)
    xmax = max(r["solves"] for r in rows)
    from scripts.publication.diag_coupling_arms import METRIC_KEYS, SUFFIXES, _mean, _rows, degenerate

    letters = iter("abcd")
    for mi, (metric, mlabel) in enumerate(METRICS):
        result_key = "result" if metric == "batc" else "result_strict_f1"
        for ax, (dom, dlabel) in zip(axes[mi], DOMAINS):
            key = METRIC_KEYS[metric][dom]
            open_runs = [_mean(_rows(f"cpl_open{s}"), key) for s in SUFFIXES
                         if not degenerate(f"cpl_open{s}")]
            base = float(np.mean(open_runs))
            sd = rows[0][result_key][dom]["run_noise_sd"]
            ax.axhspan(-2 * sd, 2 * sd, color="#dfe6ec", zorder=0)
            ax.axhline(0, color="#333333", lw=1.0, zorder=1)
            ax.scatter([0] * len(open_runs), [m - base for m in open_runs], s=22,
                       color="#777777", zorder=4)
            # No line through the means: the arms are different schedules, not points on one
            # curve, and a connecting line reads as a trend the data do not show.
            for r in rows:
                R = r[result_key][dom]
                x = r["solves"]
                pts = [m - base for m in R["arm_healthy_run_means"]]
                ax.scatter([x] * len(pts), pts, s=16, color=CONFIG.color_gt, alpha=0.55,
                           zorder=3, edgecolor="none")
                m = R["delta_healthy"]
                ax.errorbar([x], [m], yerr=[2 * R["se"]], fmt="none", ecolor=CONFIG.color_model,
                            elinewidth=1.0, capsize=3, alpha=0.8, zorder=4)
                ax.plot([x], [m], marker="_", ms=18, mew=2.6, color=CONFIG.color_model, zorder=5)
                ax.annotate(r["key"], (x, m), textcoords="offset points", xytext=(7, -3),
                            fontsize=8, fontweight="bold", color=CONFIG.color_model)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(-0.3, xmax * 2.2)
            if mi == len(METRICS) - 1:
                ax.set_xlabel("FEM re-solves per vessel")
            ax.set_title(f"({next(letters)}) {dlabel}, {mlabel}", loc="left", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
        axes[mi][0].set_ylabel(f"coupled $-$ uncoupled ({mlabel})")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#777777", ms=5,
                   label="uncoupled run"),
        plt.Line2D([], [], marker="o", ls="", color=CONFIG.color_gt, alpha=0.55, ms=5,
                   label="coupled run"),
        plt.Line2D([], [], marker="_", ls="", color=CONFIG.color_model, ms=12, mew=2.4,
                   label="arm mean $\\pm$2 SE"),
        plt.Rectangle((0, 0), 1, 1, fc="#dfe6ec", ec="none", label="$\\pm$2 SD run noise"),
    ]
    axes[0][1].legend(handles=handles, fontsize=7.2, frameon=False, loc="upper right",
                   bbox_to_anchor=(1.0, 1.14), ncol=4)
    fig.text(0.01, 0.02, "   ".join(f"{r['key']} {r['label']}" for r in rows), fontsize=8,
             color="#333333")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    stem = args.out or str(FIG_DIR / "coupling_frequency")
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{stem}.{ext}", dpi=CONFIG.dpi, bbox_inches="tight")
    print(f"[i] wrote {stem}.{CONFIG.fig_format} / .png")
    for r in rows:
        for key, tag in (("result", "BATC"), ("result_strict_f1", "F1  ")):
            w, o = r[key]["wall"], r[key]["off"]
            print(f"  {r['stem']:<9} {tag} solves {r['solves']:6.2f}  wall {w['delta_healthy']:+.4f}±"
                  f"{2 * w['se']:.4f}  off {o['delta_healthy']:+.4f}±{2 * o['se']:.4f}  "
                  f"build {r['build_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
