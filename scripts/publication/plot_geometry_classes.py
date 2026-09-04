"""Figure 3: the cohort's geometry span, and how classes are measured.

One scatter in the (narrowing, bulge) plane -- both dimensionless, normalised by each vessel's
own median lumen width -- with the aneurysm cut drawn and the designated classes marked.

WHY THIS FIGURE EARNS ITS PLACE, and it is not the obvious reason.  It shows that geometry class
is *measured* rather than asserted, which the generalization claim needs.  But it also shows,
honestly, that **the measured stenosis cut does not separate**: the three designated stenoses sit
at narrowing 0.52 / 0.53 / 0.58 while `comsol012`, a baseline, reads 0.51 -- below all three.
`geometry_class.py` keeps the cut only because it would still catch a severe unlabelled case, and
warns that it must never be retuned (refitting a cut whose classes overlap is coin-flipping).
Drawing that failure costs one panel and buys the reader's trust in every other number.

Reads packs directly; no upstream data-generation step.
"""
from __future__ import annotations
from src.utils.paths import anchor_packs_dir

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR  # noqa: E402
from scripts.publication.utils import setup_matplotlib_style  # noqa: E402

PACKS = anchor_packs_dir()
CLASS_STYLE = {
    "aneurysm": ("#c44e52", "o"),
    "stenosis": ("#4c72b0", "s"),
    "stenosis+aneurysm": ("#8172b3", "D"),
    "baseline": ("#b0b0b0", "."),
    "unknown": ("#dddddd", "x"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-data", default=str(DATA_DIR / "geometry_classes.json"))
    a = ap.parse_args()

    setup_matplotlib_style()
    from src.clot_ml.geometry_class import (
        BULGE_ANEURYSM, NARROWING_STENOSIS, USER_DESIGNATED, classify, width_stats,
    )

    rows = []
    for p in sorted(PACKS.glob("*.pt")):
        stem = p.stem
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            s = width_stats(d)
            rows.append({
                "stem": stem,
                "bulge": float(s.get("bulge", float("nan"))),
                "narrowing": float(s.get("narrowing", float("nan"))),
                "cls": str(classify(s, anchor=stem)),
                "designated": USER_DESIGNATED.get(stem),
            })
        except Exception as exc:
            print(f"  [skip] {stem}: {type(exc).__name__}: {exc}")

    rows = [r for r in rows if np.isfinite(r["bulge"]) and np.isfinite(r["narrowing"])]
    if not rows:
        print("[geometry] no packs with usable width statistics")
        return 1

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for cls, (color, marker) in CLASS_STYLE.items():
        pts = [r for r in rows if r["cls"] == cls]
        if not pts:
            continue
        ax.scatter([r["narrowing"] for r in pts], [r["bulge"] for r in pts],
                   c=color, marker=marker, s=46 if marker != "." else 26,
                   label=f"{cls} (n={len(pts)})", edgecolors="none", zorder=3)

    ax.axhline(BULGE_ANEURYSM, color="#c44e52", ls="--", lw=1.0, zorder=2)
    ax.text(ax.get_xlim()[1], BULGE_ANEURYSM, f" bulge = {BULGE_ANEURYSM}  (separates)",
            va="bottom", ha="right", fontsize=CONFIG.font_size - 3, color="#c44e52")
    ax.axvline(NARROWING_STENOSIS, color="#4c72b0", ls=":", lw=1.0, zorder=2)
    ax.text(NARROWING_STENOSIS, ax.get_ylim()[1],
            f"narrowing = {NARROWING_STENOSIS}\n(does NOT separate)",
            va="top", ha="left", fontsize=CONFIG.font_size - 3, color="#4c72b0")

    ax.set_xlabel("narrowing   (p2 / median lumen width)")
    ax.set_ylabel("bulge   (p98 / median lumen width)")
    ax.set_title("Measured geometry classes", loc="left")
    ax.legend(frameon=False, fontsize=CONFIG.font_size - 3, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"geometry_classes.{CONFIG.fig_format}"
    fig.savefig(out)
    plt.close(fig)

    Path(a.out_data).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_data).write_text(json.dumps(
        {"cuts": {"bulge_aneurysm": BULGE_ANEURYSM,
                  "narrowing_stenosis": NARROWING_STENOSIS},
         "vessels": rows}, indent=2), encoding="utf-8")

    print(f"[geometry] wrote {out}\n       {a.out_data}  ({len(rows)} vessels)")
    des = [r for r in rows if r["designated"] == "stenosis"]
    if des:
        worst = max(r["narrowing"] for r in des)
        below = [r for r in rows if r["designated"] is None and r["narrowing"] < worst]
        if below:
            print(f"  note: {len(below)} undesignated vessel(s) sit below the most-open "
                  f"designated stenosis ({worst:.2f}) -- the cut does not separate, as documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
