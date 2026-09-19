"""Plot the committed-SET operating-point curve (PUBLICATION_NOTES 7.8 item 6).

One panel per scoring domain, because wall and off-wall are different problems: the wall band
is ~18% positive and the off-wall interior ~0.3%, so a shared axis would flatten the off-wall
curve into the floor and a shared AUC-PR would be meaningless.

Each panel carries three things a reviewer asks for in this order:

  the curve          precision against recall over every threshold the OOF scores admit
  the floor          the base rate, drawn as a dashed line -- the precision a coin achieves.
                     AUC-PR is quoted as a multiple of it, because 0.60 is excellent at a
                     0.34% base rate and poor at 40%.
  the shipped cut    the precision/recall the deployed committed set actually attains,
                     marked ON the curve so "why this threshold" has a visual answer.

Reads `outputs/publication/data/operating_point.json`
(`generate_operating_point_data.py`).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from scripts.publication.config import CONFIG, DATA_DIR, FIG_DIR
from scripts.publication.utils import setup_matplotlib_style

#: Panel order and the human name for each scoring domain.
DOMAINS = (("wall", "Wall-attached"), ("off", "Off-wall"))


def _panel(ax, key: str, title: str, d: dict) -> None:
    base = d["base_rate"]
    ap = d["average_precision"]

    ax.plot(d["recall"], d["precision"], color=CONFIG.color_model, lw=1.6, zorder=3)
    ax.axhline(base, ls="--", lw=0.9, color="0.45", zorder=2)
    ax.text(0.985, base, f"  base rate {base:.4f}", color="0.35",
            fontsize=CONFIG.font_size - 3, va="bottom", ha="right", transform=ax.get_yaxis_transform())

    op = d.get("shipped_operating_point")
    if op:
        ax.plot([op["recall"]], [op["precision"]], marker="o", ms=7, mfc="white",
                mec=CONFIG.color_gt, mew=1.8, zorder=5)
        ax.annotate(f"shipped cut\nP {op['precision']:.3f} / R {op['recall']:.3f}",
                    xy=(op["recall"], op["precision"]),
                    xytext=(-8, -30), textcoords="offset points",
                    fontsize=CONFIG.font_size - 3, ha="right", color=CONFIG.color_gt,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=CONFIG.color_gt))

    spread = d.get("per_vessel_ap") or {}
    sub = f"AUC-PR {ap:.3f}  ({ap / base:.0f}x the {base:.4f} floor)"
    if spread.get("median") is not None:
        sub += f"\nper-vessel AUC-PR: median {spread['median']:.3f}, " \
               f"range {spread['min']:.3f}-{spread['max']:.3f}  (n={spread['n']})"
    ax.set_title(f"{title}\n{sub}", fontsize=CONFIG.font_size - 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)


def main() -> int:
    setup_matplotlib_style()
    src = DATA_DIR / "operating_point.json"
    if not src.is_file():
        print(f"[!] {src} missing; run generate_operating_point_data.py first")
        return 2
    payload = json.loads(src.read_text(encoding="utf-8"))
    present = [(k, t) for k, t in DOMAINS if k in payload.get("domains", {})]
    if not present:
        print("[!] no domains in payload")
        return 2

    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.0))
    if len(present) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, present):
        _panel(ax, key, title, payload["domains"][key])

    s = payload["source"]
    fig.suptitle(f"Committed-set operating point  ({s['n_vessels']} vessels, out-of-fold)",
                 fontsize=CONFIG.font_size)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dst = FIG_DIR / f"operating_point.{CONFIG.fig_format}"
    fig.savefig(dst, dpi=CONFIG.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
