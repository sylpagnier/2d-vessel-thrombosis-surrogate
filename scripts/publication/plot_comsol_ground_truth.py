"""Figure: what the surrogate is replacing -- the Cardillo-Barakat COMSOL model, and its two couplings.

`docs/publication/STORY.md` Leg 0 opens on "~48 h of coupled 12-species COMSOL". That is a cost
without a mechanism, and a reader cannot judge a surrogate without seeing what was surrogated.
This figure draws the ground-truth model: the species it carries, the two directions in which
flow and chemistry are coupled, and -- the detail the whole paper turns on -- the fact that the
flow->chemistry direction is a **threshold**, not a smooth rate.

  (left)   **Momentum.** Carreau-rheology Navier-Stokes. The clot re-enters here, and only here,
           through the `mu1(Mat)` viscosity step.
  (centre) **Chemistry.** Nine bulk species with `Reactions_9spec` plus inlet/wall/exit fluxes;
           three surface species (`M`, `Mas`, `Mat`) with `wall_surface_reactions_3spec`. `Mat`
           is the clot variable -- the label is `mu1(Mat)` crossing `viscosity_mat_crit`.
  (arrows) **The two couplings, drawn asymmetrically on purpose.**
           - flow -> chemistry is the deposition GATE
             `G_wall = [dsrx < sgt](L/gamma_m)|dsrx| + [sr < lss]` -- two Heaviside branches.
             The stagnation branch carries ~80% of deposition.
           - chemistry -> flow is a STEP in viscosity, 1 -> 80 at `Mat = 2e7`.
           Both directions are discontinuous. That is the property that makes a learned flow
           field expensive here (Leg 2.4) and it should be visible at a glance.
  (inset)  **The wound variant.** One substitution: on the injured patch `srf2` replaces `srf1`
           and the gate multiplier becomes a hard 1. Same bracket, same constants -- which is
           why Leg 0.2 is a boundary-condition branch and not a second model.

**Why we draw this rather than reproducing a figure from the paper.** Cardillo & Barakat (2025)
is open access, so reuse with attribution is likely permissible -- but their figure was drawn to
explain their model, and this one has to carry OUR argument: the two couplings, their
discontinuity, and the wound substitution, with the wall-clock cost on it. A redrawn schematic
also cannot go stale against a permissions question at submission time. **If the published
figure is reused instead, the licence must be checked and the attribution stated** -- see
EXPERIMENTS.md.

**Everything here is read off the model tree, not remembered.** Species lists, node names and the
`J0` expressions come from `comsol_models/phase2_*.mph` (`smodel.json` for structure,
`dmodel.xml` for the expressions); the physics is recorded in `docs/COMSOL_PHYSICS_VALIDATION.md`
and `docs/WOUND_PROGRESS.md` §1.

    python scripts/publication/plot_comsol_ground_truth.py
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.publication.config import CONFIG, FIG_DIR

BULK = ["rp", "ap", "apr", "aps", "at", "pt", "th", "fg", "fi"]
SURF = ["M", "Mas", "Mat"]

C_FLOW = "#1f77b4"
C_CHEM = "#2ca02c"
C_GATE = "#c23b22"
C_CLOT = "#ff7f0e"


def _box(ax, x, y, w, h, fc, ec, lw=1.6, alpha=1.0, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plt.style.use(CONFIG.style_name)
    plt.rcParams.update({"font.size": CONFIG.font_size})
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- momentum -------------------------------------------------------------------
    _box(ax, 0.03, 0.46, 0.26, 0.34, "#eaf3fa", C_FLOW)
    ax.text(0.16, 0.755, "MOMENTUM", ha="center", fontsize=9.5, weight="bold", color=C_FLOW)
    ax.text(0.16, 0.70, "Navier–Stokes, steady", ha="center", fontsize=8.5)
    ax.text(0.16, 0.645, "Carreau shear-thinning", ha="center", fontsize=8.5)
    ax.text(0.16, 0.575, r"$\mu_{\rm eff}(\dot\gamma)\;+\;\mu_1({\rm Mat})$", ha="center",
            fontsize=9.5)
    ax.text(0.16, 0.505, "no-slip wall · Re = 450", ha="center", fontsize=8, color="#555555")

    # ---- chemistry ------------------------------------------------------------------
    _box(ax, 0.40, 0.40, 0.33, 0.46, "#eaf7ec", C_CHEM)
    ax.text(0.565, 0.815, "CHEMISTRY", ha="center", fontsize=9.5, weight="bold", color=C_CHEM)
    ax.text(0.565, 0.762, "9 bulk species", ha="center", fontsize=8.5)
    ax.text(0.565, 0.715, "  ".join(BULK), ha="center", fontsize=8, family="monospace",
            color="#333333")
    ax.plot([0.425, 0.705], [0.688, 0.688], color=C_CHEM, lw=0.8, alpha=0.5)
    ax.text(0.565, 0.640, "3 surface species", ha="center",
            fontsize=7.6)
    ax.text(0.565, 0.593, "  ".join(SURF), ha="center", fontsize=8.5, family="monospace",
            color="#333333")
    _box(ax, 0.432, 0.425, 0.266, 0.075, "#fff1e2", C_CLOT, lw=1.4)
    ax.text(0.565, 0.462, r"clot label:  $\mu_1({\rm Mat})$ steps at ${\rm Mat}=2\!\times\!10^{7}$",
            ha="center", fontsize=7.8, color="#9a4c00")

    # ---- coupling: flow -> chemistry (the GATE) ---------------------------------------
    ax.add_patch(FancyArrowPatch((0.295, 0.70), (0.395, 0.70), arrowstyle="-|>",
                                 mutation_scale=16, lw=2.0, color=C_GATE, zorder=3))
    ax.text(0.345, 0.735, "GATE", ha="center", fontsize=8.5, weight="bold", color=C_GATE)
    _box(ax, 0.235, 0.885, 0.53, 0.10, "#fdeceb", C_GATE, lw=1.3)
    ax.text(0.50, 0.952, "flow → chemistry is a THRESHOLD, not a rate", ha="center",
            fontsize=9, weight="bold", color=C_GATE)
    ax.text(0.50, 0.906,
            r"$G_{\rm wall}=[\,\dot\gamma_x<{\rm sgt}\,]\cdot\frac{L}{\gamma_m}|\dot\gamma_x|"
            r"\;+\;[\,\dot\gamma<{\rm lss}\,]$",
            ha="center", fontsize=9.5, color=C_GATE)

    # ---- coupling: chemistry -> flow (the STEP) ---------------------------------------
    ax.add_patch(FancyArrowPatch((0.40, 0.462), (0.29, 0.545), arrowstyle="-|>",
                                 mutation_scale=16, lw=2.0, color=C_CLOT,
                                 connectionstyle="arc3,rad=0.25", zorder=3))
    ax.text(0.345, 0.395, "STEP  1 → 80", ha="center", fontsize=8.5, weight="bold",
            color=C_CLOT)
    ax.text(0.345, 0.355, "at gelation", ha="center", fontsize=7.5, color="#9a4c00")

    # ---- the wound substitution -------------------------------------------------------
    _box(ax, 0.775, 0.44, 0.205, 0.34, "#f6f2fa", "#7b52ab", lw=1.5)
    ax.text(0.8775, 0.735, "WOUND VARIANT", ha="center", fontsize=8.8, weight="bold",
            color="#7b52ab")
    ax.text(0.8775, 0.592, r"$srf1 \;\rightarrow\; srf2$", ha="center", fontsize=9.5)
    ax.text(0.8775, 0.535, r"gate $G_{\rm wall}\;\rightarrow\;1$", ha="center", fontsize=9.5,
            color="#7b52ab")
    ax.add_patch(FancyArrowPatch((0.735, 0.61), (0.770, 0.61), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color="#7b52ab", zorder=3))

    # ---- the cost ---------------------------------------------------------------------
    _box(ax, 0.03, 0.10, 0.60, 0.20, "#f4f4f4", "#888888", lw=1.2)
    ax.text(0.075, 0.243, "≈ 48 h", fontsize=20, weight="bold", color="#333333", va="center")
    ax.text(0.075, 0.163, "per vessel", fontsize=9, color="#555555", va="center")
    ax.plot([0.245, 0.245], [0.125, 0.275], color="#bbbbbb", lw=1.0)
    ax.text(0.265, 0.243, "coupled, transient, 12 species, quadratic mesh", fontsize=8.5,
            va="center", color="#333333")
    ax.text(0.265, 0.196, "Cardillo & Barakat, BMMB 2025",
            fontsize=8, va="center", color="#555555", style="italic")

    _box(ax, 0.66, 0.10, 0.32, 0.20, "#eaf3fa", C_FLOW, lw=1.4)
    ax.text(0.82, 0.243, "58.9 s", fontsize=20, weight="bold", color=C_FLOW, ha="center",
            va="center")
    ax.text(0.82, 0.170, "this surrogate  ·  2,934×", fontsize=9, color=C_FLOW, ha="center",
            va="center")


    stem = args.out or str(FIG_DIR / "comsol_ground_truth")
    for ext in (CONFIG.fig_format, "png"):
        fig.savefig(f"{stem}.{ext}", dpi=CONFIG.dpi, bbox_inches="tight")
    print(f"[i] wrote {stem}.{CONFIG.fig_format} / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
