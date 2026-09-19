"""Shared utilities for publication figure generation."""
from src.utils.paths import anchor_packs_dir
from pathlib import Path

import matplotlib.pyplot as plt


from scripts.publication.config import CONFIG

def setup_matplotlib_style():
    """Apply global publication styles to matplotlib."""
    try:
        plt.style.use(CONFIG.style_name)
    except OSError:
        # Fallback if style doesn't exist
        pass
    plt.rcParams.update({
        'font.size': CONFIG.font_size,
        'axes.labelsize': CONFIG.font_size,
        'axes.titlesize': CONFIG.font_size,
        'xtick.labelsize': CONFIG.font_size - 2,
        'ytick.labelsize': CONFIG.font_size - 2,
        'legend.fontsize': CONFIG.font_size - 2,
        'figure.dpi': CONFIG.dpi,
        'savefig.dpi': CONFIG.dpi,
        'savefig.format': CONFIG.fig_format,
        'savefig.bbox': 'tight',
    })

def get_pack_path(stem: str) -> Path:
    """Resolve path to a biochem anchor pack."""
    path = anchor_packs_dir() / f"{stem}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Pack not found: {path}")
    return path


#: The two published settings of the clot score, keyed by the tuple the figure path resolves:
#: ``(relax_hops, f_beta, iou_weight)``.  `severity_metric.BATC` / `BATC_0` name the same two
#: configurations for the manuscript; these are the corresponding knobs on the `guiding` path
#: (`evaluation.clot_relaxed_metrics`) that every publication script actually calls.
_PUBLISHED_SETTINGS = {
    (2, 0.5, 0.5): ("BATC_0", "Burden-Adjusted Thrombus Concordance, unadjusted"),
    (4, 1.0, 0.2): ("BATC", "Burden-Adjusted Thrombus Concordance"),
}


def metric_identity() -> dict:
    """Name the clot metric the figures are ACTUALLY computing, from its resolved settings.

    Every publication script scores through `clot_relaxed_metrics` with whatever
    `clot_guide_*` resolves to, and no script overrides them.  Those defaults are
    ``relax_hops=2, f_beta=0.5, iou_w=0.5`` -- which is BATC_0, the unadjusted variant, not
    BATC (``4, 1.0, 0.2`` plus the burden-scaled graces).  A draft that labels these BATC
    breaks the project's own rule against mixing the two scores
    (`PUBLICATION_NOTES.md` standing rule 1), and nothing in the pipeline said which one it
    was, so the label had to be remembered rather than read.

    Derived, never asserted: if the knobs move, the name moves with them and an unrecognised
    combination reports itself as `custom` rather than silently keeping a published name.
    """
    from src.evaluation.clot_relaxed_metrics import (
        clot_guide_f_beta, clot_guide_iou_weight, clot_guide_relax_hops,
    )

    hops = int(clot_guide_relax_hops())
    beta = float(clot_guide_f_beta())
    iou_w = float(clot_guide_iou_weight())
    name, long_name = _PUBLISHED_SETTINGS.get(
        (hops, beta, iou_w), ("custom", "unrecognised settings -- not a published metric"))
    return {
        "metric": name,
        "long_name": long_name,
        "settings": {"relax_hops": hops, "f_beta": beta, "iou_weight": iou_w},
        "note": ("Derived from the settings the scoring path resolved, not asserted. BATC_0 "
                 "is the unadjusted variant (2-hop tolerance, F0.5, 0.5 shape weight, no "
                 "graces); BATC is the burden-adjusted one (4-hop, F1, 0.2, with graces). "
                 "Quote the name this field gives, and never mix the two in one table."),
    }


def metric_label() -> str:
    """Short axis/caption label, e.g. ``BATC_0 (2-hop, F0.5)``."""
    mid = metric_identity()
    s = mid["settings"]
    return f"{mid['metric']} ({s['relax_hops']}-hop, F{s['f_beta']:g})"
