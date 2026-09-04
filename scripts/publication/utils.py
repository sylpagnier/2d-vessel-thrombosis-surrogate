"""Shared utilities for publication figure generation."""
import sys
from src.utils.paths import anchor_packs_dir
from pathlib import Path

import matplotlib.pyplot as plt

# Repo root by marker, not by depth: this file may move between
# scripts/ and scripts/<subdir>/ without silently resolving one level off.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
