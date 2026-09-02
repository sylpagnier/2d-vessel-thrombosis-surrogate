"""Centralized configuration for publication figures."""
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "outputs" / "publication" / "data"
FIG_DIR = REPO_ROOT / "outputs" / "publication" / "figures"
RESEARCH_SWEEP_DATA_DIR = DATA_DIR / "research_sweeps"
RESEARCH_SWEEP_FIG_DIR = FIG_DIR / "research_sweeps"

@dataclass
class PubConfig:
    # Model versions
    rgp_deq_model: str = "rgp_deq_kine"
    clot_ml_model: str = "clot_ml_0"

    # Strict nested-CV OOF (eval_strict_temporal.py --save-oof-series)
    oof_series_path: str = "outputs/publication/data/clot_ml_0_oof_series.npz"
    oof_arms: tuple = ("v5a,v5b,v5c",)
    oof_cache: str = "gt"
    oof_head_seeds: int = 4
    oof_set_masks: str = "outputs/v4_set_masks.npz"

    # Cohorts (all must appear in the OOF archive)
    fig1_vessels: tuple = ("patient020", "patient005")
    fig3_vessels: tuple = ("patient020", "patient005", "patient012", "patient041")
    fig4_vessels: tuple = ("patient020", "patient005")
    fig6_vessels: tuple = ("patient005", "patient014")

    # Research sweeps (outputs from scripts/run_research_sweep.py)
    research_sweep_root: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "research_sweeps")
    # Re is fixed at 450 across the whole project, so the four Reynolds-varying sweeps
    # (03_inlet_re, 07_stenosis_x_re, 11_aneurysm_x_re, 13_width_x_re) were deleted on
    # 2026-09-01: with Re pinned the three interaction sweeps collapse onto their first axis
    # and duplicate 01 / 02 / 04.
    research_geometry_sweeps: tuple = (
        "01_stenosis_strength",
        "02_aneurysm_strength",
        "04_inlet_width",
        "05_bendiness",
        "06_stenosis_location",
        "08_vessel_length",
        "09_stenosis_eccentricity",
        "10_pathology_length",
        "12_bend_x_stenosis",
        "14_wall_roughness",
    )
    research_wound_sweeps: tuple = (
        "16_wound_width",
        "17_wound_position",
        "18_wound_x_stenosis",
        "19_wound_vs_no_wound",
        "20_wound_stenosis_offset",
    )

    # --- Paper section mapping -------------------------------------------------------
    # The generator's script names are SEMANTIC and deliberately do not encode paper figure
    # numbers -- review reorders figures, and renaming scripts each time churns the pipeline.
    # This dict is the single place the two are tied together; update it, not the filenames.
    # Outline: docs/PUBLICATION_PLAN.md s8.
    paper_map: dict = field(default_factory=lambda: {
        "geometry_classes":  ("3", "Cohort and geometry classes"),
        "timing_cost":          ("4", "The tool -- cost"),
        "table4_kfold":           ("5", "Geometry generalization (primary evidence)"),
        "fig34_biochem_final":    ("5", "Final-time clot maps"),
        "fig34_biochem_temporal": ("5", "Temporal evolution"),
        "research_sweeps":        ("5", "Geometry-response sweeps"),
        "fig1_flow":              ("6", "Flow fields -- RGP-DEQ / FEM / GT (NOT the opener)"),
        "flow_requirement":  ("7", "What the flow surrogate must get right"),
        "fig6_failures":          ("8", "Known failure modes"),
        # Added 2026-09-02, team figure-board review -- both supplement-first (budget note,
        # PUBLICATION_PLAN.md s8). wound_ab is deliberately NOT mapped here: it is a preview,
        # not a budgeted figure, until PUBLICATION_NOTES s7.0 unblocks the wound section.
        "onset_timing":           ("8", "Onset timing -- early or late (Fig 12, new)"),
        "error_trajectories":     ("8", "Does an error compound or recover (Fig 13, new)"),
    })

    # Research sweeps: which go in the main geometry-response figure, which to the supplement.
    # Ten sweeps x four metrics is ~40 panels; the main figure takes the two pathologies plus
    # two shape axes and the rest is supplementary.
    main_sweeps: tuple = (
        "01_stenosis_strength",
        "02_aneurysm_strength",
        "05_bendiness",
        "04_inlet_width",
    )

    # Evaluation metrics
    flow_metrics: tuple = ("u", "v", "P", "shear", "dshear")
    clot_metrics: tuple = ("wall", "off", "w_reg", "w_lum", "far")

    # Plotting constants
    dpi: int = 300
    fig_format: str = "pdf" # 'pdf' or 'svg' or 'png'
    
    # Common ML Paper styling
    style_name: str = "seaborn-v0_8-paper"
    font_size: int = 10
    color_gt: str = "#1f77b4" # Blue
    color_model: str = "#ff7f0e" # Orange
    color_fem: str = "#2ca02c" # Green

CONFIG = PubConfig()

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_SWEEP_FIG_DIR.mkdir(parents=True, exist_ok=True)
