"""Centralized configuration for publication figures.

**Profiles.**  A figure is only meaningful next to the generation of models it was built from.
`--profile` selects which one the whole pipeline reads, and each profile writes to its OWN
output root, so regenerating one never overwrites another.

    python scripts/publication/generate_fig3_4_data.py                     # shipped, default

Selecting a profile does NOT move the `clot_ml_0` artifact pointer -- it only tells the figure
pipeline which arms and which cache to read.  Repointing is a separate, deliberate act.

**"split" retired 2026-09-07.** The decoupled wall/off-wall family (`RGP_DEQ_REPAIR_PLAN.md`
s18.15) was a parallel path against `DeployClotS_0`; it did not pan out and the checkpoint was
moved to `outputs/clot_ml/locked_retired/` so `clot_ml_0` (-> `clot_ml_final_0`, renamed from
`DeployClot2_0` the same day) is unambiguously the one model. `outputs/publication_split/` and
`outputs/research_sweeps_split_DeployClotS_0/` are left on disk as a historical record but
nothing regenerates them; `--profile split` now raises rather than silently building against a
checkpoint that no longer lives where this file says it does.

`--profile` is consumed HERE, at import, and removed from ``sys.argv`` before any caller's
own ``argparse`` runs -- every script in this package imports this module, so one parse
serves all twenty of them and none of them needs an extra flag definition.  It was an
environment variable (``PUB_PROFILE``) until `src/tests/test_env_registry.py` refused it:
that guard freezes the configuration surface so it can only shrink, and a knob is not exempt
from it for being the guard author's own.  A flag is also the more honest form -- it appears
in the command the log records, where an inherited environment variable does not.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.paths import get_project_root

REPO_ROOT = get_project_root()

#: Model generations the figures can be built against.  Each names the CV arms whose
#: out-of-fold scores the temporal figures replay, the feature cache those arms were trained
#: on, and the artifact the rollout figures load.  The three must agree: an OOF archive built
#: from one cache and labelled with another model's name is the failure
#: `eval_strict_temporal.py` guards against.
PROFILES: dict[str, dict] = {
    # The arms and cache here MUST match what the archive on disk was actually built from --
    # `ensure_oof_series` only regenerates when the file is missing, so a wrong entry is
    # invisible until someone deletes it and silently gets a different model's figures.  This
    # entry said `v5a,v5b,v5c` on the `gt` cache while `clot_ml_0_oof_series.npz` records
    # `{"arms": ["dc_fem_cfw025"], "cache": "v5_fem", "flow": "fem"}`, i.e. the GT-flow arms
    # against the deployed FEM ones.  Figures 3, 4 and 6 all read this archive.
    "shipped": dict(
        subdir="publication",
        clot_ml_model="clot_ml_0",
        oof_arms=("dc_fem_cfw025",),
        oof_cache="v5_fem",
        oof_series_name="clot_ml_0_oof_series.npz",
    ),
}

def _take_profile_arg(argv: list[str]) -> str:
    """Pop ``--profile NAME`` / ``--profile=NAME`` out of ``argv`` and return NAME.

    Mutates ``argv`` so the caller's own parser never sees a flag it does not define.
    Unknown names raise here rather than falling back to the default: silently building
    `split` figures under the shipped profile's output root is the exact confusion the
    two roots exist to prevent.
    """
    name = "shipped"
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok == "--profile" and i + 1 < len(argv):
            name = argv[i + 1]
            del argv[i:i + 2]
            continue
        if tok.startswith("--profile="):
            name = tok.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    name = name.strip() or "shipped"
    if name not in PROFILES:
        raise SystemExit(f"unknown --profile {name!r}; known: {', '.join(sorted(PROFILES))}")
    return name


PROFILE = _take_profile_arg(sys.argv)
_P = PROFILES[PROFILE]

DATA_DIR = REPO_ROOT / "outputs" / _P["subdir"] / "data"
FIG_DIR = REPO_ROOT / "outputs" / _P["subdir"] / "figures"
RESEARCH_SWEEP_DATA_DIR = DATA_DIR / "research_sweeps"
RESEARCH_SWEEP_FIG_DIR = FIG_DIR / "research_sweeps"

@dataclass
class PubConfig:
    # Model versions -- defaults come from the selected PUB_PROFILE, not from here
    rgp_deq_model: str = "rgp_deq_kine"
    clot_ml_model: str = _P["clot_ml_model"]
    profile: str = PROFILE

    # Strict nested-CV OOF (eval_strict_temporal.py --save-oof-series)
    oof_series_path: str = str((DATA_DIR / _P["oof_series_name"]).relative_to(REPO_ROOT))
    oof_arms: tuple = _P["oof_arms"]
    oof_cache: str = _P["oof_cache"]
    oof_head_seeds: int = 4
    oof_set_masks: str = "outputs/v4_set_masks.npz"

    # Cohorts (all must appear in the OOF archive)
    fig1_vessels: tuple = ("comsol020", "comsol005")
    fig3_vessels: tuple = ("comsol020", "comsol005", "comsol012", "comsol041")
    fig4_vessels: tuple = ("comsol020", "comsol005")
    fig6_vessels: tuple = ("comsol005", "comsol014")

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
    #
    # Each value is ``(section, description, placement)`` with placement in
    # {"main", "supplement", "dropped"}.
    #
    # **REBALANCED 2026-09-09, and the reason is a change of thesis, not of taste.** The set
    # was assembled when the TOOL was the paper's claim.  The claim is now the measured
    # division of labour (PAPER.md s1, s5), and against that the old set had two defects: the
    # lead claim had NO figure at all, while `ablation_ladder` -- which draws the 14-arm
    # feature ladder that s5.2 reports as a NULL -- sat in the main text where a reader would
    # take it for the spine.  Three figures were added and six items demoted or dropped, taking
    # the main-text count from ~20 to 12 against PUBLICATION_PLAN s8's budget of eleven.
    #
    # A "dropped" row is kept rather than deleted so the reason survives; deleting it loses the
    # argument and the next person redraws the figure.
    paper_map: dict = field(default_factory=lambda: {
        # REBUILT 2026-09-13 to docs/publication/FIGURES.md's set: sections are STORY legs, and
        # the paper carries NO supplementary figures.  Dropped rows are kept with their reason.
        # --- Leg 0: the tool works, intact and injured ------------------------------------
        "comsol_ground_truth":  ("0.1", "What we are replacing", "main"),
        "biochem_architecture": ("0.2", "The shipped architecture, wound and no-wound", "main"),
        "batc":                 ("0.3", "Why BATC: three vessels scored three ways", "main"),
        "applications":         ("0.4", "An application, with total clot mass", "main"),
        "wound_temporal":       ("0.5a", "Total clot mass over the horizon, intact and injured", "main"),
        "wound_example_comsol003": ("0.5b", "Injured vessel over time, model vs GT", "main"),
        "wound_example_comsol006": ("0.5c", "Hardest injured vessel over time, model vs GT", "main"),
        "fig6_failures":        ("0.6", "Where BATC and the eye disagree (comsol005) -- dropped on review 2026-09-16", "dropped"),
        "error_trajectories":   ("0.6b", "A bad final frame is never judged (comsol014; comsol037/040 recover)", "main"),
        "table4_kfold":         ("0", "Geometry generalization (primary evidence, a table)",
                                 "main"),
        # --- Leg 1: why it is built this way -----------------------------------------------
        "architecture_table":   ("1.0", "The shipped architecture as a methods table", "main"),
        "division_of_labour":   ("1.1", "Which domain wants physics, which wants learning",
                                 "main"),
        # --- Leg 2: the flow is solved once, with FEM --------------------------------------
        "coupling_frequency":   ("2.1", "How much coupling is needed: none measurable", "main"),
        "timing_cost":          ("2.2", "Where the time goes: the flow solve is a low share",
                                 "main"),
        # --- dropped on review 2026-09-13 --------------------------------------------------
        "closed_loop_oracle":   ("2.1", "Oracle closed loop -- the GT oracle leaks the label",
                                 "dropped"),
        "amdahl_ceiling":       ("2.2", "Separate flow-share ceiling figure -- one sentence on "
                                        "2.2 instead", "dropped"),
        "coupling_gate":        ("2.4", "Gate identity vs size -- prose only", "dropped"),
        "ablation_ladder":      ("1", "Feature-conditioning ladder -- a null, no supplement",
                                 "dropped"),
        "fig1_flow":            ("2", "Flow fields RGP-DEQ / FEM / GT", "dropped"),
        "geometry_classes":     ("0", "Cohort and geometry classes", "dropped"),
        "fig34_biochem_final":  ("0", "Final-time clot maps", "dropped"),
        "fig34_biochem_temporal": ("0", "Temporal evolution -- superseded by 0.5", "dropped"),
        "research_sweeps":      ("0", "Geometry-response sweeps -- 0.4 carries the study",
                                 "dropped"),
        "operating_point":      ("0", "Committed-set operating point", "dropped"),
        "flow_requirement":     ("2", "What the flow surrogate must get right -- prose",
                                 "dropped"),
        "onset_timing":         ("0", "Onset timing", "dropped"),
        "biochem_detail":       ("0.2", "Architecture detail slide", "dropped"),
        "rgp_architecture":     ("2", "RGP-DEQ architecture slide", "dropped"),
        # The shipped arm has 0/33 empty gates: the failure this would depict does not occur.
        "gate_seeding_mechanism": ("2", "Gate seeding -- the depicted failure no longer occurs",
                                   "dropped"),
        # The live claim is one sentence (false-alarm rate 0 in 37); the generator is gitignored.
        "preflight_validation": ("2", "Pre-flight table -- one sentence in the text", "dropped"),
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
print(f"[i] publication profile: {PROFILE}  (model={CONFIG.clot_ml_model}, "
      f"cache={CONFIG.oof_cache}, out={DATA_DIR.parent.name}/)", flush=True)

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_SWEEP_FIG_DIR.mkdir(parents=True, exist_ok=True)
