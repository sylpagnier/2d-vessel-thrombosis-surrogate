<#
.SYNOPSIS
Orchestrates the generation of all publication figures.

.DESCRIPTION
Runs the data generation scripts first (which can take a while as they do inference),
and then runs the plotter scripts to generate PDFs and CSVs in the outputs directory.
#>

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = $PSScriptRoot

Write-Host "[i] Starting Publication Figure Generation Pipeline" -ForegroundColor Cyan

Write-Host "`n=== STEP 1: Data Generation ===" -ForegroundColor Yellow
python "$SCRIPT_DIR/generate_oof_series.py"
python "$SCRIPT_DIR/generate_fig1_data.py"
python "$SCRIPT_DIR/generate_fig3_4_data.py"
python "$SCRIPT_DIR/generate_fig6_data.py"
python "$SCRIPT_DIR/generate_onset_timing_data.py"
python "$SCRIPT_DIR/generate_research_sweep_fig_data.py"
python "$SCRIPT_DIR/generate_kfold_table.py"
# Fig 9 collects artifacts produced by scripts/eval_clot_ml_0.py and
# scripts/diag_flow_sensitivity.py; it reports (exit 2) which are missing rather than failing.
python "$SCRIPT_DIR/generate_flow_diagnostics.py"
python "$SCRIPT_DIR/generate_flow_requirement_data.py"
# Timing must run on an OTHERWISE IDLE machine -- it is a wall-clock measurement, and anything
# else running here corrupts it.  Left out of the default pipeline deliberately; run it alone:
#   python scripts/publication/generate_timing_data.py --every 1
# Wound A/B (wound_comsol005 vs. comsol048) is a PREVIEW, not a Fig 12/13-budgeted item --
# the wound section stays frozen pending PUBLICATION_NOTES §7.0. Left out of the default
# pipeline deliberately; run it alone:
#   python scripts/publication/generate_wound_ab_data.py && python scripts/publication/plot_wound_ab.py

Write-Host "`n=== STEP 2: Plotting and Tables ===" -ForegroundColor Yellow
python "$SCRIPT_DIR/plot_fig1_flow.py"
python "$SCRIPT_DIR/plot_fig3_biochem_final.py"
python "$SCRIPT_DIR/plot_fig4_biochem_temporal.py"
python "$SCRIPT_DIR/plot_fig6_failures.py"
python "$SCRIPT_DIR/plot_onset_timing.py"
python "$SCRIPT_DIR/plot_error_trajectories.py"
python "$SCRIPT_DIR/plot_research_sweep_figures.py"
python "$SCRIPT_DIR/plot_geometry_classes.py"
python "$SCRIPT_DIR/plot_flow_requirement.py"
python "$SCRIPT_DIR/plot_timing.py"

Write-Host "`n[OK] Pipeline completed successfully! Outputs are in outputs/publication/" -ForegroundColor Green
