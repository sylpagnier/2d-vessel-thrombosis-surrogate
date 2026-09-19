# Run research sweeps then regenerate publication figures.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\publication\make_research_sweeps.ps1
#   powershell ... -Sweep 16_wound_width
#   powershell ... -FiguresOnly

param(
    [string] $Sweep = "",
    [switch] $All,
    [switch] $FiguresOnly,
    [switch] $ForceRebuildMesh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

if (-not $FiguresOnly) {
    $sweepArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ".\scripts\go_research_sweep.ps1")
    if ($All) { $sweepArgs += "-All" }
    elseif ($Sweep.Trim()) { $sweepArgs += @("-Sweep", $Sweep.Trim()) }
    else {
        Write-Host "[ERR] Pass -Sweep <id>, -All, or -FiguresOnly" -ForegroundColor Red
        exit 2
    }
    if ($ForceRebuildMesh) { $sweepArgs += "-ForceRebuildMesh" }
    & powershell @sweepArgs
    $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    if ($rc -ne 0) { exit $rc }
}

Write-Host "[i] Research sweep publication figures" -ForegroundColor Cyan
python scripts/publication/generate_research_sweep_fig_data.py
python scripts/publication/plot_research_sweep_figures.py
$rc2 = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
if ($rc2 -ne 0) { exit $rc2 }
Write-Host "[OK] Done" -ForegroundColor Green
