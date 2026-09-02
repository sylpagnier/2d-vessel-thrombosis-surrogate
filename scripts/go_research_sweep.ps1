# Geometry-sensitivity research sweeps (default: clot_ml_0 + FEM t=0).
#   powershell ... -File .\scripts\go_research_sweep.ps1 -List
#   powershell ... -File .\scripts\go_research_sweep.ps1 -Sweep 01_stenosis_strength
# Legacy biochem only: -WallCkpt / -MatLeg

param(
    [string] $Sweep = "",
    [switch] $All,
    [switch] $List,
    [switch] $Legacy,
    [string] $Arm = "",
    [switch] $ForceRebuildMesh,
    [switch] $Cpu,
    [string] $WallCkpt = "",
    [string] $MatLeg = ""
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
Invoke-GoResearchSweep @PSBoundParameters
