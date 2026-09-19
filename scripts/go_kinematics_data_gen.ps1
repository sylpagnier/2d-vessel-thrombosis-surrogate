# Kinematics mesh / anchor / graph generation helper.
#   powershell ... -File .\scripts\go_kinematics_data_gen.ps1 [-NumVessels 100] [-Overwrite]

param(
    [int] $NumVessels = 100,
    [switch] $Overwrite
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$RepoRoot = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
$env:PYTHONPATH = "."

Write-Host "[i] Generating straight_max vessels (n=$NumVessels)..." -ForegroundColor Cyan
$vesselArgs = @("--phase", "1", "--level", "0", "-n", "$NumVessels", "--pathology-mode", "straight_max")
if ($Overwrite) { $vesselArgs += "--overwrite" }
$rc = Invoke-HemoPython -Args (@("src/data_gen/lib/vessel_generator.py") + $vesselArgs)
if ($rc -ne 0) { Exit-GoLauncher -Rc $rc -Label "vessel_generator" }

Write-Host "[i] Generating COMSOL anchors..." -ForegroundColor Cyan
$ow = if ($Overwrite) { "True" } else { "False" }
$pyScript = @"
from src.data_gen.lib.anchor_generator import AnchorGenerator
gen = AnchorGenerator(phase='phase1')
gen.run_batch(max_new=$NumVessels, allow_overwrite=$ow)
"@
$rc = Invoke-HemoPython -Args @("-c", $pyScript)
if ($rc -ne 0) { Exit-GoLauncher -Rc $rc -Label "anchor_generator" }

Write-Host "[i] Converting meshes to PyG graphs..." -ForegroundColor Cyan
$rc = Invoke-HemoPython -Args @("src/data_gen/lib/mesh_to_graph.py", "--phase", "1", "--rheology", "newtonian")
Exit-GoLauncher -Rc $rc -Label "mesh_to_graph"
