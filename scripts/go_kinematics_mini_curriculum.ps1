param(
  [string]$Holdout = "patient007"
)

$ErrorActionPreference = "Stop"

Write-Host "====================================================================="
Write-Host "[MINI-CURRICULUM] Launching fast-track RGP-DEQ Stage A training..."
Write-Host "                  - 1000 vessels per epoch limit"
Write-Host "                  - 40 epochs total (15 foundation, 15 polish, 10 target)"
Write-Host "                  - Fully executes Phase 3 clinical geometry finetune"
Write-Host "====================================================================="

$scriptArgs = @(
    "-Fresh",
    "-Epochs", "40",
    "-AdamEpochs", "35",
    "-Stage1End", "15",
    "-Stage2End", "30",
    "-GraphCap", "1000",
    "-Holdout", $Holdout
)

& (Join-Path $PSScriptRoot "go_kinematics_production_allfix.ps1") @scriptArgs
exit $LASTEXITCODE
