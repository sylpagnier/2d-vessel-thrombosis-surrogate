# Production Stage-A kinematics (default: foundation -> polish -> clinical -> promote).
#   powershell ... -File .\scripts\go_kinematics_production_allfix.ps1
#   powershell ... -File .\scripts\go_kinematics_production_allfix.ps1 -FoundationOnly -Fresh

param(
  [switch]$Fresh,
  [switch]$FoundationOnly,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipClinicalAnchors,
  [switch]$SkipPromote,
  [switch]$RequireClinical,
  [string]$Holdout = "patient007",
  [switch]$NoContinuityFocus,
  [int]$Epochs = 100,
  [int]$AdamEpochs = 85,
  [int]$Stage1End = 40,
  [int]$Stage2End = 60,
  [int]$GraphCap = 0,
  [int]$Seed = 42,
  [switch]$Quiet
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$pyArgs = @("scripts/run_kinematics_production.py")
if ($Fresh) { $pyArgs += "--fresh" }
if ($FoundationOnly) { $pyArgs += "--foundation-only" }
if ($SkipSyntheticPolish) { $pyArgs += "--skip-synthetic-polish" }
if ($SkipClinicalAnchors) { $pyArgs += "--skip-clinical-anchors" }
if ($SkipPromote) { $pyArgs += "--skip-promote" }
if ($RequireClinical) { $pyArgs += "--require-clinical" }
if ($NoContinuityFocus) { $pyArgs += "--no-continuity-focus" }
if ($Quiet) { $pyArgs += "--quiet" }
$pyArgs += @(
  "--holdout", $Holdout
  "--epochs", "$Epochs"
  "--adam-epochs", "$AdamEpochs"
  "--stage1-end", "$Stage1End"
  "--stage2-end", "$Stage2End"
  "--graph-cap", "$GraphCap"
  "--seed", "$Seed"
)
Invoke-GoKinematicsProduction -PyArgs $pyArgs
