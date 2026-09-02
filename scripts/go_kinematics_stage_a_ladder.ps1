# Stage-A ladder: foundation -> synthetic polish -> clinical finetune -> promote.
#   powershell ... -File .\scripts\go_kinematics_stage_a_ladder.ps1 -SkipFoundation

param(
  [switch]$Fresh,
  [switch]$SkipFoundation,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipClinicalAnchors,
  [switch]$SkipPromote,
  [switch]$RequireClinical,
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_best.pth",
  [string]$Holdout = "patient007",
  [int]$SyntheticFinetuneEpochs = 40,
  [int]$ClinicalFinetuneEpochs = 25,
  [switch]$ContinuityFocus,
  [switch]$NoContinuityFocus,
  [switch]$Quiet
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$pyArgs = @("scripts/run_kinematics_production.py", "ladder")
if ($Fresh) { $pyArgs += "--fresh" }
if ($SkipFoundation) { $pyArgs += "--skip-foundation" }
if ($SkipSyntheticPolish) { $pyArgs += "--skip-synthetic-polish" }
if ($SkipClinicalAnchors) { $pyArgs += "--skip-clinical-anchors" }
if ($SkipPromote) { $pyArgs += "--skip-promote" }
if ($RequireClinical) { $pyArgs += "--require-clinical" }
if ($NoContinuityFocus) { $pyArgs += "--no-continuity-focus" }
if ($Quiet) { $pyArgs += "--quiet" }
$pyArgs += @(
  "--holdout", $Holdout
  "--synthetic-finetune-epochs", "$SyntheticFinetuneEpochs"
  "--clinical-finetune-epochs", "$ClinicalFinetuneEpochs"
)
Invoke-GoKinematicsProduction -PyArgs $pyArgs
