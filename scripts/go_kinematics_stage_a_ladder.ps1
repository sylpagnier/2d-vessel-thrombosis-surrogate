# Stage-A ladder: foundation -> synthetic polish -> comsol finetune -> promote.
#   powershell ... -File .\scripts\go_kinematics_stage_a_ladder.ps1 -SkipFoundation

param(
  [switch]$Fresh,
  [switch]$SkipFoundation,
  [switch]$SkipSyntheticPolish,
  [switch]$SkipComsolAnchors,
  [switch]$SkipPromote,
  [switch]$RequireComsol,
  [string]$Resume = "outputs/kinematics/production_allfix/kinematics_best.pth",
  [string]$Holdout = "comsol007",
  [int]$SyntheticFinetuneEpochs = 40,
  [int]$ComsolFinetuneEpochs = 25,
  [switch]$ContinuityFocus,
  [switch]$NoContinuityFocus,
  [switch]$Quiet
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$pyArgs = @("scripts/run_kinematics_production.py", "ladder")
if ($Fresh) { $pyArgs += "--fresh" }
if ($SkipFoundation) { $pyArgs += "--skip-foundation" }
if ($SkipSyntheticPolish) { $pyArgs += "--skip-synthetic-polish" }
if ($SkipComsolAnchors) { $pyArgs += "--skip-comsol-anchors" }
if ($SkipPromote) { $pyArgs += "--skip-promote" }
if ($RequireComsol) { $pyArgs += "--require-comsol" }
if ($NoContinuityFocus) { $pyArgs += "--no-continuity-focus" }
if ($Quiet) { $pyArgs += "--quiet" }
$pyArgs += @(
  "--holdout", $Holdout
  "--synthetic-finetune-epochs", "$SyntheticFinetuneEpochs"
  "--comsol-finetune-epochs", "$ComsolFinetuneEpochs"
)
Invoke-GoKinematicsProduction -PyArgs $pyArgs
