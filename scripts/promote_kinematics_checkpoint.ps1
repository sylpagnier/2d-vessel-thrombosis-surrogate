# Copy a gated Stage-A checkpoint to outputs/kinematics/kinematics_best.pth

param(
    [Parameter(Mandatory = $true)]
    [string] $Checkpoint,
    [string] $Holdout = "comsol007",
    [double] $MaxComsolRelL2 = 0.25,
    [double] $MaxSyntheticRelL2 = 0.20,
    [double] $MaxSyntheticL2RelL2 = 0.22,
    [int] $SyntheticCap = 200,
    [switch] $Force
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$RepoRoot = Initialize-HemoRepo -ScriptRoot $PSScriptRoot

$ckpt = $Checkpoint
if (-not [System.IO.Path]::IsPathRooted($ckpt)) {
    $ckpt = Join-Path $RepoRoot $Checkpoint
}

if (-not $Force) {
    $gateArgs = @(
        "scripts/check_kinematics_promotion_gates.py",
        "--checkpoint", $ckpt,
        "--holdout", $Holdout,
        "--max-comsol-rel-l2", "$MaxComsolRelL2",
        "--max-synthetic-rel-l2", "$MaxSyntheticRelL2",
        "--max-synthetic-l2-rel-l2", "$MaxSyntheticL2RelL2",
        "--synthetic-cap", "$SyntheticCap"
    )
    $rc = Invoke-HemoPython -Args $gateArgs
    if ($rc -ne 0) {
        throw "[promote] gates failed; use -Force to copy anyway."
    }
}

$dest = Join-Path $RepoRoot "outputs\kinematics\kinematics_best.pth"
Copy-Item $ckpt $dest -Force
Write-Host "[promote] copied -> $dest"
