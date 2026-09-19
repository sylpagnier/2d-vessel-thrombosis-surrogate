# Remove stale local experiment outputs (see docs/OUTPUTS_RETENTION.md).
#
#   powershell ... -File .\scripts\cleanup_stale_outputs.ps1 -WhatIf
#   powershell ... -File .\scripts\cleanup_stale_outputs.ps1

param([switch] $WhatIf)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$RepoRoot = Initialize-HemoRepo -ScriptRoot $PSScriptRoot

# Exact names to keep.  `runs` and `archive` are load-bearing and were BOTH missing: the
# split recipe's flow checkpoint lives in `outputs/runs/<arm>/`, and `outputs/archive/` holds
# the snapshotted model cohorts a paper's numbers are attached to.  Deleting either is
# unrecoverable without a full retrain, and this script deletes any top-level directory not
# named here.
$keepTop = @(
    "kinematics", "clot_ml", "research_sweeps", "customer_predict", "phase9_scores",
    "cache", "pi_corpus", "reports", "biochem", "logs",
    "runs", "archive", "deployclot"
)

# Families that legitimately GROW: one directory per flow source or per feature-cache
# generation.  Listing them by name is how the list fell four entries behind reality --
# `clot_ml_cache_v5_fem`, `clot_ml_cache_v5_split`, `temporal_transport_fem` and
# `temporal_transport_split` were all absent and would have been deleted.  Match by prefix so
# a new flow source is protected the day it is created rather than the day someone notices.
$keepPrefix = @("clot_ml_cache_", "temporal_transport", "publication")

$outRoot = Join-Path $RepoRoot "outputs"
if (-not (Test-Path $outRoot)) {
    Write-Host "[i] No outputs/ directory"
    exit 0
}

function Remove-Tree([string] $Path) {
    if (-not (Test-Path $Path)) { return }
    if ($WhatIf) {
        Write-Host "[whatif] remove $Path"
    } else {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "[OK] removed $Path"
    }
}

# Top-level dirs not on keep list
Get-ChildItem $outRoot -Directory | Where-Object {
    $name = $_.Name
    ($keepTop -notcontains $name) -and
    (-not ($keepPrefix | Where-Object { $name.StartsWith($_) }))
} | ForEach-Object {
    Remove-Tree $_.FullName
}

# biochem: keep only biochem_gnn/locked
$biochem = Join-Path $outRoot "biochem"
if (Test-Path $biochem) {
    Get-ChildItem $biochem -Directory | Where-Object { $_.Name -ne "biochem_gnn" } | ForEach-Object {
        Remove-Tree $_.FullName
    }
    $bgnn = Join-Path $biochem "biochem_gnn"
    if (Test-Path $bgnn) {
        Get-ChildItem $bgnn -Directory | Where-Object { $_.Name -ne "locked" } | ForEach-Object {
            Remove-Tree $_.FullName
        }
    }
}

Write-Host "[OK] Stale outputs cleanup done$(if ($WhatIf) { ' (whatif)' } else { '' })"
