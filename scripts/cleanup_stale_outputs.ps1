# Remove stale local experiment outputs (see docs/OUTPUTS_RETENTION.md).
#
#   powershell ... -File .\scripts\cleanup_stale_outputs.ps1 -WhatIf
#   powershell ... -File .\scripts\cleanup_stale_outputs.ps1

param([switch] $WhatIf)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
$RepoRoot = Initialize-HemoRepo -ScriptRoot $PSScriptRoot

$keepTop = @(
    "kinematics", "clot_ml", "research_sweeps", "customer_predict", "phase9_scores",
    "cache", "temporal_transport", "pi_corpus", "reports",
    "clot_ml_cache_v4", "clot_ml_cache_v4_pred", "clot_ml_cache_gt", "clot_ml_cache_pred",
    "clot_ml_cache_v5", "clot_ml_cache_v5_pred", "biochem", "logs"
)

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
Get-ChildItem $outRoot -Directory | Where-Object { $keepTop -notcontains $_.Name } | ForEach-Object {
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
