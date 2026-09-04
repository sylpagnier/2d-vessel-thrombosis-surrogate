# Cut a public release of the customer Predict bundle: build it (build_customer_bundle.ps1),
# then tag, push, and publish it as a GitHub Release with the zip attached.
#
# This is the ONE command for "ship an update to researchers": everything the bundle needs
# (checkpoints, demo geometry) only ever exists on this machine and is never sent to CI --
# see docs/PUBLISHING.md. The GitHub Release, not `git clone`, is the distribution channel.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\release_bundle.ps1 -Version 1.1
#   ... -Version 1.1 -Yes          # skip the confirm prompt (e.g. non-interactive use)

param(
    [Parameter(Mandatory = $true)][string] $Version,
    [string] $PythonVersion = "3.13.7",
    [switch] $Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI ('gh') not found. Install it (https://cli.github.com/) and run 'gh auth login' once, then re-run this script."
}
gh auth status *>$null
if ($LASTEXITCODE -ne 0) {
    throw "'gh' is installed but not authenticated. Run 'gh auth login' first."
}

$Tag = "v$Version"
$existingTag = git tag --list $Tag
if ($existingTag) {
    throw "Tag $Tag already exists locally. Pick a new -Version, or delete the tag first if this was a mistake."
}

# --- Build --------------------------------------------------------------------------------
& (Join-Path $PSScriptRoot "build_customer_bundle.ps1") -Version $Version -PythonVersion $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "Bundle build failed." }

$ZipPath = Join-Path $RepoRoot "dist\LocalFEMSolver-Predict-win64-$Version.zip"
if (-not (Test-Path $ZipPath)) { throw "Build did not produce the expected zip: $ZipPath" }
$SizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)

# --- Confirm --------------------------------------------------------------------------------
$Commit = git rev-parse --short HEAD
Write-Host ""
Write-Host "About to publish a public GitHub Release:" -ForegroundColor Cyan
Write-Host "  Tag:      $Tag (at commit $Commit)"
Write-Host "  Asset:    $ZipPath ($SizeMb MB)"
Write-Host "  Remote:   $(git remote get-url origin)"
Write-Host ""
if (-not $Yes) {
    $resp = Read-Host "Push this tag and publish the release? [y/N]"
    if ($resp -notmatch '^[Yy]') {
        Write-Host "[i] Aborted -- built zip is still at $ZipPath if you want it." -ForegroundColor Yellow
        exit 1
    }
}

# --- Tag + push -----------------------------------------------------------------------------
git tag -a $Tag -m "Local FEM Solver Predict $Tag"
git push origin $Tag

# --- Release ----------------------------------------------------------------------------------
gh release create $Tag $ZipPath `
    --title "Local FEM Solver Predict $Tag" `
    --generate-notes
if ($LASTEXITCODE -ne 0) { throw "gh release create failed (tag $Tag was already pushed -- fix the issue and re-run 'gh release create' by hand rather than re-running this script)." }

Write-Host "[OK] Published $Tag" -ForegroundColor Green
