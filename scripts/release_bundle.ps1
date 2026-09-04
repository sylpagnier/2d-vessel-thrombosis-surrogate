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
    [switch] $Yes,
    [switch] $AllowDirty
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

# build_customer_bundle.ps1 copies src/ and scripts/ straight from the working tree, not from
# git -- an uncommitted or untracked change would ship in the zip while tag $Tag's history
# shows something else, so a researcher's bug report against "v$Version" wouldn't reproduce
# against the v$Version source. Require a clean tree (pass -AllowDirty to override deliberately).
$dirty = git status --porcelain
if ($dirty -and -not $AllowDirty) {
    throw "Working tree has uncommitted/untracked changes -- the bundle would not match tag ${Tag}:`n$dirty`n`nCommit (or stash) first, or pass -AllowDirty to ship the working tree as-is anyway."
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

# --- Tag + push + release, with rollback on any failure -------------------------------------
# Each step below undoes everything staged so far before rethrowing, so a failure anywhere
# leaves no half-published tag behind -- re-running this script after fixing the problem
# always starts clean instead of hitting the "tag already exists" guard above.
git tag -a $Tag -m "Local FEM Solver Predict $Tag"
if ($LASTEXITCODE -ne 0) { throw "git tag failed." }

try {
    git push origin $Tag
    if ($LASTEXITCODE -ne 0) { throw "git push origin $Tag failed." }
} catch {
    Write-Host "[ERR] Push failed -- deleting local tag $Tag so you can re-run cleanly." -ForegroundColor Red
    git tag -d $Tag | Out-Null
    throw
}

try {
    gh release create $Tag $ZipPath `
        --title "Local FEM Solver Predict $Tag" `
        --generate-notes
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed." }
} catch {
    Write-Host "[ERR] Release publish failed -- deleting tag $Tag (local + remote) so you can re-run cleanly." -ForegroundColor Red
    git push origin --delete $Tag 2>$null
    git tag -d $Tag | Out-Null
    throw
}

Write-Host "[OK] Published $Tag" -ForegroundColor Green
