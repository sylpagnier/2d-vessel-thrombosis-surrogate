# Build a self-contained Windows bundle of the customer Predict UX (Local FEM Solver).
#
# Produces dist\LocalFEMSolver-Predict-win64-<Version>.zip: an embeddable Python + the CPU-only
# deploy dependencies (requirements-customer.txt) + the app source + the ~11 MB of clot_ml_0
# checkpoints it actually loads, with a run.bat launcher. No researcher Python env, no CUDA,
# no separate model-checkpoint hunt required on the other end -- unzip and double-click.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_customer_bundle.ps1
#   powershell ... -File .\scripts\build_customer_bundle.ps1 -Version 1.1

param(
    [string] $Version = "1.0",
    [string] $PythonVersion = "3.13.7"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$BundleName = "LocalFEMSolver-Predict-win64-$Version"
$DistDir    = Join-Path $RepoRoot "dist"
$ScratchDir = Join-Path $DistDir "_build"
$BundleDir  = Join-Path $DistDir $BundleName
$ZipPath    = Join-Path $DistDir "$BundleName.zip"

Write-Host "[i] Building $BundleName" -ForegroundColor Cyan

if (Test-Path $BundleDir) { Remove-Item -Recurse -Force $BundleDir }
if (Test-Path $ZipPath)   { Remove-Item -Force $ZipPath }
New-Item -ItemType Directory -Force -Path $ScratchDir | Out-Null
New-Item -ItemType Directory -Force -Path $BundleDir  | Out-Null

# --- 1. Embeddable Python -------------------------------------------------------------
$PyZipUrl  = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PyZipPath = Join-Path $ScratchDir "python-embed.zip"
$PyDir     = Join-Path $BundleDir "python"

Write-Host "[i] Downloading embeddable Python $PythonVersion..." -ForegroundColor DarkGray
Invoke-WebRequest -Uri $PyZipUrl -OutFile $PyZipPath
Expand-Archive -Path $PyZipPath -DestinationPath $PyDir -Force

# Enable site-packages (embeddable Python ships with `import site` commented out in its
# `._pth` file, which otherwise makes pip-installed packages invisible), and add `..` so the
# bundle root (where `src/` lives, one level up from python.exe) is on sys.path -- an
# embeddable Python's `._pth` file, once present, is the ENTIRE sys.path; it does not fall
# back to the interpreter's normal "add the script/cwd directory" behaviour the way a regular
# python.exe does, so `-m src.tools...` fails with "No module named 'src'" without this.
$PthFile = Get-ChildItem -Path $PyDir -Filter "python*._pth" | Select-Object -First 1
$PthLines = (Get-Content $PthFile.FullName) -replace '^#\s*import site', 'import site'
$PthLines = @("..") + $PthLines
Set-Content -Path $PthFile.FullName -Value $PthLines
$PyExe = Join-Path $PyDir "python.exe"

# --- 2. pip + deploy-only dependencies -------------------------------------------------
$GetPipPath = Join-Path $ScratchDir "get-pip.py"
Write-Host "[i] Bootstrapping pip..." -ForegroundColor DarkGray
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPipPath
& $PyExe $GetPipPath --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

Write-Host "[i] Installing CPU-only torch..." -ForegroundColor DarkGray
& $PyExe -m pip install --no-cache-dir --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "[i] Installing the rest of requirements-customer.txt..." -ForegroundColor DarkGray
& $PyExe -m pip install --no-cache-dir --no-warn-script-location -r (Join-Path $RepoRoot "requirements-customer.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# --- 3. App source + project marker -----------------------------------------------------
Write-Host "[i] Copying app source..." -ForegroundColor DarkGray
Copy-Item -Recurse -Force (Join-Path $RepoRoot "src") (Join-Path $BundleDir "src")
Copy-Item -Force (Join-Path $RepoRoot "pyproject.toml") $BundleDir

# `src/clot_ml/locked.py`'s readout selection (`expected_tuned` / `resid_adapt`, the deploy
# readout clot_ml_0 actually uses -- see data/reference/clot_gnn_locked.json's "readout"
# block) lazily imports helper functions from `scripts/eval_*.py` at call time. Despite the
# directory name this is deploy-reachable code, not just a research tool -- copied wholesale
# (7 MB of source, cheap) rather than hand-picking which of the ~15 possible lazy imports in
# `locked.py` a given run happens to hit.
Copy-Item -Recurse -Force (Join-Path $RepoRoot "scripts") (Join-Path $BundleDir "scripts")

$InboxDir = Join-Path $BundleDir "customer_geometries"
New-Item -ItemType Directory -Force -Path $InboxDir | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "customer_geometries\README.txt") $InboxDir

# --- 4. The ~11 MB of checkpoints this tool actually loads ------------------------------
# Traced from CustomerDeployPipeline.run() -> load_v0_bundle("clot_ml_0"): only
# outputs/clot_ml/locked/clot_ml_v0 (config/manifest) and outputs/clot_ml/locked/clot_gnn_v6
# (the ensemble + temporal readout it points at) are ever opened. No kinematics or
# wall-model checkpoint is on this path.
#
# DELIBERATELY PINNED BY NAME, not resolved dynamically: clot_ml_0 is being retrained as of
# 2026-09-03 (see the in-progress, still-manifest-less outputs/clot_ml/locked/DeployClot*
# folders) -- copying these two specific, already-validated directories means a bundle built
# today ships the last known-good artifact regardless of what's mid-training alongside it.
# Bump these two names deliberately once a retrained artifact is promoted and validated.
Write-Host "[i] Copying clot_ml_0 checkpoints..." -ForegroundColor DarkGray
$CkptSrc = Join-Path $RepoRoot "outputs\clot_ml\locked"
$CkptDst = Join-Path $BundleDir "outputs\clot_ml\locked"
foreach ($sub in @("clot_ml_v0", "clot_gnn_v6")) {
    $src = Join-Path $CkptSrc $sub
    if (-not (Test-Path $src)) { throw "Missing expected checkpoint dir: $src" }
    Copy-Item -Recurse -Force $src (Join-Path $CkptDst $sub)
}

# `src/clot_ml/locked.py` reads this pointer file UNCONDITIONALLY at the top of
# `load_ensemble()`, even on the branch that ignores its contents (an explicit `name=` is
# passed) -- so it has to exist on disk even though clot_ml_0 never uses the model it points
# at (clot_gnn_v6w). Tiny reference JSON, not a checkpoint.
$RefDst = Join-Path $BundleDir "data\reference"
New-Item -ItemType Directory -Force -Path $RefDst | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "data\reference\clot_gnn_locked.json") $RefDst

# --- 5. Launcher + README -----------------------------------------------------------------
$RunBat = @'
@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "HEREPATH=%~dp0"
set "PATHLEN=0"
for /l %%A in (0,1,300) do (
    if not "!HEREPATH:~%%A,1!"=="" set /a PATHLEN=%%A+1
)
if !PATHLEN! GTR 100 (
    echo WARNING: this folder's path is very long ^(!PATHLEN! characters^):
    echo   %~dp0
    echo Some Python packages this app depends on fail to load from long paths on
    echo Windows. If you see an error mentioning "sklearn" or "No module named", move
    echo this whole folder somewhere short first, e.g. C:\LocalFEMSolver\, then run
    echo run.bat again from there.
    echo.
)
echo Starting Local FEM Solver Predict...
echo A browser tab will open automatically once it's ready. This window logs progress -- leave it open while you use the app, close it when you're done.
"%~dp0python\python.exe" -m src.tools.customer_predict_web --cpu
pause
'@
Set-Content -Path (Join-Path $BundleDir "run.bat") -Value $RunBat -Encoding ASCII

$ReadmeTxt = @'
Local FEM Solver -- Predict
============================

Before you run it: unzip this to somewhere with a SHORT path, close to a drive root --
for example C:\LocalFEMSolver\ -- rather than deep inside Downloads or a synced folder.
A few of this app's dependencies fail to load if the unzipped path is very long (a
Windows limitation, not a bug in the app).

To run:
  1. Double-click run.bat.
  2. A black window will open and print some status lines -- that's normal, leave it open.
  3. Your web browser will open automatically to the app after a few seconds.
  4. When you're done, close the black window (or press Ctrl+C in it) to stop the app.

First run may show a Windows "protected your PC" / SmartScreen prompt, since this app isn't
code-signed yet. Click "More info" then "Run anyway" to continue.

Runs entirely on your own computer -- no data leaves your machine, no internet connection
required after this download, no GPU needed (it runs on CPU).

Questions or something not working? Contact the team that shared this with you.
'@
Set-Content -Path (Join-Path $BundleDir "README.txt") -Value $ReadmeTxt -Encoding ASCII

# --- 6. Zip --------------------------------------------------------------------------------
Write-Host "[i] Zipping to $ZipPath..." -ForegroundColor DarkGray
Compress-Archive -Path $BundleDir -DestinationPath $ZipPath -Force

Remove-Item -Recurse -Force $ScratchDir

$SizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "[OK] Built $ZipPath ($SizeMb MB)" -ForegroundColor Green
