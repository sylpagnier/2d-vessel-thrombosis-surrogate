# Build a self-contained Windows bundle of the customer Predict UX (2D Thrombosis Surrogate).
#
# Produces two zips in dist\:
#   2DThrombosisSurrogate-Predict-win64-<Version>.zip -- an embeddable Python + the CPU-only
#     deploy dependencies (requirements-customer.txt) + the app source + the ~11 MB of clot_ml_0
#     checkpoints it actually loads, with a run.bat launcher. Unzip and double-click.
#   2DThrombosisSurrogate-Predict-macos-linux-<Version>.zip -- the same app, checkpoints and demo
#     without the Windows interpreter, plus run.command / run.sh (scripts/bundle/run_unix.sh) and
#     requirements-lock.txt: the exact package versions installed into the Windows build, which
#     the launcher installs into a local venv on first run.
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

$BundleName = "2DThrombosisSurrogate-Predict-win64-$Version"
$UnixName   = "2DThrombosisSurrogate-Predict-macos-linux-$Version"
$DistDir    = Join-Path $RepoRoot "dist"
$ScratchDir = Join-Path $DistDir "_build"
$BundleDir  = Join-Path $DistDir $BundleName
$UnixDir    = Join-Path $DistDir $UnixName
$ZipPath    = Join-Path $DistDir "$BundleName.zip"
$UnixZip    = Join-Path $DistDir "$UnixName.zip"

Write-Host "[i] Building $BundleName" -ForegroundColor Cyan

if (Test-Path $BundleDir)  { Remove-Item -Recurse -Force $BundleDir }
if (Test-Path $ZipPath)    { Remove-Item -Force $ZipPath }
if (Test-Path $ScratchDir) { Remove-Item -Recurse -Force $ScratchDir }
if (Test-Path $UnixDir)    { Remove-Item -Recurse -Force $UnixDir }
if (Test-Path $UnixZip)    { Remove-Item -Force $UnixZip }
New-Item -ItemType Directory -Force -Path $ScratchDir | Out-Null
New-Item -ItemType Directory -Force -Path $BundleDir  | Out-Null

# --- 1. Embeddable Python -------------------------------------------------------------
$PyZipUrl  = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PyZipPath = Join-Path $ScratchDir "python-embed.zip"
$PyDir     = Join-Path $BundleDir "python"

# Every download that ends up inside the shipped zip is checked against a digest pinned here,
# so a tampered mirror or proxy fails the build instead of shipping. python.org publishes only
# an MD5 for the embeddable zip (its release-file API has sha256_sum empty for 3.13.x); that
# still rules out substituting a different file, which needs a second preimage. A new
# -PythonVersion needs its entry added from https://www.python.org/api/v2/downloads/release_file/.
$PythonEmbedMd5 = @{
    "3.13.7" = "77f294ec267596827a2ab06e8fa3f18c"
}
function Assert-FileHash([string] $Path, [string] $Algorithm, [string] $Expected) {
    $actual = (Get-FileHash -Path $Path -Algorithm $Algorithm).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Algorithm mismatch for $(Split-Path -Leaf $Path): expected $Expected, got $actual"
    }
}
if (-not $PythonEmbedMd5.ContainsKey($PythonVersion)) {
    throw "No pinned digest for embeddable Python $PythonVersion. Add it to `$PythonEmbedMd5 in this script."
}

Write-Host "[i] Downloading embeddable Python $PythonVersion..." -ForegroundColor DarkGray
Invoke-WebRequest -Uri $PyZipUrl -OutFile $PyZipPath
Assert-FileHash $PyZipPath "MD5" $PythonEmbedMd5[$PythonVersion]
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
# Bootstrapped from a pinned pip wheel rather than get-pip.py: that script is unversioned and
# changes under the same URL, so there is no digest to pin. pip runs straight from its own
# wheel on sys.path and installs that same verified wheel (the embeddable Python has no ensurepip).
$PipWheelName = "pip-26.2.1-py3-none-any.whl"
$PipWheelUrl  = "https://files.pythonhosted.org/packages/f3/6e/1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/$PipWheelName"
$PipWheelSha256 = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
$PipWheelPath = Join-Path $ScratchDir $PipWheelName
Write-Host "[i] Bootstrapping pip..." -ForegroundColor DarkGray
Invoke-WebRequest -Uri $PipWheelUrl -OutFile $PipWheelPath
Assert-FileHash $PipWheelPath "SHA256" $PipWheelSha256
& $PyExe -c "import sys; sys.path.insert(0, sys.argv[1]); from pip._internal.cli.main import main; sys.exit(main(['install', '--no-index', '--no-warn-script-location', sys.argv[1]]))" $PipWheelPath
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

Write-Host "[i] Installing CPU-only torch..." -ForegroundColor DarkGray
& $PyExe -m pip install --no-cache-dir --no-warn-script-location torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# The PyTorch CPU index serves an old setuptools as a torch dependency (78.1.0 in the v1.2 build,
# which carries known advisories in its package-download and sdist code). The app never imports
# those paths, but a shipped interpreter should not carry flagged packages: take PyPI's current one.
& $PyExe -m pip install --no-cache-dir --no-warn-script-location --upgrade setuptools
if ($LASTEXITCODE -ne 0) { throw "setuptools upgrade failed" }

Write-Host "[i] Installing the rest of requirements-customer.txt..." -ForegroundColor DarkGray
& $PyExe -m pip install --no-cache-dir --no-warn-script-location -r (Join-Path $RepoRoot "requirements-customer.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# pip's console-script launchers (pip.exe, f2py.exe, ...) hard-code the absolute path of the
# interpreter they were installed against -- this build machine's, username included. They are
# broken once unzipped anywhere else, and run.bat only ever calls `python.exe -m`.
Get-ChildItem -Path (Join-Path $PyDir "Scripts") -Filter "*.exe" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Force $_.FullName }

# --- 3. App source + project marker -----------------------------------------------------
Write-Host "[i] Copying app source (git-tracked files only)..." -ForegroundColor DarkGray
# Only what git tracks under `src/` and `scripts/` ships.  A wholesale copy of the working
# tree carried everything `.gitignore` keeps local -- retired code eras, investigation probes,
# history-hygiene tooling -- into a public zip, and every new local-only folder needed its own
# Remove-Item here to stay out.  The tracked set is exactly what a clean clone's test suite
# runs against, so it is self-contained by construction.
#
# `scripts/` is deploy-reachable despite its name: `src/clot_ml/locked.py`'s readout
# selection (`expected_tuned` / `resid_adapt`, see data/reference/clot_gnn_locked.json's
# "readout" block) lazily imports helpers from `scripts/eval_*.py` at call time.
$Tracked = & git -C $RepoRoot ls-files -- src scripts
if ($LASTEXITCODE -ne 0 -or -not $Tracked) { throw "git ls-files failed; the bundle needs a git checkout" }
foreach ($rel in $Tracked) {
    $dst = Join-Path $BundleDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -Force (Join-Path $RepoRoot $rel) $dst
}
Copy-Item -Force (Join-Path $RepoRoot "pyproject.toml") $BundleDir
# The project README, with the images it embeds, so a downloaded zip explains what the app is and
# what it is validated for. README.txt (written below) stays the short "how to run it" note.
Copy-Item -Force (Join-Path $RepoRoot "README.md") $BundleDir
New-Item -ItemType Directory -Force -Path (Join-Path $BundleDir "docs\assets") | Out-Null
Get-ChildItem -Path (Join-Path $RepoRoot "docs\assets") -Filter "*.png" |
    ForEach-Object { Copy-Item -Force $_.FullName (Join-Path $BundleDir "docs\assets\$($_.Name)") }

# Drop every __pycache__ the wholesale copies dragged in.
Get-ChildItem -Path $BundleDir -Directory -Recurse -Filter "__pycache__" |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }

$InboxDir = Join-Path $BundleDir "customer_geometries"
New-Item -ItemType Directory -Force -Path $InboxDir | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "customer_geometries\README.txt") $InboxDir

# Seed the inbox with a COMSOL anchor (rather than a parametric sweep vessel) so first launch
# shows a prediction immediately instead of an empty dropdown -- comsol041, a stenosis anchor
# the shipped
# clot_ml_0 was actually trained on, so a customer (or we) can sanity-check the model against
# a known-good case with a COMSOL ground truth behind it.  Every vessel in this project is
# synthetic geometry; "anchor" refers to having a COMSOL solve, not to any clinical origin. Trimmed of G_x/G_y/
# Laplacian/z_kin_pred (stale mesh operators + a cached embedding, unused by the deploy path --
# see mls_gradient.py) to cut ~20 MB of dead weight; still ~235 MB because the real 201-step
# species/velocity history (y) is what the deploy pipeline derives its own rollout step count
# from (CustomerDeployPipeline.run()), so it cannot be downsampled without changing the
# demo's timeline resolution.
Copy-Item -Force (Join-Path $RepoRoot "customer_geometries\demo_stenosis_vessel.pt") $InboxDir
$DemoMeshDir = Join-Path $BundleDir "data\raw\biochem_anchors"
New-Item -ItemType Directory -Force -Path $DemoMeshDir | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "data\raw\biochem_anchors\comsol041.msh") $DemoMeshDir

# --- 4. The ~11 MB of checkpoints this tool actually loads ------------------------------
# Traced from CustomerDeployPipeline.run() -> load_v0_bundle("clot_ml_0"): "clot_ml_0" is an
# alias resolved at RUNTIME via the pointer at data/reference/clot_gnn_locked.json ("name"),
# then load_v0_bundle walks that artifact's own base_model chain (v0 -> temporal_v4_wound ->
# temporal_v4). Only that chain is ever opened -- no kinematics or standalone wall-model
# checkpoint is on this path.
#
# DELIBERATELY PINNED BY NAME, not resolved dynamically AT BUILD TIME: copying exactly the
# validated chain below means a bundle built today ships a known-good artifact regardless of
# what's mid-training alongside it in outputs/clot_ml/locked/. This bit us once already --
# 2026-09-04's clot_ml_final_0 promotion superseded the previous clot_ml_v0/clot_gnn_v6 chain
# and this list was not updated, so every build after that date failed with a missing
# directory instead of silently shipping something stale. Re-derive the current chain with:
#   python -c "import json; m=json.load(open('data/reference/clot_gnn_locked.json')); \
#              print(m['name']); n=m['v0']['base_model']
#              while n: mm=json.load(open(f'outputs/clot_ml/locked/{n}/manifest.json')); \
#              print(n); n=(mm.get('v0') or {}).get('base_model') or mm.get('base_model')"
# and bump the list below once the new chain is promoted AND validated (not merely present).
Write-Host "[i] Copying clot_ml_0 checkpoints..." -ForegroundColor DarkGray
$CkptSrc = Join-Path $RepoRoot "outputs\clot_ml\locked"
$CkptDst = Join-Path $BundleDir "outputs\clot_ml\locked"
foreach ($sub in @("clot_ml_final_0", "clot_ml_final_w", "clot_ml_final")) {
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
    echo this whole folder somewhere short first, e.g. C:\ThrombosisSurrogate\, then run
    echo run.bat again from there.
    echo.
)
echo Starting 2D Thrombosis Surrogate Predict...
echo A browser tab will open automatically once it's ready. This window logs progress -- leave it open while you use the app, close it when you're done.
"%~dp0python\python.exe" -m src.tools.customer_predict_web --cpu
pause
'@
Set-Content -Path (Join-Path $BundleDir "run.bat") -Value $RunBat -Encoding ASCII

$ReadmeTxt = @'
2D Thrombosis Surrogate -- Predict
============================

Before you run it: unzip this to somewhere with a SHORT path, close to a drive root --
for example C:\ThrombosisSurrogate\ -- rather than deep inside Downloads or a synced folder.
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

# --- 6. macOS / Linux bundle ---------------------------------------------------------------
# Everything except the Windows interpreter and launcher. The lock file pins the exact versions
# the Windows build just installed and was tested with; `+cpu` is dropped so macOS resolves
# torch from PyPI (run_unix.sh fetches the CPU wheel explicitly on Linux).
Write-Host "[i] Assembling the macOS / Linux bundle..." -ForegroundColor DarkGray
New-Item -ItemType Directory -Force -Path $UnixDir | Out-Null
Get-ChildItem -Path $BundleDir | Where-Object { $_.Name -notin @("python", "run.bat", "README.txt") } |
    ForEach-Object { Copy-Item -Recurse -Force $_.FullName (Join-Path $UnixDir $_.Name) }
$Lock = & $PyExe -m pip freeze --all 2>$null | Where-Object { $_ -match '==' -and $_ -notmatch '^pip==' } |
    ForEach-Object { $_ -replace '\+cpu$', '' }
if ($LASTEXITCODE -ne 0 -or -not $Lock) { throw "pip freeze failed" }
[System.IO.File]::WriteAllText((Join-Path $UnixDir "requirements-lock.txt"), (($Lock -join "`n") + "`n"))
$Launcher = [System.IO.File]::ReadAllBytes((Join-Path $PSScriptRoot "bundle\run_unix.sh"))
if ($Launcher -contains 13) { throw "scripts/bundle/run_unix.sh has CRLF line endings; bash cannot run it" }
[System.IO.File]::WriteAllBytes((Join-Path $UnixDir "run.command"), $Launcher)
[System.IO.File]::WriteAllBytes((Join-Path $UnixDir "run.sh"), $Launcher)

$ReadmeUnix = @'
2D Thrombosis Surrogate -- Predict (macOS / Linux)
==================================================

Needs: an Apple Silicon (M-series) Mac, or an x86-64 Linux PC, with Python 3.12, 3.13 or
3.14 installed (https://www.python.org/downloads/). Intel Macs are not supported.

macOS:
  1. Double-click run.command.
     If macOS says it "cannot be opened because it is from an unidentified developer",
     right-click (or Control-click) run.command, choose Open, then Open again.
  2. A Terminal window opens. The FIRST launch downloads about 1 GB of packages and takes
     several minutes; later launches start in seconds.
  3. Your web browser opens the app when it is ready. Leave the Terminal window open while
     you use it; close it to stop the app.

Linux:
  In a terminal, in this folder:   ./run.sh
  Debian/Ubuntu may first need:
      sudo apt install python3-venv libglu1-mesa libopengl0 libxft2 libgomp1
  (run.sh tells you if anything is missing).

Runs entirely on your own computer: your vessels never leave your machine. After the first
launch no internet connection is needed, and no GPU is needed (it runs on CPU).
'@
[System.IO.File]::WriteAllText((Join-Path $UnixDir "README.txt"), ($ReadmeUnix -replace "`r`n", "`n"))

# --- 6. Zip --------------------------------------------------------------------------------
# scripts/bundle/make_zip.py rather than Compress-Archive: it records Unix mode bits, without
# which macOS unzips run.command non-executable and Finder will not open it.
$MakeZip = Join-Path $PSScriptRoot "bundle\make_zip.py"
foreach ($pair in @(@($BundleDir, $ZipPath), @($UnixDir, $UnixZip))) {
    Write-Host "[i] Zipping to $($pair[1])..." -ForegroundColor DarkGray
    & $PyExe $MakeZip $pair[0] $pair[1]
    if ($LASTEXITCODE -ne 0) { throw "zipping $($pair[0]) failed" }
}

Remove-Item -Recurse -Force $ScratchDir

foreach ($z in @($ZipPath, $UnixZip)) {
    $SizeMb = [math]::Round((Get-Item $z).Length / 1MB, 1)
    Write-Host "[OK] Built $z ($SizeMb MB)" -ForegroundColor Green
}
