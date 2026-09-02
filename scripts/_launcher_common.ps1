# Shared PowerShell launcher helpers for active Local FEM Solver scripts.
# Dot-source from any go_*.ps1:
#   . (Join-Path $PSScriptRoot "_launcher_common.ps1")
#   $RepoRoot = Initialize-HemoRepo -ScriptRoot $PSScriptRoot

function Initialize-HemoRepo {
    param([string] $ScriptRoot)
    $ErrorActionPreference = "Stop"
    $repoRoot = Split-Path -Parent $ScriptRoot
    Set-Location $repoRoot
    . (Join-Path $ScriptRoot "_python_rc.ps1")
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    return $repoRoot
}

function Invoke-HemoPython {
    param(
        [Parameter(Mandatory = $true)][string[]] $Args,
        [switch] $AllowCpu,
        [string] $CpuWarn = "[WARN] CPU mode (slow). CUDA is recommended."
    )
    if ($AllowCpu -and $Args -contains "--cpu") {
        Write-Host $CpuWarn -ForegroundColor Yellow
    }
    & python -u @Args
    if ($null -ne $LASTEXITCODE) { return [int]$LASTEXITCODE }
    return 0
}

function Exit-GoLauncher {
    param(
        [int] $Rc = 0,
        [string] $Label = "launcher"
    )
    if ($Rc -ne 0) {
        Write-Host "[ERR] $Label exited $Rc" -ForegroundColor Red
        exit $Rc
    }
    Write-Host "[OK] Done" -ForegroundColor Green
    exit 0
}

function Write-GoBanner {
    param(
        [string] $Title,
        [string[]] $InfoLines = @()
    )
    if ($Title) {
        Write-Host "[i] $Title" -ForegroundColor Cyan
    }
    foreach ($line in $InfoLines) {
        Write-Host "[i] $line" -ForegroundColor DarkGray
    }
}

function Invoke-GoPythonModule {
    param(
        [string] $Module,
        [string[]] $ExtraArgs = @(),
        [switch] $Cpu,
        [switch] $DirectPython,
        [string] $Label = $Module
    )
    $pyArgs = @("-m", $Module) + $ExtraArgs
    if ($Cpu) { $pyArgs += "--cpu" }
    if ($DirectPython) {
        & python -u @pyArgs
        $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    } else {
        $rc = Invoke-HemoPython -Args $pyArgs -AllowCpu:$Cpu
    }
    Exit-GoLauncher -Rc $rc -Label $Label
}

function Invoke-GoPythonScript {
    param(
        [string] $Script,
        [string[]] $PyArgs = @(),
        [switch] $Cpu,
        [string] $Label = $Script
    )
    $all = @($Script) + $PyArgs
    if ($Cpu) { $all += "--cpu" }
    $rc = Invoke-HemoPython -Args $all -AllowCpu:$Cpu
    Exit-GoLauncher -Rc $rc -Label $Label
}

function Ensure-RepoSubdir {
    param(
        [string] $RepoRoot,
        [string] $RelativePath
    )
    $path = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
    return $path
}

function Invoke-GoCustomerPredict {
    param([switch] $Cpu)
    $repo = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    $inbox = Ensure-RepoSubdir -RepoRoot $repo -RelativePath "customer_geometries"
    Write-GoBanner -Title "Local FEM Solver Predict" -InfoLines @(
        "Deploy: clot_ml_0 (RGP-DEQ t=0 + C0-tail GNN)",
        "Geometries folder: $inbox",
        "Use Open folder or Browse (starts in that folder)"
    )
    Invoke-GoPythonModule -Module "src.tools.customer_predict_app" -Cpu:$Cpu -DirectPython -Label "customer_predict"
}

function Invoke-GoCustomerPredictWeb {
    param(
        [switch] $Cpu,
        [int] $Port = 8765,
        [string] $BindHost = "127.0.0.1"
    )
    $null = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    Write-GoBanner -Title "Local FEM Solver web predict: http://${BindHost}:${Port}" -InfoLines @(
        "Deploy: clot_ml_0 (RGP-DEQ t=0 + C0-tail GNN)",
        "Open the URL above in a browser; press Ctrl+C to stop."
    )
    Invoke-GoPythonModule -Module "src.tools.customer_predict_web" `
        -ExtraArgs @("--host", $BindHost, "--port", "$Port") `
        -Cpu:$Cpu -Label "customer_predict_web"
}

function Invoke-GoResearchSweep {
    param(
        [string] $Sweep = "",
        [switch] $All,
        [switch] $List,
        [switch] $Legacy,
        [string] $Arm = "",
        [switch] $ForceRebuildMesh,
        [switch] $Cpu,
        [string] $WallCkpt = "",
        [string] $MatLeg = ""
    )
    $null = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    Write-GoBanner -Title "Research geometry-sensitivity sweeps" -InfoLines @(
        "Default model: clot_ml_0 + FEM t=0 flow"
    )
    $pyArgs = @("scripts/run_research_sweep.py")
    if ($List) {
        $pyArgs += "--list"
        if ($Legacy) { $pyArgs += "--legacy" }
    }
    elseif ($All) {
        $pyArgs += "--all"
        if ($Legacy) { $pyArgs += "--legacy" }
    }
    elseif ($Sweep.Trim()) {
        $pyArgs += @("--sweep", $Sweep.Trim())
        if ($Legacy) { $pyArgs += "--legacy" }
    }
    else {
        Write-Host "[ERR] Pass -Sweep <id>, -All, or -List" -ForegroundColor Red
        Write-Host "[i] Example: -Sweep 01_stenosis_strength" -ForegroundColor DarkGray
        exit 2
    }
    if ($Arm.Trim()) { $pyArgs += @("--arm", $Arm.Trim()) }
    if ($ForceRebuildMesh) { $pyArgs += "--force-rebuild-mesh" }
    if ($Cpu) { $pyArgs += "--cpu" }
    if ($WallCkpt.Trim()) { $pyArgs += @("--wall-ckpt", $WallCkpt.Trim()) }
    if ($MatLeg.Trim()) { $pyArgs += @("--mat-leg", $MatLeg.Trim()) }
    $rc = Invoke-HemoPython -Args $pyArgs -AllowCpu:$Cpu
    Exit-GoLauncher -Rc $rc -Label "run_research_sweep"
}

function Invoke-GoDiagnostics {
    param(
        [Parameter(Mandatory = $true)][string] $Slug,
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $DiagArgs
    )
    $null = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    $pyArgs = @("-m", "src.tools.diagnostics", $Slug) + $DiagArgs
    $rc = Invoke-HemoPython -Args $pyArgs
    Exit-GoLauncher -Rc $rc -Label "diag:$Slug"
}

function Invoke-GoKinematicsProduction {
    param([string[]] $PyArgs = @("scripts/run_kinematics_production.py"))
    $null = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    Write-Host "[i] Stage-A kinematics production" -ForegroundColor Cyan
    Write-Host "[i] Orchestrator: scripts/run_kinematics_production.py" -ForegroundColor DarkGray
    $rc = Invoke-HemoPython -Args $PyArgs
    Exit-GoLauncher -Rc $rc -Label "kinematics_production"
}
