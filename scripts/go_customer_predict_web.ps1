# HemoRGP customer browser predict UI
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_customer_predict_web.ps1
#   powershell ... -File .\scripts\go_customer_predict_web.ps1 -Cpu

param(
    [switch] $Cpu,
    [int] $Port = 8765,
    [string] $Host = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$pyArgs = @("-u", "-m", "src.tools.customer_predict_web", "--host", $Host, "--port", "$Port")
if ($Cpu) {
    $pyArgs += "--cpu"
    Write-Host "[WARN] CPU mode (slow). CUDA is recommended." -ForegroundColor Yellow
}

Write-Host "[i] HemoRGP web predict: http://${Host}:${Port}" -ForegroundColor Cyan
Write-Host "[i] Open the URL above in a browser; press Ctrl+C to stop." -ForegroundColor DarkGray
& python @pyArgs
$rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
if ($rc -ne 0) {
    Write-Host "[ERR] customer_predict_web exited $rc" -ForegroundColor Red
    exit $rc
}
exit 0
