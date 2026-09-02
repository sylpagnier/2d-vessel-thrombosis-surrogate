# Unified diagnostic probes (see src/tools/diagnostics/registry.py).
#
#   powershell ... -File .\scripts\go_diag.ps1 -List
#   powershell ... -File .\scripts\go_diag.ps1 -Slug wound-p003-causes

param(
    [switch] $List,
    [string] $Slug = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]] $DiagArgs
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
if ($List) {
    $null = Initialize-HemoRepo -ScriptRoot $PSScriptRoot
    $rc = Invoke-HemoPython -Args @("-m", "src.tools.diagnostics", "list")
    Exit-GoLauncher -Rc $rc -Label "diag list"
}
if (-not $Slug.Trim()) {
    Write-Host "[ERR] Pass -Slug <name> or -List" -ForegroundColor Red
    Write-Host "[i] Example: -Slug clot-free-headroom --cache smoke" -ForegroundColor DarkGray
    exit 2
}
Invoke-GoDiagnostics -Slug $Slug.Trim() @DiagArgs
