# Local FEM Solver customer browser predict UI
#   powershell ... -File .\scripts\go_customer_predict_web.ps1 [-Cpu] [-Port 8765]

param(
    [switch] $Cpu,
    [int] $Port = 8765,
    [string] $BindHost = "127.0.0.1"
)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
Invoke-GoCustomerPredictWeb -Cpu:$Cpu -Port $Port -BindHost $BindHost
