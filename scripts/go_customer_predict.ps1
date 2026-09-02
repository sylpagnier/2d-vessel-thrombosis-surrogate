# Local FEM Solver Customer Predict App
#   powershell ... -File .\scripts\go_customer_predict.ps1 [-Cpu]

param([switch] $Cpu)

. (Join-Path $PSScriptRoot "_launcher_common.ps1")
Invoke-GoCustomerPredict -Cpu:$Cpu
