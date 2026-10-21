[CmdletBinding()]
param([ValidateSet('cu130','cu126','cpu')][string]$DeviceExtra = 'cu130')
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Root
$env:UV_CACHE_DIR = Join-Path $Root '.uv/cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.uv/python'
$Uv = Join-Path $Root '.uv/uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) {
    $AvailableUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($AvailableUv) { $Uv = $AvailableUv.Source }
    else { throw 'Install Sammie first using install_sam3.bat, then run this installer.' }
}
& $Uv sync --frozen --extra $DeviceExtra --extra vitmatte
if ($LASTEXITCODE -ne 0) { throw 'ViTMatte environment installation failed.' }
& '.venv/Scripts/python.exe' -c 'from transformers import VitMatteForImageMatting'
if ($LASTEXITCODE -ne 0) { throw 'ViTMatte import validation failed.' }
Write-Host 'ViTMatte installed. Launch run_sam3.bat; weights download on first refinement.'
