[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Root
$env:UV_CACHE_DIR = Join-Path $Root '.uv/cache'
$Uv = Join-Path $Root '.uv/uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) {
    $AvailableUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($AvailableUv) { $Uv = $AvailableUv.Source }
    else { throw 'Run install_sam3.bat first to install uv and Python.' }
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AOV setup failed: $Executable (exit $LASTEXITCODE)" }
}
$Worker = Join-Path $Root '.runtime/aov-env'
$WorkerPython = Join-Path $Worker 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $WorkerPython)) {
    Invoke-Checked $Uv @('venv', '--python', (Join-Path $Root '.venv/Scripts/python.exe'), $Worker)
}
Invoke-Checked $Uv @('pip', 'sync', '--python', $WorkerPython, 'requirements-aov.lock', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match')
Invoke-Checked $WorkerPython @('-m', 'scripts.setup_aov', '--sources-only')
Write-Host 'AOV environment ready. Models download automatically on first Generate, or run setup_aov_models.bat.'
