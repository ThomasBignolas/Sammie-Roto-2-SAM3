[CmdletBinding()]
param([switch]$WorkerOnly)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Root
$env:UV_CACHE_DIR = Join-Path $Root '.uv/cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.uv/python'
$Uv = Join-Path $Root '.uv/uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) {
    $AvailableUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($AvailableUv) { $Uv = $AvailableUv.Source }
    else {
        New-Item -ItemType Directory -Path (Join-Path $Root '.uv') -Force | Out-Null
        $Archive = Join-Path $Root '.uv/uv.zip'
        Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.12.5/uv-x86_64-pc-windows-msvc.zip' -OutFile $Archive
        Expand-Archive -LiteralPath $Archive -DestinationPath (Join-Path $Root '.uv') -Force
    }
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Installation command failed: $Executable (exit $LASTEXITCODE)" }
}
Invoke-Checked $Uv @('python', 'install', '--no-bin', '3.12')
if (-not $WorkerOnly) {
    Invoke-Checked $Uv @('sync', '--frozen', '--extra', 'cu130')
}
$Worker = Join-Path $Root '.runtime/sam3-env'
$WorkerPython = Join-Path $Worker 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $WorkerPython)) {
    Invoke-Checked $Uv @('venv', '--python', '3.12', $Worker)
}
Invoke-Checked $Uv @('pip', 'sync', '--python', $WorkerPython, 'requirements-sam3.lock', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match')
$Commit = '7f473cfd983d5eb43040ff608f66bff0d7bdcb53'
Invoke-Checked $Uv @('pip', 'install', '--python', $WorkerPython, '--no-deps', "sam3 @ git+https://github.com/facebookresearch/sam3.git@$Commit")
Invoke-Checked $WorkerPython @('-m', 'sammie.backends.check_runtime')
Write-Host 'Installation complete. Run setup_models.bat to obtain authorized checkpoints, then run_sam3.bat.'
