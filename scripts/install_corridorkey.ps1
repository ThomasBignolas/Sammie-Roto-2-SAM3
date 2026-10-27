[CmdletBinding()]
param([switch]$SourcesOnly)
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
    if ($LASTEXITCODE -ne 0) { throw "CorridorKey setup failed: $Executable (exit $LASTEXITCODE)" }
}
$Worker = Join-Path $Root '.runtime/corridorkey-env'
$WorkerPython = Join-Path $Worker 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $WorkerPython)) {
    Invoke-Checked $Uv @('venv','--python',(Join-Path $Root '.venv/Scripts/python.exe'),$Worker)
}
Invoke-Checked $Uv @('pip','sync','--python',$WorkerPython,'requirements-corridorkey.lock','--extra-index-url','https://download.pytorch.org/whl/cu128','--index-strategy','unsafe-best-match')
if ($SourcesOnly) {
    Invoke-Checked $WorkerPython @('-m','scripts.setup_corridorkey','--sources-only')
} else {
    Invoke-Checked $WorkerPython @('-m','scripts.setup_corridorkey','--models','blue')
}
Write-Host 'CorridorKey ready. Launch run_sam3.bat, then AOVs > CorridorKey Foreground RGB.'
Write-Host 'Green weights download on first use. No footage is uploaded.'
