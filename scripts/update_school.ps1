[CmdletBinding()]
param([string]$Directory = '', [switch]$Plan, [switch]$NoShortcut)
$ErrorActionPreference = 'Stop'
if (-not $Directory) {
    if (Test-Path -LiteralPath (Join-Path $PWD 'run_sam3.bat')) { $Directory = $PWD.Path }
    else { $Directory = Join-Path $env:LOCALAPPDATA 'SammieRoto/Sammie-Roto-School' }
}
$Root = [IO.Path]::GetFullPath($Directory)
$Python = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "No installed application at $Root. Specify -Directory with its actual folder." }
Write-Host "Updating $Root. Close Sammie before running an update."
if ($Plan) { Write-Host 'Plan only: download the latest verified source, preserve local data, then reconcile dependencies.'; return }
$UpdateLock = $null
$InstallLock = $null
try {
    $Runtime = Join-Path $Root '.runtime'
    New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
    $UpdateLock = [IO.File]::Open((Join-Path $Runtime 'school-update.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $InstallLock = [IO.File]::Open((Join-Path $Runtime 'school-install.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $PortableGit = Join-Path $Runtime 'school-tools/git/cmd'
    if (Test-Path -LiteralPath $PortableGit) { $env:PATH = $PortableGit + ';' + $env:PATH }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    $Updater = Join-Path $Runtime ('school-update-' + [Guid]::NewGuid().ToString('N') + '.py')
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download/update_school.py' -OutFile $Updater
    & $Python $Updater --root $Root
    if ($LASTEXITCODE -ne 0) { throw 'Source update stopped. Resolve the error above; no forced reset was attempted.' }
    Remove-Item -LiteralPath $Updater
    $InstallLock.Dispose()
    $InstallLock = $null
    Set-Location -LiteralPath $Root
    $Options = @{Models='OnDemand'; NoLogin=$true}
    if ($NoShortcut) { $Options.NoShortcut=$true }
    $LASTEXITCODE = 0
    & (Join-Path $Root 'scripts/install_school.ps1') @Options
    if ($LASTEXITCODE -ne 0) { throw 'Source updated, but dependencies need repair. Rerun install_school.bat -Models OnDemand.' }
    Write-Host 'Update complete. Launch run_sam3.bat.'
} catch {
    Write-Host ("UPDATE STOPPED: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    if ($InstallLock) { $InstallLock.Dispose() }
    if ($UpdateLock) { $UpdateLock.Dispose() }
}
