[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'SammieRoto'),
    [ValidateSet('All','Essential','OnDemand')][string]$Models = 'All',
    [switch]$SkipSamCheckpoints,
    [switch]$NoLogin,
    [switch]$NoShortcut,
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$Release = 'https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download'
$Destination = [IO.Path]::GetFullPath($Destination)
$AppRoot = Join-Path $Destination 'Sammie-Roto-School'
Write-Host "School release: $Release"
Write-Host "Install folder: $AppRoot"
if ($Plan) { return }
try {
    if ($Destination.StartsWith('\\')) { throw 'Choose a local writable drive, not a network share.' }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    $ChecksumContent = (Invoke-WebRequest -UseBasicParsing -Uri "$Release/Sammie-Roto-School.zip.sha256").Content
    if ($ChecksumContent -is [byte[]]) { $ChecksumContent = [Text.Encoding]::UTF8.GetString($ChecksumContent) }
    $Expected = ([string]$ChecksumContent).Trim().Split(' ')[0].ToLowerInvariant()
    if ($Expected -notmatch '^[0-9a-f]{64}$') { throw 'Invalid release checksum file.' }
    $Marker = Join-Path $AppRoot '.runtime/school-package.sha256'
    if (Test-Path -LiteralPath $AppRoot) {
        if (-not (Test-Path -LiteralPath $Marker) -or (Get-Content -LiteralPath $Marker -Raw).Trim() -ne $Expected) {
            throw "An existing installation is at $AppRoot. No files overwritten. Use update_school_online.bat -Directory with this folder to update, or choose another -Destination for a fresh installation."
        }
        Write-Host 'Reusing this release in the existing folder; resuming dependency/model setup.'
    } else {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        $Archive = Join-Path $Destination ('sammie-download-' + [Guid]::NewGuid().ToString('N') + '.zip')
        Write-Host 'Downloading the application source package...'
        Invoke-WebRequest -UseBasicParsing -Uri "$Release/Sammie-Roto-School.zip" -OutFile $Archive
        if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
            throw 'Release checksum mismatch. No application files were extracted or executed.'
        }
        # The package must contain only its one app directory. Reject traversal
        # before extraction, even if a future release is packaged incorrectly.
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $Zip = [IO.Compression.ZipFile]::OpenRead($Archive)
        try {
            foreach ($Entry in $Zip.Entries) {
                $Name = $Entry.FullName.Replace('\','/')
                if (-not $Name.StartsWith('Sammie-Roto-School/') -or $Name.Split('/') -contains '..' -or $Name.Contains(':')) {
                    throw 'Unsafe path in source archive; extraction refused.'
                }
            }
        } finally { $Zip.Dispose() }
        Expand-Archive -LiteralPath $Archive -DestinationPath $Destination
        if (-not (Test-Path -LiteralPath (Join-Path $AppRoot 'scripts/install_school.ps1'))) { throw 'Release is missing its installer.' }
        New-Item -ItemType Directory -Path (Split-Path $Marker -Parent) -Force | Out-Null
        Set-Content -LiteralPath $Marker -Value $Expected -Encoding ASCII
        Remove-Item -LiteralPath $Archive
    }
    $Options = @{ Models = $Models }
    if ($NoLogin) { $Options.NoLogin = $true }
    if ($NoShortcut) { $Options.NoShortcut = $true }
    if ($SkipSamCheckpoints) { $Options.SkipSamCheckpoints = $true }
    $LASTEXITCODE = 0
    & (Join-Path $AppRoot 'scripts/install_school.ps1') @Options
    if ($LASTEXITCODE -ne 0) { throw 'Application setup failed. Fix the reported error and rerun this command to resume.' }
} catch {
    Write-Host ("SETUP STOPPED: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
