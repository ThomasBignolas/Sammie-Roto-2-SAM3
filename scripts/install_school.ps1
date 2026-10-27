[CmdletBinding()]
param(
    [ValidateSet('All','Essential','OnDemand')][string]$Models = 'All',
    [switch]$SkipSamCheckpoints,
    [switch]$NoLogin,
    [switch]$NoShortcut,
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $Root
$Steps = @(
    'Install local uv and portable Git (if missing); download Python 3.12',
    'Install locked application + SAM 3 runtime; test SAM CUDA/Triton',
    'Install ViTMatte, AOV and CorridorKey runtimes with their pinned sources'
)
if ($Models -ne 'OnDemand') {
    if (-not $SkipSamCheckpoints) { $Steps += 'Download authorized SAM 3 and SAM 3.1 checkpoints (hidden HF login if needed)' }
    $Steps += "Download $Models model set: segmentation, matting, ViTMatte, depth, normals, foreground RGB"
}
$Steps += 'Check all runtimes and CUDA; create desktop launch shortcut unless disabled'
Write-Host 'Sammie-Roto - school installation (Windows x64 / NVIDIA)'
for ($i=0; $i -lt $Steps.Count; $i++) { Write-Host ("{0}. {1}" -f ($i+1), $Steps[$i]) }
if ($Plan) { return }

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Executable. Fix the error above and rerun the installer." }
}
function Install-Tool($Spec) {
    $Directory = Join-Path $Root $Spec.directory
    $Executable = Join-Path $Directory $Spec.executable
    if (Test-Path -LiteralPath $Executable) { return $Executable }
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $Archive = Join-Path $Directory 'installer-download.zip'
    Write-Host "Downloading $($Spec.url)"
    # TLS policy and progress affect this installer process only.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $SavedProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Spec.url -OutFile $Archive
        if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Spec.sha256) {
            throw 'Installer download checksum mismatch; no downloaded program was run. Rerun to download again.'
        }
        Expand-Archive -LiteralPath $Archive -DestinationPath $Directory -Force
    } finally { $ProgressPreference = $SavedProgress }
    if (-not (Test-Path -LiteralPath $Executable)) { throw "Downloaded tool is missing: $Executable" }
    Remove-Item -LiteralPath $Archive
    return $Executable
}

$InstallLock = $null
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This installer requires 64-bit Windows.' }
    if ($Root.StartsWith('\\')) { throw 'Extract to a local writable drive first, for example D:\Sammie-Roto-School. Do not install on a network share.' }
    New-Item -ItemType Directory -Path (Join-Path $Root '.runtime') -Force | Out-Null
    try {
        $InstallLock = [IO.File]::Open((Join-Path $Root '.runtime/school-install.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { throw 'Another school installer is using this folder, or it is not writable. Close the other installer and retry.' }
    $Drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($Root).Substring(0,1))
    if ($null -ne $Drive.Free) { Write-Host ("Free space: {0:N1} GiB. Full model setup downloads tens of GB; allow 100 GB or more including environments/cache, plus shot storage." -f ($Drive.Free / 1GB)) }
    Write-Host 'Close Sammie before installation. No administrator access or system Python changes are needed.'
    Write-Host 'SAM access: https://huggingface.co/facebook/sam3 and https://huggingface.co/facebook/sam3.1'
    $ToolSpecs = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'school-tools.json') -Raw | ConvertFrom-Json
    $Uv = Install-Tool $ToolSpecs.uv
    # Use the school's existing Git when available; otherwise bootstrap a local copy.
    $Git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $Git) {
        $GitExe = Install-Tool $ToolSpecs.git
        $env:PATH = (Split-Path $GitExe -Parent) + ';' + $env:PATH
    }
    Invoke-Checked $Uv @('--version')
    Invoke-Checked 'git' @('--version')
    & (Join-Path $PSScriptRoot 'install_sam3.ps1')
    & (Join-Path $PSScriptRoot 'install_vitmatte.ps1')
    & (Join-Path $PSScriptRoot 'install_aov.ps1')
    & (Join-Path $PSScriptRoot 'install_corridorkey.ps1') -SourcesOnly

    # Optional local joke requested for this school build. Never persist the
    # answer, and never block unattended/non-interactive installations.
    if (-not $NoLogin -and $env:SAMMIE_UNATTENDED -ne '1' -and
        [Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and
        (Get-Random -Minimum 0 -Maximum 4) -eq 0) {
        $null = Read-Host 'Taille du sexe ?'
    }

    $AppPython = Join-Path $Root '.venv/Scripts/python.exe'
    if ($Models -ne 'OnDemand') {
        if (-not $SkipSamCheckpoints) {
            $LoginArgs = @('-m','scripts.setup_models','--model','both','--login-if-needed')
            if ($NoLogin) { $LoginArgs += '--no-login' }
            Invoke-Checked (Join-Path $Root '.runtime/sam3-env/Scripts/python.exe') $LoginArgs
        }
        Invoke-Checked $AppPython @('-m','scripts.setup_school_models','--models',$Models.ToLowerInvariant())
        Invoke-Checked (Join-Path $Root '.runtime/aov-env/Scripts/python.exe') @('-m','scripts.setup_aov')
        Invoke-Checked (Join-Path $Root '.runtime/corridorkey-env/Scripts/python.exe') @('-m','scripts.setup_corridorkey','--models','blue','green')
    }
    Invoke-Checked $AppPython @('-m','scripts.check_school')
    if (-not $NoShortcut) {
        try {
            $Desktop = [Environment]::GetFolderPath('Desktop')
            $Shell = New-Object -ComObject WScript.Shell
            $Shortcut = $Shell.CreateShortcut((Join-Path $Desktop 'Sammie-Roto School.lnk'))
            $Shortcut.TargetPath = Join-Path $Root 'run_sam3.bat'
            $Shortcut.WorkingDirectory = $Root
            $Shortcut.Description = 'Sammie-Roto School - SAM 3, ViTMatte and AOVs'
            $Shortcut.Save()
        } catch { Write-Warning 'Desktop shortcut could not be created. Use run_sam3.bat in this folder.' }
    }
    Write-Host ''
    Write-Host 'Installation checks passed. Launch run_sam3.bat (or the Sammie-Roto School desktop shortcut).'
    if (($Models -eq 'OnDemand' -or $SkipSamCheckpoints) -and
        (-not (Test-Path -LiteralPath (Join-Path $Root 'checkpoints/sam3.pt')) -or
         -not (Test-Path -LiteralPath (Join-Path $Root 'checkpoints/sam3.1_multiplex.pt')))) {
        Write-Host 'SAM checkpoints were skipped. Run setup_models.bat for missing SAM 3/3.1 weights.'
    }
    if ($Models -ne 'All') { Write-Host 'Some model weights download on first use. Rerun install_school.bat -Models All to prefetch every supported model.' }
} catch {
    Write-Host ("INSTALLATION STOPPED: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    if ($InstallLock) { $InstallLock.Dispose() }
}
