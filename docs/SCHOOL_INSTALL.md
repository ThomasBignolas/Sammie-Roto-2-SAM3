# School installation (Windows / NVIDIA)

The GitHub repository and release are **public**, at the owner's request.
The application is intended for school use. Staff can also share the source ZIP
through an internal share or USB drive. No GitHub login is needed to download it. Dependencies and models download directly from their official providers.
No personal footage, projects, account credentials or downloaded model weights
are included. The two small original upstream example clips are retained.

## One command in CMD

Open a normal CMD window and paste:

```bat
curl.exe -fL https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download/install_school_online.bat -o "%TEMP%\install-sammie.bat" && "%TEMP%\install-sammie.bat"
```

This downloads the release, verifies its ZIP SHA-256, extracts it into
`%LOCALAPPDATA%\SammieRoto\Sammie-Roto-School`, and runs the full installer.
The hidden Hugging Face token prompt still requires your approved account.
An existing, different installation is never overwritten. Rerunning the command
resumes this release when its package marker matches. To choose a drive, append
`-Destination D:\SchoolApps` to the final command; the application is installed
inside `D:\SchoolApps\Sammie-Roto-School`. `-Models`, `-NoLogin`, `-NoShortcut`
and `-SkipSamCheckpoints` are also supported.

## ZIP installation

1. Download **Sammie-Roto-School.zip** from the
   [latest school release](https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest),
   or obtain that ZIP from your school administrator.
2. Extract it to a **local writable folder**, for example `D:\Sammie-Roto-School`.
   Avoid Program Files, network shares and very long folder paths. Do not run
   from inside the ZIP. Keep the application in that folder after installation.
3. Double-click **install_school.bat**, or open CMD in the extracted folder and run:

   ```bat
   install_school.bat
   ```

4. When asked, paste a Hugging Face **read** token in the hidden local prompt.
   Use an account approved for both [SAM 3](https://huggingface.co/facebook/sam3)
   and [SAM 3.1](https://huggingface.co/facebook/sam3.1).
   Create the token at [Hugging Face settings](https://huggingface.co/settings/tokens).
   The installer reuses an existing HF login. The token is saved by HF in the
   Windows user's profile, **not** in the project, package or Git credentials.
   Do not distribute a personal token with the ZIP. A school-managed account
   must have its own authorized access; installing our code does not grant access.
5. Launch **run_sam3.bat** or the **Sammie-Roto School** desktop shortcut.

Git, Python and pip do not have to be installed beforehand. The installer uses
local tools and environments; it does not need administrator access or change
system Python/PATH. It cannot install a GPU driver or bypass school IT policies.
If the school blocks PowerShell, executable downloads or GPU drivers, IT needs
to allow those prerequisites.

**Hardware:** 64-bit Windows, a current NVIDIA driver compatible with CUDA 13,
and an RTX 30-series or newer GPU with BF16 support for the complete SAM 3 stack.
16 GB VRAM is the tested target. Smaller cards require lower settings and may
not fit every model. Allow **100 GB or more** for all models, environments and
installer caches, plus separate room for footage and generated sequences.
This is a planning allowance, not a measured maximum. Setup downloads tens of
GB; use a reliable connection and expect a substantial first installation.

## What the single installer does

- Downloads checksum-verified uv and, if Git is missing, portable MinGit.
- Downloads Python 3.12 and installs the **existing locked dependencies**.
- Creates four separate environments: app + ViTMatte, SAM 3/3.1, AOVs, CorridorKey.
- Downloads the exact pinned SAM/AOV/CorridorKey source revisions with their notices.
- By default prefetches **every model currently supported by the application**:
  SAM 2 Large/Base/Anime/EfficientTAM; SAM 3/3.1; MatAnyone/MatAnyone2;
  VideoMaMa + SVD VAE; ViTMatte **Base (larger, default)** and Small;
  MiniMax and ProPainterX; DVD/Wan, NormalCrafter and VDA Small;
  CorridorKey Blue 1.0 and Green 1.0.
- Validates imports, CUDA operations and the actual SAM Triton kernel, and makes
  a desktop shortcut. It does not automatically process a shot.

**CorridorKey version:** the app's supported optional foreground pass uses the
public CK **1.0** models. The separate CK2 experiments and their test footage are
not part of this release. Existing alpha is preserved by the foreground pass.
See [CorridorKey workflow](CORRIDORKEY.md) and [model attribution](MODEL_LICENSES.md).

## Other CMD options

```bat
rem Inspect the steps without installing or requesting credentials:
install_school.bat -Plan

rem Standard roto/matting plus all AOV/foreground models; omit alternate
rem SAM 2 variants and object-removal weights until first use:
install_school.bat -Models Essential

rem Install all software, then fetch model weights when needed:
install_school.bat -Models OnDemand
setup_models.bat

rem Install while SAM account approval is pending:
install_school.bat -SkipSamCheckpoints

rem For IT automation: use an already provisioned HF login, no prompts/shortcut:
set SAMMIE_UNATTENDED=1
install_school.bat -NoLogin -NoShortcut
```

`All` is the default. Both `All` and `Essential` include the larger ViTMatte Base
model and download SAM 3/3.1 unless explicitly skipped. `OnDemand` skips all
weight downloads; SAM 3/3.1 still require `setup_models.bat` before use.
`-NoLogin` never asks for a token; gated downloads fail if access is unavailable.
Do not put tokens in CMD arguments, scripts or deployment logs. A nonfunctional,
optional French joke prompt may appear in interactive setup; Enter skips it,
its answer is discarded, and unattended/`-NoLogin` setup suppresses it.

Interrupted setup can be rerun with the same command. Completed, verified model
downloads are reused, and the classic-model downloader resumes partial transfers
when the server supports ranges. The installer stops on errors and returns a
nonzero exit code; it does not report a complete installation after a failed stage.
Model-specific downloaders retain their existing verification behavior: classic
models use registry MD5, ViTMatte and CorridorKey verify SHA-256, SAM checks pinned
revision/file size, and AOVs use pinned HF snapshots with file-presence checks.

For a subsequent download retry, the individual `setup_models.bat`,
`setup_aov_models.bat`, `install_corridorkey.bat` and application model dialogs
remain available. Run this command to check installed runtimes independently:

```bat
.venv\Scripts\python.exe -m scripts.check_school
```

## Obtaining the source with Git instead

For anyone with Git already installed:

```bat
git clone --branch feature/aov-depth-normals https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3.git Sammie-Roto-School && cd Sammie-Roto-School && install_school.bat
```

Public cloning and downloading do not require a GitHub login. Hugging Face
approval is still required for the gated SAM checkpoints.

## Updating and moving between PCs

Newer builds also offer **Help → Check for School Updates → Save, close and update**.
This waits for Sammie to close, runs the same updater and reopens the app on
success. Its log is saved under `.runtime/ui-update-*.log`.

Close Sammie, then paste this **one line in CMD**:

```bat
curl.exe -fL https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest/download/update_school_online.bat -o "%TEMP%\update-sammie.bat" && "%TEMP%\update-sammie.bat"
```

Run it from the installed application folder, or from anywhere if you used the
default `%LOCALAPPDATA%\SammieRoto\Sammie-Roto-School` location. For a custom
location, append `-Directory "D:\Sammie-Roto-School"` to the final command.
The online updater also supports the first `school-2026.10.27` ZIP installation,
which did not include an update launcher. Updated installations can instead run
**`update_school.bat`** directly. `-Plan` previews the destination without changing
files; `-NoShortcut` preserves the current desktop shortcuts.

The updater downloads the **latest published school release**, verifies the ZIP
and per-file SHA-256 manifest, and updates packaged source only. It preserves
downloaded models, environments, settings, active sessions and personal projects.
It then reconciles the locked dependencies with `-Models OnDemand -NoLogin`, so
existing model downloads are reused and no token prompt appears.

For ZIP installations it keeps replaced source in
`.runtime/update-backups/<id>/previous`. Local source edits or filename collisions
stop the update before changes; resolve them or use a separate installation.
Failed source replacements trigger rollback. If a locked file prevents rollback,
the error identifies the files and backup to restore. A dependency-installation
failure stops with an error; repair with `install_school.bat -Models OnDemand`.
Completed source updates are repeatable. Downloaded ZIPs/backups can be removed
after verifying an update if disk space is needed.

For Git checkouts of this repository, the updater fetches the published commit
and only performs a fast-forward; it never force-resets, switches branches or
overwrites local commits. Uncommitted tracked changes or divergent/newer history
stop the Git update. Use this school's launchers, not the original upstream
updater, which targets a different build. A working internet connection is
required for the online updater and dependency reconciliation.

Share the **source ZIP**, not your live installation directory: virtual
environments contain machine-specific paths, and runtime folders contain downloaded
third-party code/models. Install separately on each PC. Copy saved `.sammie`
projects and footage separately as needed; they are not in the application ZIP.
Keep source file locations available when opening projects.

For the new working format, move **both** `shot.sammieproj` and its sibling
`shot.sammie-data` folder. Alternatively, use **File → Pack Portable Project**
to share a single `.sammie` archive. **File → Set Project** establishes the
workspace on each PC. See [working projects and Nuke handoff](WORKFLOW.md).

The ZIP includes `SCHOOL_BUILD.json` with the exact commit and SHA-256 of every
packaged file. Its companion `.zip.sha256` verifies the archive:

```bat
certutil -hashfile Sammie-Roto-School.zip SHA256
```

Maintainers build a reproducible source ZIP from a committed revision with
`.venv\Scripts\python.exe -m scripts.package_school --output dist/Sammie-Roto-School.zip`.
The package includes required checkpoint JSON configurations and the tiny
mask-smoothing weight already distributed by upstream. Downloaded large weights,
secrets, local settings and media are excluded.

## Validation and limitations

See [school installer validation](SCHOOL_VALIDATION.md) for checks actually run,
including the distinction between clean-directory installation on this workstation
and an installation on another physical school PC. Existing shot/GPU validation
is recorded in [general validation](VALIDATION.md), [AOV validation](AOV_VALIDATION.md),
[ViTMatte validation](VITMATTE_VALIDATION.md), [CorridorKey validation](CORRIDORKEY_VALIDATION.md)
and [opaque-core comparisons](OPAQUE_CORE.md). Those tests are not a guarantee of
matte quality or memory use on every shot.
