# School package validation

Validation date: 2026-10-27. Release: `school-2026.10.27`.

The school package includes the previously tested application features and a new
Windows installer. Setup checks are distinct from model inference and matte-quality
benchmarks; see the linked feature validation documents in [SCHOOL_INSTALL.md](SCHOOL_INSTALL.md).

## Executed checks

- **148 automated tests passed**, both in the existing application environment
  and in a newly installed application environment in a separate directory.
  This includes the existing segmentation, refinement, AOV, foreground, contact
  seam, opaque core and GUI/export tests, plus 14 school installer tests.
- Installer tests cover verified-download reuse, range resume, servers ignoring
  ranges, corrupt downloads, HTTP failures, interruption, credential-host limits,
  package exclusions/config retention and deterministic manifests/checksums.
- Windows bootstrap tests execute real PowerShell with mocked network responses:
  extraction/installer invocation, resuming the same package, refusal to overwrite
  another installation, checksum rejection, traversal rejection and nonzero exit
  status on an installer failure. Plan mode does not download or ask questions.
- Both PowerShell entry points parse successfully in Windows PowerShell.
- **Full installer executed successfully in a fresh source directory**, with Git
  and uv deliberately removed from the child process PATH. It downloaded and
  SHA-256-verified uv 0.12.5 and MinGit 2.56.0.windows.4, downloaded Python 3.12.14,
  created four new environments and checked out the pinned external sources.
- The `All` model setup completed. Missing SAM 2 variants, MatAnyone2, MiniMax,
  ProPainterX and both CorridorKey models were actually downloaded and checked.
  Existing SAM 3/3.1, VideoMaMa, ViTMatte and AOV model files were reused from
  this workstation; the setup used the existing authorized HF login.
- All four installed runtimes passed imports and a real CUDA tensor operation on
  the **RTX 5060 Ti (15.93 GiB reported VRAM)**. The SAM runtime additionally ran
  the official Triton distance-transform kernel. Application torch is
  **2.11.0+cu130**; the three isolated workers use **2.10.0+cu128**. No dependency
  version changes were needed.
- Rerunning the full installer also succeeded and reused completed model files.
  Fresh-directory setup took **261.1 seconds**, and the cached rerun took
  **39.9 seconds** on this machine. The first run reused the package-download
  cache and several large checkpoints; **these are not all-download installation
  benchmarks or estimates for another PC**.
- Before making the repository public at the owner's request, a scan of **669
  historical blobs** across the published branches and current commit found no
  HF/GitHub token/private-key patterns, personal `.sammie`/Nuke projects or user
  footage. The only historical video files were two original upstream examples.
  A pattern scan does not establish the absence of every possible secret.

## Scope of validation

The installation was performed in a clean directory on this workstation, **not
on a second physical school PC**. Existing driver/system components and the
authorized user login were available. No new full-shot neural inference or GPU
performance/peak-memory benchmark was run for this packaging change. Feature
validation documents retain the actual prior model/shot results.

Desktop shortcut creation was disabled during the clean test to preserve the
user's desktop. Interactive token entry was not repeated; a saved HF login was
used. Newly provisioned school accounts still need model approval, and IT policies,
driver compatibility, disk quotas and network restrictions may differ by PC.

The release ZIP is built from an exact commit, includes a per-file SHA-256
manifest, and is checked for ZIP integrity and required application/config files.
Neither Python environments, experimental CK2 runs, downloaded model weights,
user projects nor credentials are packaged. The tiny upstream smoothing weight
is retained for the original mask-smoothing feature.
