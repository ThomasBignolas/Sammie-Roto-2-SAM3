# CorridorKey integration validation

Executed locally on 2026-10-21, Windows, NVIDIA RTX 5060 Ti 16 GB, 64 GB system
RAM. CorridorKey worker: Python 3.12, PyTorch 2.10.0+cu128, pinned official
CorridorKey 1.0 code and weights. These tests exercise the new optional pass;
they are not a claim of ground-truth foreground accuracy.

## Automated tests

`.venv\Scripts\python.exe -m unittest discover -s tests -q`

**95 tests passed in 11.219 seconds.** Tests include the existing SAM backend,
multi-text prompts, media ordering, ViTMatte and AOV suites, plus:

- Selected original, refined and temporally validated alpha; exact preservation.
- Opaque RGB preservation, HDR/ACEScg conversion, exposure and despill controls.
- Numbered EXR mapping, original plate delivery and combined depth/normal/foreground EXR.
- Cache reuse for artist controls; invalidation for source, alpha and inference changes.
- Missing model/refinement errors, corrupt cache repair, cancellation and cleanup.
- Current-frame tests preserving the rest of the cached sequence.
- Official ViTMatte cache portability and UI presets.

The suite uses controlled fake models where appropriate. Its printed mock GPU
statistics are not measurements. Real model measurements are listed below.

## Real UI, GPU and export tests

The validation utility creates the actual main window, runs the owned worker
through the application's AOV action and exports via the existing EXR dialog.
It uses a disposable copy of the saved project cache, not the user's original.
The selected alpha was **ViTMatte Base Composition-1k, temporally validated**.
VideoMaMa and ViTMatte inference were not rerun in these tests.

| Test | Original scene-linear plate | Imported display plate |
| --- | --- | --- |
| Frames | 3 consecutive frames | 1 frame |
| Source / export size | 4608 x 3164 | 4608 x 3164 |
| Input color space | ACEScg original EXR | sRGB imported image |
| Model | CorridorKeyBlue 1.0 | CorridorKeyBlue 1.0 |
| Model input | 2048 x 2048 | 1024 x 1024 |
| Precision | CUDA FP16 autocast | CUDA FP16 autocast |
| Input exposure | -2 EV, reversed in recovered RGB | -2 EV, reversed in recovered RGB |
| Inference time per frame | 1.892 / 1.377 / 1.432 s | 1.552 s |
| Peak allocated CUDA memory | 5049.49 MiB (4.93 GiB) | 1585.38 MiB (1.55 GiB) |
| Peak reserved CUDA memory | 9774 MiB (9.54 GiB) | 2110 MiB (2.06 GiB) |
| Worker duration | 30.261 s | 6.338 s |
| Maximum alpha error | 0 | 0 |
| Maximum opaque RGB error | 0 | 0 |

Worker duration includes model initialization, loading plate/alpha and writing
prediction caches. It excludes the UI's alpha-preparation phase and the final
EXR export. Inference times include the adapter's device transfers but exclude
cache compression and full-resolution merge. These short runs do not establish
sustained throughput or a full-shot maximum memory requirement.

Both runs verified original plate RGB, exact `A == foreground.A`, matching
cached foreground values in `foreground.R/G/B`, unchanged fully opaque RGB,
source dimensions and export numbering. Changing strength to 0.75 and back
reused predictions. The current-frame test also reused cached predictions.

The combined geometry/foreground EXR test uses mocked depth and normals with
real OpenEXR I/O; these runs did not regenerate DVD or NormalCrafter models.

## CPU / green model

The official Green 1.0 checkpoint was downloaded and checksum-verified.
CPU FP32 inference at 1024 on synthetic RGB/alpha completed in **6.849 s**;
model initialization took **1.780 s**. Finite float32 output, evaluation mode,
no gradients and unchanged input alpha were verified. This validates loading
and API compatibility, not green-screen image quality or full-shot CPU speed.

## Installation and portability

`install_corridorkey.bat`'s installer was executed successfully on this PC.
It installs the pinned isolated environment without changing the main UI's
PyTorch environment. A source ZIP contains this modified source and the
existing upstream small smoothing weight, with no downloaded model code,
checkpoints, user footage, Nuke files or credentials. Official dependencies
are downloaded locally on each PC. A second physical PC has **not** been tested.

Use **run_sam3.bat** to launch the modified app; the original upstream updater
does not carry these local changes.

## Limitations

- Public CorridorKey 1.0 is used, not CorridorKey 2.0.
- The pass estimates RGB independently per frame. Stable input alpha does not
  guarantee stable RGB; the whole 194-frame shot was not validated end to end.
- No clean plate or ground-truth foreground was available. Lower contamination
  or a preferred composite is not a quantitative accuracy measurement.
- Blue sky differs from a controlled blue screen; blue clothes, backlit hair
  and highlights need visual review. Opaque plate RGB is preserved exactly.
- The viewer uses sRGB, not the shot's ARRI Reveal OCIO view. Inspect exported
  scene-linear AOVs in Nuke under the intended color management.
- 1536 resolution and CUDA FP32 are available but were not separately benchmarked.

Machine-readable, path-sanitized results: [JSON](corridorkey-validation-results.json).
Reproducible harnesses: `python -m scripts.validate_foreground_ui --help` and
`python -m scripts.validate_foreground_model --help`. Run from the project root.
Use a copied test session for the UI harness, which changes its test settings.
