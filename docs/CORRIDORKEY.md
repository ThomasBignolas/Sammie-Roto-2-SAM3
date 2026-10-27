# CorridorKey foreground RGB — local school build

This optional post-matting pass uses the **public CorridorKey 1.0 / CorridorKeyBlue
1.0** model to estimate clean foreground colors. It is not CorridorKey 2.0.
It does not replace VideoMaMa or ViTMatte, rerun them, or change their alpha.

## Install and launch

For the whole current application on another PC, use `install_school.bat` or the
[one-command download and setup](SCHOOL_INSTALL.md). The individual steps below
remain useful when adding only CorridorKey to an existing installation.

On this PC, run `install_corridorkey.bat` once, then `run_sam3.bat`.
Installation was executed with the pinned worker environment. Use this direct
launcher; the original upstream launcher/updater does not carry this school change.

On another Windows/NVIDIA school PC:

1. Extract the source ZIP into a writable local folder. Install Git for
   Windows and a current NVIDIA driver supporting CUDA 13.0 for the existing
   main application installer (the isolated CorridorKey worker uses CUDA 12.8).
2. Run `install_sam3.bat` to install the existing application and SAM runtime.
3. Run `install_vitmatte.bat` if you will generate new ViTMatte results.
4. Run `install_corridorkey.bat`. It creates `.runtime/corridorkey-env`, obtains
   the pinned official CorridorKey source and downloads the blue checkpoint.
   Green weights download when first selected. Public CorridorKey weights do
   not require a Hugging Face token.
5. Run `setup_models.bat` for the SAM models you need. SAM 3/3.1 still require
   the user's approved Hugging Face access. Credentials are not part of the ZIP.
6. Launch `run_sam3.bat`, then open the `.sammie` project separately.

Do not copy a `.venv` between PCs; rebuild it with the installers. Installers use
paths relative to their own folder. The main UI's Torch/Transformers environment
is unchanged by the CorridorKey installer. `install_aov.bat` is only required
if you also want to generate depth/normals.

The optional source and models live in ignored `.runtime` folders. No CorridorKey
code or weights, user footage, Nuke scripts, caches or credentials are included
in the school source ZIP. Existing project caches travel inside `.sammie` files;
external original EXR plates must travel separately and be relinked on the new PC.
The original upstream 97 KB mask-smoothing weight is retained for compatibility;
downloaded SAM, ViTMatte, CorridorKey and AOV weights are excluded.
Official pinned ViTMatte output caches can now be read without the originating
machine's absolute checkpoint path. Custom checkpoint changes still invalidate
their cache.

## Workflow

1. Generate your SAM masks, VideoMaMa matte and optional **ViTMatte Base** refinement.
2. In Matting, select the alpha view you want: **Original**, **Refined**, or
   **Temporally validated**. Enable ViTMatte if using its refined/validated view.
3. Open the fourth **AOVs** tab, enable **CorridorKey Foreground RGB**, and select
   **Matting (selected ViTMatte view)** as the alpha source. Segmentation masks
   are also supported, but have less useful transparency information.
4. Disable Depth/Normals if you only need color recovery.
5. Pick the current frame and press **Test CorridorKey on current frame**.
   Compare **Foreground on background**, **Recovered foreground**, and
   **Original plate**. The usual background picker includes Checkerboard.
6. Tune the controls, then press **Generate / Reuse cached AOVs** for timeline
   In/Out. Use **Export multilayer AOV EXR sequence** for delivery.

The selected alpha is passed into CorridorKey as its hint. The network internally
predicts both RGB and alpha, but only its RGB output is used here. The delivered
alpha is copied from the selected existing matte at float32 precision. Multiple
objects use Sammie's existing maximum/union alpha; individual alpha layers remain
in the export. This produces one combined foreground RGB pass, not a separate
RGB decomposition for overlapping objects.

Missing/stale selected ViTMatte results block generation with a clear error.
Choose Original explicitly if you want to use VideoMaMa instead. Changes to the
actual selected alpha invalidate the matching foreground results; the old RGB
will not be exported with a new alpha silently.

## Tuned settings

| Preset | Model | Model size | Exposure | Strength | Edge coverage | Despill |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Blue sky / hair (tested) | Blue 1.0 | 2048 | −2 stops | 1 | 4 | 0 |
| Blue screen / neutral exposure | Blue 1.0 | 2048 | 0 | 1 | 4 | 0 |
| Green screen / neutral exposure | Green 1.0 | 2048 | 0 | 1 | 4 | 0 |
| Fast blue preview | Blue 1.0 | 1024 | −2 stops | 1 | 4 | 0 |

Blue was chosen for the tested shot because the hair edges contain blue sky.
This is a shot-specific adaptation of a blue-screen model. It is not an automatic
choice for every outdoor shot, and blue sky is not a controlled blue screen.
Green selects different trained weights; it is not just a channel switch.

- **Exposure:** applied in linear Rec.709 before inference and reversed on the
  recovered RGB afterward. The original plate and alpha are not exposed. −2
  stops gave a useful result on the supplied shot; other shots may need 0.
- **Strength:** overall RGB cleanup. 0 gives exactly the original RGB.
- **Edge coverage:** blend weight is `strength × (1 − alpha^coverage)`.
  Higher values affect more near-opaque edge pixels. Fully opaque RGB is always
  copied exactly; the matte is never eroded or changed by this control.
- **Despill:** an additional, optional suppression of excess blue/green in the
  estimated edge color, conserving linear Rec.709 luminance. Default 0. It can
  also affect legitimate blue/green edge colors; judge hair and clothing.
- **Resolution:** native model input is square (the official API's behavior).
  2048 is the quality setting exercised on the 16 GB RTX 5060 Ti. 1024 is useful
  for preview; 1536 is intermediate. These are model-input sizes, not output sizes.
- **Precision:** FP16 autocast on CUDA by default, FP32 optional. CPU uses FP32
  even when FP16 is selected, and is much slower. The CPU adapter has a smoke
  test; it is not a recommended full-shot workflow.

Strength, coverage and despill operate on cached RGB predictions and do not run
the model again. Exposure, checkpoint, resolution, precision, input source or
alpha changes require Generate. Model inference always runs in evaluation and
inference mode. There is no training or weight fine-tuning here.

## Color, resolution and Nuke

The default input is the application's imported display RGB (sRGB-like PNG/JPEG).
For scene-linear VFX delivery, choose the **first original RGB EXR** under the
optional plate override and explicitly select its actual color space:

- ACEScg (linear AP1/D60)
- Linear Rec.709 (D65)
- sRGB (encoded Rec.709 primaries)

Do not choose a camera LogC plate and label it ACEScg. Convert unsupported camera
log, gamut or OCIO spaces first. No inverse film/display LUT is used. ACEScg uses
the matrix from OCIO's ACES 1.3 configuration to/from linear Rec.709, which is the
model's expected gamut. The model handles positive values; out-of-gamut negatives
are clamped in its sRGB encoding. Opaque source RGB is still restored exactly.
Model foreground predictions are bounded; HDR edge highlights cannot be assumed
to be perfectly recovered.

Set **EXR number at project frame 0** explicitly. For the supplied 993-start
sequence, project frame 58 maps to EXR 1051. The In point does not reset this
mapping. EXR and alpha dimensions must match exactly; cropped/nonzero-origin EXR
windows are rejected. Output always retains source dimensions and frame order.

With the pass enabled, multilayer delivery contains:

| Channels | Meaning |
| --- | --- |
| R/G/B | Original straight input plate, if enabled. The EXR override is used when selected. |
| A and object `.Y` layers | Existing selected mattes, if enabled. |
| foreground.R/G/B | Recovered **straight**, unpremultiplied RGB, in the same selected input color space. |
| foreground.A | Exact union alpha used for recovery, even if other matte export layers are disabled. |
| N.x/y/z, depth.Z, Z, Z_raw | Existing selected geometry passes, unchanged. |

Channels are float32 ZIP EXR with frame-rate and processing metadata. The
source-sized foreground is reconstructed with the official Lanczos resampling
order. The viewer displays an sRGB preview, not your Nuke project's ARRI Reveal
view. Judge final scene-linear color in Nuke under the intended OCIO view.

In Nuke, Shuffle the `foreground` layer into RGBA, assign the documented input
color space, then **Premult once** before Merge. Leave the original plate layers
available for A/B comparison. Do not multiply foreground by alpha twice.

## Memory, cache and cancellation

SAM is offloaded and the matting/removal models are released before inference.
Geometry and CorridorKey run sequentially in separate subprocesses. The worker
keeps one model loaded across the requested frames; cancellation terminates only
that owned worker and releases its GPU allocations.

Each prediction is cached at native model resolution in a compressed NumPy
archive, not as a full-resolution RGBA image. FP16 model output is stored as
FP16 without an additional rounding step; FP32 mode remains FP32. Alpha inputs
are lossless float32 archives. Completed frames survive cancellation, and
Generate repairs corrupt/incomplete files. Current-frame tests retain other
cached frames with matching inference settings. Different model/exposure trials
may accumulate separate caches under the project's `foreground` directory.

For archiving a small project, you can remove that generated `foreground` cache
from a **closed backup copy** and regenerate it later. Do not remove `matting`,
`matting_float` or `vitmatte` if you need the saved original alpha. Keeping caches
allows other PCs to preview/export without rerunning CorridorKey; installing the
model is needed only for new inference.

## Limits and attribution

CK1.0 is a per-image model. It has no CK2.0 time-awareness mechanism. Stable
VideoMaMa/ViTMatte alpha does not guarantee stable estimated RGB. Review the
result in motion, particularly occlusions, backlit hairs and blue clothing.
Using the original alpha with a jointly predicted RGB is an adaptation; it can
leave residual contamination or darken/change fine hair. No clean plate or
ground-truth foreground was available, so the shot comparisons are diagnostic.

Official source: [CorridorKey](https://github.com/nikopueringer/CorridorKey),
revision `97e55a453060745bead1befd293f6e523c4b845c`.
Blue weights: [CorridorKeyBlue 1.0](https://huggingface.co/nikopueringer/CorridorKeyBlue_1.0).
Green weights: [CorridorKey 1.0](https://huggingface.co/nikopueringer/CorridorKey_v1.0).
Revision and SHA256 checksums are pinned in `sammie/foreground/versions.py`.
The adapter uses the official Hiera/decoder/refiner and input normalization.

CorridorKey's [license](https://github.com/nikopueringer/CorridorKey/blob/97e55a453060745bead1befd293f6e523c4b845c/LICENSE)
is CC BY-NC-SA with additional terms, including restrictions on repackaging,
resale, paid inference services and commercial software integration. This school workflow installs it locally as an optional external dependency. It does
not grant permission for a later commercial or public distribution. Existing
Sammie, SAM, ViTMatte and other upstream notices remain applicable.

See [executed validation](CORRIDORKEY_VALIDATION.md) for measured results.
