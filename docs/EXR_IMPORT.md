# EXR plates and sequences

Use **File > Open** or drop an `.exr` onto the viewer. Select one frame to load
its numbered sequence, or choose **Single image** in the import dialog.
This feature is included in the school installer; no additional packages or
Python edits are necessary. Update an existing installation with
[`update_school.bat` or the online CMD command](SCHOOL_INSTALL.md#updating-and-moving-between-pcs).

## Import controls

- **RGB layer:** choose `R / G / B`, `beauty`, or another complete RGB layer.
  Single-part multilayer files are supported. Depth/normal-only files are not RGB plates.
- **Input color space:** linear Rec.709, ACEScg (linear AP1), or already encoded
  sRGB. ACEScg chromaticities are recognized when present; otherwise linear
  Rec.709 is suggested. Confirm the choice against your plate's color pipeline.
  LogC, ACES2065-1, custom OCIO views and LUTs require conversion before import.
- **Preview / model exposure:** adjust the display preview in stops. This affects
  the frames sent to segmentation, matting, depth and normal models. It does not
  change the retained HDR plate or the original-RGB EXR export.
- **Frame rate:** defaults to `framesPerSecond` metadata or 24 when absent.
  Ratios such as `24000/1001` are supported and retained in EXR exports.

Import retains the original EXRs verbatim and makes full-resolution display RGB
PNG frames for the existing models. Linear inputs are converted to linear
Rec.709, exposed, encoded to sRGB, clipped and quantized to 8 bits for the model
frames only. This is not an ACES/OCIO display rendering transform. Very bright
highlights may require lowering the proxy exposure for useful model detail.

RGB is read as stored: **input alpha is ignored**, and no automatic
unpremultiplication is performed. Use an original, unpremultiplied plate.
Generated alpha follows the existing segmentation/matting workflow.

## Frame alignment and export

Matching numbered files with the same name prefix and padding load in numerical
order. Missing or duplicate frame numbers stop import instead of silently
shortening the shot. Dimensions, display-window origin and pixel aspect must
stay consistent. Sequence export defaults to the first source frame number;
the application's timeline continues to use its existing zero-based indices.

The display window defines the full working canvas. Cropped data windows are
placed at their correct offset, with zero outside the available data. Overscan
outside the display window is excluded from model frames and generated exports.
The source files still retain their complete original data.

In **EXR sequence export**, enable **Include original** to deliver retained HDR
RGB. The **AOVs multilayer EXR** export also uses that HDR plate for `R/G/B`, with
the generated alpha, depth and normals selected as usual. Negative and above-one
RGB values are preserved as float32; proxy exposure is not baked into them.
The export writes a full-canvas data window, original display-window origin,
pixel aspect, rational frame rate and the selected linear-space chromaticities.
Other arbitrary input header attributes and unselected source layers are not
copied into generated files. Existing alpha/channel naming stays unchanged.

PNG/video exports and the viewer use the display RGB working frames. They are
not HDR plate delivery paths. The optional CorridorKey pass reads retained HDR
RGB and inherits its declared color space; an explicit plate override retains
its own color-space controls.

Saving a `.sammie` project includes the original EXRs, import choices and model
frames. These remain available when the original sequence is moved. Projects
and temporary sessions consequently need space for both EXRs and proxies.
Import can be cancelled between frames; preparation errors/cancellation preserve
the previous session. Existing video and ordinary image import are unchanged.

## Supported files and limits

Tested half/float RGB, scanline/tiled EXRs and single-part multilayer RGB.
Multipart, deep, subsampled RGB, missing RGB and non-finite pixels are rejected
with an error. Flatten or convert these before import. The reader processes one
frame at a time on CPU, but the OpenEXR Python reader loads all channels in that
frame; very large AOV files can use substantial RAM. Reading uses the installed
[official OpenEXR Python API](https://openexr.com/en/latest/python.html).

## Executed validation (27 October 2026)

Automated tests cover real generated EXRs, layers, HDR/negative values, transfer
functions, cropped/nonzero windows, tiled images, rational FPS, sequence gaps,
cancellation, project save/reopen, the actual import dialog/MainWindow and both
EXR writers. AOV and refinement pipeline integration tests use fake inference
models to check the proxy/native-plate separation.

A separate real CUDA smoke test used **ViTMatte Base Composition-1k**, FP32, on
three **640 x 360** imported ACEScg frames with 384-pixel tiles and 64-pixel
overlap, on the RTX 5060 Ti 16 GB. Refinement took **8.78 seconds**, including
initial model loading; subsequent frames took **0.20 and 0.17 seconds**. Peak
PyTorch allocation was **562.4 MiB**, reservation **668 MiB**. Those figures are
for this small test process, not total workstation VRAM or a 4K estimate.
All three float-alpha/HDR EXR exports passed resolution, range, HDR-value and
rational-FPS checks.

Reproduce with `.venv\Scripts\python.exe -m scripts.validate_exr_import`.
It uses the bundled example clip and synthetic elliptical baseline alpha.
This validates execution and file formats, **not matte accuracy**; it is not a
new end-to-end SAM/VideoMaMa benchmark or a production-footage quality claim.
