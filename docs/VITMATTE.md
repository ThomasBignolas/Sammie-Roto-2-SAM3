# Optional ViTMatte edge refinement

ViTMatte refines the uncertain boundary of a VideoMaMa matte using the original RGB frame. It is an image model: it can improve fine edges on some shots and introduce errors on others. Keep the original available, compare in motion, and inspect the final composite. This feature does not replace segmentation, VideoMaMa or MatAnyone.

## Editing settings without losing the result (2.6.0)

ViTMatte controls now configure the **next run**. Changing strength, thresholds,
resolution, model or other parameters keeps the last completed refinement selected
for preview, EXR export, Send to Nuke and CorridorKey input. A status message marks
unapplied settings; click **Refine cached mattes / Resume** to apply them. Temporarily
invalid thresholds while editing no longer switch the display back to VideoMaMa.
**Restore settings of completed result** recovers the values used to make that
matte, including after saving/reopening. Mouse-wheel changes are disabled on the
ViTMatte dropdowns and numeric fields to prevent accidental edits while scrolling.
Use their buttons, keyboard or typed values instead.

The enable checkbox and A/B/C selector remain immediate, intentional comparison
controls. Changing the underlying RGB, VideoMaMa alpha or an applied custom
checkpoint still invalidates affected results. Export metadata records applied
ViTMatte settings separately from the controls' current values. No model rerun is
needed merely to recover a result hidden by the old settings-mismatch behavior,
provided its files and input dependencies still match.

## Windows installation and launch

Use the current public repository's `feature/aov-depth-normals` branch, or the [school installer](SCHOOL_INSTALL.md). For a manual installation:

```powershell
git clone --branch feature/aov-depth-normals https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3.git
cd Sammie-Roto-2-SAM3
.\install_sam3.bat
.\setup_models.bat
.\install_vitmatte.bat
.\run_sam3.bat
```

For an existing installation on this branch, close Sammie, run `install_vitmatte.bat`, then `run_sam3.bat`. No Python source edits are needed. Re-running the original base installer can remove optional extras; run the ViTMatte installer again afterward.

The optional `vitmatte` extra pins Transformers 5.8.0 in `pyproject.toml` / `uv.lock`. The existing PyTorch, torchvision, NumPy, Diffusers and Hugging Face Hub versions are unchanged. SAM 3's separate environment is unchanged. No Detectron2 build is required. The frozen installer uses CUDA 13.0 by default, matching the existing Windows installer. Alternate supported base extras:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_vitmatte.ps1 -DeviceExtra cu126
# CPU-only installation (ViTMatte can refine existing mattes; VideoMaMa requires CUDA):
powershell -ExecutionPolicy Bypass -File scripts/install_vitmatte.ps1 -DeviceExtra cpu
```

**ViTMatte Base is the default model.** This is the larger of the official Base/Small models. Its first refinement downloads approximately 369 MiB of public weights from [`hustvl/vitmatte-base-composition-1k`](https://huggingface.co/hustvl/vitmatte-base-composition-1k), pinned to revision `bf486d01a7d9e3dbcc8400f7942835caf0eaf76e`. Small remains selectable (approximately 99 MiB, revision `6a58ad7646403c1df626fbd746900aec7361ea1d`). No Hugging Face token or approval is required for either checkpoint. Files are cached in `checkpoints/vitmatte/<revision>/`. The download is cancellable between network chunks, uses a temporary file and verifies the pinned weight SHA-256 before loading. Offline use works once the files are present.

The official Base repository provides `pytorch_model.bin`; the loader verifies its SHA-256 and uses PyTorch's restricted `weights_only=True` loading through Transformers. Small uses `model.safetensors`. Automatic loading explicitly selects the verified format, even if another file is present. No dependency versions change when switching models.

You can select a **local checkpoint folder** containing a compatible Transformers `config.json` and `model.safetensors`. Select the matching Base (768 hidden channels) or Small (384) model in the UI. Custom binary checkpoints and Detectron2 `.pth` files are not accepted. Architecture, missing weights and unexpected weights are checked; failure is reported without substituting another model. Official Base and Small have been exercised on the GPU; other locally supplied checkpoints have not been validated.

## Workflow

1. Segment and track normally using any supported SAM backend.
2. Choose **VideoMaMa** in the existing **Matting** tab. Scroll to **ViTMatte Edge Refinement** and check its title checkbox. It is initially OFF; controls are hidden for other matting models.
3. Select **ViTMatte Base (larger) / Composition-1k**, choose a preset and press **Run Matting**. VideoMaMa completes and releases its GPU models before the refinement worker starts. New runs with refinement enabled cache unquantized VideoMaMa alpha as well as ordinary PNG mattes. Older projects retain their saved model selection; change Small to Base here to upgrade them.
4. For an existing VideoMaMa project, or after changing refinement settings, press **Refine cached mattes / Resume**. This never invokes VideoMaMa or SAM. Existing PNG mattes are supported, but their prior 8-bit quantization cannot be undone. The completion status reports this case.
5. Scrub or play the usual Matting view and switch **Preview / export** between **A Original VideoMaMa**, **B ViTMatte merged**, and **C Temporally validated**. This is an A/B/C toggle in the existing matte, background-color and alpha views. It also selects the source of subsequent exports. When temporal validation is disabled, B and C are identical. The first frame has no temporal reference, so B and C also match there.
6. Export using the existing formats. For Nuke, use a **Matting-Matte EXR sequence** to retain float alpha. Uncheck refinement to restore the original application rendering/export path immediately.

Settings are saved with the project. A complete pass becomes active atomically. Cancelled, failed or stale refinement does not become a partially refined export: original mattes remain the fallback. The completion indicator reports errors and cancellation. Cancellation waits for the current model tile or download chunk to finish; it does not interrupt a CUDA kernel. Completed frames remain reusable by **Resume**.

Refinement uses the current in/out range, independently per object, including combined object 0. Every object must have a matte for every selected frame. Choose the same in/out range to resume identical temporal context. Changing the range start changes the first temporal reference. Frames outside the completed range use original mattes. Clear Matting also clears float and refinement caches.

## Explicit presets

Widths below are radii in pixels at a 1080-pixel short edge; they scale with processing resolution. Total erosion radius is unknown radius + extra expansion. Resolution is a maximum short edge, with no upscaling.

| Parameter | Fast | Balanced (default) | Quality |
| --- | ---: | ---: | ---: |
| Processing short edge | 720 | 1080 | 2160 |
| Tile size | 768 | 768 | 1024 |
| Unknown radius | 8 | 12 | 16 |
| Extra expansion | 2 | 4 | 6 |
| Refinement strength | 0.65 | 0.75 | 0.85 |
| Temporal validation | Off | On | On |
| Temporal strength | 0.35 (if enabled) | 0.65 | 0.80 |

All presets leave foreground/background thresholds at their current values (initially **0.98 / 0.02**), precision/device unchanged (initially **FP32 / Auto**), and model/checkpoint/debug selection unchanged. The presets deliberately change only the parameters listed. The shared default tile overlap is 128 pixels and merge feather is 3 pixels at 1080p. These last two advanced values are saved as `vitmatte_overlap` and `vitmatte_feather` in session settings; the normal UI does not require editing them.

For **1080p**, start with Balanced and inspect hair against contrasting backgrounds. Reduce strength if it changes a stable edge unnecessarily. For **4K or larger**, Balanced is a quicker proxy workflow; Quality retains more spatial detail, using overlapping 1024-pixel tiles. **Source** resolution is available for native-resolution tiles, at higher cost; it has not been benchmarked over the entire supplied high-resolution shot. Larger unknown widths may recover missed hair or motion blur but also allow more background contamination. A missing structure outside the unknown band cannot be recovered by this pass.

**Use FP32 for Base final mattes.** Optional CUDA BF16 uses autocast and checks for finite output; CPU always uses FP32. On the Base precision check, BF16 differed from CUDA FP32 by alpha MAE 0.0233 and maximum 0.721. This was an all-unknown test input, not a ground-truth accuracy test; finite output alone does not establish acceptable precision. BF16 remains experimental and requires visual comparison. CPU is practical for small crops/tests; full-shot CPU performance has not been established. CUDA out-of-memory errors are reported; reduce tile size, processing resolution, or use CPU. No automatic quality reduction happens silently.

## Alpha, color and export behavior

The app imports display frames through its existing OpenCV/PyAV RGB pipeline. ViTMatte receives **original, unpremultiplied RGB**, never a composited foreground. Its RGB normalization follows the [original ViTMatte configuration](https://github.com/hustvl/ViTMatte/blob/8cd7ef068380977c3962c4cb733cb1fe7f2241a5/configs/common/model.py): mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]` after dividing uint8 by 255. Trimap values 0/128/255 are divided by 255 without RGB normalization. Inputs are zero-padded on the right/bottom to multiples of 32 and predictions are cropped back before merging.

The published [HF processor configuration](https://huggingface.co/hustvl/vitmatte-small-composition-1k/blob/6a58ad7646403c1df626fbd746900aec7361ea1d/preprocessor_config.json) instead specifies 0.5/0.5 RGB normalization (Base has the same discrepancy). This integration explicitly follows the original project's configuration; the choice is part of the cache identity and unit tests. Checkpoint weights are the official organization's Transformers conversions.

Linear EXR/HDR source paths are rejected for this refinement. Supply a display-referred sRGB/Rec.709 proxy with identical geometry, frame count and order. This is not an OCIO-aware HDR import upgrade. The original application's video color metadata and RGB export behavior are retained. Alpha is linear coverage and receives no RGB transfer function.

Processing and compressed caches store float32 alpha clamped to [0,1]. VideoMaMa still computes in its existing FP16 mode. Display, PNG and video exports retain their existing 8-bit rendering path; **EXR matte channels receive float32 directly**, without passing through the display quantization. Existing names (`Object_0.Y`, or named object channels), numbering offsets, channel precision and metadata behavior remain. Gamma/shrink/grow controls still apply after A/B/C selection, using float operations before EXR export. No FPS/frame-number transformations are added.

## Trimaps and temporal behavior

Foreground and background confidence maps are eroded separately. Pixels between thresholds remain uncertain, and erosion expands uncertainty around both sides of boundaries, including nearby low-alpha pixels. Replicated image borders preserve masks that reach the frame edge. Empty/full mattes avoid unnecessary model calls; disconnected regions and thin structures remain supported. The inference trimap is restored with nearest-neighbor sampling; that same trimap defines the merge support. Known regions retain their exact original alpha, rather than being forced to 0 or 1. Inside the unknown band, a feathered strength weight blends the prediction with the baseline. Debug PNG trimaps are written to `temp/vitmatte/trimaps/<frame>/<object>.png`.

Temporal validation tracks **refinement deltas**, not the previous final alpha. OpenCV Farneback flow runs in both directions on RGB grayscale proxies capped at 640 pixels on the long edge. Forward/backward consistency, bounds, photometric error and baseline-alpha agreement gate confidence. Large novel delta changes and unreliable correspondence reduce the contribution of ViTMatte toward the **current** VideoMaMa matte. Past mattes are never averaged into the current image. Cuts and disocclusion therefore cannot drag an old silhouette into a new frame through alpha averaging.

This is a conservative heuristic, not a learned video matting model. Small hair, large motion, lighting changes and motion blur can defeat flow. Returning toward the baseline may remove a real improvement or itself alter temporal variation. The initial frame and gaps reset history. Review the result in motion and disable validation when B looks better than C. Original VideoMaMa overlap feedback/blending remains in its own stage; refinement never feeds predictions back into that baseline.

## Memory, cache and recovery

SAM is offloaded using the existing manager. VideoMaMa runs first; its model references are dropped and the allocator cache is released at the model boundary. Only then does one ViTMatte model load for the pass. Each frame uses fixed-grid overlapping tiles containing unknown pixels; known-only tiles are skipped. Feather weights normalize overlap predictions. Tiles share a 128-pixel context overlap, but they cannot reproduce the global context of an unrestricted whole-frame transformer. Native-resolution tiling can introduce context-dependent differences despite seam blending. No temporal video chunking is added by this stage.

The worker holds only current arrays and small preceding flow/reference arrays per object. Its model is always released in `finally`, including failure and cancellation; no per-frame `empty_cache()` is added. Reported GPU allocated/reserved peaks are measured by PyTorch in this process, not total workstation usage or a VRAM guarantee.

`temp/matting/` remains the original PNG matte cache. `temp/matting_float/` contains optional compressed original float alpha with hashes binding it to source/PNG content. `temp/vitmatte/` keeps one refinement variant per frame/object, with compressed refined/validated alpha and trimap. Signatures include source RGB, source alpha, PNG content, model weights/config, normalization/implementation version, settings, frame/object IDs and preceding temporal signature. Interrupted passes reuse completed matching frames and reconstruct temporal history before continuing. Changing ViTMatte settings never reruns VideoMaMa. Temporal-only changes currently rerun ViTMatte as well; the baseline remains cached.

Only a completed manifest selects refinement for display/export. It validates input revisions; restored archive timestamps are accepted after content hashing. Changing a local checkpoint invalidates selection. Moving to a new workstation or checkpoint directory may require **Resume** to rebind the manifest. Starting a new refinement run overwrites one cached variant rather than accumulating variants. Editing controls alone does not overwrite or deselect the completed result. An interrupted replacement can leave affected frames unavailable until Resume finishes; the original VideoMaMa cache remains intact. Incompressible 4K float mattes can still be large: two float alpha planes alone are about 63 MiB/frame at 3840×2160 before compression, plus baseline/trimap. Save projects include these caches. **Clear Matting** removes all stages; source frames are retained.

## Reproducible checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\validate_vitmatte_gpu.py
.venv\Scripts\python.exe scripts\benchmark_vitmatte.py --project YourProject.sammie --output .test-output\comparison --start 0 --frames 12 --preset Balanced
.venv\Scripts\python.exe scripts\validate_vitmatte_ui.py --session .test-output\comparison\input-session
```

The benchmark reads `.sammie` archives without changing them and extracts selected source/matte frames under its output directory. Alternatively, `--session` operates on an existing session copy and replaces that copy's refinement cache. It defaults to Base; use `--model small-composition-1k` to reproduce Small measurements. The UI validator also accepts `--model base-composition-1k` to override a copied project's saved choice. The benchmark runs B and C separately and records per-object-frame time, peak allocated/reserved GPU memory, output dimensions, alpha/edge differences and unaligned temporal variation. A uses existing mattes: its original inference time is explicitly unknown. The separate GPU validator measures fresh VideoMaMa followed by refinement.

Optional `--ground-truth <folder>` reads `<frame>/<object>.npy` float alpha or integer PNG and computes SAD (sum absolute error / 1000), MSE and a documented Gaussian/Sobel gradient MAE. The latter is not the standard benchmark's exact gradient-error implementation. Without ground truth, every metric is diagnostic. Alpha changes or lower variation do not prove improved accuracy. Comparison composites and detail crops are saved locally, including a later frame where temporal validation is active. See [executed results](VITMATTE_VALIDATION.md).

## Code map and attribution

`sammie/refinement/algorithms.py` owns trimaps, merging and temporal logic; `model.py` owns weight validation, preprocessing and tiling; `cache.py` owns atomic data and manifests; `pipeline.py` orchestrates the sequential pass; `display.py` selects a shared preview/export matte; `ui.py` owns controls and cancellable Qt worker. Small integration changes are in `sammie_main.py`, `sammie/matting.py`, `sammie/sammie.py`, `sammie/settings_manager.py`, `sammie/export_workers.py` and `videomama/pipeline_svd_mask_numpy.py`.

ViTMatte: Jingfeng Yao, Xinggang Wang, Shusheng Yang and Baoyuan Wang, *ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers* (2023), [official project](https://github.com/hustvl/ViTMatte). Original source is MIT; the official organization's [Base](https://huggingface.co/hustvl/vitmatte-base-composition-1k) and [Small](https://huggingface.co/hustvl/vitmatte-small-composition-1k) converted checkpoint cards declare Apache-2.0, as does Transformers. Existing application and other model licenses remain applicable. Weights and user footage are downloaded/read locally, not committed or redistributed with this change.
