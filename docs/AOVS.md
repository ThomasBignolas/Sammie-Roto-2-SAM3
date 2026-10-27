# Depth and normal AOVs

The fourth sidebar tab, **AOVs**, follows Object Removal. It operates on the original imported RGB frames and does not require segmentation or matting. This integration uses LiveActionAOV's vendored NormalCrafter and Video Depth Anything implementations, with a separate official DVD adapter. It does not embed LiveActionAOV's separate desktop application or all its other passes.

## Install and launch on Windows

Use branch `feature/aov-depth-normals` of this private repository. Existing users run **install_aov.bat** once, then launch **run_sam3.bat** as usual. The AOV environment and checkpoints have already been installed on the development workstation.

For a fresh checkout:

```powershell
git clone --branch feature/aov-depth-normals https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3.git
cd Sammie-Roto-2-SAM3
.\install_sam3.bat
.\setup_models.bat
.\install_vitmatte.bat
.\install_aov.bat
.\run_sam3.bat
```

`install_aov.bat` creates `.runtime/aov-env`, synchronizes `requirements-aov.lock`, and checks out pinned upstream sources under `.runtime/aov-sources`. It never installs AOV dependencies into the UI or SAM environments. The worker uses Python 3.12, PyTorch 2.10/CUDA 12.8, Diffusers 0.36, Transformers 4.57.6 and NumPy 1.26.4; the UI retains its existing versions. No SageAttention, xFormers, DeepSpeed, CUDA compiler or ComfyUI installation is required.

Public checkpoints download automatically on first Generate. For advance download, run **setup_aov_models.bat**. All AOV weights total approximately 15.3 GB on disk; the runtime and source checkouts need additional space. No video or project content is uploaded. Network access is only needed for installation/downloads; completed local model downloads work offline.

## Workflow

1. Import footage or open a `.sammie` project. Set timeline In/Out to one continuous shot.
2. Open **AOVs**. Select **Depth AOV**, **Normal AOV**, or both.
3. Choose **DVD v1.1** for depth, or **Video Depth Anything Small** for a faster option. Normals use **NormalCrafter**.
4. Start with a **512-pixel processing long edge**, model CPU offload, and the default temporal settings. Try **768** for more spatial detail. These are inference dimensions, not delivery dimensions.
5. Click **Generate / Reuse cached AOVs**. The progress window supports cancellation; errors identify missing dependencies, invalid settings, or inference failures.
6. Select **Depth**, **Normals**, or **Original plate** in Timeline preview and scrub normally. A missing/stale AOV is explicitly labeled in the viewer.
7. In **Multilayer EXR contents**, choose **Include original RGB plate** and **Include generated alpha + object layers** (both default on). Choose **Matting** to use the selected ViTMatte comparison stage, or **Segmentation masks** for tracked masks. Generate the chosen alpha first; for AOV-only footage, uncheck alpha inclusion.
8. Choose the first output number (1001 by default; original sequence start is used when available), then **Export multilayer AOV EXR sequence** to a new folder. Every frame contains all selected layers in one file. Only the current In/Out range is exported, in order. Existing sequence files and metadata are never overwritten automatically.

The tab is scrollable on small windows. AOV settings, numbering and cached data are included in the existing project-save workflow. The regular mask/matte exports remain available through their existing controls.

### Background checkerboard

In **Segmentation-BGcolor** or **Matting-BGcolor**, choose **Checkerboard** beside the color swatch. Choose **Solid color** to restore the color picker. The choice is saved with the project. BGcolor exports bake in the chosen background; alpha/matte exports retain their original transparency/data behavior.

## Channels and Nuke

Each `aov.1001.exr`, etc. is a source-sized ZIP-compressed **32-bit float**, single-part multichannel EXR (commonly called multilayer EXR). Depth, normals and alpha are data: read with **raw / no color transform** in Nuke. Apply a color transform only to the RGB plate if needed.

| Channel | Meaning |
| --- | --- |
| `R`, `G`, `B` | Optional original imported RGB plate, normalized to 0–1, **straight/unpremultiplied**. Values retain the display-referred PNG/JPEG color encoding used by Sammie; they are not automatically converted to scene-linear. Original camera RAW/HDR latitude is not reconstructed. |
| `A` | Optional generated alpha: per-pixel maximum of the object mattes, matching Sammie's union behavior. Uses existing results, with no inference during export. |
| `Object_<id>.Y` or `<id>_<name>.Y` | Separate object alphas using the existing mask export naming convention. Combined-object matting exports `0_Combined.Y`, because that matting operation produced one combined alpha. |
| `Z`, `depth.Z` | Identical relative depth: 0 near, 1 far. The raw disparity-like output is normalized over the whole processed range and inverted. This is an artistic depth control, not a calibrated physical Z-buffer. |
| `Z_raw` | Unnormalized model output. Both selected depth checkpoints use larger raw values for nearer surfaces. Preserve this for custom grading/normalization. |
| `N.x`, `N.y`, `N.z` | Signed, unit-length normal components. The default preserves NormalCrafter's native axes; **Flip Y and Z** supports an alternative camera convention. Normals are renormalized after source-resolution interpolation. |

Use a Shuffle to view `N.x/y/z` as RGB, then map `[-1,1]` to `[0,1]` for display only. `depth.Z` is provided for Nuke depth-channel selection. Read the EXR's `framesPerSecond` attribute or `aov_metadata.json` for the original rational frame rate; Nuke script FPS may still need to be set explicitly. The JSON also records source indices, output numbering, settings, model revisions and channel conventions. This preserves metadata, rather than implying EXR playback automatically controls the host application's timeline.

For compositing, use a raw Read, isolate RGB, convert that plate from its known source display encoding to your working space, then Premult with `A` as needed. Keep `N`, `depth`, raw Z and object masks out of that RGB transform. RGB is deliberately not multiplied by `A`, so the original background remains recoverable.

Matting export honors the selected **Original VideoMaMa / ViTMatte refined / Temporally validated** view and its existing cache-validity/fallback behavior. Float VideoMaMa and refinement caches retain float precision; legacy PNG mattes retain their existing precision. Grow/gamma settings apply. Segmentation alpha uses saved masks with the existing hole/dot/border/grow processing; optional neural antialiasing is not run by the AOV exporter. Missing individual-object frames are zero and listed in metadata; a frame with no selected alpha at all stops export rather than silently delivering an empty result. Export-only checkboxes, alpha source and output numbering do not invalidate depth/normal caches.

Normal axes are model predictions, not a camera solve. The upstream LiveActionAOV adapter contains contradictory comments about OpenCV/OpenGL axes; this integration deliberately preserves the model's native output by default and records the chosen conversion. Check your relighting setup against a known surface before delivery. No world-space normals, camera intrinsics, metric distances, calibrated position pass or geometry are inferred by this tab.

## Memory and temporal processing

- SAM is offloaded and matting/removal models unloaded before the worker starts. DVD/VDA and NormalCrafter run sequentially. The worker exits after the job, releasing its CUDA context. SAM is restored afterward.
- DVD uses BF16, inference mode, tiled VAE encode/decode, and official DiffSynth offloading. Its upstream inference path omits stage-onload calls; the adapter reconnects these at VAE encode, DiT inference, and VAE decode. LoRA parameters are loaded strictly and merged as in official inference, then redundant merged adapter storage is released.
- DVD defaults to **17-frame windows with 5-frame overlap**. Overlap scale/shift alignment and blending follow its released inference approach. Short clips are padded to `4n+1`, then cropped to the exact original frame count. Smaller windows can lose temporal context; long sequences can still drift. Run separate shots separately.
- NormalCrafter defaults to **14-frame windows, 4-frame overlap, decode batch 2**. Its official overlapping latent propagation remains intact. RGB image-encoder/VAE inputs are encoded in small batches instead of placing the entire RGB clip on the GPU at once. There is no independent per-window restart of its tracker/state.
- Decode batch size affects VAE temporal context as well as memory. Larger batches, up to the upstream default of 7, may help temporal decoding but require more memory. Compare moving footage before final delivery.
- CPU offload is configurable. Sequential offload is slower. The measured tests use model offload; sequential offload remains an available upstream facility, without a separate full-shot benchmark.
- Inputs are resized with preserved aspect ratio, padded to the model's spatial alignment, and cropped accurately. The adapter avoids LiveActionAOV's hardcoded 1024×576 stretch and the upstream NormalCrafter numpy-unpadding layout problem.
- DVD requires CUDA/BF16. VDA Small and NormalCrafter support CPU execution, which is intended for fallback/small tests and is much slower.

For **1080p or 4K delivery**, begin at 512, then compare 768 on a representative shot. Export always restores original dimensions; upsampling does not recover details never inferred at working resolution. 1024 is available but is not guaranteed to fit 16 GB on every clip. The UI reports actual peak allocated memory; other applications and reserved CUDA memory also consume VRAM.

## Cache and interruption behavior

Float arrays are cached at inference resolution under the project session's `aov` folder. Keys include source-frame content/stat signatures, processed frame range, relevant per-pass settings, source revisions and checkpoint revisions. A depth-only setting change keeps normals reusable, and vice versa. Changing display mode or export numbering requires no inference. Truncated/nonfinite cached arrays are recomputed.

Only complete passes are activated. Cancellation preserves previously completed passes; an interrupted pass restarts from its beginning. There is no claim of resuming mid-DVD-window or mid-NormalCrafter latent propagation. Cached settings variants are retained so you can switch back; there is currently no automatic cache eviction. Full-resolution EXRs are written only on export, not duplicated in the working cache.

Cancellation stops only the job-owned worker. Partial export files already completed remain available; choose a new export folder when restarting. An incomplete `.tmp` file is never presented as a completed EXR.

## Models, attribution and selection

Reviewed on 2026-10-21:

| Component | Selected version / terms |
| --- | --- |
| [LiveActionAOV](https://github.com/lettidude/LiveActionAOV) | Source `8c2a231785bdf6a016a39c722010f90026b035f7`, MIT, Leonardo Paolini. Used for vendored video-normal/depth inference code and channel conventions. |
| [DVD](https://github.com/EnVision-Research/DVD) | Official v1.1 checkpoint `dvd_1.1.safetensors`; code Apache-2.0; **weights CC BY-NC 4.0**. Not licensed for paid/commercial VFX by this integration. |
| [NormalCrafter](https://github.com/Binyr/NormalCrafter) | Official `Yanrui95/NormalCrafter`; code MIT, official model card Apache-2.0. Built from Stable Video Diffusion: review the upstream/base-model terms for your intended commercial use; the authors provide a business-license contact. |
| [Video Depth Anything Small](https://github.com/DepthAnything/Video-Depth-Anything) | Official Small checkpoint, Apache-2.0. Included as the lighter depth alternative. Do not infer the same license for other model sizes. |
| [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Original 1.3B backbone and VAE for DVD, Apache-2.0; no text-encoder weights downloaded. |

Exact immutable checkpoint revisions are in `sammie/aov/versions.py`. Source/license files are downloaded directly from upstream and retained locally. No third-party weights are committed to this repository.

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) was also reviewed. It targets joint spatial geometry, depth and poses and has a separate streaming implementation. This version implements the specifically requested DVD/NormalCrafter video pipeline plus the lighter VDA option; it does not claim one model is universally best or that every recent geometry model is integrated. Accuracy depends on the shot; reflective/transparent surfaces, sky, motion blur and fine silhouettes remain difficult.

## Validation and development

See [AOV validation](AOV_VALIDATION.md). Core inference is tested with real public checkpoints on the 16 GB RTX 5060 Ti and copied user footage. No ground-truth depth/normals are available, so reported differences and timings are diagnostics, not accuracy evidence.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.runtime\aov-env\Scripts\python.exe -m scripts.validate_aov --source my_project.sammie --model both --count 18 --edge 512
.venv\Scripts\python.exe -m scripts.validate_aov_ui --session .test-output\my_copied_session --edge 640 --last 2 --matte-project my_project.sammie
```

The validation scripts create/use isolated `.test-output` directories. They do not overwrite the supplied project. `validate_aov_ui` requires an isolated copy and runs actual inference/export through the tab; only unrelated SAM startup and native folder selection are replaced for automation.

## Optional recovered foreground RGB

The AOV tab also supports the local optional **CorridorKey Foreground RGB** pass.
It can run by itself, or alongside depth and normals, and requires an existing
selected alpha. See [the CorridorKey workflow](CORRIDORKEY.md) before using it.
`foreground.R/G/B` is straight recovered foreground color; `foreground.A` is the
unchanged alpha used for that recovery. Existing plate, alpha, depth and normal
layers keep their names. An explicitly selected original EXR override supplies
both the plate RGB and the foreground input, in the selected color space.
