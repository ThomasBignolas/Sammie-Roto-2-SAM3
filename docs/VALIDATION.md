# Validation record

Date: 2026-10-21. Platform: Windows, NVIDIA RTX 5060 Ti (16 GB), driver 610.62. These are reproducible short smoke tests, not sustained performance or production-quality benchmarks.

## Executed tests

| Check | Result |
| --- | --- |
| Automated tests (`unittest`, 23 tests) | Passed: coordinates/XYWH, binary masks, selected-ID matching and pruning, multi-object corrections, occlusion clearing, cancellation, error cleanup, subprocess protocol, seed persistence, UI control visibility, 4K proxy/source restoration, real PNG/EXR/FFV1 exports, exact rational rate, frame ordering/numbering, multiplex cache/refinement regressions. SAM 3 inference is mocked in these tests. |
| Original application and worker environments | Installed independently; original frozen lock preserved. |
| Worker runtime native CUDA check | Passed: official builders, BF16, Windows Triton distance-transform kernel. |
| Real SAM 2.1 Base checkpoint | Passed: six 1920x1080 frames, two objects, forward/backward, intermediate positive/negative correction, unchanged other-object mask, reversible preview, exact replay-mask comparison. |
| Real MatAnyone checkpoint | Passed: six full-size 1920x1080 alpha mattes, 720 working short edge, sequential model loading. |
| Official SAM 3 network with **random/untrained weights** | Passed: model construction, session, multi-object point prompts, forward/backward, intermediate corrections/restart, text and normalized-box API requests. |
| Official SAM 3.1 network with **random/untrained weights** | Same API checks passed after fixing the Windows SDPA backend selection. |
| MainWindow / embedded controls | Passed with real trained SAM 3 and SAM 3.1: single-image import, model button, text selection, point correction, one-frame tracking, independent manual objects and two-click box selection. Switching back to SAM 2 and point segmentation passed. Qt ran offscreen; confirmation/instance selection dialogs were answered by the test harness. |
| Official trained SAM 3 checkpoint | Passed six- and 60-frame 1920x1080 acceptance: box discovery, four text detections/two selected instances, forward/backward tracking, intermediate positive/negative correction, sequential MatAnyone, PNG alpha exports, identity restoration after reloading, and further tracking after matting. |
| Official trained SAM 3.1 checkpoint | Same checks passed on six and 60 frames; the final six-frame run also resumed propagation after matting/reconstruction. The initial run exposed masks being dropped on uncached frames; a second exposed a first-refinement tensor mismatch. Both were corrected in the compatibility layer and covered by regression tests before the successful runs. |
| Visual/alpha review | Inspected output masks and the actual interface screenshot. Exported segmentation alpha contains 0/255 values; MatAnyone output contains all 256 alpha levels. No ground-truth quality score is claimed. |

The random-weight diagnostics prove execution of real official network/kernel paths, not checkpoint compatibility, useful detections, mask quality or trained-model performance. They are explicitly isolated from production loading.

## Measurements from executed short tests

| Test | Measured value | Scope |
| --- | --- | --- |
| SAM 2 Base load + initialization | 0.872 s | Already-local checkpoint, six-frame fixture |
| SAM 2 Base forward tracking | 0.745 s / 6 frames (8.06 fps) | Two objects, includes mask writing; not a long-clip benchmark |
| SAM 2 peak allocated VRAM | 774.4 MiB | Torch allocation statistic in application process |
| MatAnyone load + processing | 4.922 s | Includes first-run ResNet download/load; not a steady-state speed estimate |
| MatAnyone peak allocated VRAM | 1271.6 MiB | Six frames, one object, 720 working short edge |
| Untrained SAM 3 API diagnostic peak | 5690 MiB | Four 320x180 images, fixed official encoder, synthetic prompts |
| Untrained SAM 3.1 API diagnostic peak | 6618 MiB | Same tiny fixture; Windows SDPA fallback |
| Trained SAM 3, six-frame forward tracking | 1.683 s (3.57 fps) | Two objects, source-size masks, includes transport/writes |
| Trained SAM 3, peak allocated VRAM | 5784 MiB | Includes discovery and both correction passes, six-frame fixture |
| Trained SAM 3, 60-frame forward tracking | 25.864 s (2.32 fps) | Two objects, 1080p, includes transport/writes |
| Trained SAM 3, 60-frame peak | 6159 MiB allocated / 6440 MiB reserved | Includes intermediate correction and both directions |
| Trained SAM 3.1, six-frame forward tracking | 1.792 s (3.35 fps) | Two objects, source-size masks, includes transport/writes |
| Trained SAM 3.1, six-frame peak | 6426 MiB | Includes discovery and both correction passes |
| Trained SAM 3.1, 60-frame forward tracking | 26.679 s (2.25 fps) | Two objects, 1080p, first 60 bundled dance frames |
| Trained SAM 3.1, 60-frame peak | 7260 MiB allocated / 7966 MiB reserved | Includes intermediate correction and both directions; MatAnyone runs after SAM exits |

Allocated VRAM excludes driver/context overhead, some third-party allocations and other applications. These results do not establish a memory bound for arbitrarily long clips. GPU memory statistics in the UI refer to the SAM worker, separately from the application process.

The post-matting two-frame continuation used a new worker and reached 5896 MiB for SAM 3 and 6623 MiB for SAM 3.1. Model-load/primary-tracking statistics are recorded separately from this restored session. Selected machine-readable reports are in [validation-results.json](validation-results.json).

## Reproduce

From the repository root after installation:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.runtime\sam3-env\Scripts\python.exe -m sammie.backends.check_runtime
.venv\Scripts\python.exe -m scripts.gpu_smoke
```

`gpu_smoke` uses the first six frames of the bundled `examples/example_dance.mp4`, resized to 1920x1080, and requires the SAM 2 Base and MatAnyone checkpoints. The original download dialogs are used if necessary. Reports and masks go into ignored `.test-output` directories.

After obtaining the authorized SAM 3 checkpoints:

```powershell
.venv\Scripts\python.exe -m scripts.validate_sam3 --model "SAM 3"
.venv\Scripts\python.exe -m scripts.validate_sam3 --model "SAM 3.1"
.venv\Scripts\python.exe -m scripts.validate_sam3 --model "SAM 3" --frames 60
.venv\Scripts\python.exe -m scripts.validate_sam3 --model "SAM 3.1" --frames 60
.venv\Scripts\python.exe -m scripts.ui_smoke
```

These commands use the actual production worker bridge. They require nonempty box and text detections, select instances, track two objects, apply intermediate positive/negative corrections, propagate both ways, release SAM memory, run MatAnyone, export source-sized PNG alpha sequences with numbering starting at 1001, and recreate the tracking session after matting. Each run records timings, GPU statistics, errors and `passed` in `results.json`. A missing checkpoint is a failure, never replaced with random weights. Inspect the exported masks visually as well: dimension and identity checks do not establish roto quality.

For native API debugging without gated weights only:

```powershell
.runtime\sam3-env\Scripts\python.exe -m scripts.sam3_api_smoke --version sam3
.runtime\sam3-env\Scripts\python.exe -m scripts.sam3_api_smoke --version sam3.1
```

## Remaining validation and limits

- Manually review text-selected identities, correction quality, small details, occlusion and reappearance on representative production footage. The bundled dance fixture is not a complete VFX quality benchmark.
- Test shots longer than the 60-frame fixture and 4K originals under realistic object counts. Only proxy/resize/export dimensions have been verified at 4K; no sustained 4K trained-model benchmark is claimed.
- SAM 3.1 lacks tracking-state CPU offload in the pinned API. State memory can increase with shot length; automatic reset-based chunking is deliberately absent because it would change temporal tracking.
- Text/box discovery is one initial query per selection. Boxes are not per-object correction boxes. SAM 3 reversible hover preview is unavailable.
- Rebuilding after matting or project load replays prompts, not every prior tracking action; exact bitwise equivalence is not promised. Seed identity is checked using masks before continuing.
- Variable-rate video exports at a constant rate; original decoded timestamps are stored separately. Audio and timecode handling retain upstream limitations.
- Large/Anime/EfficientTAM, MatAnyone2, VideoMaMa, and removal backends retain their implementation but were not run with real weights in this validation.
