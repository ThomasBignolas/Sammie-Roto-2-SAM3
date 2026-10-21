# ViTMatte validation — 2026-10-21

This is a tested optional integration, not a certification of production matte quality. No ground-truth alpha was supplied. Numerical alpha/edge changes and temporal variation below are **diagnostic**, not accuracy measurements. The original VideoMaMa matte remains selectable.

Hardware: Windows, NVIDIA RTX 5060 Ti 16 GB, PyTorch 2.11.0+cu130, Transformers 5.8.0, FP32 ViTMatte unless stated. The initial measurements below used official **ViTMatte Small**, revision `6a58ad7646403c1df626fbd746900aec7361ea1d`, verified SHA-256 `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`. **Base is now the default**; its separate measured results appear below. The official source configuration was inspected at `8cd7ef068380977c3962c4cb733cb1fe7f2241a5`.

## Executed checks

- **49 automated tests passed** with `.venv/Scripts/python.exe -m unittest discover -s tests -q`. This includes the 23 existing SAM/backend/import/export tests and 26 refinement tests. Coverage includes confidence trimaps, empty/full masks, thin/disconnected structures, boundary contact, semi-transparent and low-alpha edges, resolution scaling, exact preservation outside unknown regions, zero-strength merge, overlapping tiles, temporal changes/cuts/gaps, cache and checkpoint invalidation, project timestamp restoration, corrupted cache recovery, checkpoint-loading errors, cancellation/cleanup, disabled preview behavior, settings/presets, and float EXR numbering/channel equality. The Base follow-up adds model/architecture matching, restricted binary loading with hash/path checks, and model-switch cache invalidation; UI tests also verify Base defaults and persisted model selection.
- Real pretrained ViTMatte **CUDA FP32, CUDA BF16 and CPU FP32** inference executed. Non-multiple-of-32 geometry and strict checkpoint loading passed. No inference was substituted by another matting model.
- Real **VideoMaMa → float cache → unload → ViTMatte** executed on four 1920×1080 frames from the bundled example, with VideoMaMa at short edge 480 and chunk 3 / overlap 1. The rough ellipse segmentation masks deliberately test plumbing, not segmentation quality. The overlap boundary produced four aligned float32 alpha files and zero legacy-precision fallbacks.
- Default VideoMaMa uint8 output was compared with explicitly requested float output, resetting the RNG for the same two-frame inputs. Quantizing the float return reproduced the default uint8 frames **exactly**.
- The **actual MainWindow** refinement button, QThread, modal progress, A/B/C preview and EXR sequence export executed on six copied source-size frames from the supplied project. Exported `Object_0.Y` float values exactly matched the selected C matte, with 1001-based sequence numbering. The Matting tab's view switching and scrolling were exercised. The test isolated startup/console setup; model inference, worker and exports were real.
- The actual **Run Matting** button also completed the automatic VideoMaMa → unload → ViTMatte handoff on the four-frame 1080p test session, followed by A/B/C preview and float EXR export.
- The documented Windows `install_vitmatte.ps1` completed against the frozen lock. Lock inspection showed **no existing package version changes**. SAM 3's worker environment was not changed.

Qt's offscreen platform emits a missing bundled-font-directory warning; UI validation explicitly loads Windows Segoe UI. Existing worker protocol error tests intentionally print unknown-command tracebacks. These are expected test diagnostics.

## Initial Small measurements: cached VideoMaMa mattes

The user's read-only project contains 194 frames at **4608×3164, 24 fps**. Isolated copies of frames **0–5** and **60–83** were tested. The whole 194-frame clip was not processed. Its existing mattes are 8-bit PNG; the original VideoMaMa inference duration is unavailable. Timings below are the **added refinement pass**, including lazy ViTMatte loading, frame I/O, merging and cache writes, excluding initial download/verification. Each frame contains one object. The source dimensions were preserved in all results.

| Sample / preset | Temporal validation | Seconds/frame | Peak allocated MiB | Peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| First 6 / Balanced, 1080 processing short edge | Off | 2.003 | 811 | 1238 |
| First 6 / Balanced | On | 2.013 | 811 | 1244 |
| First 6 / Quality, 2160 processing short edge | Off | 3.131 | 1402 | 1874 |
| First 6 / Quality | On | 3.057 | 1402 | 1886 |
| Frames 60–83 / Balanced | Off | 1.621 | 811 | 1238 |
| Frames 60–83 / Balanced | On | 1.741 | 811 | 1244 |

Small timing reversals reflect cache/warmup/system variability, not a speed benefit from extra temporal work. GPU figures are process-local PyTorch allocator peaks, not total VRAM use including other applications. Source RGB and most intermediate arrays live in system memory. Native source-resolution processing and a full shot on CPU have not been benchmarked.

On frames 60–83, mean absolute alpha change from VideoMaMa was 0.001060 without temporal validation and 0.001044 with it. Unknown-region change was 0.01166 and 0.01149. Unaligned refinement-delta variation decreased from 0.0005732 to 0.0005556. Total unaligned matte variation was 0.0014759 for VideoMaMa, 0.0015701 for B and 0.0015692 for C. These measures include legitimate motion and do not establish which matte is more accurate.

On the first-six-frame Quality test, total matte variation actually increased after temporal validation (0.005960 to 0.006348), although refinement-delta variation decreased. This is a concrete reason to keep validation optional and compare C against B. The gate's changing confidence can affect variation. Fine hair appeared different in the inspected composites; there is no ground-truth basis for claiming a quantitative improvement.

Local visual artifacts (not published with user footage):

- `.test-output/vitmatte-balanced/edge-comparison-0.jpg`
- `.test-output/vitmatte-quality/edge-comparison-0.jpg`
- `.test-output/vitmatte-motion/edge-comparison-00072-0.jpg`
- `.test-output/vitmatte-content/vitmatte-ui.png`

## Small: fresh VideoMaMa integration and precision

The four-frame 1080p integration run measured **7.36 s** for VideoMaMa including loading, with **4050 MiB** peak allocated GPU memory. After unload, allocated memory was **9.1 MiB** before refinement. The 720-short-edge ViTMatte pass with temporal validation took **1.91 s**, peak **778 MiB** allocated. It read float baseline caches for all four frames. These low-resolution VideoMaMa settings should not be interpreted as a measurement of 1080p internal VideoMaMa inference.

For one 256×144 input with an all-unknown trimap, CUDA BF16 differed from CUDA FP32 by mean absolute alpha error **0.00293**, maximum **0.02585**. CPU FP32 differed by MAE **0.000315**, maximum **0.00173** (CUDA backend/TF32 choices affect numeric equivalence). CPU ran successfully; this small check is not a CPU throughput benchmark. FP32 remains the default, and BF16 is opt-in.

Machine-readable measured runs are in [vitmatte-validation-results.json](vitmatte-validation-results.json). Reproduce using [the commands and settings in the workflow guide](VITMATTE.md). Synthetic/unit-test expectations are correctness checks, not trained-model accuracy results.

## Base: larger model follow-up

Official `hustvl/vitmatte-base-composition-1k`, revision `bf486d01a7d9e3dbcc8400f7942835caf0eaf76e`, was downloaded and verified against SHA-256 `b2521bcc4b719fb24611c39605b6642162fd7502e69b3cc846506ca921757b41`. The loaded model contains **96,686,561 parameters**, with a 768-channel backbone. Strict loading succeeded using the official PyTorch state dictionary and `weights_only=True`.

The same six supplied frames 0–5 were refined at both presets, with original **4608×3164** output dimensions. Measurements are the added refinement stage, including model load, frame I/O, merge and cache writes; download/checkpoint resolution and separate diagnostic calculations are outside the timer. No VideoMaMa rerun was needed.

| Base preset / stage | Seconds/frame | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: |
| Balanced / B | 2.540 | 1256 | 1394 |
| Balanced / C | 2.452 | 1254 | 1394 |
| Quality / B | 5.313 | 2938 | 3128 |
| Quality / C | 5.219 | 2940 | 3140 |

These are process allocator measurements on this sample, not total workstation VRAM or a worst-case guarantee. Small timing reversals reflect system/warmup variability. Balanced uses short edge 1080 and tile 768; Quality uses short edge 2160 and tile 1024. Both restore alpha to source dimensions.

The actual MainWindow with Base selected completed the refinement worker on the six copied frames, A/B/C preview and exact float `Object_0.Y` EXR export with 1001-based numbering. A separate fresh four-frame VideoMaMa test took **8.88 s** including loading, peaked at **4050 MiB**, and released GPU allocation to **9.1 MiB** before Base loaded. Base refinement at short edge 720 took **2.93 s** total and peaked at **1186 MiB**. All four frames used unquantized baseline caches. Default VideoMaMa uint8 output again matched quantized float output exactly for the same inputs/RNG.

Base ran on **CUDA FP32, CUDA BF16 and CPU FP32**. On the 256×144 all-unknown test input, BF16 differed from CUDA FP32 by MAE **0.02331**, maximum **0.72074**. CPU FP32 differed by MAE **0.00106**, maximum **0.05136**; the application's CUDA/TF32 settings affect this comparison. **FP32 remains recommended for Base final mattes.** These comparisons measure numerical sensitivity, not accuracy against ground truth.

Quality's mean alpha change from the original was 0.00750 for B and 0.00663 for C. Mean unaligned refinement-delta variation decreased from 0.00324 to 0.00293, while total matte variation increased from 0.00572 to 0.00614 (original 0.00569). Thus temporal validation remains optional; reduced delta variation does not establish a more accurate or smoother final matte. No quantitative superiority over Small or the original matte is claimed.

Detailed data: [vitmatte-base-validation-results.json](vitmatte-base-validation-results.json). Local visual comparisons are in `.test-output/vitmatte-base-balanced/` and `.test-output/vitmatte-base-quality/`; user images are not committed. Reproduce with `scripts/benchmark_vitmatte.py --session <copy> --frames 6 --preset Quality --model base-composition-1k` and `scripts/validate_vitmatte_gpu.py --model base-composition-1k`. The UI check accepts the same model option. Use `--model small-composition-1k` to reproduce earlier Small tests.

## Remaining limits

No ground-truth hair, fur, motion-blur, transparency or occlusion dataset was evaluated. No full-shot accuracy, universal temporal improvement, native 4K throughput or 16-GB worst-case guarantee is claimed. Tiling limits transformer context; proxy resolution can lose hair; flow can fail. Existing mattes may already have quantization loss. PNG/video exports remain on the original 8-bit rendering path; use EXR for float alpha. A local checkpoint, another GPU, or a different driver/runtime can change results. Keep refinement reversible and review the exported matte in the intended Nuke composite.
