# AOV integration validation

Executed on Windows, 2026-10-21. This report covers the fourth AOVs tab, checkerboard background choice, and combined RGB/alpha/depth/normal EXR export. It does not establish ground-truth depth, normal accuracy or production suitability for every shot.

## Automated checks

**76 tests passed**, using `.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Twenty AOV/checkerboard tests cover aspect/padding, temporal-window coverage, overlap alignment, normal normalization, invalid settings, independent settings invalidation, changed sources, corrupted cache recovery, model failure, nonfinite output, model cleanup, missing frames, export numbering/dimensions/float channels/rational FPS, whole-shot depth normalization, RGB channel order and straight RGB preservation, combined and independent object layers, VideoMaMa float precision, all three ViTMatte comparison stages, alpha-source selection, UI persistence and actual job-owned subprocess cancellation. Model predictions in these unit tests use explicit test doubles; real checkpoint validation is separate below.

The complete run also passes the pre-existing SAM backends, additive text concepts, import/export, UI, ViTMatte algorithms, temporal validation, cache invalidation and cancellation tests. Expected negative-test errors and Qt offscreen font/deprecation warnings are not model validation results.

## Real checkpoint execution

Hardware: **NVIDIA RTX 5060 Ti, 16 GB VRAM**, driver 610.62. Another user application remained open and occupied GPU memory; it was not terminated. Pinned AOV runtime: Python 3.12, PyTorch 2.10.0+cu128, Diffusers 0.36.0, Transformers 4.57.6, NumPy 1.26.4. Checkpoints were downloaded publicly and actually loaded; there were no substituted predictions in these runs.

Source: copies of the supplied 194-frame project, **4608 × 3164, 24 fps**, showing a person and camera. Runs used the first 18 frames unless stated otherwise. Original projects were not changed or uploaded. There is no reference depth/normal ground truth.

| Model / processing long edge | Frames | Time, including model loading | Seconds/frame | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| DVD v1.1 / 512 | 18 | 23.85 s | 1.325 | 3075 MiB | 3296 MiB |
| NormalCrafter / 512 | 18 | 30.41 s | 1.690 | 3910 MiB | 4112 MiB |
| DVD v1.1 / 768 | 18 | 36.64 s | 2.035 | 3257 MiB | 3476 MiB |
| NormalCrafter / 768 | 18 | 15.40 s | 0.856 | 4865 MiB | 5440 MiB |
| Video Depth Anything Small / 512 | 5 | 4.37 s | 0.874 | 1572 MiB | 2154 MiB |

Timings include model initialization, inference and float caching, excluding downloads and EXR delivery. These are diagnostic runs under different system load, not controlled throughput comparisons: a CPU validation was running during the 512 sequential run, and warm OS/model caches differ. In particular, the faster recorded 768 normal run does not imply that increasing resolution makes inference faster. GPU figures are process-local PyTorch allocation/reservation, not total workstation VRAM. The measured 512 run retained 9.125 MiB of allocated library buffers after model destruction; the worker then exited to release its CUDA context. DVD and normals were processed sequentially.

The 18-frame runs exercise more than one DVD window and more than one NormalCrafter temporal window, preserving overlap processing. Inference dimensions were 512 × 352 and 768 × 527 before model-specific padding. Source-sized EXR exports restored **4608 × 3164**. Raw numbers/settings are retained in [aov-validation-results.json](aov-validation-results.json). The earlier 768 run measures model execution; the final export convention (near=0, far=1) was subsequently checked at 512 and through the UI at 640.

CPU fallback also executed actual weights on **one 96 × 64 frame only**: VDA Small 1.83 s; NormalCrafter 50.23 s, with its short-clip padding. This is a functionality check, not a full-resolution CPU benchmark. DVD intentionally rejects CPU execution.

## Actual application workflow and multilayer delivery

The validation script instantiated the real MainWindow and used its fourth tab and QProcess worker. It checked Generate, source-aligned timeline previews, a second Generate using cached results, and Export. Only unrelated SAM startup, console redirection and the native destination chooser were replaced for automation.

The final run exported **three 4608 × 3164 multilayer EXRs** using cached real-model DVD/NormalCrafter results and the actual VideoMaMa PNG alpha copied from the supplied project. Each file contained:

```text
R G B A Object_0.Y Z depth.Z Z_raw N.x N.y N.z
```

Every RGB and alpha value was checked against the original imported plate/matte. Every depth/normal channel was checked against the source-sized float cache conversion, with a tolerance for a few float32 interpolation ULPs between isolated OpenCV versions. Float ViTMatte export and independent/combined objects are separately verified in automated tests. This final real-content export used the existing VideoMaMa matte, not newly generated ViTMatte output.

UI screenshots were captured and inspected for aligned normal/depth previews and checkerboard BGcolor. All media artifacts remain local and ignored by Git. The validation script can be reproduced with the commands in [AOVS.md](AOVS.md); its optional `--matte-project` mode checks frame identity before copying alpha, and expects one existing unrefined object matte.

## Limits of this validation

- No ground-truth error or claimed accuracy improvement. A plausible preview and lower temporal variation do not establish correct geometry.
- No full 194-frame inference run, 1024-edge run, long-shot drift benchmark, camera solve or calibrated XYZ position pass.
- Full-shot sequential offload and cancellation during a live CUDA kernel were not separately benchmarked. Owned-worker cancellation was executed with a real Qt subprocess; model-run cancellation uses that same process mechanism.
- NormalCrafter's native axes need checking against the target relighting setup. Depth is relative, not metric or a physically calibrated Z-buffer.
- EXRs were read and channel-validated with OpenEXR; Nuke itself was not available for an interactive import test.
- DVD weights are non-commercial. Review the model terms described in [MODEL_LICENSES.md](MODEL_LICENSES.md) before paid use.
