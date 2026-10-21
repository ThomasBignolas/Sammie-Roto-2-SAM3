# Architecture and compatibility

## Inspected sources

| Repository | Inspected revision | Finding |
| --- | --- | --- |
| [Zarxrax/Sammie-Roto-2](https://github.com/Zarxrax/Sammie-Roto-2) | `e759816843a741d25411db7f360b0ce286259d76` (2.5.0) | Authoritative upstream; PySide6 UI, SAM 2 predictor, model-specific matting managers, PNG mask cache and PyAV/OpenEXR exports. |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3/tree/7f473cfd983d5eb43040ff608f66bff0d7bdcb53) | `7f473cfd983d5eb43040ff608f66bff0d7bdcb53` | Official request/session predictors for SAM 3 and SAM 3.1; distinct model builders and tracking behavior. |
| [FrameCGI/Sammie-Roto-SAM3](https://github.com/FrameCGI/Sammie-Roto-SAM3) | `23d14c6ca119630b17fb0b2d652fb857118e98b8` | At this revision, code inspection found the SAM 2 implementation and no SAM 3 inference integration. The repository name alone is not evidence of support. |
| [tenetmotion/Samosa](https://github.com/tenetmotion/Samosa) | `94b925ba1f4174c680d83b8401a6deee6f885b0b` | Inspected its isolated worker/environment and SAM 3.1 setup approach. No Samosa implementation code was copied. |

SAM 3.1 is an official release, announced March 27, 2026 in Meta's [release notes](https://github.com/facebookresearch/sam3/blob/7f473cfd983d5eb43040ff608f66bff0d7bdcb53/RELEASE_SAM3p1.md). It uses Object Multiplex and `sam3.1_multiplex.pt`, not the SAM 3 checkpoint. Its Hugging Face repository does not provide a Transformers integration; this application uses Meta's native implementation for both versions.

## Application boundary

`SamManager` in `sammie/backends/manager.py` owns model selection and lifecycle. Both backends implement loading, session initialization, point editing/replay, removal/reset, and `iter_masks`. `TrackingMixin` supplies the existing tracking controls and source-size PNG writing. UI-only text/box controls live in `sammie/sam3_controls.py`.

`Sam2Backend` contains the original point replay, intermediate correction and reversible preview behavior. `Sam3Backend` translates application coordinates and selected IDs, retains a prompt journal and restores masks to source dimensions. It communicates through a persistent subprocess (`bridge.py`) with the official model runtime (`worker.py`). There is one model per worker, with independent native sessions for selected text/box concepts. The worker does not import Qt, the bundled SAM 2 package or MatAnyone.

JSON requests pass control information. Masks use acknowledged, reusable PNG transport files; only the selected masks are written into the existing `temp/masks/00000/0.png` layout. This is a necessary GPU-to-CPU boundary for the original CPU display/matting/export interface. Backpressure prevents the worker from replacing a mask before the UI has consumed it. Cancellation closes the current iterator without loading another model.

## API adaptations

- Points use normalized XY coordinates and explicit positive/negative labels. Boxes use normalized **XYWH**, as required by the official video API, not SAM 2 pixel XYXY.
- `out_binary_masks` are binary masks, not SAM 2 logits. Masks become source-size uint8 0/255 arrays. Absent objects produce explicit zero masks to erase stale propagation outputs.
- Detector IDs map to independent application object IDs. Unselected detections are removed with `is_user_action=False`. Saved selections use compressed seed masks and an IoU threshold of 0.8 to match identity after session recreation. A failed match is an error, not an arbitrary new identity assignment.
- `concepts.py` splits explicit lists (`person, camera`, `person and camera`) into independent detector queries; double quotes preserve compound descriptions. The official [SAM 3 semantic prompt path](https://github.com/facebookresearch/sam3/blob/7f473cfd983d5eb43040ff608f66bff0d7bdcb53/sam3/model/sam3_video_inference.py#L865) resets its session; SAM 3.1 does likewise. A new query therefore gets its own native session, sharing the existing checkpoint. Temporary discovery sessions close after copying candidates to CPU. Accepting re-detects and matches only selected seeds. Existing sessions are untouched by discovery/cancellation. Selected concepts retain independent histories, including intermediate point corrections.
- Worker object IDs encode session scope plus native ID (`group * 1,000,000 + native`); IDs remain ordinary sequential object IDs in the application/export. The worker advances all selected sessions once per frame and emits a single aligned result; missing/misaligned concept frames fail explicitly. Cancellation closes every native iterator. Point requests route back to the correct concept. Saved records retain their query scope, and older single-query records default to scope 0. Restoring after matting reconstructs all selected concepts without union prompts or approximating detector masks with points.
- SAM 3 propagation uses `force_tracker_propagation=True` to track selected objects. SAM 3.1 lacks that flag: the adapter temporarily marks all selected IDs for tracker refinement using its `add_action_history` API before each directional pass. It removes that synthetic marker after the pass so the next real user correction still performs the official singleton object extraction. This avoids treating later passes as cached result fetches without incorrectly marking detector objects as user-refined.
- The pinned SAM 3.1 `_build_sam2_output` returns early when a frame has no cached detector output, dropping fresh tracker masks. The compatibility wrapper seeds an empty cache entry before calling that method, preserving the official merge/suppression behavior on newly visited frames. Real trained-checkpoint tests and a regression test cover this case.
- Official backward propagation excludes the starting frame; forward propagation includes it. The backend validates every returned index, detects duplicates/missing frames, and uses source indices unchanged in the disk cache.
- The pinned base predictor's `start_session` forwards `offload_state_to_cpu` to SAM 3.1, whose `init_state` does not accept it. `worker.py` filters initialization arguments by the actual model signature, then registers the session using the base predictor's session structure. This private registration boundary is intentionally localized and tied to the source commit pin.
- The official asynchronous frame loader eagerly retains the whole decoded video. `frame_cache.py` replaces only that loader in the worker with a bounded eight-frame cache, retaining official image decode/normalization and all temporal model state.
- On Windows, PyTorch reports FlashAttention unavailable. The SAM 3.1 decoder still requests FlashAttention even with `use_fa3=False`. `compat.py` narrowly replaces its SDPA context with available cuDNN/efficient/math choices. Both the fallback and actual attention call were exercised on CUDA.

Do not change the official source pin without re-running the API and trained-checkpoint acceptance tests. Compilation and FA3 are disabled in this Windows profile. No automatic model-state chunk resets or lossy temporal compression are implemented.

## Dependency isolation

| Component | Python | Torch / torchvision | NumPy | Configuration |
| --- | --- | --- | --- | --- |
| Original application, SAM 2, MatAnyone | 3.12 | 2.11.0+cu130 / 0.26.0 | Original `uv.lock` (2.5.2) | `.venv`; original frozen dependency lock |
| SAM 3 / SAM 3.1 | 3.12 | 2.10.0+cu128 / 0.25.0 | 1.26.4 | `.runtime/sam3-env`; `requirements-sam3.lock`; official source installed separately with `--no-deps` |

The Windows worker lock includes Triton Windows 3.6.0.post26 and all transitive runtime dependencies. The installer does not silently upgrade/downgrade the application's critical packages. `requirements-sam3.in` is the editable source for regenerating the worker lock; the checked-in lock is Windows-specific. Linux/macOS SAM 3 installation is not validated by this upgrade.

`SAMMIE_SAM3_PYTHON` may select an alternate compatible worker interpreter. Model manifests pin checkpoint revisions and expected sizes. Checkpoints, environments, research clones, credentials, caches and generated test media are excluded from Git.

## Feature comparison

| Capability in this application | SAM 2.1 | SAM 3 | SAM 3.1 |
| --- | --- | --- | --- |
| Still image / video points, positive and negative | Yes | Yes | Yes |
| Independent objects | Original behavior | Up to 16 selected | Up to 16 selected |
| Text, multiple detected instances, selection dialog | No | Yes | Yes |
| Box discovery, optionally with text | Not exposed in original UI | Yes | Yes |
| Intermediate-frame point corrections and restart | Yes | Yes | Yes |
| Forward / backward / one-frame tracking | Yes | Yes | Yes |
| Shift-hover reversible preview | Yes | No | No |
| Multiple descriptions and successive text/box additions | N/A | Yes; independent sessions | Yes; independent sessions |
| CPU frame offload / bounded worker image cache | Original loader | Yes | Yes |
| Official tracking-state CPU offload used | Original behavior | Yes | Unavailable in pinned API |
| MatAnyone and original mask/alpha exports | Yes | Common mask adapter | Common mask adapter |
| Actual trained checkpoint validation | Passed for Base | Passed | Passed |

Original Large, EfficientTAM and Anime options remain in the selector. MatAnyone2, VideoMaMa and object-removal code remain present, but their model inference was not revalidated in this work.

The initial validation above predates the VideoMaMa/ViTMatte work; see [ViTMatte validation](VITMATTE_VALIDATION.md) for subsequent tests. Multi-concept tracking retains full temporal state per concept and increases memory versus one concept. The 16-object limit is a guard, not a guarantee that 16 independent concepts fit every 16 GB workload. See [multi-concept validation](SAM3_MULTI_CONCEPTS.md).
