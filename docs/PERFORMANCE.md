# Performance and responsiveness — 2.6.0

Update with `update_school.bat` and restart with `run_sam3.bat`. No new model
downloads or dependency changes are required. An already running application
continues using its old code until restarted.

## What changed

- Timeline decoding/compositing runs in a separate CPU process. Rapid seeks keep
  only the newest pending request; an obsolete result cannot replace the current
  frame. Point editing waits for the requested frame to appear. Preview keeps
  source resolution; export dimensions and numbering are unchanged.
- Native RGB and eligible rendered views use a byte-limited LRU cache. The default
  combined budget is 256 MiB, reduced further when available RAM is low. Changed
  settings, prompts, source files and masks invalidate affected results. Matting,
  ViTMatte and AOV previews still validate their complete dependency chains;
  their rendered results are not cached by the new display cache.
- Segmentation overlays and checkerboards allocate fewer temporary arrays.
  ViTMatte previews decompress only the requested alpha stage and trimap, cache
  small metadata reads, and reuse content hashes while file stamps are unchanged.
- Model loading/session initialization, committed point prompts, SAM text searches,
  prompt replay, tracking, matting and object removal run through a background
  operation dialog. Qt widgets remain on the UI thread. The dialog shows the
  current stage, elapsed time and reported progress. Stages without finer progress
  say so after 15 seconds; elapsed time is not a fabricated percentage.
- VideoMaMa crop scanning and input preparation report individual frames. SAM
  stays offloaded through optional ViTMatte, then restores after that stage.
  Existing ViTMatte, AOV, shot-queue and Nuke dialogs also show elapsed/waiting status.
- Media imports, working-project copying, portable project packing/unpacking and
  single-image rendering/export run in workers. Frame installation and cleanup
  of the old scratch session also happen outside the UI thread. Invalid images
  fail explicitly instead of silently creating missing frame numbers. Cancellation
  before import commit leaves the previous session intact.
- Video decode and image writing each use at most two workers. The decoded-frame
  write queue is limited to 128 MiB (or less under RAM pressure), with at most four
  arrays; an individual oversized frame is still allowed. Direct BGR conversion
  removes a redundant full-frame copy while preserving the original conversion.
- CPU/OpenCV processing defaults to at most four threads, leaving capacity for
  other applications. The Windows preview process runs below normal priority.
  The status bar displays available host RAM. Removal checks known frame-buffer
  requirements before allocating a clip that would consume most available RAM.
- SAM 2 keeps a small CPU frame cache (at most four tensors / 48 MiB), preserving
  its exact float32 preprocessing. SAM 3 reports when it reuses prepared working
  frames; creating a working project preserves those proxies and their timestamps.
- Cancelled owned workers stop their Windows process tree, including Python venv
  launcher children. Console history is capped at 3,000 blocks, and point-table
  refreshes suppress intermediate repaints.
- Editing ViTMatte parameters keeps the last completed result in preview/export
  until Refine / Resume runs. Pending values are labeled, applied settings can be
  restored, and wheel events no longer change ViTMatte controls accidentally.
  [Details and source-invalidation behavior](VITMATTE.md#editing-settings-without-losing-the-result-260).

## Opening and settings

Projects open for preview without automatically loading SAM or replaying prompts.
Click **Load Model** when ready to edit or track. Creating a working project with
Save As also releases the previous model session; saved selections remain available.
Working `.sammieproj` files avoid repeatedly unpacking portable `.sammie` archives.

Under **Settings → General → Performance**:

| Setting | Default | Guidance |
| --- | --- | --- |
| Preview cache | 256 MiB | Use 512–1024 MiB only with spare RAM; 0 disables caching. This is not total app memory. |
| CPU threads | Auto, up to 4 | Restart after changing. Affects the app and previews; isolated model workers retain their conservative four-thread default. |
| Load SAM when opening projects | Off | Enable to restore automatic initialization with the new progress dialog. |

For 1080p, start with defaults. For 4K/larger source, cached revisits are faster,
but uncached compositing still costs CPU time and memory. Playback waits for the
current preview rather than building an unbounded backlog; the requested FPS is
an upper limit, not a guarantee. Keep projects/caches on a fast local SSD.

## Cancellation and remaining limits

Cancellation is cooperative: an active CUDA kernel, checkpoint read or file write
may need to finish before cleanup. The dialog explains that wait. Installing an
already prepared import completes its filesystem transaction rather than aborting
mid-rename. Completed tracking frames remain on disk. A cancelled model load
unloads the incomplete session; use Load Model to retry.

This update does not make every legacy operation asynchronous. SAM 2's optional
Shift-hover prompt preview still runs synchronously to preserve its temporary
predictor-state snapshot/restore semantics. Legacy deduplication and some bulk
clear/unload operations can still pause the UI. Avoid Shift-hover on slow devices;
committed point prompts use the new background path. Initial Python imports and
device setup can also delay startup before model-loading progress appears.

The preview cache is only part of memory use: model weights, decoded/composited
buffers, Qt copies and the Python/PyTorch runtime are additional. A bounded queue
does not guarantee that Windows cannot page or stall, particularly with other
large applications active. Neither low-memory preflight nor the progress timer
can predict model-specific allocations or prove a model is making progress.

## Validation on 28 October 2026

All **211 automated tests passed**, including real Qt event-loop/process tests,
latest-seek and cache invalidation, cancellation, owned-process-tree cleanup,
source-preserving import failure, frame numbering, exact export pixels, and a
mocked VideoMaMa → ViTMatte sequence with failure cleanup. Existing project,
queue, EXR and Nuke tests are included. GPU inference was **not tested for this
update**, at the user's request. Prior GPU results in other documents do not
validate these new threading paths.

CPU measurements used Windows, 14 logical processors, about 63.4 GiB installed
RAM and four processing threads. Other applications remained open. The native
fixture contains two 4608 × 3164 frames from the overlapping-people shot; personal
footage is excluded from GitHub and the package. These are short diagnostic runs,
not long-shot throughput or matte-accuracy benchmarks.

With equal four-thread settings, uncached segmentation-edit rendering fell from
a median **824 ms to 512 ms**. Plain plate and matte rendering did not materially
improve. Original and new pixel hashes matched for all four tested view modes
on both native frames. The main benefit for expensive views is moving their work
off the UI thread and reusing recent previews, not changing image quality.

| Measured operation | Result |
| --- | --- |
| Cached native source/edit/matte/checkerboard preview | Median 74–79 ms including Qt transfer |
| Preview UI heartbeat | Largest gap 54 ms in the final run |
| Preview worker RAM | 790 MiB RSS snapshot, including runtime and caches |
| Two-frame video import, before → after | 0.67 → 0.62 s; identical pixels and PNG sizes |
| Import UI heartbeat, before → after | Largest gap 338 → 12 ms |
| Real SAM 2 Base load + saved prompt, CPU | 10.84 s; largest UI gap 59 ms |
| Real SAM 2 forward tracking, two frames, CPU | 3.29 s; largest UI gap 11 ms |
| Real ViTMatte Base load + prediction, 384 × 216, CPU | 2.37 s; finite alpha in [0, 1]; largest UI gap 92 ms |

The first preview worker startup took about 1.3 seconds. Cached timings apply to
recent frames fitting the cache, not an entire shot. The 128 MiB import queue
holds at most two 4608 × 3164 RGB arrays (about 83 MiB), versus the previous
100-array queue's potential 4.1 GiB; decoder, encoder and live buffers are extra.

See [machine-readable measurements](performance-validation-results.json) for
native preview/cache timings, UI heartbeat gaps, real CPU SAM 2 and ViTMatte Base
inference, import timings and the worker RAM snapshot. A 10 ms Qt timer measures
UI responsiveness; its gaps include scheduling, callbacks and native image
transfer. The worker RSS figure is a snapshot, not a measured peak. No new peak
VRAM or CUDA speed claim is made.

To reproduce tests and preview diagnostics from the installed folder:

```bat
set CUDA_VISIBLE_DEVICES=-1
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m scripts.benchmark_preview --root "D:\shots\shot.sammie-data" --frames 145 257 --output preview-timing.json
```

The preview utility uses CPU only and reads existing working frames/masks. Choose
frame indices that exist in your project. It skips matting timing when no matting
outputs exist. It does not run segmentation or change project data. Automated
tests mock GPU behavior; CUDA inference must be validated separately on a free GPU.
