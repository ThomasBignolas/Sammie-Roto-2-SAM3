# Working projects, shot processing and Nuke

These features are in the school build. No additional Python dependencies are
needed. Update with `update_school.bat`, then launch `run_sam3.bat`.

Since 2.6.0, opening a project does not load SAM automatically. Click **Load Model**
before editing/tracking, or enable automatic loading in Settings → General.
Working Save As preserves prepared SAM 3 proxies, releases the old model session
and leaves saved selections available. Copying/packing/unpacking now shows
background progress. See [performance settings and validation](PERFORMANCE.md).

## Set Project, like a Maya workspace

Choose **File → Set Project…** and select a folder for the production or exercise.
Sammie creates `scenes`, `footage`, `exports` and `nuke`, and remembers this workspace
on this PC. Open/save dialogs default to the appropriate folder; normal exports
default to `exports`. Setting a workspace does not move an already open shot or
copy source footage automatically.

1. Import footage or open an existing `.sammie` archive.
2. Choose **File → Save Project** (`Ctrl+S`). Give the shot a name in `scenes`.
3. Continue working and saving with `Ctrl+S`.

The first save creates **`shot.sammieproj` plus `shot.sammie-data/`**. It copies the
current session once, including working frames and cached results. Subsequent
saves write only the settings, selections and small descriptor, using atomic file
replacement. They do not enumerate, hash, compress or copy frame/cache files.
Generated results are written directly into the working data folder as processing
runs. Closing and reopening is therefore not a way to undo a processing run.

**Keep the descriptor and its data folder together.** Move/copy the pair into the
same directory on another PC; relative lookup supports relocating them together.
The original source path remains descriptive metadata; the saved working frames
are available without the original movie. External foreground-plate overrides and
custom checkpoint paths still need to exist or be reselected on the destination.
Save As always requires a new name and creates an independent copy.

**File → Pack Portable Project…** creates a self-contained `.sammie` archive for
sharing/backup. This explicitly copies all included data and can still take time
and substantial disk space. Cancel leaves an existing archive unchanged. Old
`.sammie` files remain supported: opening one extracts a temporary session; saving
creates a working project. Opening other footage detaches a saved working project
before replacing the temporary session.

Only one Sammie writer can hold a working project at once. The operating system
releases the lock after a crash. On reopening, interrupted queue writes are
recovered before the project is used. `Ctrl+S` is not version control or a backup;
keep periodic portable packs or filesystem backups. Local SSD storage is advised;
network-share locking and filesystem durability have not been validated.

## Repair a range and reuse intermediate results

Set timeline In/Out markers, then choose **Workflow → Repair Selected Range…**.
The dialog uses **zero-based, inclusive timeline indices**, independent of source
sequence numbers. Select only the steps that need rerunning:

| Step | Inputs and behavior |
| --- | --- |
| SAM tracking | Replays saved prompts, then propagates forward/backward from the current frame clamped into the processing range. |
| Matting | Uses existing SAM masks and the selected VideoMaMa/MatAnyone settings. |
| ViTMatte | Reads existing VideoMaMa alpha; it does not rerun VideoMaMa. |
| Depth / Normals | Processes the **whole shot**, even for a small selected range, to retain temporal context and consistent depth normalization. |
| Foreground RGB | Runs the existing CorridorKey recovery using the selected matte source. |
| Nuke delivery | Exports only the requested range, using cached results. |

Context handles default to **16 frames on each side** for tracking/matting/
refinement/foreground. Handles are **processed and saved**, not discarded. The
dialog shows the effective range and counts of existing cache files. File presence
is not proof of freshness. Inspect the repair boundaries; restarting a temporal
model on a shorter interval is not guaranteed to match a full-shot run. Increase
handles or process the full shot when continuity is critical.

Unchecked downstream steps are not run automatically. After tracking changes,
rerun matting and any affected refinements/recovery. After changing VideoMaMa,
rerun ViTMatte and foreground as needed. Existing downstream files are retained;
the dialog explains these dependencies. ViTMatte cache validation is now per
frame: the current frame and previous raw-refinement inputs govern temporal
validity. Repairing one range no longer discards unrelated valid refined frames.
Changing ViTMatte model/settings still invalidates incompatible results.

## Sequential shot queue

Use **Workflow → Shot Queue…**, add saved `.sammieproj` shots, and choose the range
and stages for each. Tracking and matting need object selections already saved in
each project. This is a queue of prepared shots, not automatic subject selection
for arbitrary movies. Each queue item captures the project's settings/prompts;
if they change afterwards, remove and add the item again rather than silently
running an obsolete configuration.

Stages run in separate processes, one at a time, using the existing app/SAM/AOV/
CorridorKey runtimes. The interactive models are released before processing.
Project editing is blocked while the queue dialog owns the worker. After closing
it, **Load Model** restores interactive segmentation; preview/export remains
available without reloading a neural model.

The queue is saved at `<workspace>/.sammie/queue.json`, or
`<installation>/.runtime/shot-queue.json` without a workspace. Reopen it and press
**Start / Resume** after an interruption. Completed stages are skipped only if
their input, output, settings and implementation signatures still match. A failed
shot is reported and other pending shots can continue. Cache verification hashes
media and can take time on large shots; ordinary project saves do not do this.

**Stop and keep completed stages** terminates the active worker tree on Windows
and restores the interrupted output range/pointers from a journaled backup. The
backup temporarily needs disk space comparable to the affected existing output.
Completed prior stages stay available. Range repairs have their own persisted
queue under `.workflow/repair-queue.json`; use Start / Resume in that dialog after
stopping. After closing/restarting the app, reopening Repair creates a new repair
item; use the main Shot Queue for stage-level resume across restarts.

## Send to Nuke

Choose **File → Send to Nuke…**. Sammie saves the working project, exports the
timeline range to a new folder under the workspace's `nuke` directory and opens
the generated `sammie.nk`. If Nuke is not found, select its executable; cancelling
that selection keeps the delivery available to open manually.

Each delivery contains its own multilayer float EXR sequence, `.nk`, and metadata.
The script uses paths relative to itself, so move the entire delivery folder.
It includes the original plate, selected generated alpha, checkerboard composite
and any **valid, already generated** depth, normals and foreground RGB. It does
not start those models automatically. Missing/stale selected ViTMatte alpha is an
error, rather than an unnoticed fallback to the original matte. Switch the
ViTMatte comparison to Original VideoMaMa if that is the intended delivery.

Plate RGB in this delivery is a **linear Rec.709 copy**. Retained EXR input is
converted from its declared input space, including ACEScg; display PNG/JPEG plates
are converted from sRGB. The source project and ordinary export behavior remain
unchanged. Nuke reads these EXRs as raw data, with built-in Nuke color management;
adapt the viewer/project configuration for a studio OCIO pipeline. Alpha and AOV
data receive no RGB transfer function. Depth remains relative, not metres.

Source dimensions, ordering, sequence start numbers and rational FPS metadata are
retained. The Nuke timeline uses the source frame numbers. The graph exposes plate,
alpha, recovered foreground and diagnostic normals/depth branches. Cancelled or
crashed delivery staging is cleaned on recovery; completed deliveries are kept.

## Update from inside Sammie

**Help → Check for School Updates…** shows the installed commit and latest
published school release notes. **Save, close and update** saves the current
session, starts the existing verified updater, waits for this app process to exit,
updates source/dependencies and reopens Sammie on success. Saved working projects
are remembered on launch. Logs are in `.runtime/ui-update-*.log`; an update error
leaves details there and can be retried with `update_school.bat`.

The updater preserves models, settings and personal projects. It refuses to
overwrite locally edited tracked code; Git updates must fast-forward. This UI
does not update from upstream Sammie or run unreviewed development branches.
The [one-line CMD updater](SCHOOL_INSTALL.md#updating-and-moving-between-pcs)
remains available for old installations and external administration.

## Executed validation (27 October 2026)

A disposable copy of an existing **194-frame, 4608×3164** project contained
**2.70 GiB** of retained working media/cache. Creating its first working copy took
**29.54 s**. Five subsequent metadata saves took **0.49–1.48 s** (median **1.15 s**),
versus **176.54 s** to create a portable archive of that working copy. Media file
sizes/timestamps and the original `.sammie` SHA-256 were unchanged. These are local
storage measurements while other model checks were active, not a controlled
storage benchmark, a cold disk/network-share estimate, or a timing of the old
upstream compression implementation. The architectural improvement is that save
cost no longer scales with the amount of cached media.

The expanded automated suite passes **186 tests**, including actual Qt worker
processes, two sequential deliveries, worker-tree cancellation with output
rollback, working-project save/detach in MainWindow, legacy archive handling,
interrupted recovery, cache invalidation/reuse, and updater handoff arguments.
No new dependency versions were introduced.

A real three-frame **384×216** synthetic sequence ran through queued **SAM 2 Base
→ VideoMaMa → ViTMatte Base → Nuke delivery** on the RTX 5060 Ti 16 GB. Wall times
including process startup/model loading were 9.00 s, 183.72 s, 4.43 s and 1.46 s
respectively. ViTMatte's measured inference/load portion was 2.36 s, with peak
PyTorch allocation **489.68 MiB**, reserved **558 MiB**. This small synthetic test
verifies execution and alignment; it does not establish production quality or
1080p/4K performance. VideoMaMa and SAM 2 peak VRAM were not separately recorded.

Additional actual queue workers passed on the same small RGB fixture:

| Model/stage | Wall time including worker setup | Peak allocated / reserved MiB |
| --- | ---: | ---: |
| SAM 3 point-guided tracking | 61.49 s | Backend reported 5590 MiB peak |
| Video Depth Anything Small | 88.01 s | 577.21 / 724 |
| NormalCrafter, 3-frame window, decode chunk 1 | 194.08 s | 3222.04 / 3466 |
| CorridorKey Blue, 1024 processing resolution | 25.49 s | 1585.38 / 2742 |
| Nuke delivery including depth, normals and foreground | 6.67 s | CPU export |

These measurements include model/checkpoint setup and were collected while other
validation/storage work was active. They are execution diagnostics, not steady
state throughput comparisons. The additional foreground run used cached original
VideoMaMa alpha. SAM 3.1, DVD and MatAnyone were not rerun through this new queue
in this validation; their earlier direct application tests remain documented in
the existing feature reports.

Actual **Nuke 17.0v3** opened a generated graph, sampled original/alpha/foreground/
normal/depth branches, checked composite connections, saved and reopened it.
Multi-object and EXR color/precision behavior are additionally covered by the
existing regression tests. Long multi-shot neural queues, other Nuke versions,
network workspaces and another physical school PC have not been validated here.

For a reproducible small CUDA execution check, run:

```bat
.venv\Scripts\python.exe -m scripts.validate_workflow
```

It creates an isolated synthetic shot under `.test-output/workflow`, runs real
ViTMatte Base plus Nuke delivery in workers, verifies receipt reuse, and records
JSON timings/VRAM and logs. It requires the installed model/runtime. It does not
alter existing projects or claim matting accuracy.
