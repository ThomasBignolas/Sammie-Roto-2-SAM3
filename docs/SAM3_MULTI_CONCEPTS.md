# Multiple text concepts in one selection

This fix is on `feature/vitmatte-refinement`. Restart `run_sam3.bat` after updating; no package or model download is needed.

## Cause and behavior

Previously, the entire text field was sent as one SAM concept. `person and camera` or `person, camera` therefore asked the detector for a compound concept; it did not request the union of two searches. The application also prohibited text discovery after an accepted selection. Calling the native text API twice on the same session would not solve this: both pinned official predictors reset their semantic session on each new text prompt.

The application now searches each description separately using the same loaded model. Selected concepts receive independent native tracking sessions and independent application object IDs. This retains the exact detector masks and each concept's temporal state; detections are not approximated by bounding boxes or sampled points. Forward/backward passes advance all sessions once per frame and emit aligned masks together.

## Use it

1. Load SAM 3 or SAM 3.1. Type **`person, camera`** or **`person and camera`** and press **Find objects**.
2. Select the labeled person and camera instances, or use **Select all**, then accept.
3. Track normally. You can correct either object independently on intermediate frames.
4. For one final matte containing both, select **Combine All Objects** in the Matting tab before **Run Matting**. Otherwise, keep separate mattes/channels. Optional ViTMatte refinement follows the same matting choice.

You can also find/select `person` first, then find/select `camera`. The second operation adds objects. Cancel and searches with no matches preserve committed selections and their tracker histories. Partial failures to find a description retain successful candidates and report the missing description in the status text. Clear All remains the explicit way to replace the entire selection.

Commas, semicolons, newlines and the standalone word `and` are separators. Use double quotes for one compound description: `"person in a red and white coat"`. `person holding a camera` is still one relational description. Repeated identical descriptions in one entry are searched once. Different descriptions can return overlapping objects; inspect the labeled thumbnails and select the intended instances. Repeated separate searches can intentionally add the same object again, so do not select an existing instance twice unless desired.

## Identity, memory and compatibility

Temporary discovery states close after their masks reach CPU. Acceptance re-detects the selected concepts and matches saved masks by IoU, rejecting unreliable matches. Existing selected sessions are never reset by discovery. Each session gets a scoped worker ID namespace so native detector ID 0 for person cannot overwrite native ID 0 for camera. The application still uses ordinary object IDs 0, 1, etc.; output numbering, dimensions and EXR channel names are unchanged.

Saved prompt records contain the per-concept scope; older single-query projects remain compatible. Unloading SAM for matting releases every session. Editing/tracking afterward reloads the model and reconstructs the separate queries, selections and corrections. A propagation error rejects missing/misaligned frame sequences. Cancellation closes all active native iterators between complete frames.

One model is loaded, but tracking-state memory grows with the number of concepts and shot length. The 16-object limit does not guarantee that 16 independent concepts fit 16 GB in every shot. Start with the objects you actually need; GPU allocator measurements exclude other applications and driver memory. No arbitrary temporal chunk resets are added. SAM 2 behavior is unchanged.

## Executed validation — 2026-10-21

**56 automated tests passed**, including prior segmentation, media, refinement and export tests. New coverage includes both text syntaxes, quoted compound descriptions, duplicate phrases, colliding detector IDs, additive selection/cancel/no-match behavior, per-concept corrections, saved-selection replay/removal, routing to the right native session, merged frame alignment, missing frames and cancellation of every stream.

Real trained-checkpoint tests ran on the RTX 5060 Ti 16 GB using four copied frames from the supplied project, **4608×3164**, working proxy long edge 1920. The original project was not modified. Both SAM 3 and SAM 3.1 detected one person and one camera using both comma and `and` syntax. Each selected seed was **pixel-identical to its corresponding independent search** (IoU 1.0), and both remained nonempty on all four tracked frames. Tests also executed intermediate camera corrections while preserving the person mask, tracking restart, backward propagation, complete worker unload/reload, restored object IDs, and additive selection/cancellation.

| Model | Four-frame forward tracking | Peak allocated GPU memory observed, including corrections |
| --- | ---: | ---: |
| SAM 3 | 1.83 s | 6145 MiB |
| SAM 3.1 | 2.01 s | 7343 MiB |

Times exclude checkpoint loading and initial detection. These are short integration diagnostics, not full-shot benchmarks or accuracy evaluations. One initial SAM 3.1 worker launch exited during model loading before any selection test; retry completed the full validation. The initial exit cause was not established. Worker-exit errors now include the exit code and bounded diagnostic text.

The actual MainWindow was also exercised with SAM 3 and a source-sized still from the project. The real candidate dialog displayed both concept labels; **Select all**, acceptance, tracking and sequential person-then-camera addition passed. The existing combined-mask function exactly matched the pixelwise union of both masks. The exported `mask.1001.exr` retained source dimensions and separate `Object_0.Y` / `Object_1.Y` channels exactly matching the masks. Model inference, selection dialogs and exports were real; only startup and confirmation handling were isolated by the test harness. Detailed reports are in [sam3-multi-concept-results.json](sam3-multi-concept-results.json). UI screenshots remain local under `.test-output/concept-ui-1e55b3b7/`.

The existing real `scripts.ui_smoke` regression also passed: SAM 3 and SAM 3.1 single-description instance selection, point correction/tracking, independent manual objects and box selection, followed by switching back to SAM 2 Base and real point inference. Local screenshots: `.test-output/ui-ebb67b54/`.

Reproduce from the application environment:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe -u -m scripts.validate_sam3_concepts --model "SAM 3" --frames <copied-frame-folder>
.venv\Scripts\python.exe -u -m scripts.validate_sam3_concepts --model "SAM 3.1" --frames <copied-frame-folder>
.venv\Scripts\python.exe -u -m scripts.validate_sam3_concepts_ui --model "SAM 3" --image <test-image.png>
```

The video validator expects `00000.png` through `00003.png` in the copied folder and a fixture containing a person and camera. Outputs, reports and screenshots stay under `.test-output`; user footage and checkpoints are not published.
