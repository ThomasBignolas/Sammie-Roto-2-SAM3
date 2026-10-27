# Narrow seams between touching foreground objects

SAM predicts separate instance masks. Their edges can leave a thin strip that
neither instance owns, even where the RGB shows two opaque objects touching.
Unioning the masks cannot recover those missing pixels. The strip can become
false background guidance for VideoMaMa and then survive edge refinement.

## Use the optional correction

1. Restart Sammie after updating the code; reopen the project.
2. In **Segmentation → Postprocessing → Contact seams**, start at **8**.
   **0 disables the correction** and restores the exact original union.
3. Preview **Segmentation-Matte** with all objects visible. Scrub the contact,
   separation and occlusion frames. Use the lowest effective value.
4. Enable **Combine All Objects** for matting. Rerun VideoMaMa, then ViTMatte and any
   downstream RGB recovery that used the old alpha. Changing this setting does
   not alter previously computed matting or refinement results.

The setting is stored in the project. Width is measured in pixels at a 1080-pixel
short edge and scales to the actual mask resolution (8 corresponds to about
23 pixels for a 4608×3164 plate). It is not a global mask grow. SAM repropagation
is unnecessary: saved per-object PNGs, prompts and tracker memory remain intact.
The edit view and individual-object exports retain the original instance masks;
combined segmentation previews/exports and combined matting guides use the repair.
MatAnyone's combined guide uses the same correction as VideoMaMa.

## How it works and limitations

Only missing pixels inside a morphological closing of the combined silhouette
are considered. Candidates must lie away from the exterior of that closed
silhouette and be close to **two different object masks**. The sum of their
distances to those objects must fit the configured width. A completion step then
fills narrow enclosed background remnants only if they touch newly repaired
contact pixels. Their maximum distance to surrounding foreground must fit half
the width plus one source pixel of rasterization tolerance. Components reaching
the frame boundary and broad enclosed gaps are excluded. This prevents tiny
width variations from leaving detached holes along an otherwise repaired seam.
Existing foreground pixels are never removed. Unrelated single-object holes, thin structures, isolated objects
and wider gaps are not targets. Processing uses full-resolution binary guides;
soft alpha is never run through this operation.

This is an explicit geometric assumption for touching opaque foreground objects.
It cannot distinguish every genuine narrow background opening from a bad SAM
seam. It is therefore off by default and reversible. Wide missing regions need
SAM corrections; do not increase the width until all gaps disappear. There is
no temporal averaging or optical flow: changes in mask geometry can still change which pixels qualify.
Inspect moving contacts before accepting the result.

## Residual lines in the zoom-shot demo (frames 145 and 257)

The initial width-8 demo was insufficient on these frames. Parts of the actor
contact exceed that tolerance, so narrower parts filled while wider sections
remained as disconnected holes. For this particular shot, compare **Contact
seams 20** and **Remove Holes 8**, with **Shrink/Grow 0**. Remove Holes is the
existing small-hole control (contour area bounded by the squared slider value,
in source pixels); it is distinct from contact width. Do not apply these values
blindly to shots containing real fine gaps or translucent objects.

The updated contact algorithm also completes narrow enclosed remnants adjoining
its own repairs. This operates on segmentation guides, before VideoMaMa. It
does not force all foreground alpha opaque and does not change existing cached
mattes. **Rerun combined VideoMaMa, then ViTMatte** to update alpha. Select
**C · Temporally validated** to view the temporal result; selecting B intentionally
shows the unvalidated ViTMatte merge.

If the neural matte still develops false transparency inside the corrected
guide, use the separate optional [Opaque core](OPAQUE_CORE.md) control for
opaque subjects. Its inset preserves an edge band, and its strength is reversible
without model inference. The linked document includes the matched real GPU
tests around both reported frames and their limits.

## Validation on the supplied zoom shot

The saved project contains 307 frames at 4608×3164, 24 fps, two SAM 3.1 person
tracks seeded at frame 171. A real CUDA reproduction covered frames 145–205
with the official checkpoint. Instrumentation found no union pixels removed by
the final object-exclusivity step in that run. Missing contact pixels were
already present before that step; independently upsampling the instance logits
also widened some seams. No upstream tracker behavior was changed.

At width 8, inspected frames 200, 240 and 280 lost their long torso contact seams,
while the larger head separation remained open. Frame 150 retained wider gaps
outside the correction's scope. All 307 guides passed dimension, binary-value
and existing-foreground-preservation checks. 159 frames received additions;
the first was frame 138. Mean CPU correction time was 0.202 seconds per frame
at 4608×3164 on this workstation, excluding PNG decoding.

Matched real CUDA tests on frames **196–207** ran VideoMaMa at 1080 short edge
on the same application-selected crop, followed by **ViTMatte Small Composition-1k
at native resolution**, FP32, 1024-pixel tiles / 128 overlap, Quality parameters,
and temporal validation. Only the contact-width setting differed (0 versus 8).
Each run began with the same generator seed and independent temporal state.

Across these 12 frames, the mean of per-frame alpha averages on the repaired
pixels rose from **0.9102 to 0.9885** after ViTMatte. On frame 200 it rose from
0.9063 to 0.99976, and the visible torso seam disappeared. These are contact-region
diagnostics, not accuracy metrics: no ground truth exists, some boundary pixels
remain translucent, and neural processing can change pixels beyond the repaired
guide. The full 307-frame neural pipeline was **not** rerun.

VideoMaMa took 31.28 / 32.26 seconds for the two 12-frame runs, with 7229 MiB
peak PyTorch allocation and 4.22 seconds shared model loading. ViTMatte took
49.11 / 43.57 seconds, with about 1402 MiB peak allocation. VideoMaMa was unloaded
before ViTMatte. These are workstation timings with other tasks open, not isolated
benchmarks. Both 24 fps review videos and all native float outputs were validated.

The first neural-test harness attempt failed its output checks because it had not
loaded the intended frame-range settings; no result from that attempt is included.
The corrected harness asserts the range and explicitly sets both contact widths.
Automated tests cover disabled behavior, empty/full masks, single-object holes,
thin structures, wide gaps, resolution scaling, frame boundaries, immutable
source masks, and consistent preview/VideoMaMa/MatAnyone guide loading.
All 13 new tests passed. The full suite ran 108 tests: 107 passed in the sandbox;
the remaining AOV process-launch test passed when its 20-test module was rerun
outside the sandbox, which had blocked Qt's child-process creation.

Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mask_contacts.py -v`
from the repository.
