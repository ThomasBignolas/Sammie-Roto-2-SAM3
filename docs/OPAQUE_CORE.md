# Optional opaque core for flickering interior holes

Use this for **opaque people and props** whose interiors incorrectly become
transparent. Restart Sammie with `run_sam3.bat`, then open
**Matting → Postprocessing**:

- **Opaque core: 0%** is OFF and preserves the previous result exactly.
- **Opaque core: 100%** makes the deep interior of the current SAM guide opaque.
- **Core inset: 16** starts protection 16 pixels inside the guide at a 1080-pixel
  short edge. A second 16-pixel band smoothly ramps up to full opacity. At
  4608 × 3164 this is approximately 47 pixels untouched, then 47 pixels feathered.
  Increase the inset to protect more uncertain hair/blur near the silhouette.

The result updates from cached mattes without running VideoMaMa or ViTMatte again.
It applies after the selected A/B/C matte and grow/gamma controls. Preview, matte
sequence export, multilayer AOV alpha and CorridorKey's input alpha use the same
result. Set strength back to zero for an immediate unmodified comparison. Stored
VideoMaMa/ViTMatte arrays are never overwritten. Changing alpha invalidates an
existing CorridorKey RGB result; regenerate that downstream pass when needed.

## Why there are two fixes

On the supplied zoom shot, the original **Contact seams 8** demo retained wider
pieces of the line between the two actors. Their changing widths produced
disconnected holes. The revised contact correction completes narrow enclosed
remnants adjoining its own repairs. **Contact seams 20**, plus the existing
**Remove Holes 8**, improved the guides around frames 145 and 257.

VideoMaMa and ViTMatte can still predict transparency inside a corrected guide.
The optional opaque core addresses that second problem. It requires the shot's
interiors to be opaque; it does not infer whether transparency is legitimate.

For this shot, the tested sequence is:

1. **Segmentation → Postprocessing:** Contact seams **20**, Remove Holes **8**,
   Shrink/Grow **0**. Inspect genuine background openings.
2. Run **combined VideoMaMa**, then ViTMatte. Guide changes require this rerun;
   SAM retracking is unnecessary.
3. Select **C · Temporally validated**. B displays the unvalidated ViTMatte merge.
4. Set **Matting → Postprocessing → Opaque core 100%, Core inset 16** and compare
   with strength zero. These two settings require no model rerun.

## Constraints and limitations

The current frame's processed SAM guide supplies the core. The method measures
distance inside that guide and applies a smoothstep weight:
`alpha_out = alpha + strength * weight * (1 - alpha)`.
Outside the inset and inside actual guide holes, weight is zero and alpha is
preserved exactly. No previous-frame alpha or motion estimates are used, so
there is no temporal averaging or carried foreground trail. Geometry changes
can still affect the transition band; this is not a general temporal solver.

Do not enable it for transparent fabric, glass, smoke or subjects with genuine
interior transparency. It cannot correct a wrong silhouette or a missing SAM
region. A bad guide can mark real background as opaque. A larger contact-width
setting may also bridge genuine narrow gaps: review the moving result.

Two source-resolution float core masks are cached in RAM (about 111 MiB at this
shot's dimensions). Guide file size/mtime, object IDs, inset and segmentation
postprocessing settings determine reuse. No alpha or additional disk cache is
created. Missing mattes are not manufactured from segmentation, and missing
guides leave alpha unchanged. Processing is CPU-based and adds no model VRAM.

## Executed validation, 22 October 2026

- **134 automated tests passed**. New tests cover disabled identity, immutable alpha, exact edge-band
  preservation, strength, scaling, empty/full/thin shapes, moving guides without
  trails, guide cache invalidation, missing individual mattes, UI settings and
  matching preview/inspector/EXR output.
- All **307 source-sized segmentation guides** were checked for dimensions,
  binary values and preservation of existing foreground.
- Matched real GPU comparisons ran **139–150** and **251–262**, 12 frames per
  range in each of two branches (48 neural frame results). Both branches used
  VideoMaMa at a 1080 processing short edge, chunk 16 / overlap 4, then ViTMatte
  **Small Composition-1k, source resolution, FP32, 1024 tiles / 128 overlap**, with
  temporal validation 0.8. The earlier branch used the old contact algorithm at
  width 8 / Remove Holes 0; the revised branch used completion at width 20 /
  Remove Holes 8. Thus this is a combined configuration comparison, not an
  isolated measurement of the completion code.
- At frame 145, mean final alpha on newly repaired guide pixels increased from
  **0.7279 to 0.9916** before opaque-core protection. The remaining interior grey
  patch was removed by the optional core. At frame 257 the interior contact-hole
  component was opaque after refinement. Whole-image changed-pixel statistics
  also include silhouette/leg-gap pixels and must not be interpreted as accuracy.
- VideoMaMa took **28.6–31.2 seconds per 12-frame run**. Peak allocated GPU memory
  was **6,772–6,962 MiB**. ViTMatte used about **1,402 MiB**; the models were loaded
  sequentially. These are workstation measurements, not isolated benchmarks.
- The core pass was checked on all 24 revised native float frames: pixels with
  zero core weight remained **bit-exact**, and weight-one pixels became exactly
  opaque at strength 100%. Uncached guide loading and core processing averaged
  approximately **0.62–0.67 seconds/frame** on this 4608 × 3164 footage. Adjusting
  strength on the same cached frame is cheaper.
- Standard matte EXR and multilayer-alpha EXR readbacks for frames **145 and
  257** exactly matched the native float preview, retaining dimensions and frame
  numbers. The original project and its saved mattes were not overwritten.

Local comparisons and logs are in `.test-output/alpha-holes/Review.html` and the
same directory. Only the selected 24 unique frames received new neural matting;
the full 307-frame neural sequence was not rerun. No ground-truth alpha exists:
diagnostic alpha values and imposed opacity are not proof of accuracy.
