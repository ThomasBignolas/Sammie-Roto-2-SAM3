# Matte precision, repeatability and inspection

Restart Sammie with `run_sam3.bat` to use these changes. No new dependencies or checkpoints are required. Existing saved mattes are unchanged until you rerun processing.

## Float VideoMaMa chunk handoff

When ViTMatte is enabled, VideoMaMa writes float alpha caches. Previously it nevertheless quantized the overlap guidance to 8-bit before feeding it into the next chunk. The handoff now retains floating point values and accepts mixed float overlap frames and uint8 SAM guides. This removes an unnecessary precision loss in subtle edge values. The model still uses its configured inference precision.

Chunk length, overlap, frame ordering, fixed crop and boundary blending are unchanged. Blending combines two predictions for the **same source frame**, not adjacent source frames. The legacy VideoMaMa-only uint8 path remains available. This fix cannot recover precision from old PNGs or guarantee better-looking edges on every shot.

## Repeatable VideoMaMa sampling

The inference seed now controls VAE sampling as well as the existing UNet noise input. Both use local generators; inference does not reseed or consume the application's global PyTorch random state. Previously the VAE sampled from that global state, so repeated runs could differ despite the same inference seed.

Rerendered mattes may therefore differ from mattes made before this change. Repeatability was verified on one CUDA workstation with the same weights, settings, frame range and software. Changing chunk boundaries, devices, model precision or library versions may change the result; bit-exact cross-platform output is not promised. Repeatability is not temporal stabilization.

## Inspect Matte Quality

In **Matting**, click **Inspect Matte Quality** on a processed frame. The dialog inspects all objects and the current Matting preview alpha, including its selected ViTMatte A/B/C stage and grow/gamma adjustments. If that refinement cache is unavailable or stale, the existing preview's base-alpha fallback also applies here.

- **Magenta:** alpha below 0.95 well inside the combined segmentation guide.
- **Cyan:** alpha above 0.05 well outside the guide.
- **Ignore edge band:** default 16 pixels at a 1080-pixel short edge, scaled to the source dimensions. This excludes the most ambiguous silhouette area.
- Adjust thresholds, then click **Inspect with these thresholds**. Switch between RGB, alpha, guide and flagged views; use **100%** and scroll to inspect fine detail.

These are review candidates, not error classifications. Transparent objects, hair, motion blur, missed objects and incorrect SAM guides can trigger them. An unflagged matte is not necessarily accurate. The inspector does not repair alpha, change settings, write caches, add export layers or assess temporal flicker. Close it to return to editing; reopen after changing the frame or matte.

For narrow gaps where two tracked actors touch, use the separate optional [Contact seams correction](CONTACT_SEAMS.md). Start with the lowest effective width, inspect real gaps, and rerun **combined** matting/refinement. Avoid growing or filling the entire alpha to hide a local tracking problem.

## Executed validation (22 October 2026)

- Full local automated suite: **121 tests passed**, including 13 contact-seam tests, five VideoMaMa precision/repeatability tests and eight inspector tests. Coverage includes float/uint8 normalization, global RNG isolation, real chunk-loop ordering, scaling, empty/full guides, immutable alpha, missing frames, combined object IDs, Qt controls and preview selection.
- Actual RTX 5060 Ti 16 GB inference: frames **196–215** from `D:/SammieZoom.sammie`, 20 source frames at **4608 × 3164**, VideoMaMa processing short edge 1080, chunk 16 / overlap 4, Contact seams 8. Two runs with different global seeds but the same inference seed produced **identical float arrays for all 20 frames**. The second chunk received four float overlap guides and four uint8 segmentation guides. Global CPU and CUDA RNG states were unchanged.
- Measured matting times: **62.33 s** and **68.09 s**, excluding model loading. Peak PyTorch allocated GPU memory: **8,590.55 MiB** in each run; this is not total GPU usage. Other desktop applications were open.
- The precision/repeatability test did not rerun ViTMatte. Separately, the contact-seam comparison ran real VideoMaMa then native-resolution ViTMatte Small on two matched 12-frame ranges, as documented in [CONTACT_SEAMS.md](CONTACT_SEAMS.md).
- The inspector was exercised on actual cached, temporally validated ViTMatte alpha at frame 202. Its loaded float array exactly matched the cached result; source dimensions remained 4608 × 3164. The displayed flags are diagnostics, not accuracy measurements.

Local logs, hashes and UI captures are under `.test-output/videomama-quality/`; footage and generated media are intentionally excluded from Git. No ground-truth alpha is available, so these checks do not establish improved matting accuracy or long-shot temporal robustness.
