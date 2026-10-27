# Sammie-Roto 2 with SAM 3, ViTMatte and video AOVs

**School build — easiest Windows installation:** download and extract
[Sammie-Roto-School.zip from the release](https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/tag/school-2026.10.27),
then run **`install_school.bat`**. It installs Python, dependencies and all supported
models, including ViTMatte **Base (larger)**. Launch **`run_sam3.bat`** afterwards.
Staff can share the source ZIP internally; recipients do not need GitHub access.
SAM 3/3.1 require an approved Hugging Face account and a hidden local read-token
login. [Full school setup, CMD options, updates and troubleshooting](docs/SCHOOL_INSTALL.md).

**Local school build: CorridorKey foreground RGB.** Run `install_corridorkey.bat`, then launch `run_sam3.bat`. In **AOVs → CorridorKey Foreground RGB**, enable the optional pass, choose the alpha source and use **Test CorridorKey on current frame**. It uses the selected VideoMaMa/ViTMatte alpha as a hint, preserves that alpha exactly, and adds `foreground.R/G/B/A` to the multilayer EXR. Cleanup strength and despill can be adjusted from cache. [Setup, school PC transfer, presets, color handling and limitations](docs/CORRIDORKEY.md). CorridorKey's code and weights are downloaded separately into a local runtime; they are not bundled in this source build. Distributed through this school repository; see the school installer above.

This development version adds Meta's official SAM 3 and SAM 3.1 video predictors to the existing Sammie-Roto interface. SAM 2.1, MatAnyone, projects, image sequences and exports remain available.

**New AOVs tab:** on branch `feature/aov-depth-normals`, run `install_aov.bat` once, launch `run_sam3.bat`, then open the fourth tab after Object Removal. Generate depth with **DVD v1.1** or **Video Depth Anything Small**, and normals with **NormalCrafter** through LiveActionAOV's model implementation. Preview on the timeline and export one source-sized float multilayer EXR per frame, including original RGB, generated alpha, object layers, `depth.Z` and `N.x/y/z`. Start at a 512-pixel processing long edge on a 16 GB GPU. See [installation, licensing and workflow](docs/AOVS.md) and [actual validation](docs/AOV_VALIDATION.md).

**Checkerboard background:** choose **Checkerboard** in the background color controls in Segmentation-BGcolor or Matting-BGcolor. Alpha and matte exports keep their original behavior.

**ViTMatte after VideoMaMa:** use branch `feature/aov-depth-normals`, run `install_vitmatte.bat` after installing the application, then launch `run_sam3.bat`. In **Matting → VideoMaMa**, check **ViTMatte Edge Refinement** and select **ViTMatte Base (larger)**, the default model for new projects. Use **Refine cached mattes / Resume** for existing projects. Original mattes remain available, with A/B/C preview and export selection. See the [ViTMatte workflow, presets and limitations](docs/VITMATTE.md) and [executed validation results](docs/VITMATTE_VALIDATION.md). Refinement is OFF by default.

**Individual component setup (advanced):** install [Git for Windows](https://git-scm.com/downloads/win), obtain this repository's `feature/aov-depth-normals` branch, then run these files in order:

1. `install_sam3.bat` — creates the application and isolated SAM 3 environments.
2. `setup_models.bat` — downloads the official checkpoints after your Hugging Face access is approved.
3. `run_sam3.bat` — launches the upgraded application directly.

Load footage, select **SAM 3** or **SAM 3.1**, click **Load Model**, then add points or use **Find objects**. Select the detected instances, track, refine in the existing Matting tab, and export.

**Multiple text selections:** `person, camera` and `person and camera` now search both concepts independently. Select both labeled results; subsequent searches add objects without clearing earlier selections. Enable **Combine All Objects** in Matting if you want one combined matte. See [multi-concept behavior and validation](docs/SAM3_MULTI_CONCEPTS.md).

**Touching objects with a black seam:** optional **Segmentation → Postprocessing → Contact seams** repairs narrow interior gaps in the combined guide, leaving original instance masks intact. Start at 8, inspect Segmentation-Matte, then rerun combined matting and refinement. Zero disables it. See [operation, testing and limitations](docs/CONTACT_SEAMS.md).

Read the [Windows setup and workflow](docs/SAM3_WINDOWS.md), [architecture and compatibility](docs/SAM3_ARCHITECTURE.md), [test results and limitations](docs/VALIDATION.md), and [model license notes](docs/MODEL_LICENSES.md).

**Matte quality tools:** VideoMaMa now preserves float overlap guidance when ViTMatte is enabled and seeds VAE sampling for repeatable runs. **Matting → Inspect Matte Quality** highlights possible interior holes and exterior leakage without changing alpha. See [operation, limits and real GPU validation](docs/MATTE_QUALITY.md).

**Flickering interior holes:** the contact correction now completes narrow enclosed remnants along repaired seams. Optional **Matting → Postprocessing → Opaque core** holds opaque interiors solid with a feathered inset, shared by preview and EXR export. It is OFF by default and adjustable from cached mattes. See [settings and the frame 145/257 comparisons](docs/OPAQUE_CORE.md).

**Validation status:** 148 automated tests pass. Real checkpoint CUDA tests have exercised SAM 2, SAM 3/3.1, matting, DVD v1.1, NormalCrafter and VDA Small on a 16 GB RTX 5060 Ti. SAM 3 / 3.1 tests include text/box detection, independent objects, bidirectional tracking, corrections, sequential MatAnyone refinement and full-resolution alpha exports. The actual MainWindow AOV workflow exports original RGB, existing VideoMaMa alpha, depth and normals in full-resolution multilayer EXRs. The new [CorridorKey validation](docs/CORRIDORKEY_VALIDATION.md) includes real GPU/UI/export runs using ViTMatte Base alpha and a CPU green-model smoke test. See the validation documents for measured results and limits; arbitrarily long shots and production-quality results are not guaranteed by these tests.

Use `run_sam3.bat` for this version: the original upstream launcher/update mechanism targets upstream releases. The original project documentation follows.

---

# Original Sammie-Roto 2 documentation
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto 2 screenshot](https://github.com/user-attachments/assets/bc2c99c8-4039-49f1-94ed-65f104a83e8d)

[![GitHub Downloads](https://img.shields.io/github/downloads/Zarxrax/Sammie-Roto-2/total)](https://github.com/Zarxrax/Sammie-Roto-2/releases)
[![GitHub Code License](https://img.shields.io/github/license/Zarxrax/Sammie-Roto-2)](LICENSE)
[![Discord](https://img.shields.io/discord/1437589475369811970?label=Discord&color=blue)](https://discord.gg/jb5qrFyGFF)

**Please add a Github Star if you find it useful!**

Sammie-Roto 2 is an easy-to-use, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone), [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), and [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) and [ProPainterX](https://github.com/Zarxrax/ProPainterX)

Sammie-Roto 2 is free and open source, but runs models produced by several external projects and organizations. Some models may have restrictions on commercial usage. Please check with the relevant model provider if you have questions regarding licensing.

### Documentation and Tutorials:
## [Wiki Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

[![Quick Start Video](https://img.youtube.com/vi/m0iZpxsZJcE/0.jpg)](https://www.youtube.com/watch?v=m0iZpxsZJcE)

*This video does not cover new features and changes added since version 2.0

## Need more help? [Join the Discord](https://discord.gg/jb5qrFyGFF)!

### Installation
#### Windows:
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive to any location that doesn't restrict write permissions (don't put it in Program Files)
- Run 'install.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software (or double click the desktop shortcut).

#### Mac
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Unzip Sammie-Roto-2 into a folder in your home directory, e.g. ~/Sammie-Roto-2 (avoid Desktop, Documents and Downloads).
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command in the terminal: `bash install.sh` then follow the prompt.
- When it finishes, launch from the Sammie-Roto-2 icon on your Desktop, from Spotlight, or with `bash run_sammie.sh`.

#### Linux
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command in the terminal: `bash install.sh` then follow the prompt.
- To launch the program, execute `bash run_sammie.sh` in the terminal or find the shortcut in the applications menu.

### Uninstall
If you want to remove the application, simply delete the Sammie-Roto folder. Shortcuts on your desktop or elsewhere can also be deleted.

### Updates
**Full Changelog can be seen under [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)**
- [10/04/2026] 2.5.0 - Adds an anime segmentation model, ProPainterX object removal, and several improvements and fixes.
- [09/05/2026] 2.4.1 - Premilinary support for AMD GPUs on Windows, some bugfixes.
- [08/22/2026] 2.4.0 - New installer/updater, additional segmentation tracking options, many small fixes and improvements.
- [04/17/2026] 2.3.3 - Several large performance optimizations, and colorspace conversions are now handled correctly.
- [04/10/2026] 2.3.2 - Improved temporal stability for VideoMaMa.
- [04/02/2026] 2.3.1 - Added a live preview during segmentation while holding the shift key.
- [03/27/2026] 2.3.0 - Added VideoMaMa model, added option to combine objects when matting, fixed major segmentation bug, and more.
- [03/08/2026] 2.2.0 - Added MatAnyone2 model.
- [01/18/2026] 2.1.1 - Rebuilt the export dialog, slightly faster application startup, bug fixes.
- [12/16/2025] 2.1.0 - Added In/Out markers. Modifying points no longer deletes tracking data. Enabled half-precision for much faster segmentation. Added EfficientTAM model.
- [11/23/2025] 2.0.0 - First stable release. Includes several new features and bugfixes. New quick-start video tutorial and [Discord server](https://discord.gg/jb5qrFyGFF).
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone) & [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone code)
* [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
* [ProPainter](https://github.com/sczhou/ProPainter)
* Some icons by [Yusuke Kamiyamane](http://p.yusukekamiyamane.com/)
