# Windows installation and workflow

## Requirements

- Windows x64, Git for Windows, internet access, and a current NVIDIA driver.
- A CUDA GPU with BF16 support. The tested workstation is an RTX 5060 Ti with 16 GB VRAM and driver 610.62. The SAM 3 backend currently requires CUDA/BF16; SAM 2 retains its existing device support.
- Allow several tens of GB of free disk space for two Python environments, package caches, model files and footage. Each SAM 3 checkpoint is approximately 3.5 GB. Full resolution frame caches can be much larger than the compressed source video.
- Use a writable folder outside Program Files. Do not install over an unrelated existing project.

## Install

Clone the development repository (your GitHub account needs access to this private repository):

```powershell
git clone --branch feature/vitmatte-refinement https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3.git
cd Sammie-Roto-2-SAM3
.\install_sam3.bat
```

Alternatively, double-click **install_sam3.bat** in the checkout. Git must be available on PATH; the script obtains uv and Python 3.12 if needed. No system Python packages are replaced.

The installer uses the original `uv.lock` for `.venv`, including PyTorch 2.11.0/CUDA 13.0. It separately installs PyTorch 2.10.0/CUDA 12.8, NumPy 1.26.4, Triton for Windows, and the pinned official SAM 3 source into `.runtime/sam3-env`. The SAM 3 install must not be done with pip inside `.venv`: the two dependency sets conflict.

At the end, the installer checks CUDA, BF16, official model imports and the Triton distance-transform kernel. A CUDA toolkit installation is not required for the tested wheel configuration. If the check fails, retain the printed error and update the NVIDIA driver before retrying. Do not arbitrarily change Torch, NumPy or Triton versions in one of the environments.

To reinstall only the worker:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_sam3.ps1 -WorkerOnly
```

## Obtain authorized model weights

1. Sign in at [Hugging Face](https://huggingface.co/join).
2. Request access and accept the terms on [facebook/sam3](https://huggingface.co/facebook/sam3) and, separately, [facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
3. After approval, create a **read** token at [Hugging Face token settings](https://huggingface.co/settings/tokens). With a fine-grained token, grant read access to the gated repositories your account can access.
4. Run **setup_models.bat** and paste the token only into its hidden local prompt. It uses Hugging Face's normal credential storage, outside the project. Do not put tokens in chat, Git, project files or environment configuration committed to the repository.

The script downloads the revisions pinned in `model_versions.json` and verifies their file sizes. You can download one version at a time:

```powershell
.\setup_models.bat --model sam3
.\setup_models.bat --model sam3.1 --no-login
```

Alternatively, download these files from the approved model pages and place them here:

```text
checkpoints/sam3.pt
checkpoints/sam3.1_multiplex.pt
```

Use Meta's native `.pt` files, not Transformers safetensors or a different model's checkpoint. The application reports missing checkpoints without preventing access to SAM 2 or existing exports. SAM 2 and MatAnyone use the original application's model download dialogs when first needed.

## Launch and segment

Run **run_sam3.bat**. It opens the existing Sammie-Roto application, bypassing the upstream release updater. You can also pass a footage path:

```powershell
.\run_sam3.bat "D:\Footage\shot001.mov"
```

1. Import a video, a still image, or an image sequence using the existing controls.
2. Choose **SAM 3** or **SAM 3.1** in the Segmentation tab. Choose the working resolution, then click **Load Model**. Wait for the loading status to complete.
3. For points, choose an object ID, left-click foreground, and right-click or Ctrl+left-click background. Use another object ID for another independently tracked object.
4. For text, type a short description such as `person`, or multiple descriptions such as **`person, camera`** / **`person and camera`**. Click **Find objects**, check the desired labeled instances (or **Select all**), and accept. Each selected instance receives its own application object ID. You can then search for another description and add its results without clearing previous objects.
5. For a box, enable **Box: click two corners**, then click opposite corners on the image. Text may be combined with the box. Boxes perform initial discovery; correct an existing tracked object using points.
6. Use the original full-range, forward, backward or one-frame tracking controls. For a selection made in the middle of a shot, track forward and then backward from the selected frame. In/out markers restrict processing without resetting the session.
7. Scrub to a problematic frame, select the correct object ID, add positive/negative points, and propagate again in each required direction. Previously exported media is not automatically regenerated.
8. Save the project with the existing controls. Text/box selections are stored as selection records, not fabricated point prompts. On reopening, the backend matches detections to saved seed masks; it stops if identities cannot be matched reliably.

Text/box discovery is additive. Cancelling the selection dialog or finding no matches preserves existing objects and their tracking history. Use **Clear All** when you intend to replace everything. Commas, semicolons and the word `and` separate independent searches; use double quotes for one description containing those words, e.g. `"person in a red and white coat"`. `person holding a camera` remains a single relational description. A box, when present, is applied to each listed description. Up to 16 objects are supported; each independent concept adds tracking-state memory, so the practical limit depends on footage and available VRAM. Shift-hover live preview remains available for SAM 2; SAM 3 corrections are applied by clicking.

## Matting and export

Use the original Matting tab and select **MatAnyone**. Start with its 720 working resolution on a 16 GB GPU. SAM 3 exits its worker before MatAnyone loads, releasing model and session GPU memory. Prompts and cached masks remain on disk. After matting, SAM 3 reloads only when you edit or track again; viewing and exporting do not reload it.

For **one matte containing both the person and camera**, track both selected objects, then enable **Combine All Objects** in the Matting tab before **Run Matting**. This unions their segmentation masks before matting. Leave it unchecked when you need independent mattes/channels. VideoMaMa and optional ViTMatte Base use the same object selection; see [ViTMatte setup](VITMATTE.md).

Export **Segmentation-Matte**, **Segmentation-Alpha**, **Matting-Matte** or **Matting-Alpha** through the existing export dialog. PNG sequences preserve dimensions and numbering; multilayer EXR includes stable channels even on empty frames. FFV1 and the other original codecs remain available. Verify your compositor's sequence start frame against the project setting (for example, 1001).

Source images are retained at their imported resolution. A 4K source can use a 1920 working proxy; masks are resized back to 4K before the original matting/export pipeline. This preserves dimensions, not lost fine detail: compare against source-resolution working frames for small features. The official model's encoder remains fixed at 1008 pixels internally.

Constant frame-rate video retains its exact rational rate (for example, 24000/1001). Decoded source frame order and count are preserved. Variable frame-rate input timestamps are recorded in `temp/source_timing.json`; video export is constant frame-rate at the reported average rate. Use an image sequence plus that timing record when exact VFR timing matters. Audio and source timecode are not added by this upgrade.

## Memory and recovery

- Begin at 1920 long edge, one or two objects, and a short shot. The status shows processing time and peak allocated VRAM; driver and other-application memory are additional.
- The worker caches eight decoded frames on CPU, streams masks to disk, and retains the model's temporal tracking state between ranges. It does not reset tracking at arbitrary chunk boundaries.
- SAM 3 supports state CPU offload. The pinned SAM 3.1 API does not; its tracker state can grow with shot length and object count. Neither version has a guaranteed constant-memory bound for arbitrarily long videos.
- Cancel takes effect between frames. Completed masks remain available; re-run the required range to finish. CUDA request failures stop the worker and preserve committed prompts for a clean rebuild on the next operation.
- If memory is exhausted, close other GPU applications, reduce selected object count, or process shorter shots at real cuts. Lowering proxy size helps input/output storage and transfer costs but does not reduce the fixed encoder to a smaller network. Splitting a shot into independent projects loses cross-boundary identity/state and requires manual review.
- The proxy cache is reused when the source files and resolution are unchanged. Saving/reopening or unloading for matting rebuilds the tracker from prompts rather than serializing GPU state. Exact bitwise masks after a rebuild are not guaranteed.

To validate the installed trained models, follow [VALIDATION.md](VALIDATION.md).
