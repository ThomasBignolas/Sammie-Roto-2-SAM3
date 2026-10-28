# sammie/sammie.py
import cv2
import json
import os
import numpy as np
import shutil
import re
import glob
import zipfile
from pathlib import Path
import threading
import queue
import multiprocessing
import av
from tqdm import tqdm
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from sammie import core
from sammie.smooth import run_smoothing_model, prepare_smoothing_model
from sammie.duplicate_frame_handler import replace_similar_matte_frames
from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import show_message_dialog
from sammie.model_downloader import ensure_models

smoothing_model = None  # global variable needed to avoid complexity of passing the model around

# .........................................................................................
# SAM2 / EfficientTAM segmentation
# .........................................................................................

from sammie.backends.manager import SamManager


# .........................................................................................
# Smoothing model
# .........................................................................................

def load_smoothing_model():
    global smoothing_model
    if smoothing_model is None:
        device = core.DeviceManager.get_device()
        try:
            smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)
        except Exception as e:
            print(f"Warning: Could not load antialiasing model: {e}")
            smoothing_model = None


# .........................................................................................
# View / display handlers
# .........................................................................................

def update_image(slider_value, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Main image update function - delegates to specific view handlers

    Args:
        slider_value: Frame number
        view_options: Dictionary of view options
        points: List of point dictionaries
        return_numpy: If True, return numpy array; if False, return QPixmap
        object_id_filter: If specified, only process masks for this object ID
        preview_mask: If specified, a live (uncommitted) mask to substitute
            for preview_object_id's normal saved mask
        preview_object_id: Which object preview_mask belongs to - every
            other object's normal saved mask is still loaded and shown as
            usual, unaffected by the preview

    Returns:
        QPixmap or numpy array depending on return_numpy parameter
    """
    view_mode = view_options.get("view_mode", "Segmentation-Edit")

    if view_mode == "Segmentation-Edit":
        return _handle_segmentation_edit_view(slider_value, view_options, points, return_numpy, object_id_filter, preview_mask, preview_object_id)
    elif view_mode == "Segmentation-Matte":
        return _handle_segmentation_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-BGcolor":
        return _handle_segmentation_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-Alpha":
        return _handle_segmentation_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Matte":
        return _handle_matting_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-BGcolor":
        return _handle_matting_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Alpha":
        return _handle_matting_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "ObjectRemoval":
        return _handle_object_removal_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode in ('AOV-Foreground', 'AOV-Foreground-BGcolor'):
        from sammie.foreground.display import preview
        image = preview(slider_value, points, view_options)
        return image if return_numpy else _convert_to_qpixmap(image)
    elif view_mode in ('AOV-Depth', 'AOV-Normals'):
        from sammie.aov.display import preview
        options = get_settings_manager().get_session_setting('aov_settings', {})
        image = preview(core.temp_dir, 'depth' if view_mode == 'AOV-Depth' else 'normals', slider_value, options)
        if image is None:
            image = core.load_base_frame(slider_value)
            if image is not None:
                image = image.copy()
                cv2.putText(image, 'AOV unavailable - Generate this range/settings', (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 200, 80), 2, cv2.LINE_AA)
        return image if return_numpy else _convert_to_qpixmap(image)
    elif view_mode == "None":
        return _handle_none_view(slider_value, return_numpy)
    else:
        print(f"Unknown view mode: {view_mode}")
        return None


def load_removal_frame(frame_number):
    """
    Load the object removal frame image from disk.
    If the frame does not exist, load the base frame instead.
    """
    frame_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return core.load_base_frame(frame_number)


def _convert_to_qpixmap(image):
    """Convert NumPy array to QPixmap"""
    if image is None:
        return QPixmap.fromImage(QImage())
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    if len(image.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif image.shape[2] == 3:  # RGB
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:  # RGBA
        bytes_per_line = 4 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

    return QPixmap.fromImage(q_image)


def _get_display_mask(frame_number, points, object_id_filter, folder, postprocess_fn, float_alpha=False):
    """
    Load the combined mask for a view and postprocess it. Returns None (without
    calling postprocess_fn) if no mask exists for this frame/object filter —
    callers are responsible for deciding what to do in that case.
    """
    if folder == core.matting_dir:
        from sammie.refinement.display import selected_alpha
        alpha = selected_alpha(frame_number, points, object_id_filter)
        if alpha is not None:
            from sammie.opaque_core import apply_to_preview
            alpha = apply_to_preview(alpha, frame_number, points, object_id_filter)
            return alpha if float_alpha else np.round(alpha*255).astype(np.uint8)
    loader_points = points
    if (folder == core.matting_dir and get_settings_manager().get_session_setting('matany_combined',False)
            and len({p['object_id'] for p in points if 'object_id' in p}) > 1):
        loader_points = [{'object_id':0}]
    mask = core.load_masks_for_frame(frame_number, loader_points, return_combined=True,
                                      object_id_filter=object_id_filter, folder=folder)
    if mask is None:
        return None
    mask = postprocess_fn(mask)
    if folder == core.matting_dir and get_settings_manager().get_session_setting('matany_core_strength', 0):
        from sammie.opaque_core import apply_to_preview
        from sammie.refinement.algorithms import alpha_float
        mask = alpha_float(apply_to_preview(mask, frame_number, points, object_id_filter))
        return mask if float_alpha else np.round(mask*255).astype(np.uint8)
    return mask


def _maybe_antialias_mask(mask_3channel, view_options):
    """
    Apply the smoothing model to a 3-channel mask if antialiasing is enabled
    in view_options (default True) and the model is available. Used only by
    segmentation views — matting views have never applied antialiasing.
    """
    if not view_options.get("antialias", True):
        return mask_3channel
    global smoothing_model
    if smoothing_model is None:
        load_smoothing_model()
    if smoothing_model is not None:
        device = core.DeviceManager.get_device()
        mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
    return mask_3channel


def _composite_bgcolor(image, mask, bgcolor, background_style='solid'):
    """Blend `image` with a solid `bgcolor` background using `mask` (single-channel, 0-255) as alpha."""
    if background_style == 'checkerboard':
        h, w = image.shape[:2]
        cell = max(8, round(min(h, w) / 40))
        rows = ((np.arange(h) // cell) % 2).astype(np.uint8)
        cols = ((np.arange(w) // cell) % 2).astype(np.uint8)
        grid = rows[:, None] ^ cols[None, :]
        bg = np.repeat((grid[..., None] * 56 + 104), 3, axis=2)
    else:
        bg = np.full_like(image, bgcolor)
    alpha = mask.astype(np.float32) / 255.0
    return cv2.blendLinear(image, bg, alpha, 1.0 - alpha)


def _composite_alpha(image, mask):
    """Merge `image` (RGB) and `mask` (single-channel, 0-255) into an RGBA image."""
    return cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2], mask])


def _handle_none_view(frame_number, return_numpy=False):
    """Handle None view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_edit_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Handle Segmentation-Edit view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    image = apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter, preview_mask, preview_object_id)

    highlighted_points = view_options.get('highlighted_point', None)

    if highlighted_points is None:
        highlighted_points = None
    elif isinstance(highlighted_points, list):
        highlighted_points = highlighted_points.copy()
    else:
        highlighted_points = [highlighted_points]

    image = draw_points(image, frame_number, points, highlighted_points)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Matte view"""
    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return None

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_segmentation_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    image = _composite_bgcolor(image, mask_3channel[:, :, 0], bgcolor, view_options.get('background_style', 'solid'))

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.mask_dir, core.apply_mask_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask_3channel = np.stack([mask] * 3, axis=-1)
    mask_3channel = _maybe_antialias_mask(mask_3channel, view_options)
    mask = mask_3channel[:, :, 0]

    image_rgba = _composite_alpha(image, mask)

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_matting_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Matte view"""
    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing,
                             float_alpha=return_numpy and view_options.get('float_alpha', False))
    if mask is None:
        return None

    mask_3channel = np.stack([mask] * 3, axis=-1)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_matting_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    image = _composite_bgcolor(image, mask, bgcolor, view_options.get('background_style', 'solid'))

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_matting_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = _get_display_mask(frame_number, points, object_id_filter, core.matting_dir, core.apply_matany_postprocessing)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    image_rgba = _composite_alpha(image, mask)

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_object_removal_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Object Removal view"""
    image = load_removal_frame(frame_number)
    if image is None:
        return None

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None

    mask = core.apply_mask_postprocessing(mask)

    settings_mgr = get_settings_manager()
    grow = settings_mgr.get_session_setting("inpaint_grow", 0)
    mask = core.grow_shrink(mask, grow)

    if view_options.get("show_removal_mask", True):
        image = draw_removal_overlay(image, mask)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def draw_masks(image, processed_masks):
    """Draw masks on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    combined_colored_mask = np.zeros_like(image, dtype=np.uint8)
    mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)

    for object_id, mask in processed_masks.items():
        color = core.PALETTE[object_id % len(core.PALETTE)]
        mask_bin = cv2.compare(mask, 0, cv2.CMP_GT)
        cv2.bitwise_or(mask_binary, mask_bin, dst=mask_binary)
        # OpenCV's masked copy avoids several large NumPy boolean gather/scatter
        # temporaries at source resolution. Overlap order is unchanged.
        color_image = np.broadcast_to(np.asarray(color, np.uint8), image.shape)
        cv2.copyTo(color_image, mask_bin, combined_colored_mask)

    if np.any(mask_binary):
        overlay = image.copy()
        blend = cv2.addWeighted(image, 0.5, combined_colored_mask, 0.5, 0)
        cv2.copyTo(blend, mask_binary, overlay)
        return overlay
    else:
        return image
    

def draw_removal_overlay(image, mask):
    """Draw an outline (no fill) around the removal mask, using object 0's color"""
    return draw_contours(image, {0: mask})


def draw_contours(image, processed_masks):
    """Draw colored contours on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    overlay = image.copy()
    kernel = np.ones((3, 3), np.uint8)

    for object_id, mask in processed_masks.items():
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        border_color = core.PALETTE[object_id % len(core.PALETTE)]
        overlay[edges > 0] = border_color

    return overlay


def draw_points(image, frame_number, points, highlighted_points=None):
    """Draw points on image"""
    frame_points = [p for p in points if p['frame'] == frame_number and p.get('kind') != 'sam3_seed']
    if not frame_points:
        return image
    image = image.copy()  # Cached source pixels must never acquire drawn prompts.

    highlighted_set = set()
    if highlighted_points:
        highlighted_set = {(p['frame'], p['x'], p['y']) for p in highlighted_points}

    for point in frame_points:
        is_highlighted = (point['frame'], point['x'], point['y']) in highlighted_set
        center = (point['x'], point['y'])
        point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
        if is_highlighted:
            cv2.circle(image, center, 9, (0, 128, 255), 3)
        cv2.circle(image, center, 5, (255, 255, 0), 2)
        cv2.circle(image, center, 4, point_color, -1)

    return image

def apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter=None, preview_mask=None, preview_object_id=None):
    """Apply postprocessing to masks and draw them on the image for display"""
    raw_masks = core.load_masks_for_frame(
        frame_number, points, return_combined=False, object_id_filter=object_id_filter
    )

    if raw_masks:
        processed_masks = {
            object_id: core.apply_mask_postprocessing(mask) for object_id, mask in raw_masks.items()
        }
    else:
        processed_masks = {}

    # Substitute the preview mask for the selected object if provided
    if preview_mask is not None and preview_object_id is not None:
        processed_masks[preview_object_id] = core.apply_mask_postprocessing(preview_mask)

    if view_options.get("show_masks", True):
        image = draw_masks(image, processed_masks)
    if view_options.get("show_outlines", True):
        image = draw_contours(image, processed_masks)

    return image


# .........................................................................................
# Mask / session utilities
# .........................................................................................

def deduplicate_masks(parent_window):
    """Deduplicate similar masks using settings threshold"""
    settings_mgr = get_settings_manager()
    threshold = settings_mgr.app_settings.dedupe_threshold
    return replace_similar_matte_frames(parent_window, threshold)


def remove_backup_mattes():
    if os.path.exists(core.backup_dir):
        shutil.rmtree(core.backup_dir)


# .........................................................................................
# Video / image I/O
# .........................................................................................

def load_video(video_file, parent_window=None):
    from .ingest import prepare_dialog, commit, apply_info
    from .projects import MARKER
    if (Path(core.temp_dir)/MARKER).exists():
        raise ValueError('Detach the working project before importing new media.')
    manager = get_settings_manager()
    prepared = prepare_dialog(video_file, core.temp_dir, manager, parent_window)
    if prepared is None: return 0
    stage, info = prepared
    from .background_io import run_io
    run_io(parent_window, 'Installing frames and cleaning the previous session',
           lambda progress, cancel: commit(stage, core.temp_dir))
    apply_info(info, manager)
    return info['total_frames']


def detect_image_sequence(image_path):
    """
    Detect if an image is part of a sequence based on common naming patterns.
    Returns (is_sequence, sequence_files, base_name) or (False, [], None).
    base_name is the filename prefix with the trailing frame number removed
    """
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    patterns = [
        r'^(.+?)(\d{4,})$',
        r'^(.+?)(\d{3})$',
        r'^(.+?)(\d{2})$',
        r'^(.+?)_(\d+)$',
        r'^(.+?)\-(\d+)$',
        r'^(.+?)\.(\d+)$',
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            base_name = match.group(1)

            if pattern == r'^(.+?)(\d{4,})$':
                glob_pattern = f"{base_name}*{ext}"
            elif pattern == r'^(.+?)(\d{3})$':
                glob_pattern = f"{base_name}???{ext}"
            elif pattern == r'^(.+?)(\d{2})$':
                glob_pattern = f"{base_name}??{ext}"
            else:
                separator = pattern.split('(\\d+)')[0][-1]
                glob_pattern = f"{base_name}{separator}*{ext}"

            search_path = os.path.join(directory, glob_pattern)
            potential_files = glob.glob(search_path)

            sequence_files = []
            for file_path in potential_files:
                file_name = os.path.basename(file_path)
                file_base = os.path.splitext(file_name)[0]
                candidate = re.match(pattern, file_base)
                if candidate and candidate.group(1) == base_name:
                    sequence_files.append(file_path)

            def natural_sort_key(path):
                base_name = os.path.splitext(os.path.basename(path))[0]
                match = re.match(pattern, base_name)
                if match:
                    return (match.group(1), int(match.group(2)))
                return (base_name, 0)

            sequence_files.sort(key=natural_sort_key)

            if len(sequence_files) > 1:
                return True, sequence_files, base_name

    return False, [], None


def get_sequence_start_frame(files):
    """
    Return the frame number of the first file if the files form a consecutively
    numbered sequence (each frame number is exactly one more than the previous).
    Returns 0 for single images, sequences with gaps/duplicates, or files without
    a trailing frame number.
    """
    if len(files) < 2:
        return 0

    numbers = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r'(\d+)$', stem)
        if not match:
            return 0
        numbers.append(int(match.group(1)))

    first = numbers[0]
    if all(n == first + i for i, n in enumerate(numbers)):
        return first
    return 0


def load_image_sequence(image_path, parent_window=None):
    from .ingest import prepare_dialog, commit, apply_info
    from .projects import MARKER
    if (Path(core.temp_dir)/MARKER).exists():
        raise ValueError('Detach the working project before importing new media.')
    manager = get_settings_manager()
    prepared = prepare_dialog(image_path, core.temp_dir, manager, parent_window, images=True)
    if prepared is None: return 0
    stage, info = prepared
    from .background_io import run_io
    run_io(parent_window, 'Installing frames and cleaning the previous session',
           lambda progress, cancel: commit(stage, core.temp_dir))
    apply_info(info, manager)
    return info['total_frames']


def resume_session():
    if os.path.exists(core.temp_dir):
        if os.path.exists(core.frames_dir) and os.listdir(core.frames_dir):
            print("Resuming previous session...")
            QApplication.processEvents()
            restore_video_info()
            return core.VideoInfo.total_frames


def restore_video_info():
    if not os.path.exists(core.frames_dir):
        return 0
    from PIL import Image
    path = Path(core.frames_dir)/f'00000.{core.get_frame_extension()}'
    if path.is_file():
        # Read the image header; decoding every full-resolution pixel is deferred
        # to the preview process. Missing/invalid metadata still raises clearly.
        with Image.open(path) as image:
            width, height = image.size
        core.VideoInfo.width = width
        core.VideoInfo.height = height
        core.VideoInfo.fps = 24
        core.VideoInfo.fps_rational = ''
        extension = core.get_frame_extension()
        core.VideoInfo.total_frames = len([f for f in os.listdir(core.frames_dir) if f.endswith(f".{extension}")])

        settings_mgr = get_settings_manager()
        if settings_mgr.session_exists():
            session_width = settings_mgr.get_session_setting("video_width", 0)
            session_height = settings_mgr.get_session_setting("video_height", 0)
            session_fps = settings_mgr.get_session_setting("video_fps", 0)
            session_frames = settings_mgr.get_session_setting("total_frames", 0)

            if session_width > 0 and session_height > 0:
                core.VideoInfo.width = session_width
                core.VideoInfo.height = session_height
            if session_fps > 0:
                core.VideoInfo.fps = session_fps
                core.VideoInfo.fps_rational = settings_mgr.get_session_setting('video_fps_rational', '')
            if session_frames > 0:
                core.VideoInfo.total_frames = session_frames


def load_project(file_name, parent_window):
    from sammie.projects import MARKER
    if (Path(core.temp_dir)/MARKER).exists():
        raise ValueError('Open portable projects through the project controller; this is a saved working folder.')
    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.temp_dir, exist_ok=True)
    progress = None

    try:
        with zipfile.ZipFile(file_name, 'r') as zipf:
            file_list = zipf.namelist()
            total_files = len(file_list)

        if total_files == 0:
            return True

        progress = QProgressDialog("Extracting files...", "", 0, total_files, parent_window)
        progress.setWindowTitle("Extracting Backup")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        with zipfile.ZipFile(file_name, 'r') as zipf:
            for i, file_name in enumerate(file_list):
                progress.setValue(i)
                progress.setLabelText(f"Extracting: {os.path.basename(file_name)}")
                QApplication.processEvents()
                zipf.extract(file_name, core.temp_dir)

        progress.setValue(total_files)
        progress.setLabelText("Extraction completed!")
        QApplication.processEvents()
        return True

    except Exception as e:
        if progress:
            progress.close()
        raise e

    finally:
        if progress:
            progress.close()


def save_project(file_name, parent_window):
    total_files = 0
    all_files = []

    for root, dirs, files in os.walk(core.temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, core.temp_dir)
            all_files.append((file_path, arcname))
            total_files += 1

    progress = QProgressDialog("Backing up files...", "Cancel", 0, total_files, parent_window)
    progress.setWindowTitle("Creating Backup")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    QApplication.processEvents()

    try:
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_STORED) as zipf:
            for i, (file_path, arcname) in enumerate(all_files):
                if progress.wasCanceled():
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                    return 0

                progress.setValue(i)
                progress.setLabelText(f"Adding: {os.path.basename(file_path)}")
                QApplication.processEvents()
                zipf.write(file_path, arcname)

        progress.setValue(total_files)
        QApplication.processEvents()

    except Exception as e:
        progress.close()
        try:
            os.remove(file_name)
        except Exception:
            pass
        raise e

    finally:
        if not progress.wasCanceled():
            progress.close()

    return 1
