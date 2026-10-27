"""Non-destructive spatial checks against a segmentation guide.

These are review candidates, not a prediction of matte accuracy. A segmentation
guide cannot distinguish a hole from real transparency or a missed object.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class QualityFlags:
    interior: np.ndarray
    exterior: np.ndarray
    margin_pixels: int


def _normalized(image):
    a = np.asarray(image)
    if a.ndim != 2 or not a.size:
        raise ValueError('Expected a non-empty single-channel image.')
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    if a.dtype == np.bool_:
        return a.astype(np.float32)
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
        raise ValueError('Use uint8 or finite float alpha in [0, 1].')
    if a.min() < 0 or a.max() > 1:
        raise ValueError('Float alpha must be in [0, 1].')
    return a.astype(np.float32, copy=False)


def inspect_alpha(alpha, guide, margin=16, foreground=.95, background=.05):
    """Flag low interior alpha / high exterior alpha, excluding an edge band.

    Margin is measured at a 1080px short edge and scaled to the source image.
    Frame boundaries replicate the guide, so an object may touch the frame.
    The guide is binarized only for diagnostics; alpha is never modified.
    """
    if not np.isfinite([margin, foreground, background]).all() or margin < 0:
        raise ValueError('Margin must be finite and non-negative.')
    if not 0 <= background < foreground <= 1:
        raise ValueError('Use 0 <= background < foreground <= 1.')
    a, g = _normalized(alpha), _normalized(guide)
    if a.shape != g.shape:
        raise ValueError('Matte and segmentation dimensions differ.')
    g = (g >= .5).astype(np.uint8)
    radius = max(1, round(margin * min(a.shape) / 1080)) if margin else 0
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        inside = cv2.erode(g, kernel, borderType=cv2.BORDER_REPLICATE) > 0
        outside = cv2.dilate(g, kernel, borderType=cv2.BORDER_REPLICATE) == 0
    else:
        inside, outside = g > 0, g == 0
    return QualityFlags(inside & (a < foreground), outside & (a > background), radius)


def load_inspection(frame, points):
    """Load the same all-object alpha as Matting preview, and its current guide."""
    from pathlib import Path
    from sammie import core
    from sammie.settings_manager import get_settings_manager
    from sammie.sammie import _get_display_mask
    ids = sorted({p['object_id'] for p in points if 'object_id' in p})
    manager = get_settings_manager()
    matte_ids = [0] if manager.get_session_setting('matany_combined', False) and len(ids)>1 else ids
    if not matte_ids or not any((Path(core.matting_dir)/f'{frame:05d}'/f'{i}.png').is_file() for i in matte_ids):
        raise ValueError('No matte exists on this frame. Run matting first or choose a processed frame.')
    guide, loaded = core.load_combined_mask(frame, ids)
    if not loaded:
        raise ValueError('No segmentation guide exists on this frame.')
    guide = core.apply_mask_postprocessing(guide)
    rgb = core.load_base_frame(frame)
    if rgb is None:
        raise ValueError('The source frame is missing.')
    # Keep original IDs for optional opaque-core guidance, even though combined
    # matting is stored in object 0.
    alpha = _get_display_mask(frame, points, None, core.matting_dir,
                              core.apply_matany_postprocessing, float_alpha=True)
    if alpha is None:
        raise ValueError('The matte could not be loaded.')
    alpha = _normalized(alpha)
    if rgb.shape[:2] != alpha.shape or guide.shape != alpha.shape:
        raise ValueError('Source, matte and guide dimensions differ. Reprocess this frame.')
    return rgb, alpha, guide
