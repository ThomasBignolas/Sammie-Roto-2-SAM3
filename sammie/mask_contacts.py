"""Optional, non-destructive repair of narrow seams in a combined SAM guide.

This is a user-controlled geometric assumption about touching opaque objects,
not a new segmentation prediction. Individual object masks remain unchanged.
"""
import math
import cv2
import numpy as np


def combine_masks(masks, contact_width=0):
    """Union uint8 masks, optionally bridging interior gaps between distinct IDs.

    ``contact_width`` is the gap tolerance in pixels at a 1080-pixel short edge.
    Repairs need two different nearby objects, must lie inside a morphological
    closing of the union, and must be away from its exterior silhouette. Single
    object holes, outer silhouette growth and wide gaps are excluded. Narrow
    enclosed remnants touching repaired pixels are completed, with one pixel
    of radius tolerance for rasterization. A real,
    narrow gap between two objects is still ambiguous: keep this OFF unless the
    shot calls for treating the objects as one opaque foreground group.
    """
    masks = list(masks)
    if not masks:
        return None
    union = masks[0].copy()
    for mask in masks[1:]:
        if mask.shape != union.shape:
            raise ValueError('Cannot combine masks with different dimensions')
        np.maximum(union, mask, out=union)
    if contact_width <= 0 or len(masks) < 2:
        return union
    if not math.isfinite(contact_width) or contact_width > 20:
        raise ValueError('Contact seam width must be between 0 and 20 pixels at 1080p')
    if union.ndim != 2 or union.dtype != np.uint8:
        raise ValueError('Contact repair requires single-channel uint8 segmentation masks')
    width = max(1, round(float(contact_width) * min(union.shape) / 1080))
    radius = max(1, math.ceil(width / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    foreground = (union > 0).astype(np.uint8)
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel,
                            borderType=cv2.BORDER_REPLICATE)
    candidate = (closed > 0) & (foreground == 0)
    if not candidate.any():
        return union
    interior = cv2.distanceTransform(closed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    candidate &= interior > 2*radius
    ys, xs = np.nonzero(candidate)
    if not len(ys):
        return union
    nearest = np.full(len(ys), np.inf, np.float32)
    second = nearest.copy()
    for mask in masks:
        if not np.any(mask):
            continue
        distance = cv2.distanceTransform((mask == 0).astype(np.uint8),
                                        cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[ys, xs]
        second = np.minimum(second, np.maximum(nearest, distance))
        nearest = np.minimum(nearest, distance)
    accepted = nearest + second <= width + 1
    union[ys[accepted], xs[accepted]] = 255
    # Pointwise distance/closing tests can leave tiny disconnected islands of
    # background along an otherwise repaired contact. Complete only narrow,
    # enclosed remnants whose boundary actually touches this repair. Existing
    # single-object holes and gaps open to the exterior remain unchanged.
    added = ((union > 0) & (foreground == 0)).astype(np.uint8)
    if added.any():
        _complete_contact_remnants(union, added, width)
    return union


def _complete_contact_remnants(union, added, width):
    background = (union == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    adjacent = cv2.dilate(added, np.ones((3, 3), np.uint8)) & background
    candidates = np.unique(labels[adjacent != 0])
    h, w = union.shape
    for component in candidates:
        if component == 0:
            continue
        x, y, cw, ch, _ = stats[component]
        if x == 0 or y == 0 or x+cw == w or y+ch == h:
            continue  # Includes gaps reaching the frame edge.
        region = labels[y:y+ch, x:x+cw] == component
        # Padding supplies real foreground around the enclosed component.
        distance = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        if distance.max() <= width/2 + 1:
            view = union[y:y+ch, x:x+cw]
            view[region] = 255
