"""Optional opaque-interior constraint, shared by preview and matte exports.

This is an explicit opaque-subject assumption, not inferred transparency. It
uses only the current frame's guide and never blends alpha from other frames.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from .mask_contacts import combine_masks
from .refinement.algorithms import alpha_float, scaled_radius


def core_weight(guide, inset=16):
    """Zero at/near the guide silhouette, smoothly reaching one deeper inside."""
    if not np.isfinite(inset) or inset < 1:
        raise ValueError('Opaque-core inset must be at least one pixel at 1080p.')
    g = alpha_float(guide) >= .5
    radius = scaled_radius(inset, g.shape)
    distance = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = np.clip((distance-radius)/max(radius, 1), 0, 1)
    return (t*t*(3-2*t)).astype(np.float32)


def protect_core(alpha, weight, strength=1.):
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Opaque-core strength must be between zero and one.')
    if strength == 0:
        return alpha  # Exact disabled behavior, including dtype.
    a = alpha_float(alpha)
    if weight.shape != a.shape or not np.isfinite(weight).all() or weight.min()<0 or weight.max()>1:
        raise ValueError('Opaque core and alpha must be aligned, finite and in [0, 1].')
    return (a + (1-a)*weight*strength).astype(np.float32)


@lru_cache(maxsize=2)
def _cached_weight(paths, options, inset):
    # Only two source-sized float masks are retained. File size/mtime and every
    # guide postprocessing setting participate in this key, not the alpha.
    masks=[]
    for path, size, mtime in paths:
        mask=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:raise ValueError(f'Cannot read opaque-core guide: {path}')
        masks.append(mask)
    settings=dict(options)
    guide=combine_masks(masks,settings['contact_seam_width'])
    from . import core
    for name, function in [('holes',core.fill_small_holes),('dots',core.remove_small_dots),
                           ('border_fix',core.apply_border_fix),('grow',core.grow_shrink)]:
        if settings[name]:guide=function(guide,settings[name])
    weight=core_weight(guide,inset);weight.setflags(write=False)
    return weight


def apply_from_files(alpha, root, frame, object_ids, settings):
    strength=settings.get('matany_core_strength',0)
    if not strength:return alpha
    paths=[]
    for oid in sorted(set(object_ids)):
        path=Path(root)/'masks'/f'{frame:05d}'/f'{oid}.png'
        if path.is_file():
            stat=path.stat();paths.append((str(path.resolve()),stat.st_size,stat.st_mtime_ns))
    if not paths:return alpha  # No guide, no manufactured foreground.
    options=tuple((key,settings.get(key,0)) for key in ('contact_seam_width','holes','dots','border_fix','grow'))
    weight=_cached_weight(tuple(paths),options,settings.get('matany_core_inset',16))
    return protect_core(alpha,weight,strength)


def apply_to_preview(alpha, frame, points, object_id_filter=None):
    from . import core
    from .settings_manager import get_settings_manager
    settings=vars(get_settings_manager().session_settings)
    if not settings.get('matany_core_strength',0):return alpha
    ids=sorted({p['object_id'] for p in points if 'object_id' in p})
    combined=settings.get('matany_combined',False) and len(ids)>1
    if combined:
        if object_id_filter not in (None,0):return alpha
        matte_ids=[0]
    else:
        if object_id_filter is not None:ids=[i for i in ids if i==object_id_filter]
        ids=[i for i in ids if (Path(core.matting_dir)/f'{frame:05d}'/f'{i}.png').is_file()]
        matte_ids=ids
    # The legacy mask loader returns zero for a missing matte. Do not turn that
    # into an opaque guide and accidentally conceal unfinished processing.
    if not any((Path(core.matting_dir)/f'{frame:05d}'/f'{i}.png').is_file() for i in matte_ids):return alpha
    return apply_from_files(alpha,core.temp_dir,frame,ids,settings)
