"""Single selection point shared by preview and existing export workers."""
from pathlib import Path
import numpy as np
from sammie import core
from sammie.settings_manager import get_settings_manager
from .algorithms import RefinementSettings
from .cache import RefinementCache, load_base


def selected_alpha(frame, points, object_id_filter=None):
    manager = get_settings_manager()
    if not manager.get_session_setting('vitmatte_enabled', False) or manager.get_session_setting('matany_model') != 'VideoMaMa':
        return None
    try:
        settings = RefinementSettings.from_session(manager)
    except ValueError:
        return None  # Users may temporarily cross thresholds while editing controls.
    stage = manager.get_session_setting('vitmatte_view', 'validated')
    root = Path(core.temp_dir)
    ids = sorted({p['object_id'] for p in points if 'object_id' in p})
    if manager.get_session_setting('matany_combined', False) and len(ids) > 1:
        ids = [0]
    if object_id_filter is not None:
        ids = [i for i in ids if i == object_id_filter]
    result = np.zeros((core.VideoInfo.height, core.VideoInfo.width), np.float32)
    cache = RefinementCache(root)
    source = root/'frames'/f'{frame:05d}.{core.get_frame_extension()}'
    for oid in ids:
        png = root/'matting'/f'{frame:05d}'/f'{oid}.png'
        if not png.exists():
            continue
        alpha = cache.selected(frame, oid, settings, stage) if stage != 'original' else None
        if alpha is None:
            alpha, _ = load_base(root, frame, oid, source, png)
        result = np.maximum(result, alpha)
    grow = manager.get_session_setting('matany_grow', 0)
    gamma = manager.get_session_setting('matany_gamma', 1.0)
    if grow:
        result = core.grow_shrink(result, grow)
    if gamma != 1:
        result = np.power(result, 1/max(gamma, .01))
    return np.clip(result, 0, 1).astype(np.float32)
