"""Read existing matte results for multilayer delivery; never run a model."""
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
import re
import cv2
import numpy as np


class MatteLayers:
    def __init__(self, job, shape):
        self.root = Path(job['root'])
        self.shape = tuple(shape)
        self.session = job.get('session', {})
        self.source = job.get('matte_source', 'matting')
        if self.source not in ('matting', 'segmentation'):
            raise ValueError('Choose Matting or Segmentation as the exported alpha source.')
        self.folder = self.root/('matting' if self.source == 'matting' else 'masks')
        self.extension = job.get('extension', 'png')
        self.ids = sorted({int(p['object_id']) for p in job.get('points', []) if 'object_id' in p})
        self.guide_ids = list(self.ids)
        if any(i < 0 for i in self.ids):
            raise ValueError('Invalid object ID in alpha export.')
        self.combined = (self.source == 'matting' and self.session.get('matany_combined', False)
                         and len(self.ids) > 1)
        if self.combined:
            self.ids = [0]
        if not self.ids:
            raise ValueError('No objects available for alpha export. Generate mattes, or disable Include generated alpha.')
        indices = range(job['first'], job['last']+1)
        self.missing = {}
        for oid in self.ids:
            missing = [i for i in indices if not self.path(i, oid).is_file()]
            if len(missing) == len(indices) and not job.get('allow_empty_objects', False):
                raise ValueError(f'No {self.source} alpha for object {oid} in this range. '
                                 'Generate it first or choose the correct Alpha source.')
            if missing:
                self.missing[str(oid)] = missing
        for frame in indices:
            if all(not self.path(frame, oid).is_file() for oid in self.ids):
                raise ValueError(f'Missing {self.source} alpha at source frame {frame}. Generate this range first.')
        names = self.session.get('object_names', {})
        self.names = {}
        for oid in self.ids:
            label = 'Combined' if self.combined else names.get(str(oid), '')
            name = re.sub(r'_+', '_', re.sub(r'[^\w]', '_', label)).strip('_')[:80]
            self.names[oid] = f'{oid}_{name}.Y' if name else f'Object_{oid}.Y'

    def path(self, frame, oid):
        return self.folder/f'{frame:05d}'/f'{oid}.png'

    @contextmanager
    def context(self):
        # The app runs exports in a separate CPU worker. Also restore globals for
        # direct library/test callers; do not change or save the user's settings.
        from sammie import core, settings_manager as sm
        previous = sm._settings_manager
        old_root, old_h, old_w = core.temp_dir, core.VideoInfo.height, core.VideoInfo.width
        manager = sm.SettingsManager.__new__(sm.SettingsManager)
        keys = {f.name for f in fields(sm.SessionSettings)}
        manager.session_settings = sm.SessionSettings(**{k: v for k, v in self.session.items() if k in keys})
        manager.session_settings.frame_format = self.extension
        manager.app_settings = sm.ApplicationSettings()
        sm._settings_manager = manager
        core.temp_dir = str(self.root)
        core.VideoInfo.height, core.VideoInfo.width = self.shape
        try:
            yield
        finally:
            sm._settings_manager = previous
            core.temp_dir, core.VideoInfo.height, core.VideoInfo.width = old_root, old_h, old_w

    def channels(self, frame):
        from sammie import core
        from sammie.refinement.algorithms import alpha_float
        from sammie.refinement.cache import load_base
        from sammie.refinement.display import selected_alpha
        channels = {}
        union = np.zeros(self.shape, np.float32)
        with self.context():
            for oid in self.ids:
                png = self.path(frame, oid)
                if not png.is_file():
                    alpha = np.zeros(self.shape, np.float32)
                elif self.source == 'matting' and self.session.get('matany_model') == 'VideoMaMa':
                    # Match the selected ViTMatte view, including its safe fallback
                    # to original VideoMaMa, without quantizing the float cache.
                    alpha = selected_alpha(frame, [{'object_id': oid}], oid)
                    if alpha is None:
                        source = self.root/'frames'/f'{frame:05d}.{self.extension}'
                        alpha, _ = load_base(self.root, frame, oid, source, png)
                        grow = self.session.get('matany_grow', 0)
                        gamma = self.session.get('matany_gamma', 1.)
                        if grow:
                            alpha = core.grow_shrink(alpha, grow)
                        if gamma != 1:
                            alpha = np.power(alpha, 1/max(gamma, .01))
                else:
                    raw = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                    if raw is None:
                        raise ValueError(f'Cannot read alpha: {png}')
                    fn = core.apply_matany_postprocessing if self.source == 'matting' else core.apply_mask_postprocessing
                    alpha = alpha_float(fn(raw))
                alpha = alpha_float(alpha)
                if self.source == 'matting' and png.is_file():
                    from sammie.opaque_core import apply_from_files
                    alpha = apply_from_files(alpha, self.root, frame,
                        self.guide_ids if self.combined else [oid], self.session)
                if alpha.shape != self.shape:
                    raise ValueError(f'Alpha dimensions do not match the plate at frame {frame}, object {oid}.')
                channels[self.names[oid]] = alpha
                np.maximum(union, alpha, out=union)
        channels['A'] = union
        return channels

    def metadata(self):
        from dataclasses import asdict
        from sammie.refinement.cache import RefinementCache
        applied = RefinementCache(self.root).applied_settings()
        return dict(source=self.source, combined=self.combined,
                    channels={str(k): v for k, v in self.names.items()},
                    union='A: maximum of object alphas, matching Sammie union semantics',
                    missing_object_frames_as_zero=self.missing,
                    vitmatte_view=self.session.get('vitmatte_view', 'validated'),
                    vitmatte_enabled=self.session.get('vitmatte_enabled', False),
                    vitmatte_applied_settings=asdict(applied) if applied is not None else None,
                    session_parameters={k: v for k, v in self.session.items()
                                        if k.startswith(('matany_', 'vitmatte_')) or k in ('holes', 'dots', 'grow', 'border_fix')})
