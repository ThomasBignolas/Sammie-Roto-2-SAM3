"""Sequential post-VideoMaMa pass, independent of Qt and SAM model lifetimes."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
from .algorithms import make_trimap, merge_alpha, TemporalValidator
from .cache import RefinementCache, load_base, key, stamp, VERSION
from .model import VitMatteModel, resolve_checkpoint, digest, check_cancel, predict_tiled


def run_refinement(root, frame_range, settings, extension='png', progress=lambda n, total, text: None,
                   cancel=lambda: False, model_factory=VitMatteModel, checkpoint_resolver=resolve_checkpoint):
    settings.validate()
    root = Path(root)
    frame_range = list(frame_range)
    session_file = root/'session_settings.conf'
    if session_file.exists():
        source_path = str(json.loads(session_file.read_text()).get('video_file_path', '')).lower()
        if source_path.endswith(('.exr', '.hdr')):
            raise ValueError('Use a display-referred RGB proxy of linear EXR/HDR footage before ViTMatte refinement.')
    cache = RefinementCache(root)
    jobs = [(f, p) for f in frame_range for p in sorted((root/'matting'/f'{f:05d}').glob('*.png'))]
    if not jobs:
        raise ValueError('No cached VideoMaMa mattes in the selected frame range. Run VideoMaMa first.')
    # Require a complete set per object over the range; do not export a mixture
    # of refined frames and gaps after an interrupted upstream matting pass.
    frames = list(frame_range)
    objects = sorted({p.stem for _, p in jobs})
    if len(jobs) != len(frames)*len(objects):
        raise ValueError('The selected range has missing object mattes. Complete VideoMaMa or choose a complete range.')
    for frame in frames:
        if not (root/'frames'/f'{frame:05d}.{extension}').is_file():
            raise ValueError(f'Missing original RGB frame {frame}.')
    cache.invalidate()
    progress(0, len(jobs), 'Checking ViTMatte checkpoint...')
    checkpoint, model_identity = checkpoint_resolver(settings, lambda t: progress(0, len(jobs), t), cancel)
    config_key = key([VERSION, asdict(settings), model_identity])
    model = model_factory(settings, checkpoint)
    validators, previous_keys, entries, inputs = {}, {}, {}, {}
    report = {'frames': [], 'cached_frames': 0, 'legacy_alpha_frames': 0,
              'model': model_identity, 'settings': asdict(settings), 'device': str(model.device)}
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    begin = time.perf_counter()
    try:
        for index, (frame, png) in enumerate(jobs):
            check_cancel(cancel)
            start = time.perf_counter()
            oid = png.stem
            source = root/'frames'/f'{frame:05d}.{extension}'
            # Cached app frames are display-referred PNG/JPEG. Reject direct EXR.
            if source.suffix.lower() == '.exr':
                raise ValueError('Convert linear EXR RGB to a display-referred input sequence before refinement.')
            rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError(f'Cannot decode RGB frame: {source}')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            base, base_path = load_base(root, frame, oid, source, png)
            if base.shape != rgb.shape[:2]:
                raise ValueError(f'Frame {frame}: source and matte dimensions differ.')
            if base_path == png:
                report['legacy_alpha_frames'] += 1
            for path in {source, png, base_path}:
                inputs[path.relative_to(root).as_posix()] = {'stamp': stamp(path), 'sha256': digest(path)}
            signature = key([config_key, frame, oid, digest(source), digest(png), digest(base_path),
                             previous_keys.get(oid) if settings.temporal else None])
            previous_keys[oid] = signature
            result = cache.read(frame, oid, signature)
            validator = validators.setdefault(oid, TemporalValidator(settings.temporal_strength))
            if result is None:
                h, w = base.shape
                scale = min(1, settings.short_edge/min(h, w)) if settings.short_edge else 1
                size = (max(1, round(w*scale)), max(1, round(h*scale)))
                small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
                small_base = cv2.resize(base, size, interpolation=cv2.INTER_AREA) if scale < 1 else base
                tri_small = make_trimap(small_base, settings)
                prediction = predict_tiled(model, small_rgb, tri_small, settings, cancel,
                                            lambda t: progress(index, len(jobs), f'Frame {frame}, object {oid}: {t}'))
                # Inference trimap is authoritative: nearest restoration keeps the
                # merge support consistent with what the model was told was known.
                tri = cv2.resize(tri_small, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1 else tri_small
                pred = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1 else prediction
                refined = merge_alpha(base, pred, tri, settings)
                validated = validator.apply(rgb, base, refined, frame) if settings.temporal else refined.copy()
                result = dict(refined=refined, validated=validated, trimap=tri)
                check_cancel(cancel)
                cache.write(frame, oid, signature, **result)
            else:
                report['cached_frames'] += 1
                if settings.temporal:
                    validator.apply(rgb, base, result['refined'], frame)  # restore history for exact resume
            if settings.debug:
                debug = cache.path/'trimaps'/f'{frame:05d}'/f'{oid}.png'
                debug.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(debug), result['trimap']):
                    raise OSError(f'Could not write debug trimap: {debug}')
            entries[f'{frame}/{oid}'] = signature
            report['frames'].append({'frame': frame, 'object': oid, 'seconds': time.perf_counter()-start,
                                     'width': rgb.shape[1], 'height': rgb.shape[0]})
            progress(index+1, len(jobs), f'Refined frame {frame}, object {oid}')
        check_cancel(cancel)
        files = [Path(checkpoint)/name for name in ('config.json', 'model.safetensors')]
        model_files = {str(p.resolve()): stamp(p) for p in files if p.is_file()}
        cache.publish(settings, entries, inputs, model_identity, model_files)
        report['seconds'] = time.perf_counter()-begin
        if model.device.type == 'cuda':
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(model.device)/1048576
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved(model.device)/1048576
        return report
    finally:
        model.close()
