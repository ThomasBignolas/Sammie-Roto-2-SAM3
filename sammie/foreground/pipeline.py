"""Prepare exact application mattes on CPU, then infer in an isolated runtime."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import uuid
import cv2
import numpy as np
from sammie.aov.cache import atomic_json, fingerprint
from .config import ForegroundSettings
from .color import load_rgb, source_path, to_linear709
from .cache import SCHEMA, alpha_hash, frame_key, save_npz, load_array, manifest
from .versions import SOURCE, WEIGHTS


def prepare(job, progress=lambda done,total,message: None):
    from sammie.aov.matte_layers import MatteLayers
    settings = ForegroundSettings.from_dict(job.get('foreground_settings', {})).validate()
    root = Path(job['root'])
    from sammie.exr_media import project_info
    info = project_info(root)
    imported = str(job.get('session',{}).get('video_file_path','')).lower()
    if not settings.plate_path:
        expected_space = info['options']['color_space'] if info else 'srgb'
        if settings.color_space != expected_space:
            raise ValueError(f'The imported plate uses {expected_space}; select that CorridorKey plate color space.')
        if not info and imported.endswith(('.exr','.hdr')):
            raise ValueError('Reimport this EXR through File > Open, or choose an original EXR plate override and its actual color space.')
    first, last = int(job['first']), int(job['last'])
    if first < 0 or last < first:
        raise ValueError('Invalid CorridorKey frame range.')
    shape = load_rgb(source_path(root, first, job.get('extension','png'), settings)).shape[:2]
    mattes = MatteLayers(job, shape)
    require_refinement = (job.get('matte_source','matting') == 'matting' and
        job.get('session',{}).get('vitmatte_enabled',False) and
        job.get('session',{}).get('vitmatte_view','validated') != 'original')
    if require_refinement:
        from sammie.refinement.cache import RefinementCache
        from sammie.refinement.algorithms import RefinementSettings
        from sammie.settings_manager import get_settings_manager
        with mattes.context():
            refinement = RefinementSettings.from_session(get_settings_manager())
        for frame in range(first,last+1):
            for oid in mattes.ids:
                if mattes.path(frame,oid).is_file() and RefinementCache(root).selected(frame,oid,refinement,job['session'].get('vitmatte_view','validated')) is None:
                    raise ValueError('Selected ViTMatte result is missing or stale. Run ViTMatte again or explicitly choose its Original VideoMaMa view before CorridorKey.')
    records = {}
    for frame in range(first, last+1):
        progress(frame-first,last-first+1,f'Preparing selected alpha, frame {frame}...')
        path = source_path(root, frame, job.get('extension','png'), settings)
        source = fingerprint(path)
        rgb = load_rgb(path)
        if rgb.shape[:2] != shape:
            raise ValueError(f'Plate dimensions changed at frame {frame}.')
        alpha = mattes.channels(frame)['A']
        sha = alpha_hash(alpha)
        key = frame_key(source['sha256'], sha, settings)
        alpha_path = root/'foreground'/'inputs'/f'{sha}.npz'
        cached = load_array(alpha_path, 'alpha', shape)
        if cached is None or alpha_hash(cached) != sha:
            save_npz(alpha_path, 'alpha', alpha.astype(np.float32))
        records[str(frame)] = dict(key=key, alpha_sha256=sha, source_sha256=source['sha256'])
    data = dict(schema=SCHEMA, records=records, first=first, last=last, shape=shape,
                inference=settings.inference(), settings=asdict(settings), code=SOURCE, weights=WEIGHTS[settings.screen],
                root=str(root), extension=job.get('extension','png'), fps=job.get('fps','24/1'),
                alpha=mattes.metadata())
    path = root/'foreground'/'jobs'/f'{uuid.uuid4().hex}.prepared.json'
    atomic_json(path,data)
    return dict(prepared=str(path))


def run_job(job, progress=lambda done,total,message: None, factory=None):
    data = json.loads(Path(job['prepared']).read_text(encoding='utf-8'))
    root = Path(data['root'])
    settings = ForegroundSettings.from_dict(data['settings']).validate()
    began = time.perf_counter()
    model = None
    reused = 0
    timings = []
    gpu = None
    torch = None
    try:
        for frame in range(data['first'],data['last']+1):
            record = data['records'][str(frame)]
            path = source_path(root, frame, data['extension'], settings)
            if fingerprint(path)['sha256'] != record['source_sha256']:
                raise ValueError('Source changed during CorridorKey processing; restart generation.')
            output = root/'foreground'/'predictions'/f"{record['key']}.npz"
            expected = (settings.resolution, settings.resolution, 3)
            if load_array(output, 'fg', expected) is not None:
                reused += 1
            else:
                alpha = load_array(root/'foreground'/'inputs'/f"{record['alpha_sha256']}.npz", 'alpha', data['shape'])
                if alpha is None or alpha_hash(alpha) != record['alpha_sha256']:
                    raise ValueError('Prepared alpha is missing or corrupt; restart Generate.')
                source = load_rgb(path)
                if source.shape[:2] != tuple(data['shape']):
                    raise ValueError('Source resolution changed during processing.')
                if model is None:
                    if factory is None:
                        import torch
                        from .model import CorridorModel
                        torch.set_num_threads(min(6, torch.get_num_threads()))
                        if settings.device == 'cuda' and torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        factory = CorridorModel
                    model = factory(settings, lambda msg: progress(0,0,msg))
                progress(frame-data['first'],len(data['records']),f'CorridorKey {settings.screen}: frame {frame}...')
                started = time.perf_counter()
                try:
                    raw = model.predict(to_linear709(source, settings.color_space)*(2.**settings.exposure_ev), alpha)
                except RuntimeError as error:
                    if 'out of memory' in str(error).lower():
                        raise RuntimeError('CorridorKey ran out of GPU memory. Try 1024 resolution or close other GPU applications; completed frames are cached.') from error
                    raise
                if raw.shape != expected or raw.dtype not in (np.float16,np.float32) or not np.isfinite(raw).all():
                    raise ValueError('CorridorKey returned invalid foreground RGB.')
                timings.append(time.perf_counter()-started)
                save_npz(output, 'fg', raw)
                del raw, alpha, source
            progress(frame-data['first']+1,len(data['records']),f'CorridorKey frame {frame} complete.')
        for frame, record in data['records'].items():
            if fingerprint(source_path(root,int(frame),data['extension'],settings))['sha256'] != record['source_sha256']:
                raise ValueError('Source changed during inference; results were not activated.')
        if torch is not None and settings.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu = dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        data.update(seconds=time.perf_counter()-began, gpu=gpu, reused=reused,
                    cached=reused == len(data['records']), inference_seconds=timings)
        previous = manifest(root, settings)
        if previous and tuple(previous['shape']) == tuple(data['shape']):
            data['records'] = dict(previous['records'], **data['records'])
            data['first'] = min(int(f) for f in data['records'])
            data['last'] = max(int(f) for f in data['records'])
        atomic_json(root/'foreground'/'active.json',data)
        return dict(passes={'foreground':data})
    finally:
        del model
        gc.collect()
        if torch is not None and settings.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
