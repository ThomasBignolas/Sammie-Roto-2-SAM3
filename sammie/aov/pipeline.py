"""AOV jobs preserve frame indices; caches store float data at inference resolution."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .config import AOVSettings, resize_rgb
from .cache import SCHEMA, atomic_json, save_array, fingerprint, signature, complete
from .versions import SOURCES, WEIGHTS


def run_job(job, progress=lambda done, total, message: None, factory=None):
    settings = AOVSettings.from_dict(job['settings']).validate()
    root = Path(job['root'])
    indices = list(range(int(job['first']), int(job['last']) + 1))
    if not indices or indices[0] < 0:
        raise ValueError('Invalid frame range.')
    extension = job.get('extension', 'png').lstrip('.')
    if extension not in ('png', 'jpg', 'jpeg'):
        raise ValueError('AOV models require display-referred PNG/JPEG frames. Convert linear EXR/HDR to a display proxy first.')
    files = [root/'frames'/f'{i:05d}.{extension}' for i in indices]
    sources, frames, shape = [], [], None
    for i, file in enumerate(files):
        progress(i, len(files), f'Reading source frame {indices[i]}...')
        source = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f'Cannot read {file}')
        if shape and shape != source.shape[:2]:
            raise ValueError('Source frame dimensions changed within the range.')
        shape = source.shape[:2]
        frames.append(resize_rgb(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), settings.long_edge))
        sources.append(fingerprint(file))
    if factory is None:
        import torch
        from .models import DVDModel, NormalModel, VDAModel
        if settings.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use CPU with VDA Small, or install the NVIDIA driver and AOV runtime.')
        def factory(kind, settings, status):
            cls = NormalModel if kind == 'normals' else (DVDModel if settings.depth_model == 'dvd' else VDAModel)
            return cls(settings, status)
    report = {'passes': {}, 'source_shape': shape, 'frame_count': len(indices)}
    for kind, enabled in (('depth', settings.depth), ('normals', settings.normals)):
        if not enabled:
            continue
        key = signature(sources, settings, kind)
        folder = root/'aov'/key
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder/'manifest.json'
        expected_frame = frames[0].shape[:2] + ((3,) if kind == 'normals' else ())
        if complete(folder, indices, expected_frame):
            atomic_json(root/'aov'/f'{kind}.json', {'key': key})
            report['passes'][kind] = {'cached': True, 'key': key}
            progress(1, 1, f'Reusing cached {kind}.')
            continue
        model = None
        began = time.perf_counter()
        gpu = None
        try:
            import torch
            if settings.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model = factory(kind, settings, lambda text: progress(0, 0, text))
            result = np.asarray(model.predict(frames, progress), dtype=np.float32)
            expected = (len(indices), *frames[0].shape[:2]) + ((3,) if kind == 'normals' else ())
            if result.shape != expected or not np.isfinite(result).all():
                raise RuntimeError(f'{kind} returned invalid data: {result.shape}; expected {expected}.')
            for i, frame in enumerate(indices):
                progress(i, len(indices), f'Caching {kind}, frame {frame}...')
                save_array(folder/f'{frame:05d}.npy', result[i])
            if settings.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu = {'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                       'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
            manifest = dict(schema=SCHEMA, key=key, kind=kind, frames=indices, sources=sources,
                settings=settings.pass_settings(kind), source_shape=shape, inference_shape=result.shape[1:3],
                range=[float(result.min()), float(result.max())], fps=job.get('fps', '24/1'),
                sequence_start=job.get('sequence_start', 0), seconds=time.perf_counter()-began, gpu=gpu,
                sources_version=SOURCES, weights_version=WEIGHTS,
                # Both published checkpoints predict disparity-like values.
                # DVD trains with trunc_disparity; larger raw values are nearer.
                depth_direction='near_is_larger',
                depth_units='relative; not metres', normal_axes=settings.normal_axes)
            # Do not mark a pass complete if frames were replaced during inference.
            if sources != [fingerprint(f) for f in files]:
                raise RuntimeError('Source frames changed during inference; results were not activated.')
            atomic_json(manifest_path, manifest)
            atomic_json(root/'aov'/f'{kind}.json', {'key': key})
            report['passes'][kind] = manifest
            del result
        finally:
            del model
            gc.collect()
            # Called once after destroying live model tensors, never per frame.
            if 'torch' in locals() and settings.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if kind in report['passes'] and report['passes'][kind].get('gpu'):
                    report['passes'][kind]['gpu']['remaining_allocated_mib'] = torch.cuda.memory_allocated()/2**20
                    atomic_json(manifest_path, report['passes'][kind])
    progress(1, 1, 'AOV generation complete.')
    return report
