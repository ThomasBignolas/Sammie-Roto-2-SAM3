"""Resumable per-frame native RGB predictions; alpha is a separate lossless input."""
import hashlib
import json
import os
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
from sammie.aov.cache import atomic_json, fingerprint
from .config import ForegroundSettings
from .color import load_rgb, source_path, merge_foreground
from .versions import SOURCE, WEIGHTS

SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def alpha_hash(alpha):
    return hashlib.sha256(np.ascontiguousarray(alpha, np.float32).tobytes()).hexdigest()


def frame_key(source, alpha, settings):
    return digest(dict(schema=SCHEMA, source=source, alpha=alpha, settings=settings.inference(),
                       code=SOURCE, weights=WEIGHTS[settings.screen]))


def save_npz(path, name, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **{name: array})
    os.replace(tmp, path)


def load_array(path, name, shape=None):
    try:
        with np.load(path, allow_pickle=False) as data:
            a = data[name]
        if a.dtype not in (np.float16, np.float32) or not np.isfinite(a).all() or (shape is not None and a.shape != tuple(shape)):
            return None
        return a
    except (OSError, ValueError, KeyError, BadZipFile, EOFError):
        return None


def manifest(root, settings):
    try:
        m = json.loads((Path(root)/'foreground'/'active.json').read_text(encoding='utf-8'))
        if m['schema'] != SCHEMA or m['inference'] != settings.inference() or m['code'] != list(SOURCE) or m['weights'] != list(WEIGHTS[settings.screen]):
            return None
        return m
    except (OSError, ValueError, KeyError):
        return None


def read_result(job, frame, settings=None, mattes=None, alpha=None):
    """Validate current source AND current selected alpha before preview/export."""
    from sammie.aov.matte_layers import MatteLayers
    settings = settings or ForegroundSettings.from_dict(job.get('foreground_settings', {}))
    m = manifest(job['root'], settings)
    if not m or str(frame) not in m['records']:
        raise ValueError('CorridorKey result is missing or inference settings changed. Generate this range first.')
    record = m['records'][str(frame)]
    path = source_path(job['root'], frame, job.get('extension','png'), settings)
    if fingerprint(path)['sha256'] != record['source_sha256']:
        raise ValueError('CorridorKey source plate changed. Generate this range again.')
    if alpha is None:
        if mattes is None:
            mattes = MatteLayers(dict(job, first=frame, last=frame, allow_empty_objects=True), m['shape'])
        alpha = mattes.channels(frame)['A']
    if alpha_hash(alpha) != record['alpha_sha256']:
        raise ValueError('The selected alpha changed (including ViTMatte). Regenerate CorridorKey for this alpha.')
    key = record['key']
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid CorridorKey cache key.')
    raw = load_array(Path(job['root'])/'foreground'/'predictions'/f'{key}.npz', 'fg', (settings.resolution, settings.resolution, 3))
    if raw is None:
        raise ValueError('CorridorKey cache is missing or corrupt. Generate to repair it.')
    source = load_rgb(path)
    if source.shape[:2] != tuple(m['shape']):
        raise ValueError('Plate resolution changed. Regenerate CorridorKey.')
    return merge_foreground(source, alpha, raw, settings), alpha, source, m
