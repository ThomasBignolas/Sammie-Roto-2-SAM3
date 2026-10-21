"""Float AOV caches; only completed passes become active for preview/export."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from .versions import SOURCES, WEIGHTS

SCHEMA = 2  # Includes explicit near-is-larger raw depth and delivery conventions.

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def save_array(path, array):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.save(f, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)

def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    digest = _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    return dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)

@lru_cache(maxsize=4096)
def _file_hash(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def signature(sources, settings, kind):
    data = dict(schema=SCHEMA, sources=sources, settings=settings.pass_settings(kind),
                code=SOURCES, weights=WEIGHTS, kind=kind)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def complete(folder, indices, expected_shape):
    """Reject truncated/nonfinite files instead of repeatedly reusing a broken cache."""
    try:
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA or manifest['frames'] != indices:
            return False
        for frame in indices:
            value = np.load(folder/f'{frame:05d}.npy', allow_pickle=False, mmap_mode='r')
            valid = value.shape == expected_shape and value.dtype == np.float32 and np.isfinite(value).all()
            del value
            if not valid:
                return False
        return True
    except (OSError, ValueError, KeyError, EOFError):
        return False

def active(root, kind, settings=None):
    root = Path(root)
    try:
        pointer = json.loads((root/'aov'/f'{kind}.json').read_text(encoding='utf-8'))
        key = pointer['key']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            return None
        folder = root/'aov'/key
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA:
            return None
        if settings and manifest['settings'] != settings.pass_settings(kind):
            return None
        for source in manifest['sources']:
            path = root/'frames'/source['name']
            stat = path.stat()
            if stat.st_size != source['size'] or stat.st_mtime_ns != source['mtime_ns']:
                # Project archives may change mtimes while preserving frame contents.
                if fingerprint(path)['sha256'] != source['sha256']:
                    return None
        return folder, manifest
    except (OSError, KeyError, ValueError):
        return None

def read_frame(root, kind, frame, settings=None):
    found = active(root, kind, settings)
    if not found or frame not in found[1]['frames']:
        return None
    folder, manifest = found
    try:
        return np.load(folder/f'{frame:05d}.npy', allow_pickle=False), manifest
    except (OSError, ValueError):
        return None
