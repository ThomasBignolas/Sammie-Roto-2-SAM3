"""Bounded, resumable float caches. Original PNG mattes are never replaced here."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from zipfile import BadZipFile
from pathlib import Path
import cv2
import numpy as np
from .algorithms import alpha_float
from .model import digest

VERSION = 1


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def atomic_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


@lru_cache(maxsize=2048)
def _digest_at_stamp(path, size, mtime_ns):
    # Project archives restore different mtimes. Verify content once per file
    # revision rather than discarding valid refinement on every project reopen.
    return digest(path)


def save_base(root, frame, object_id, alpha, source, png):
    """Cache unquantized VideoMaMa alpha, bound to both source and legacy PNG."""
    dest = Path(root)/'matting_float'/f'{frame:05d}'/f'{object_id}.npz'
    atomic_arrays(dest, alpha=alpha_float(alpha))
    atomic_json(dest.with_suffix('.json'), {'png': digest(png), 'source': digest(source)})


def load_base(root, frame, object_id, source, png):
    dest = Path(root)/'matting_float'/f'{frame:05d}'/f'{object_id}.npz'
    meta = dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        data = json.loads(meta.read_text())
        if data == {'png': digest(png), 'source': digest(source)}:
            with np.load(dest, allow_pickle=False) as a:
                return alpha_float(a['alpha']), dest
    a = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f'Cannot read original VideoMaMa alpha: {png}')
    return alpha_float(a), Path(png)


class RefinementCache:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root/'vitmatte'
        self.active = self.path/'active.json'

    def invalidate(self):
        self.active.unlink(missing_ok=True)

    def paths(self, frame, oid):
        p = self.path/f'{frame:05d}'/f'{oid}.npz'
        return p, p.with_suffix('.json')

    def read(self, frame, oid, signature):
        p, meta = self.paths(frame, oid)
        try:
            if json.loads(meta.read_text())['signature'] != signature:
                return None
            with np.load(p, allow_pickle=False) as a:
                result = {k: a[k] for k in ('refined', 'validated', 'trimap')}
                if any(result[k].shape != result['trimap'].shape or not np.isfinite(result[k]).all()
                       for k in ('refined', 'validated')):
                    return None
                return result
        except (OSError, ValueError, KeyError, BadZipFile, EOFError):
            return None

    def write(self, frame, oid, signature, **arrays):
        p, meta = self.paths(frame, oid)
        atomic_arrays(p, **arrays)
        atomic_json(meta, {'signature': signature})

    def publish(self, settings, entries, inputs, model, model_files=None):
        atomic_json(self.active, {'version': VERSION, 'settings': asdict(settings),
                                 'entries': entries, 'inputs': inputs, 'model': model,
                                 'model_files': model_files or {}})

    def selected(self, frame, oid, settings, stage):
        """Read only complete runs whose settings and source files still match.

        Stat validation over the input manifest also invalidates following temporal
        frames if an earlier source matte changes. Content hashes govern resume.
        """
        try:
            active = json.loads(self.active.read_text())
            if active['version'] != VERSION or active['settings'] != asdict(settings):
                return None
            for p, data in active['inputs'].items():
                current = stamp(self.root/p)
                if isinstance(data, list):  # pre-release manifests
                    if current != data: return None
                elif current != data['stamp']:
                    if _digest_at_stamp(str(self.root/p), *current) != data['sha256']:
                        return None
            if any(stamp(p) != s for p, s in active.get('model_files', {}).items()):
                return None
            entry = active['entries'].get(f'{frame}/{oid}')
            if entry is None:
                return None
            a = self.read(frame, oid, entry)
            return a[stage] if a is not None else None
        except (OSError, ValueError, KeyError):
            return None
