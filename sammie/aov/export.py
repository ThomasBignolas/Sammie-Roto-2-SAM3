"""One float32 multilayer EXR per frame: plate, alpha, normals and depth."""
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .cache import active, atomic_json
from .config import AOVSettings, unit_normals


def channel_data(data, manifest, kind):
    h, w = manifest['source_shape']
    resized = cv2.resize(data, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if kind == 'normals':
        n = unit_normals(resized)
        return {f'N.{axis}': np.ascontiguousarray(n[..., i]) for i, axis in enumerate('xyz')}
    lo, hi = manifest['range']
    # One range for the entire shot. Never normalize each frame independently.
    z = (resized-lo) / max(hi-lo, 1e-8)
    if manifest['depth_direction'] == 'near_is_larger':
        z = 1-z
    z = np.clip(z, 0, 1).astype(np.float32)
    return {'Z': z, 'depth.Z': z, 'Z_raw': resized}


def write_exr(path, channels, metadata):
    import OpenEXR
    import Imath
    height, width = next(iter(channels.values())).shape
    if any(a.shape != (height, width) or not np.isfinite(a).all() for a in channels.values()):
        raise ValueError('EXR channel dimensions or values are invalid.')
    header = OpenEXR.Header(width, height)
    header['channels'] = {k: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for k in channels}
    header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    header['sammie:aov'] = json.dumps(metadata).encode('utf-8')
    from fractions import Fraction
    fps = Fraction(str(metadata['fps']))
    header['framesPerSecond'] = Imath.Rational(fps.numerator, fps.denominator)
    temporary = str(path) + '.tmp'
    output = OpenEXR.OutputFile(temporary, header)
    try:
        output.writePixels({k: np.ascontiguousarray(v, dtype=np.float32).tobytes() for k, v in channels.items()})
    finally:
        output.close()
    os.replace(temporary, path)


def export_sequence(job, progress=lambda done, total, message: None):
    settings = AOVSettings.from_dict(job['settings'])
    enabled = [kind for kind, yes in [('depth', settings.depth), ('normals', settings.normals)] if yes]
    if not enabled:
        raise ValueError('Select at least one AOV to export.')
    passes = {}
    indices = list(range(job['first'], job['last']+1))
    if not indices or indices[0] < 0:
        raise ValueError('Invalid export frame range.')
    for kind in enabled:
        found = active(job['root'], kind, settings)
        if not found or not set(indices).issubset(found[1]['frames']):
            raise ValueError(f'{kind} is missing or its settings changed. Generate the requested range first.')
        passes[kind] = found
    shapes = {tuple(m['source_shape']) for _, m in passes.values()}
    if len(shapes) != 1:
        raise ValueError('Cached AOV dimensions differ. Regenerate this range.')
    shape = shapes.pop()
    mattes = None
    if job.get('include_mattes', False):
        from .matte_layers import MatteLayers
        mattes = MatteLayers(job, shape)  # Validate before writing any frames.
    destination = Path(job['destination'])
    destination.mkdir(parents=True, exist_ok=True)
    # Preflight all names before writing so unrelated exports are never overwritten.
    paths = [destination/f"aov.{job['start_number']+i:04d}.exr" for i in range(len(indices))]
    if any(p.exists() for p in paths) or (destination/'aov_metadata.json').exists():
        raise FileExistsError('An AOV sequence with these frame numbers already exists. Choose a new folder or start number.')
    metadata = dict(fps=job.get('fps', '24/1'), source_frames=indices, start_number=job['start_number'],
                    depth='Z and depth.Z: shot-normalized relative depth, 0 near / 1 far; Z_raw: model output',
                    normals='N.x/y/z: signed unit vectors; ' + settings.normal_axes,
                    passes={k: m for k, (_, m) in passes.items()})
    if job.get('include_plate', False):
        metadata['plate'] = 'R/G/B: original imported display-referred RGB, unpremultiplied, no color transform; not scene-linear'
    if mattes:
        metadata['alpha'] = mattes.metadata()
    for i, frame in enumerate(indices):
        channels = {}
        for kind, (folder, manifest) in passes.items():
            value = np.load(folder/f'{frame:05d}.npy', allow_pickle=False)
            channels.update(channel_data(value, manifest, kind))
        if job.get('include_plate', False):
            source = Path(job['root'])/'frames'/f"{frame:05d}.{job.get('extension', 'png')}"
            rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != shape:
                raise ValueError(f'Missing or misaligned original plate at frame {frame}.')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            channels.update({c: rgb[..., j] for j, c in enumerate('RGB')})
        if mattes:
            channels.update(mattes.channels(frame))
        frame_meta = dict(metadata, source_frame=frame, export_frame=job['start_number']+i)
        # Full manifests go in JSON, compact conventions/provenance in each EXR.
        frame_meta['passes'] = {k: {'key': m['key'], 'settings': m['settings']} for k, (_, m) in passes.items()}
        write_exr(paths[i], channels, frame_meta)
        progress(i+1, len(indices), f'Exported {paths[i].name}')
    atomic_json(destination/'aov_metadata.json', metadata)
    return dict(exported=len(paths), destination=str(destination), channels=sorted(channels))
