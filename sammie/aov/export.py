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
    if metadata.get('exr_source'):
        from sammie.exr_media import export_header
        export_header(header, metadata['exr_source'], metadata['fps'])
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
    if not enabled and not settings.foreground:
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
    foreground = None
    if settings.foreground:
        from sammie.foreground.config import ForegroundSettings
        from sammie.foreground.cache import manifest, alpha_hash, read_result
        from sammie.foreground.color import source_path, load_rgb
        from .cache import fingerprint
        foreground = ForegroundSettings.from_dict(job.get('foreground_settings', {})).validate()
        foreground_manifest = manifest(job['root'], foreground)
        if not foreground_manifest or not set(map(str, indices)).issubset(foreground_manifest['records']):
            raise ValueError('CorridorKey foreground is missing or settings changed. Generate the requested range first.')
        shapes.add(tuple(foreground_manifest['shape']))
    if len(shapes) != 1:
        raise ValueError('Cached AOV dimensions differ. Regenerate this range.')
    shape = shapes.pop()
    mattes = None
    if job.get('include_mattes', False) or foreground:
        from .matte_layers import MatteLayers
        mattes = MatteLayers(job, shape)  # Validate before writing any frames.
    if foreground:
        # Reject changed selected alpha/plate before creating a partial delivery.
        for frame in indices:
            record = foreground_manifest['records'][str(frame)]
            if alpha_hash(mattes.channels(frame)['A']) != record['alpha_sha256']:
                raise ValueError('Selected alpha changed (including ViTMatte). Regenerate CorridorKey before export.')
            path = source_path(job['root'], frame, job.get('extension','png'), foreground)
            if fingerprint(path)['sha256'] != record['source_sha256']:
                raise ValueError('CorridorKey source plate changed. Regenerate before export.')
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
    from sammie.exr_media import project_info, project_rgb
    exr_source = project_info(job['root'])
    if exr_source:
        metadata['exr_source'] = exr_source
        if foreground and foreground.plate_path:
            metadata['exr_source'] = dict(exr_source, options=dict(exr_source['options'],color_space=foreground.color_space))
    if job.get('include_plate', False):
        metadata['plate'] = 'R/G/B: original imported display-referred RGB, unpremultiplied, no color transform; not scene-linear'
        if foreground:
            metadata['plate'] = f'R/G/B: original CorridorKey input plate, straight {foreground.color_space}; no exposure change'
        if exr_source and not (foreground and foreground.plate_path):
            metadata['plate'] = f"R/G/B: original imported EXR {exr_source['options']['color_space']} RGB as stored, float HDR; proxy exposure is not applied"
    if mattes and job.get('include_mattes', False):
        metadata['alpha'] = mattes.metadata()
    if foreground:
        from dataclasses import asdict
        metadata['foreground'] = dict(channels='foreground.R/G/B: recovered straight RGB; foreground.A: unchanged selected alpha',
            color_space=foreground.color_space, settings=asdict(foreground),
            model='Public CorridorKey 1.0 / CorridorKeyBlue 1.0, not CK2.0',
            code=foreground_manifest['code'], weights=foreground_manifest['weights'],
            alpha=mattes.metadata(), temporal='Independent frames; existing temporally validated alpha is preserved')
    for i, frame in enumerate(indices):
        channels = {}
        matte_channels = mattes.channels(frame) if mattes else {}
        foreground_source = None
        if foreground:
            recovered, alpha, foreground_source, _ = read_result(job, frame, foreground, mattes, matte_channels['A'])
            channels.update({f'foreground.{c}': recovered[..., j] for j,c in enumerate('RGB')})
            channels['foreground.A'] = alpha
        for kind, (folder, manifest) in passes.items():
            value = np.load(folder/f'{frame:05d}.npy', allow_pickle=False)
            channels.update(channel_data(value, manifest, kind))
        if job.get('include_plate', False):
            if foreground_source is not None:
                rgb = foreground_source
            elif exr_source:
                rgb = project_rgb(job['root'], frame)
            else:
                source = Path(job['root'])/'frames'/f"{frame:05d}.{job.get('extension', 'png')}"
                rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if rgb is None or rgb.shape[:2] != shape:
                    raise ValueError(f'Missing or misaligned original plate at frame {frame}.')
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            channels.update({c: rgb[..., j] for j, c in enumerate('RGB')})
        if mattes and job.get('include_mattes', False):
            channels.update(matte_channels)
        frame_meta = dict(metadata, source_frame=frame, export_frame=job['start_number']+i)
        # Full manifests go in JSON, compact conventions/provenance in each EXR.
        frame_meta['passes'] = {k: {'key': m['key'], 'settings': m['settings']} for k, (_, m) in passes.items()}
        write_exr(paths[i], channels, frame_meta)
        progress(i+1, len(indices), f'Exported {paths[i].name}')
    atomic_json(destination/'aov_metadata.json', metadata)
    return dict(exported=len(paths), destination=str(destination), channels=sorted(channels))
