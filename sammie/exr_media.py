"""EXR plates retained verbatim beside explicit, display-referred model proxies."""
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import re
import shutil
import uuid
import cv2
import numpy as np

SCHEMA = 1
MANIFEST = 'source_exr/import.json'


@dataclass(frozen=True)
class ImportOptions:
    layer: str = ''
    color_space: str = 'linear709'
    exposure: float = 0.
    fps: str = '24'

    def validate(self):
        if self.color_space not in ('linear709', 'acescg', 'srgb'):
            raise ValueError('Choose linear Rec.709, ACEScg or display sRGB input.')
        if not np.isfinite(self.exposure) or not -12 <= self.exposure <= 12:
            raise ValueError('Proxy exposure must be between -12 and +12 stops.')
        try:
            rate = Fraction(self.fps)
        except (ValueError, ZeroDivisionError):
            raise ValueError('Frame rate must be a positive number or ratio, e.g. 24000/1001.') from None
        if not 0 < rate <= 1000:
            raise ValueError('Frame rate must be greater than 0 and at most 1000.')
        return self


def _box(value):
    return [[int(v) for v in point] for point in value]


def inspect(path):
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True, header_only=True) as source:
        if len(source.parts) != 1:
            raise ValueError('Multi-part EXR is not supported. Export the RGB plate as a single-part EXR first.')
        h = source.header()
        if h.get('type') in (OpenEXR.deepscanline, OpenEXR.deeptile):
            raise ValueError('Deep EXR needs flattening to an RGB image before import.')
        channels = {c.name: c for c in h['channels']}
        layers = sorted({name[:-1] for name in channels if name.endswith('R')
                         and name[:-1]+'G' in channels and name[:-1]+'B' in channels})
        if not layers:
            raise ValueError('EXR has no RGB layer. Choose a beauty/plate EXR containing R, G and B.')
        for layer in layers:
            if any(channels[layer+c].xSampling != 1 or channels[layer+c].ySampling != 1 for c in 'RGB'):
                raise ValueError('Subsampled RGB channels are unsupported; export full-resolution RGB channels.')
        display, data = _box(h['displayWindow']), _box(h['dataWindow'])
        width, height = [display[1][i]-display[0][i]+1 for i in (0, 1)]
        if min(width, height) < 1 or width*height > 200_000_000:
            raise ValueError('EXR display window is empty or exceeds the 200 megapixel import limit.')
        fps = h.get('framesPerSecond', 24)
        try:
            fps = str(Fraction(str(fps)).limit_denominator(1_000_000))
        except (ValueError, ZeroDivisionError):
            fps = '24'
        chroma = h.get('chromaticities')
        space = 'linear709'
        if chroma is not None and np.allclose(np.asarray(chroma).reshape(-1),
                [.713,.293,.165,.830,.128,.044,.32168,.33767], atol=.002):
            space = 'acescg'
        return dict(layers=layers, display_window=display, data_window=data, shape=[height, width],
                    pixel_aspect=float(h.get('pixelAspectRatio', 1.)), fps=fps, suggested_space=space)


def read_rgb(path, layer='', expected=None):
    """Place cropped data in its display window, preserving float HDR RGB values."""
    import OpenEXR
    info = inspect(path)
    if layer not in info['layers']:
        raise ValueError(f'Missing RGB layer {layer or "R/G/B"} in {Path(path).name}.')
    if expected and (info['display_window'] != expected['display_window'] or
                     info['pixel_aspect'] != expected['pixel_aspect']):
        raise ValueError(f'Frame dimensions, origin or pixel aspect changed in {Path(path).name}.')
    with OpenEXR.File(str(path), separate_channels=True) as source:
        channels = source.channels()
        rgb = np.stack([channels[layer+c].pixels.astype(np.float32) for c in 'RGB'], -1)
    if not np.isfinite(rgb).all():
        raise ValueError(f'Non-finite RGB in {Path(path).name}. Repair NaN/Inf pixels before import.')
    display, data = info['display_window'], info['data_window']
    canvas = np.zeros((*info['shape'], 3), np.float32)
    left, top = [max(display[0][i], data[0][i]) for i in (0, 1)]
    right, bottom = [min(display[1][i], data[1][i])+1 for i in (0, 1)]
    if right > left and bottom > top:
        canvas[top-display[0][1]:bottom-display[0][1], left-display[0][0]:right-display[0][0]] = \
            rgb[top-data[0][1]:bottom-data[0][1], left-data[0][0]:right-data[0][0]]
    return canvas, info


def display_rgb(rgb, options):
    from .foreground.color import to_linear709, linear_to_srgb
    linear = to_linear709(rgb, options.color_space) * (2.**options.exposure)
    return np.rint(np.clip(linear_to_srgb(linear), 0, 1)*255).astype(np.uint8)


def sequence(path):
    path = Path(path).resolve()
    match = re.fullmatch(r'(.*?)(\d+)', path.stem)
    if not match:
        return [path], [0]
    prefix, digits = match.groups()
    matches = []
    for candidate in path.parent.iterdir():
        found = re.fullmatch(re.escape(prefix)+r'(\d+)', candidate.stem)
        if candidate.is_file() and candidate.suffix.lower() == '.exr' and found and len(found[1]) == len(digits):
            matches.append((int(found[1]), candidate))
    matches.sort(key=lambda item: item[0])
    return [p for _, p in matches], [n for n, _ in matches]


def prepare(files, options, parent, progress=lambda n,total,text: None, cancel=lambda: False):
    """Build a complete new session before touching the current session or settings."""
    options.validate()
    files = [Path(p).resolve() for p in files]
    if not files:
        raise ValueError('Choose at least one EXR.')
    numbers = [int(m[0]) if (m := re.search(r'\d+$', p.stem)) else 0 for p in files]
    if len(files) > 1 and numbers != list(range(numbers[0], numbers[0]+len(files))):
        raise ValueError('The EXR sequence has missing or duplicate frame numbers. Fill the gaps or load one image; frames are never silently skipped.')
    parent = Path(parent).resolve()
    stage = parent / ('.exr-import-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        for folder in ('frames', 'source_exr', 'masks', 'matting'):
            (stage/folder).mkdir()
        first = inspect(files[0])
        for index, path in enumerate(files):
            if cancel():
                raise InterruptedError('EXR import cancelled. The previous project is unchanged.')
            progress(index, len(files), f'Importing EXR {index+1}/{len(files)}: {path.name}')
            # Decode the copied file so the proxy and retained plate always match.
            plate = stage/'source_exr'/f'{index:05d}.exr'
            shutil.copy2(path, plate)
            rgb, _ = read_rgb(plate, options.layer, first)
            proxy = display_rgb(rgb, options)
            if not cv2.imwrite(str(stage/'frames'/f'{index:05d}.png'), cv2.cvtColor(proxy, cv2.COLOR_RGB2BGR)):
                raise OSError(f'Could not write display proxy for {path.name}.')
        if cancel():
            raise InterruptedError('EXR import cancelled. The previous project is unchanged.')
        metadata = dict(schema=SCHEMA, options=asdict(options), **first,
                        source_files=[p.name for p in files], source_numbers=numbers, proxy='png-srgb-u8',
                        canvas='displayWindow; outside data filled with zero; outside display overscan excluded')
        metadata['fps'] = str(Fraction(options.fps))
        (stage/MANIFEST).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        progress(len(files), len(files), 'EXR import ready')
        return stage, metadata
    except BaseException:
        discard(stage)
        raise


def discard(stage):
    path = Path(stage).resolve()
    if not path.name.startswith('.exr-import-'):
        raise ValueError('Refusing to remove a directory not created by the EXR importer.')
    if path.exists():
        shutil.rmtree(path)


def commit(stage, target):
    stage, target = Path(stage).resolve(), Path(target).resolve()
    from .projects import MARKER
    if (target/MARKER).exists():
        raise ValueError('Cannot import over a saved working project. Open a new session first.')
    if stage.parent != target.parent or not stage.name.startswith('.exr-import-') or target == target.parent:
        raise ValueError('Invalid EXR import staging location.')
    backup = target.parent / ('.exr-import-backup-' + uuid.uuid4().hex)
    if target.exists():
        target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup.exists():
            backup.rename(target)
        raise
    if backup.exists():
        discard(backup)


@lru_cache(maxsize=8)
def _manifest(path, stamp, size):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or data.get('proxy') != 'png-srgb-u8':
        raise ValueError('Unsupported EXR import manifest. Reimport this plate.')
    return data


def project_info(root):
    path = Path(root).resolve()/MANIFEST
    if not path.is_file():
        return None
    stat = path.stat()
    return _manifest(str(path), stat.st_mtime_ns, stat.st_size)


def project_rgb(root, frame):
    info = project_info(root)
    if info is None or not 0 <= frame < len(info['source_files']):
        raise ValueError('Missing imported EXR plate or invalid frame index.')
    return read_rgb(Path(root)/'source_exr'/f'{frame:05d}.exr', info['options']['layer'], info)[0]


def export_header(header, info, fps=None):
    """Export full display-window pixels at the original origin and frame rate."""
    import Imath
    window = info['display_window']
    box = Imath.Box2i(Imath.V2i(*window[0]), Imath.V2i(*window[1]))
    header['displayWindow'] = box
    header['dataWindow'] = box
    header['pixelAspectRatio'] = info['pixel_aspect']
    rate = Fraction(fps or info['fps'])
    header['framesPerSecond'] = Imath.Rational(rate.numerator, rate.denominator)
    header['sammie:exrImport'] = json.dumps(dict(options=info['options'], canvas=info['canvas'])).encode('utf-8')
    if info['options']['color_space'] != 'srgb':
        values = ([.713,.293,.165,.830,.128,.044,.32168,.33767] if info['options']['color_space'] == 'acescg'
                  else [.64,.33,.30,.60,.15,.06,.3127,.3290])
        header['chromaticities'] = Imath.Chromaticities(*[Imath.V2f(*values[i:i+2]) for i in range(0,8,2)])
