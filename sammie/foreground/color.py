"""Explicit straight RGB color handling, shared by preview and EXR delivery."""
from pathlib import Path
import re
import cv2
import numpy as np

# OCIO 2.4.2 built-in ACES 1.3 CG config: ACEScg -> Linear Rec.709 (sRGB).
# Includes AP1/D60 -> Rec.709/D65 adaptation; no display view or tone mapping.
AP1_TO_709 = np.array([[1.7050509453, -.6217921376, -.0832588747],
                       [-.1302564144, 1.1408047676, -.0105483187],
                       [-.0240033567, -.1289689690, 1.1529723406]], np.float32)
REC709_TO_AP1 = np.linalg.inv(AP1_TO_709).astype(np.float32)


def srgb_to_linear(rgb):
    x = np.maximum(np.asarray(rgb, np.float32), 0)
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4).astype(np.float32)


def linear_to_srgb(rgb):
    x = np.maximum(np.asarray(rgb, np.float32), 0)
    return np.where(x <= .0031308, x*12.92, 1.055*x**(1/2.4)-.055).astype(np.float32)


def to_linear709(rgb, space):
    if space == 'srgb':
        return srgb_to_linear(rgb)
    if space == 'acescg':
        return np.asarray(rgb, np.float32) @ AP1_TO_709.T
    if space == 'linear709':
        return np.asarray(rgb, np.float32).copy()
    raise ValueError('Unknown input color space.')


def from_linear709(rgb, space):
    if space == 'srgb':
        return linear_to_srgb(rgb)
    if space == 'acescg':
        return np.asarray(rgb, np.float32) @ REC709_TO_AP1.T
    if space == 'linear709':
        return np.asarray(rgb, np.float32).copy()
    raise ValueError('Unknown output color space.')


def source_path(root, frame, extension, settings):
    if not settings.plate_path:
        from sammie.exr_media import project_info
        if project_info(root):
            return Path(root)/'source_exr'/f'{frame:05d}.exr'
        if extension not in ('png', 'jpg', 'jpeg'):
            raise ValueError('Use display-referred PNG/JPEG frames or select an original EXR override.')
        return Path(root)/'frames'/f'{frame:05d}.{extension}'
    first = Path(settings.plate_path)
    match = re.search(r'(\d+)(?=\.exr$)', first.name, flags=re.I)
    if match is None:
        raise ValueError('Choose a numbered EXR, e.g. plate.0993.exr; project frame 0 maps to the specified first frame.')
    number = str(settings.plate_first + frame).zfill(len(match[0]))
    return first.with_name(first.name[:match.start()] + number + first.name[match.end():])


def load_rgb(path):
    path = Path(path)
    if path.suffix.lower() == '.exr':
        from sammie.exr_media import project_info, project_rgb
        if path.parent.name == 'source_exr' and project_info(path.parent.parent):
            return project_rgb(path.parent.parent, int(path.stem))
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(str(path))
        try:
            h = f.header()
            dw, display = h['dataWindow'], h['displayWindow']
            if (dw.min.x, dw.min.y) != (0, 0) or str(dw) != str(display):
                raise ValueError('EXR must have an uncropped zero-origin data/display window for exact matte alignment.')
            shape = (dw.max.y+1, dw.max.x+1)
            if not all(c in h['channels'] for c in 'RGB'):
                raise ValueError('EXR plate needs R, G, B channels.')
            rgb = np.stack([np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(shape) for c in 'RGB'], -1)
        finally:
            f.close()
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f'Expected an 8-bit display RGB frame: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    if not np.isfinite(rgb).all():
        raise ValueError(f'Non-finite RGB in {path}')
    return rgb


def merge_foreground(source, alpha, prediction, settings):
    """Prediction is the official model's native-size straight sRGB foreground.

    Preserve opaque source RGB exactly. Alpha is read-only and never regenerated.
    Optional despill conserves linear Rec.709 luminance, then shares the edge mask.
    """
    if source.shape[:2] != alpha.shape or source.shape[2] != 3:
        raise ValueError('Source RGB and alpha dimensions differ.')
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError('Alpha must be finite float values in [0,1].')
    if not np.isfinite(prediction).all():
        raise ValueError('Invalid CorridorKey foreground prediction.')
    if settings.strength == 0:
        return source.copy()
    h, w = alpha.shape
    # Matches the official CPU postprocess resampling, including its order before decoding.
    predicted = cv2.resize(np.asarray(prediction, np.float32), (w,h), interpolation=cv2.INTER_LANCZOS4)
    linear = srgb_to_linear(predicted) / (2.**settings.exposure_ev)
    if settings.despill:
        channel = 2 if settings.screen == 'blue' else 1
        other = [i for i in range(3) if i != channel]
        excess = np.maximum(linear[..., channel]-np.maximum(linear[...,other[0]], linear[...,other[1]]), 0)*settings.despill
        luma = np.array([.2126,.7152,.0722], np.float32)
        linear[...,channel] -= excess
        compensation = excess*luma[channel]/sum(luma[i] for i in other)
        for i in other:
            linear[...,i] += compensation
    fg = from_linear709(linear, settings.color_space)
    weight = (settings.strength*(1-alpha**settings.edge_power))[...,None]
    out = fg*weight + source*(1-weight)
    out[alpha == 1] = source[alpha == 1]
    if not np.isfinite(out).all():
        raise ValueError('Foreground merge produced non-finite RGB.')
    return out.astype(np.float32)
