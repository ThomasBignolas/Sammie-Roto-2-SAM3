"""Viewer-only remapping. Signed normals and raw depth remain float in the cache."""
import cv2
import numpy as np
from .cache import read_frame
from .config import AOVSettings
from .export import channel_data


def preview(root, kind, frame, options):
    settings = AOVSettings.from_dict(options)
    found = read_frame(root, kind, frame, settings)
    if found is None:
        return None
    data, manifest = found
    channels = channel_data(data, manifest, kind)
    if kind == 'normals':
        rgb = np.stack([channels[f'N.{axis}'] for axis in 'xyz'], axis=-1) * .5 + .5
    else:
        rgb = np.repeat(channels['Z'][..., None], 3, axis=-1)
    return np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)
