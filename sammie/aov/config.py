"""Settings and geometry shared by UI, isolated worker and tests."""
from dataclasses import asdict, dataclass, fields
import numpy as np
import cv2


@dataclass(frozen=True)
class AOVSettings:
    depth: bool = True
    normals: bool = True
    foreground: bool = False
    depth_model: str = 'dvd'
    long_edge: int = 512
    dvd_window: int = 17
    dvd_overlap: int = 5
    normal_window: int = 14
    normal_overlap: int = 4
    decode_chunk: int = 2
    offload: str = 'model'
    seed: int = 42
    device: str = 'cuda'
    normal_axes: str = 'native'

    def validate(self):
        if not self.depth and not self.normals and not self.foreground:
            raise ValueError('Select Depth, Normals, or CorridorKey foreground.')
        if self.depth_model not in ('dvd', 'vda-small'):
            raise ValueError('Unknown depth model.')
        if self.long_edge not in (384, 512, 640, 768, 1024):
            raise ValueError('Unsupported processing resolution.')
        if not (5 <= self.dvd_window <= 81 and self.dvd_window % 4 == 1):
            raise ValueError('DVD window must be 4n+1 frames between 5 and 81.')
        if not 1 <= self.dvd_overlap < self.dvd_window:
            raise ValueError('DVD overlap must be smaller than its window.')
        if not 2 <= self.normal_window <= 32 or not 1 <= self.normal_overlap < self.normal_window:
            raise ValueError('NormalCrafter overlap must be smaller than its window (2–32 frames).')
        if not 1 <= self.decode_chunk <= 7 or self.offload not in ('model', 'sequential'):
            raise ValueError('Invalid memory settings.')
        if self.device not in ('cuda', 'cpu') or self.normal_axes not in ('native', 'flip_yz'):
            raise ValueError('Invalid device or normal axes.')
        if self.device == 'cpu' and self.depth and self.depth_model == 'dvd':
            raise ValueError('DVD requires CUDA/BF16. Select Video Depth Anything Small for CPU inference.')
        return self

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

    def pass_settings(self, kind):
        common = ('long_edge', 'device', 'seed')
        keys = common + (('depth_model', 'dvd_window', 'dvd_overlap', 'offload') if kind == 'depth'
                         else ('normal_window', 'normal_overlap', 'decode_chunk', 'offload', 'normal_axes'))
        return {key: asdict(self)[key] for key in keys}


def resize_rgb(rgb, long_edge):
    h, w = rgb.shape[:2]
    scale = min(1., long_edge / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if size != (w, h) else rgb


def pad_rgb(rgb, multiple):
    """Bottom/right padding preserves the plate aspect and pixel alignment."""
    h, w = rgb.shape[:2]
    return cv2.copyMakeBorder(rgb, 0, (-h) % multiple, 0, (-w) % multiple, cv2.BORDER_REPLICATE)


def unit_normals(normals):
    n = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(n).all():
        raise ValueError('Model produced non-finite normals.')
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), np.array([0., 0., 1.], np.float32))


def windows(length, size, overlap):
    """Cover each frame, retaining a full final window where possible."""
    if length < 1 or not 0 < overlap < size:
        raise ValueError('Invalid temporal window.')
    result = [(0, min(length, size))]
    while result[-1][1] < length:
        start = min(result[-1][0] + size - overlap, max(0, length - size))
        result.append((start, min(start + size, length)))
    return result


def align_depth(current, reference):
    """DVD global affine overlap alignment, with a guarded constant-depth case."""
    x, y = current.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    mx, my = x.mean(), y.mean()
    variance = np.mean((x - mx) ** 2)
    scale = np.clip(np.mean((x - mx) * (y - my)) / variance, .7, 1.5) if variance > 1e-12 else 1.
    return float(scale), float(my - scale * mx)
