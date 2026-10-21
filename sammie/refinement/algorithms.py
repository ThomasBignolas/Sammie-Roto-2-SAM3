"""Float alpha, trimaps, boundary merging and conservative motion validation."""
from dataclasses import asdict, dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class RefinementSettings:
    model: str = 'small-composition-1k'
    checkpoint: str = ''
    strength: float = 0.75
    foreground: float = 0.98
    background: float = 0.02
    radius: int = 12                 # radius, not diameter, at a 1080px short edge
    expansion: int = 4
    feather: int = 3
    short_edge: int = 1080           # 0 = source resolution, never upscale
    tile_size: int = 768
    overlap: int = 128
    temporal: bool = True
    temporal_strength: float = 0.65
    precision: str = 'fp32'          # conservative default; BF16 is opt-in
    device: str = 'auto'
    debug: bool = False

    def validate(self):
        if not 0 <= self.background < self.foreground <= 1:
            raise ValueError('Background confidence must be below foreground confidence, within [0, 1].')
        if not 0 <= self.strength <= 1 or not 0 <= self.temporal_strength <= 1:
            raise ValueError('Refinement and temporal strengths must be in [0, 1].')
        if min(self.radius, self.expansion, self.feather, self.short_edge) < 0:
            raise ValueError('Widths and resolution cannot be negative.')
        if self.tile_size < 256 or not 32 <= self.overlap < self.tile_size // 2:
            raise ValueError('Use tiles >=256px and overlap >=32px and below half the tile size.')
        if self.precision not in ('fp32', 'bf16') or self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Unsupported precision or device.')
        return self

    @classmethod
    def from_session(cls, manager):
        return cls(**{k: manager.get_session_setting('vitmatte_' + k, v)
                      for k, v in asdict(cls()).items()}).validate()


PRESETS = {
    'Fast': dict(short_edge=720, radius=8, expansion=2, temporal=False,
                 temporal_strength=0.35, tile_size=768, strength=0.65),
    'Balanced': dict(short_edge=1080, radius=12, expansion=4, temporal=True,
                     temporal_strength=0.65, tile_size=768, strength=0.75),
    'Quality': dict(short_edge=2160, radius=16, expansion=6, temporal=True,
                    temporal_strength=0.8, tile_size=1024, strength=0.85),
}


def alpha_float(alpha):
    a = np.asarray(alpha)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Alpha must be a finite two-dimensional image.')
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    return np.clip(a, 0, 1).astype(np.float32, copy=False)


def scaled_radius(radius, shape):
    return max(1, round(radius * min(shape[:2]) / 1080)) if radius else 0


def make_trimap(alpha, settings):
    settings.validate()
    a = alpha_float(alpha)
    radius = scaled_radius(settings.radius + settings.expansion, a.shape)
    fg = (a >= settings.foreground).astype(np.uint8)
    bg = (a <= settings.background).astype(np.uint8)
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        # Replicate border: an object reaching the image boundary is not surrounded by BG.
        fg = cv2.erode(fg, kernel, borderType=cv2.BORDER_REPLICATE)
        bg = cv2.erode(bg, kernel, borderType=cv2.BORDER_REPLICATE)
    trimap = np.full(a.shape, 128, np.uint8)
    trimap[bg != 0] = 0
    trimap[fg != 0] = 255
    return trimap


def merge_alpha(base, prediction, trimap, settings):
    base, prediction = alpha_float(base), alpha_float(prediction)
    if base.shape != prediction.shape or base.shape != trimap.shape:
        raise ValueError('RGB, alpha, trimap and prediction must remain aligned.')
    weight = (trimap == 128).astype(np.float32)
    feather = scaled_radius(settings.feather, base.shape)
    if feather:
        weight *= np.clip(cv2.distanceTransform(weight.astype(np.uint8), cv2.DIST_L2, 5) / feather, 0, 1)
    # Exact preservation outside unknown, including nonbinary confident alpha.
    return np.clip(base + settings.strength * weight * (prediction-base), 0, 1).astype(np.float32)


class TemporalValidator:
    """Gate refinement deltas using bidirectional flow; never average past alpha.

    Occlusion, unreliable flow, cuts and changes in baseline alpha fall back toward
    the current VideoMaMa matte. Only the preceding raw refinement is retained.
    """
    def __init__(self, strength=0.65, flow_edge=640):
        self.strength, self.flow_edge = strength, flow_edge
        self.previous = None

    def apply(self, rgb, base, refined, frame):
        base, refined = alpha_float(base), alpha_float(refined)
        h, w = base.shape
        scale = min(1, self.flow_edge / max(h, w))
        size = (max(8, round(w*scale)), max(8, round(h*scale)))
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
        small_base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
        delta = cv2.resize(refined-base, size, interpolation=cv2.INTER_AREA)
        previous = self.previous
        self.previous = (frame, gray, small_base, delta)
        if previous is None or previous[0] != frame-1 or previous[1].shape != gray.shape:
            return refined.copy()
        _, old_gray, old_base, old_delta = previous
        args = (None, .5, 3, 21, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(gray, old_gray, *args)
        forward = cv2.calcOpticalFlowFarneback(old_gray, gray, *args)
        y, x = np.mgrid[:size[1], :size[0]].astype(np.float32)
        mx, my = x + backward[..., 0], y + backward[..., 1]
        def warp(a):
            return cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        error = np.linalg.norm(backward + warp(forward), axis=-1)
        photo = np.abs(gray.astype(np.float32)-warp(old_gray).astype(np.float32))/255
        valid = ((mx >= 0) & (mx <= size[0]-1) & (my >= 0) & (my <= size[1]-1)
                 & (error < 1.5) & (photo < .12) & (np.abs(small_base-warp(old_base)) < .15))
        # Even accurate motion cannot validate a large, novel refinement delta.
        disagreement = np.clip((np.abs(delta-warp(old_delta))-.04)/.16, 0, 1)
        suspicion = np.where(valid, disagreement, 1).astype(np.float32)
        suspicion = cv2.resize(suspicion, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(base + (refined-base)*(1-self.strength*suspicion), 0, 1).astype(np.float32)
