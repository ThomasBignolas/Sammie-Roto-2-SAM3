"""Explicit mask and coordinate conversions at the backend boundary."""
import base64
import zlib
import cv2
import numpy as np


def binary_mask(value, width, height):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f'Expected HxW mask, received {mask.shape}')
    mask = (mask > 0).astype(np.uint8) * 255
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def normalize_points(points, width, height):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite')
    if np.any(points < 0) or np.any(points >= [width, height]):
        raise ValueError('Point outside source image')
    return (points / [width, height]).tolist()


def normalize_box(box, width, height):
    x0, y0, x1, y1 = map(float, box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Box must have positive area and lie inside the source image')
    # Official SAM3 concept prompts use normalized XYWH, not SAM2 XYXY.
    return [x0 / width, y0 / height, (x1-x0) / width, (y1-y0) / height]


def pack_mask(mask):
    mask = np.asarray(mask) > 0
    return {'shape': list(mask.shape),
            'data': base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode('ascii')}


def unpack_mask(data):
    shape = tuple(map(int, data['shape']))
    if len(shape) != 2 or min(shape) < 1 or np.prod(shape) > 100_000_000:
        raise ValueError('Invalid seed mask dimensions')
    size = int(np.prod(shape))
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(data['data'], validate=True), (size+7)//8 + 1)
    if len(raw) != (size+7)//8 or not decoder.eof:
        raise ValueError('Invalid seed mask data')
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:size].reshape(shape).astype(bool)


def match_seeds(seeds, candidates):
    """Re-identify saved selections by mask overlap, never by detector ordering."""
    available = dict(candidates)
    mapping = {}
    for seed in seeds:
        target = unpack_mask(seed['seed_mask'])
        scores = []
        for oid, mask in available.items():
            mask = binary_mask(mask, target.shape[1], target.shape[0]) > 0
            union = np.count_nonzero(target | mask)
            scores.append((np.count_nonzero(target & mask) / max(union, 1), oid))
        if not scores or max(scores)[0] < 0.8:
            raise RuntimeError('Saved detection could not be matched reliably. Re-select objects; identities were not reassigned.')
        _, oid = max(scores)
        mapping[oid] = seed['object_id']
        del available[oid]
    return mapping
