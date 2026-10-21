"""Pinned official ViTMatte weights, strict loading and bounded tile inference."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import requests
import torch

MODEL_SPECS = {
    'base-composition-1k': {
        'label': 'ViTMatte Base (larger) / Composition-1k',
        'repo': 'hustvl/vitmatte-base-composition-1k',
        'revision': 'bf486d01a7d9e3dbcc8400f7942835caf0eaf76e',
        'filename': 'pytorch_model.bin',
        'sha256': 'b2521bcc4b719fb24611c39605b6642162fd7502e69b3cc846506ca921757b41',
        'hidden_size': 768,
    },
    'small-composition-1k': {
        'label': 'ViTMatte Small / Composition-1k',
        'repo': 'hustvl/vitmatte-small-composition-1k',
        'revision': '6a58ad7646403c1df626fbd746900aec7361ea1d',
        'filename': 'model.safetensors',
        'sha256': 'bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb',
        'hidden_size': 384,
    },
}
NORMALIZATION = 'official-imagenet-rgb-v1'


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_cancel(cancel):
    if cancel():
        raise InterruptedError('ViTMatte refinement cancelled; completed frames can be resumed.')


def resolve_checkpoint(settings, progress=lambda message: None, cancel=lambda: False):
    if settings.model not in MODEL_SPECS:
        raise ValueError('Unsupported ViTMatte model. Select Base or Small / Composition-1k.')
    spec = MODEL_SPECS[settings.model]
    if settings.checkpoint:
        root = Path(settings.checkpoint).expanduser().resolve()
        if not root.is_dir() or not all((root/f).is_file() for f in ('config.json', 'model.safetensors')):
            raise ValueError('Local ViTMatte checkpoint must be a Transformers directory containing config.json and model.safetensors, not a Detectron2 .pth file.')
        weights = root/'model.safetensors'
    else:
        root = Path('checkpoints/vitmatte') / spec['revision']
        weights = root/spec['filename']
        root.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'preprocessor_config.json', spec['filename']):
            check_cancel(cancel)
            dest = root/name
            if not dest.exists():
                part = dest.with_suffix(dest.suffix+'.part')
                try:
                    with requests.get(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}",
                                      stream=True, timeout=(15, 30)) as response:
                        response.raise_for_status()
                        total = int(response.headers.get('Content-Length', 0))
                        received = 0
                        with part.open('wb') as f:
                            for block in response.iter_content(1024*1024):
                                check_cancel(cancel)
                                f.write(block)
                                received += len(block)
                                progress(f'Downloading {name}: {received/1048576:.1f} / {total/1048576:.1f} MiB')
                    part.replace(dest)
                finally:
                    part.unlink(missing_ok=True)
        progress(f"Verifying {spec['label']} checkpoint...")
        if digest(weights) != spec['sha256']:
            raise ValueError(f'ViTMatte checksum mismatch. Remove {weights} and retry the download.')
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != 'vitmatte' or config.get('backbone_config', {}).get('num_channels') != 4:
        raise ValueError('Checkpoint is not a four-channel RGB + trimap ViTMatte model.')
    if config.get('hidden_size') != spec['hidden_size']:
        raise ValueError(f"Checkpoint size does not match selected {spec['label']}. Choose the matching model or clear the local checkpoint folder.")
    identity = digest(weights) + ':' + digest(root/'config.json') + ':' + NORMALIZATION + ':transformers-5.8.0'
    return root, identity


class VitMatteModel:
    def __init__(self, settings, checkpoint):
        self.settings, self.checkpoint = settings, Path(checkpoint)
        self.model = None
        device = settings.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable. Select CPU or Auto.')
        self.device = torch.device(device)

    def load(self):
        try:
            from transformers import VitMatteForImageMatting
        except ImportError as e:
            raise RuntimeError('ViTMatte is optional. Run install_vitmatte.bat, then restart Sammie-Roto.') from e
        try:
            # Base is officially distributed as a PyTorch state dictionary. Only
            # accept that binary in its pinned, checksum-verified download folder.
            spec = MODEL_SPECS[self.settings.model]
            safe = bool(self.settings.checkpoint) or spec['filename'].endswith('.safetensors')
            if not safe:
                expected = (Path('checkpoints/vitmatte')/spec['revision']).resolve()
                if (self.checkpoint.resolve() != expected
                        or digest(expected/spec['filename']) != spec['sha256']):
                    raise ValueError('Binary ViTMatte weights must be the verified official download; local custom checkpoints require safetensors.')
            model, info = VitMatteForImageMatting.from_pretrained(
                str(self.checkpoint), local_files_only=True, use_safetensors=safe,
                weights_only=True, output_loading_info=True, dtype=torch.float32)
            if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'Checkpoint architecture mismatch: {info}')
            self.model = model.eval().requires_grad_(False).to(self.device)
        except Exception:
            self.close()
            raise

    @torch.inference_mode()
    def predict(self, rgb, trimap):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('ViTMatte requires display-referred RGB uint8, not linear EXR or premultiplied foreground.')
        if rgb.shape[:2] != trimap.shape or not np.isin(trimap, [0, 128, 255]).all():
            raise ValueError('ViTMatte trimap must be aligned uint8 values 0, 128 and 255.')
        if self.model is None:
            self.load()
        h, w = trimap.shape
        # Original hustvl configs/common/model.py normalization. The published HF
        # processor uses 0.5/0.5 instead; do not silently adopt that discrepancy.
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()/255
        mean = image.new_tensor([.485, .456, .406])[:, None, None]
        std = image.new_tensor([.229, .224, .225])[:, None, None]
        tri = torch.from_numpy(np.ascontiguousarray(trimap)).to(self.device).float()[None]/255
        pixels = torch.cat(((image-mean)/std, tri), dim=0)[None]
        pixels = torch.nn.functional.pad(pixels, (0, (-w) % 32, 0, (-h) % 32))
        mixed = self.settings.precision == 'bf16' and self.device.type == 'cuda'
        if mixed and not torch.cuda.is_bf16_supported():
            raise RuntimeError('This GPU does not support BF16; select FP32.')
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=mixed):
            result = self.model(pixel_values=pixels).alphas[0, 0, :h, :w].float()
        if not torch.isfinite(result).all():
            raise RuntimeError('ViTMatte returned non-finite alpha. Try FP32 or a different checkpoint.')
        return result.clamp(0, 1).cpu().numpy()

    def close(self):
        self.model = None
        gc.collect()
        if self.device.type == 'cuda':
            # Only at model lifetime boundary, after live model references are gone.
            torch.cuda.empty_cache()


def predict_tiled(model, rgb, trimap, settings, cancel=lambda: False, progress=lambda message: None):
    """Fixed grid, context overlap and normalized feather weights. Skip known tiles."""
    h, w = trimap.shape
    tile, overlap = settings.tile_size, settings.overlap
    def starts(length):
        return sorted(set(list(range(0, max(1, length-tile+1), tile-overlap)) + [max(0, length-tile)]))
    result, weights = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    boxes = [(y, x) for y in starts(h) for x in starts(w)
             if np.any(trimap[y:y+tile, x:x+tile] == 128)]
    for i, (y, x) in enumerate(boxes):
        check_cancel(cancel)
        progress(f'ViTMatte tile {i+1}/{len(boxes)}')
        tri = trimap[y:y+tile, x:x+tile]
        prediction = model.predict(rgb[y:y+tile, x:x+tile], tri)
        th, tw = tri.shape
        wy, wx = np.ones(th, np.float32), np.ones(tw, np.float32)
        oy, ox = min(overlap, th), min(overlap, tw)
        if y: wy[:oy] = np.linspace(1/(oy+1), 1, oy)
        if x: wx[:ox] = np.linspace(1/(ox+1), 1, ox)
        if y+th < h: wy[-oy:] = np.linspace(1, 1/(oy+1), oy)
        if x+tw < w: wx[-ox:] = np.linspace(1, 1/(ox+1), ox)
        weight = wy[:, None]*wx[None]
        result[y:y+th, x:x+tw] += prediction*weight
        weights[y:y+th, x:x+tw] += weight
    if np.any((trimap == 128) & (weights == 0)):
        raise RuntimeError('Tile grid did not cover the unknown region.')
    return np.divide(result, weights, out=result, where=weights > 0)
