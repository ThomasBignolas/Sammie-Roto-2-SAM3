"""Adapter to the installed official engine, isolated from the UI's PyTorch."""
import os
import subprocess
import sys
import numpy as np
from .versions import SOURCE, SOURCE_PATH, ROOT, ensure_weights


class CorridorModel:
    def __init__(self, settings, status=lambda text: None):
        import torch
        from safetensors.torch import load_file
        if not (SOURCE_PATH/'CorridorKeyModule'/'inference_engine.py').is_file():
            raise RuntimeError('CorridorKey is not installed. Run install_corridorkey.bat on this PC.')
        revision = subprocess.check_output(['git','-C',str(SOURCE_PATH),'rev-parse','HEAD'], text=True).strip()
        if revision != SOURCE[1]:
            raise RuntimeError('CorridorKey source revision differs from the tested version. Run setup in a fresh local runtime folder.')
        if settings.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select CPU, or install the NVIDIA driver and CorridorKey runtime.')
        os.environ['CORRIDORKEY_SKIP_COMPILE'] = '1'
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT/'.runtime'/'corridorkey-compile'))
        sys.path.insert(0, str(SOURCE_PATH))
        from CorridorKeyModule.inference_engine import CorridorKeyEngine
        checkpoint = ensure_weights(settings.screen, status)
        status(f'Loading CorridorKey {settings.screen}, {settings.resolution}px, {settings.device}...')
        self.settings = settings
        self.engine = CorridorKeyEngine(str(checkpoint), device=settings.device, img_size=settings.resolution,
            use_refiner=True, mixed_precision=settings.device == 'cuda' and settings.precision == 'fp16',
            model_precision=torch.float32)
        # Upstream loads strict=False. Refuse any missing or unexpected keys here.
        state = load_file(str(checkpoint))
        state = {k.removeprefix('_orig_mod.'): v for k,v in state.items()}
        actual = self.engine.model.state_dict()
        if set(state) != set(actual):
            raise ValueError('Incompatible CorridorKey checkpoint keys.')
        for k,v in state.items():
            if v.shape != actual[k].shape and 'pos_embed' not in k:
                raise ValueError(f'Incompatible CorridorKey tensor: {k}')
        self.engine.model.eval()

    def predict(self, linear709, alpha):
        import torch
        engine = self.engine
        with torch.inference_mode():
            rgb = torch.from_numpy(np.ascontiguousarray(linear709)).permute(2,0,1)[None].to(engine.device)
            mask = torch.from_numpy(np.ascontiguousarray(alpha))[None,None].to(engine.device)
            inputs = engine._preprocess_input(rgb, mask, input_is_linear=True)
            del rgb, mask
            with torch.autocast(device_type=engine.device.type, dtype=torch.float16, enabled=engine.mixed_precision):
                result = engine.model(inputs)
            # Cache the native output before full-size interpolation; FP16 is lossless
            # for FP16 model output. FP32 mode retains float32 instead.
            fg = result['fg'][0].permute(1,2,0).cpu().numpy().copy()
            if not np.isfinite(fg).all():
                raise RuntimeError('CorridorKey returned non-finite RGB. Try FP32.')
            return fg
