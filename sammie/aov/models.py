"""Adapters to pinned LiveActionAOV and DVD code. Imported only in the worker."""
import gc
import os
import sys
import numpy as np
import torch
from PIL import Image
from .versions import source_path, ensure_weights
from .config import pad_rgb, unit_normals, windows, align_depth


def prepare_sources():
    os.environ.setdefault('MPLCONFIGDIR', str(source_path('DVD').parent/'matplotlib-cache'))
    for name in ('LiveActionAOV', 'DVD'):
        path = source_path(name)
        if not path.is_dir():
            raise RuntimeError('AOV dependencies are missing. Run install_aov.bat first.')
        sys.path.insert(0, str(path/'src' if name == 'LiveActionAOV' else path))


class DVDModel:
    def __init__(self, settings, status):
        self.settings, self.status, self.pipe = settings, status, None

    def load(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('DVD needs an NVIDIA CUDA GPU with BF16 support.')
        prepare_sources()
        from diffsynth.pipelines.wan_video_new_determine import WanVideoPipeline, ModelConfig
        from peft import LoraConfig, inject_adapter_in_model
        from peft.tuners.lora.layer import Linear as LoraLinear
        from safetensors.torch import load_file
        dvd = ensure_weights('dvd', self.status)
        wan = ensure_weights('wan', self.status)
        self.status('Loading DVD v1.1 and Wan VAE on CPU...')
        pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cpu',
            model_configs=[ModelConfig(path=str(wan/'diffusion_pytorch_model.safetensors')),
                           ModelConfig(path=str(wan/'Wan2.1_VAE.pth'))],
            tokenizer_config=ModelConfig(path=str(wan)), skip_download=True)
        pipe.scheduler.set_timesteps(training=True, denoise_step=.5)
        pipe.dit = inject_adapter_in_model(LoraConfig(r=512, lora_alpha=512,
            target_modules=['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']), pipe.dit, adapter_name='default')
        state = load_file(str(dvd/'dvd_1.1.safetensors'), device='cpu')
        state = {k.removeprefix('pipe.dit.'): v for k, v in state.items() if k.startswith('pipe.dit.')}
        pipe.dit.load_state_dict(state, strict=True)
        del state
        for module in pipe.dit.modules():
            if isinstance(module, LoraLinear):
                module.merge()
        # Remove the merged adapter storage; keep identical fused base weights.
        for name, module in list(pipe.dit.named_modules()):
            if isinstance(module, LoraLinear):
                parent_name, _, child = name.rpartition('.')
                parent = pipe.dit.get_submodule(parent_name) if parent_name else pipe.dit
                setattr(parent, child, module.base_layer)
        pipe.eval().requires_grad_(False)
        pipe.device = torch.device('cuda')
        # Official layer offloading keeps the VAE and DiT from needing simultaneous residency.
        free_gib = torch.cuda.mem_get_info()[0] / 2**30
        if self.settings.offload == 'sequential':
            pipe.enable_vram_management(num_persistent_param_in_dit=0)
        else:
            pipe.enable_vram_management(vram_limit=max(2., min(10., free_gib - 2.)))
        # DVD's inference path omits the stock DiffSynth stage-onload calls.
        # Reconnect them at the three actual stage boundaries, otherwise its
        # official offloading wrappers leave VAE weights on CPU.
        encode, decode, model_fn = pipe.vae.encode, pipe.vae.decode, pipe.model_fn
        def encode_stage(*args, **kwargs):
            pipe.load_models_to_device(['vae'])
            return encode(*args, **kwargs)
        def decode_stage(*args, **kwargs):
            pipe.load_models_to_device(['vae'])
            return decode(*args, **kwargs)
        def dit_stage(*args, **kwargs):
            pipe.load_models_to_device(['dit'])
            return model_fn(*args, **kwargs)
        pipe.vae.encode, pipe.vae.decode, pipe.model_fn = encode_stage, decode_stage, dit_stage
        self.pipe = pipe
        gc.collect()

    def predict(self, frames, status):
        self.load()
        h, w = frames[0].shape[:2]
        clips = windows(len(frames), self.settings.dvd_window, self.settings.dvd_overlap)
        output = None
        previous_end = 0
        for i, (start, end) in enumerate(clips):
            status(i, len(clips), f'DVD temporal window {i+1}/{len(clips)}')
            images = [pad_rgb(f, 16) for f in frames[start:end]]
            count = len(images)
            images += [images[-1]] * ((1-count) % 4)
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.
            _, t, _, ph, pw = tensor.shape
            with torch.inference_mode():
                result = self.pipe(prompt=[''], negative_prompt=[''], mode='regression', height=ph, width=pw,
                    num_frames=t, batch_size=1, input_image=tensor[:, 0], extra_images=tensor,
                    extra_image_frame_index=torch.ones((1, t), device='cuda'), input_video=tensor,
                    cfg_scale=1, seed=self.settings.seed, tiled=True, denoise_step=.5)
            current = np.asarray(result['depth'][0, :count, :h, :w], np.float32).mean(axis=-1)
            if output is None:
                output = current
            else:
                overlap = previous_end - start
                scale, shift = align_depth(current[:overlap], output[-overlap:])
                current = np.maximum(current * scale + shift, 0)
                blend = np.linspace(0, 1, overlap, dtype=np.float32)[:, None, None]
                mixed = output[-overlap:] * (1-blend) + current[:overlap] * blend
                output = np.concatenate([output[:-overlap], mixed, current[overlap:]])
            previous_end = end
        return output


class NormalModel:
    def __init__(self, settings, status):
        self.settings, self.status, self.pipe = settings, status, None

    def load(self):
        prepare_sources()
        from diffusers import AutoencoderKLTemporalDecoder
        from live_action_aov.vendored.normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
        from live_action_aov.vendored.normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
        path = str(ensure_weights('normalcrafter', self.status))
        device = self.settings.device
        dtype = torch.float16 if device == 'cuda' else torch.float32
        self.status('Loading NormalCrafter...')
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(path,
            subfolder='unet', torch_dtype=dtype, local_files_only=True)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(path, subfolder='vae', torch_dtype=dtype, local_files_only=True)
        pipe = NormalCrafterPipeline.from_pretrained(path, unet=unet, vae=vae,
            torch_dtype=dtype, variant='fp16', local_files_only=True)
        for model in (pipe.unet, pipe.vae, pipe.image_encoder):
            model.eval().requires_grad_(False)
        if device == 'cuda':
            if self.settings.offload == 'sequential':
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to('cpu')
        self.pipe = pipe

    def predict(self, frames, status):
        self.load()
        # Preserve aspect, explicitly pad to 64, and crop ourselves. Upstream's
        # unpad_image assumes a different layout than its numpy output.
        h, w = frames[0].shape[:2]
        images = [Image.fromarray(pad_rgb(f, 64)) for f in frames]
        count = len(images)
        images += [images[-1]] * max(0, self.settings.normal_window - count)
        pipe = self.pipe
        original_generate = pipe.generate
        calls = 0
        total = len(windows(len(images), self.settings.normal_window, self.settings.normal_overlap))
        def generate(*args, **kwargs):
            nonlocal calls
            calls += 1
            status(calls-1, total, f'NormalCrafter temporal window {calls}/{total}')
            return original_generate(*args, **kwargs)
        pipe.generate = generate
        # Encode RGB in small batches; official default encodes the entire clip
        # on GPU at once. Latents and overlapping-window propagation stay intact.
        original_encode = pipe._encode_image
        original_vae = pipe.ecnode_video_vae
        chunk = self.settings.decode_chunk
        def encode(images, *args, **kwargs):
            return torch.cat([original_encode(images[i:i+chunk], *args, **kwargs)
                              for i in range(0, len(images), chunk)])
        def encode_vae(images, chunk_size=2):
            return torch.cat([original_vae(images[i:i+chunk], chunk_size=chunk)
                              for i in range(0, len(images), chunk)])
        pipe._encode_image, pipe.ecnode_video_vae = encode, encode_vae
        try:
            with torch.inference_mode():
                result = pipe(images, window_size=self.settings.normal_window,
                    time_step_size=self.settings.normal_window-self.settings.normal_overlap,
                    decode_chunk_size=chunk,
                    generator=torch.Generator(device=self.settings.device).manual_seed(self.settings.seed))
            normals = np.asarray(result.frames[0][:count, :h, :w], np.float32)
            if self.settings.normal_axes == 'flip_yz':
                normals[..., 1:] *= -1
            return unit_normals(normals)
        finally:
            pipe.generate, pipe._encode_image, pipe.ecnode_video_vae = original_generate, original_encode, original_vae


class VDAModel:
    def __init__(self, settings, status):
        self.settings, self.status, self.model = settings, status, None

    def predict(self, frames, status):
        prepare_sources()
        from live_action_aov.vendored.video_depth_anything.video_depth import VideoDepthAnything
        path = ensure_weights('vda-small', self.status)/'video_depth_anything_vits.pth'
        model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384], metric=False)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.model = model.to(self.settings.device).eval().requires_grad_(False)
        status(0, 1, 'Video Depth Anything: temporally aligned inference...')
        with torch.inference_mode():
            depth, _ = self.model.infer_video_depth(np.stack(frames), target_fps=24,
                input_size=min(518, min(frames[0].shape[:2])), device=self.settings.device,
                fp32=self.settings.device == 'cpu')
        return np.asarray(depth, np.float32)
