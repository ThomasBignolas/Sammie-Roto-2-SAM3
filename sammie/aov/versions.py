"""Audited source and weight revisions, kept separate from the UI environment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'LiveActionAOV': ('lettidude/LiveActionAOV', '8c2a231785bdf6a016a39c722010f90026b035f7'),
    'DVD': ('EnVision-Research/DVD', '62f52d16faa2cac10e31eb7815ece6e61d5449ec'),
}
WEIGHTS = {
    'dvd': ('FayeHongfeiZhang/DVD', '769534dec2d77f8da3667719568fbe57fe69cd9b', ['dvd_1.1.safetensors']),
    'wan': ('Wan-AI/Wan2.1-T2V-1.3B', '37ec512624d61f7aa208f7ea8140a131f93afc9a', ['diffusion_pytorch_model.safetensors', 'Wan2.1_VAE.pth', 'config.json']),
    'normalcrafter': ('Yanrui95/NormalCrafter', '7e24d68d86ae008fe08ef50b4e51cd2fc2c8cf57', [
        'model_index.json', 'feature_extractor/preprocessor_config.json', 'scheduler/scheduler_config.json',
        'image_encoder/config.json', 'image_encoder/model.fp16.safetensors',
        'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors']),
    'vda-small': ('depth-anything/Video-Depth-Anything-Small', '256875362cff76724b920335dfb4b29dd611f66e', ['video_depth_anything_vits.pth']),
}

def source_path(name):
    return ROOT / '.runtime' / 'aov-sources' / name

def model_path(name):
    return ROOT / 'checkpoints' / 'aov' / name / WEIGHTS[name][1]

def ensure_weights(name, progress=lambda text: None):
    from huggingface_hub import snapshot_download
    repo, revision, patterns = WEIGHTS[name]
    progress(f'Checking / downloading {name} weights...')
    kwargs = dict(repo_id=repo, revision=revision, allow_patterns=patterns, local_dir=str(model_path(name)))
    marker = model_path(name) / '.download-complete'
    complete = all((model_path(name)/filename).is_file() and (model_path(name)/filename).stat().st_size > 0
                   for filename in patterns)
    if not marker.exists() or not complete:
        snapshot_download(**kwargs)
        marker.write_text(revision, encoding='ascii')
    return model_path(name)
