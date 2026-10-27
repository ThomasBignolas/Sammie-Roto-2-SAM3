"""Check school runtimes without loading checkpoints or changing projects."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    'app': ('.venv', ['PySide6.QtWidgets', 'cv2', 'av', 'OpenEXR', 'transformers.models.vitmatte', 'sammie_main']),
    'sam3': ('.runtime/sam3-env', ['sam3.model_builder']),
    'aov': ('.runtime/aov-env', ['diffusers', 'transformers', 'safetensors', 'sammie.aov.worker']),
    'corridorkey': ('.runtime/corridorkey-env', ['timm', 'safetensors', 'sammie.foreground.pipeline']),
}


def check_environment(name):
    import importlib
    import torch
    for module in ENVIRONMENTS[name][1]:
        importlib.import_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError(f'{name}: CUDA unavailable. Install a current NVIDIA driver through school IT, then rerun setup.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The full school stack requires an NVIDIA GPU with BF16 support (RTX 30 series or newer).')
    with torch.inference_mode():
        x = torch.ones((32, 32), device='cuda')
        assert (x @ x).mean().item() == 32
    torch.cuda.synchronize()
    print(json.dumps({'runtime': name, 'python': sys.version.split()[0], 'torch': torch.__version__,
                      'gpu': torch.cuda.get_device_name(), 'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                      'imports_and_cuda': 'passed'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=ENVIRONMENTS)
    args = parser.parse_args()
    if args.environment:
        check_environment(args.environment)
        return
    for name, (folder, _) in ENVIRONMENTS.items():
        python = ROOT / folder / 'Scripts/python.exe'
        if not python.is_file():
            raise SystemExit(f'Missing {name} runtime. Run install_school.bat first.')
        subprocess.run([str(python), '-m', 'scripts.check_school', '--environment', name], cwd=ROOT, check=True)
    subprocess.run([str(ROOT / ENVIRONMENTS['sam3'][0] / 'Scripts/python.exe'), '-m',
                    'sammie.backends.check_runtime'], cwd=ROOT, check=True)
    print('All four runtimes passed. Checkpoints are validated by downloaders; no shot or full model inference was run.')


if __name__ == '__main__':
    main()
