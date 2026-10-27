"""Package an exact Git revision for school use, never the working folder."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'Sammie-Roto-School/'
FORBIDDEN = {'.git', '.runtime', '.research', '.uv', '.venv', '.test-output', '__pycache__',
             'temp', 'dist', 'build', '.cache', '.codex', '.agents', '.env', '.ssh', '.workflow', '.sammie'}
MEDIA = {'.sammie', '.sammieproj', '.nk', '.exr', '.pt', '.pth', '.safetensors', '.bin', '.mov', '.mp4', '.pyc', '.pfx', '.pem', '.key'}
BUNDLED_WEIGHT = 'checkpoints/1x_binary_mask_smooth.pth'
BUNDLED_EXAMPLES = {'examples/example_bunny.mp4', 'examples/example_dance.mp4'}
REQUIRED = ('install_school.bat', 'run_sam3.bat', 'uv.lock', 'scripts/install_school.ps1',
            'sammie/opaque_core.py', 'sammie/mask_contacts.py', 'sammie/matte_quality.py',
            'checkpoints/videomama/model_index.json', 'checkpoints/minimax/vae/config.json',
            'docs/SCHOOL_INSTALL.md', 'LICENSE', BUNDLED_WEIGHT, *sorted(BUNDLED_EXAMPLES),
            'sammie/exr_media.py', 'sammie/exr_import_ui.py', 'update_school.bat',
            'scripts/update_school.py', 'scripts/update_school.ps1', 'sammie/projects.py',
            'sammie/project_ui.py', 'sammie/workflow/ui.py', 'sammie/workflow/worker.py',
            'sammie/workflow/nuke.py', 'sammie/school_updates.py')


def allowed(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or FORBIDDEN.intersection(path.parts):
        return False
    if path.name == 'sammie.workspace.json' or any(p.endswith('.sammie-data') for p in path.parts):
        return False
    if path.name in ('sammie_settings.json', 'sammie_settings.conf', '.git-credentials', 'token', 'stored_tokens'):
        return False
    if path.name.startswith('.env') or path.suffix.lower() in MEDIA:
        return name == BUNDLED_WEIGHT or name in BUNDLED_EXAMPLES
    # Configs are required; downloaded weights and HF cache metadata are not.
    if 'checkpoints' in path.parts and path.suffix.lower() != '.json':
        return False
    return True


def build(root=ROOT, revision='HEAD', output=None):
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--verify', revision + '^{commit}'], text=True).strip()
    source = subprocess.check_output(['git', '-C', str(root), 'archive', '--format=zip', commit])
    output = output or root / 'dist' / f'Sammie-Roto-School-{commit[:12]}.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'commit': commit, 'distribution': 'school source; models and credentials excluded', 'sha256': {}}
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        names = [n for n in original.namelist() if not n.endswith('/') and allowed(n)]
        missing = set(REQUIRED) - set(names)
        if missing:
            raise RuntimeError(f'Revision is missing school package files: {sorted(missing)}. Commit the implementation first.')
        for name in sorted(names):
            data = original.read(name)
            info = zipfile.ZipInfo(PREFIX + name, date_time=original.getinfo(name).date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            manifest['sha256'][name] = hashlib.sha256(data).hexdigest()
        info = zipfile.ZipInfo(PREFIX + 'SCHOOL_BUILD.json')
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Package CRC verification failed')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    return output, digest, len(manifest['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output, digest, count = build(revision=args.revision, output=args.output)
    print(f'{output}\n{count} files, {output.stat().st_size / 2**20:.1f} MiB\nSHA256 {digest}')


if __name__ == '__main__':
    main()
