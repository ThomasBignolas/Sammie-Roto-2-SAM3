"""Prefetch the app's existing model registry and verified official ViTMatte weights."""
import argparse
import hashlib
from pathlib import Path
import os
import re
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ESSENTIAL = ('Base', 'matanyone', 'matanyone2', 'videomama', 'svd_vae')


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def download(spec, root=ROOT, get=None, token=None, progress=print):
    """Resume when supported; verify before replacing an existing file.

    The legacy registry uses MD5 integrity checks. TLS and fixed source URLs
    remain the trust boundary; MD5 is not a security signature. HF credentials
    are sent only to huggingface.co, never to other providers.
    """
    if get is None:
        import requests
        get = requests.get
    destination = root / spec.final_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and checksum(destination) == spec.md5:
        progress(f'Verified: {spec.final_path}')
        return destination
    part = destination.with_suffix(destination.suffix + '.part')
    if part.is_file() and checksum(part) == spec.md5:
        part.replace(destination)
        return destination
    offset = part.stat().st_size if part.is_file() else 0
    headers = {'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    if token and urlsplit(spec.url).hostname == 'huggingface.co':
        headers['Authorization'] = f'Bearer {token}'
    progress(f'Downloading {spec.final_path}' + (f' (resume at {offset / 2**20:.1f} MiB)' if offset else ''))
    with get(spec.url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if response.status_code == 416:
            part.unlink(missing_ok=True)
            raise RuntimeError(f'Stale download for {spec.filename}. Rerun setup to restart this file.')
        if response.status_code not in (200, 206):
            raise RuntimeError(f'{spec.filename}: HTTP {response.status_code}. Check network access and HF model permissions.')
        if response.status_code == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset:
                raise RuntimeError(f'Invalid resume response for {spec.filename}; existing download was kept.')
        else:
            offset = 0
        received = offset
        last_report = time.monotonic()
        with part.open('ab' if offset else 'wb') as stream:
            for block in response.iter_content(4 * 1024 * 1024):
                stream.write(block)
                received += len(block)
                if time.monotonic() - last_report >= 5:
                    progress(f'  {spec.filename}: {received / 2**20:.0f} MiB')
                    last_report = time.monotonic()
    if checksum(part) != spec.md5:
        # A failed transfer above keeps its partial file for resume. A complete
        # but corrupt response must restart; the old destination stays intact.
        part.unlink(missing_ok=True)
        raise RuntimeError(f'Checksum mismatch for {spec.filename}; rerun to download again.')
    part.replace(destination)
    progress(f'Verified: {spec.final_path}')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', choices=('all', 'essential'), default='all')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    from sammie.model_downloader import MODEL_REGISTRY
    keys = tuple(MODEL_REGISTRY) if args.models == 'all' else ESSENTIAL
    print('Application models: ' + ', '.join(keys))
    print('ViTMatte: Base (default, larger) and Small')
    if args.plan:
        return
    os.chdir(ROOT)
    from huggingface_hub import get_token
    failures = []
    for key in keys:
        try:
            download(MODEL_REGISTRY[key], token=get_token())
        except Exception as error:
            print(f'FAILED {key}: {type(error).__name__}. Check connection, disk space and model access; rerun to retry.', flush=True)
            failures.append(key)
    from sammie.refinement.algorithms import RefinementSettings
    from sammie.refinement.model import resolve_checkpoint, MODEL_SPECS
    for model in MODEL_SPECS:
        try:
            last_report = [0.0]
            def report(message):
                now = time.monotonic()
                if now - last_report[0] >= 5 or 'Verifying' in message:
                    print(message, flush=True)
                    last_report[0] = now
            resolve_checkpoint(RefinementSettings(model=model), progress=report)
            print(f'Verified: ViTMatte {model}', flush=True)
        except Exception as error:
            print(f'FAILED ViTMatte {model}: {type(error).__name__}. Rerun setup; see docs/VITMATTE.md for checkpoint repair.', flush=True)
            failures.append(model)
    if failures:
        raise SystemExit('Model setup incomplete: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
