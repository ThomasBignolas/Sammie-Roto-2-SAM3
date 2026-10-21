"""Download pinned upstream sources and model weights. No user footage is uploaded."""
import argparse
import subprocess
from sammie.aov.versions import SOURCES, WEIGHTS, source_path, ensure_weights

def sources():
    from filelock import FileLock
    root = source_path('DVD').parent
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root/'setup.lock'), timeout=600):
        _sources()

def _sources():
    for name, (repo, revision) in SOURCES.items():
        path = source_path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', f'https://github.com/{repo}.git', str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{path} has unexpected revision {actual}; expected {revision}. No files overwritten.')
        print(f'{name}: {revision}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--models', nargs='+', choices=list(WEIGHTS), default=list(WEIGHTS))
    args = parser.parse_args()
    sources()
    if not args.sources_only:
        print('DVD weights: CC BY-NC 4.0. NormalCrafter: review upstream SVD terms before commercial use.', flush=True)
        for name in args.models:
            ensure_weights(name, lambda s: print(s, flush=True))

if __name__ == '__main__':
    main()
