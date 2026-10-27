"""Install optional upstream CorridorKey locally; no redistribution or media upload."""
import argparse
import subprocess
from sammie.foreground.versions import SOURCE, SOURCE_PATH, ensure_weights


def sources():
    repo, revision = SOURCE
    if not SOURCE_PATH.exists():
        SOURCE_PATH.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',f'https://github.com/{repo}.git',str(SOURCE_PATH)],check=True)
        subprocess.run(['git','-C',str(SOURCE_PATH),'checkout','--detach',revision],check=True)
    current = subprocess.check_output(['git','-C',str(SOURCE_PATH),'rev-parse','HEAD'],text=True).strip()
    if current != revision:
        raise RuntimeError(f'Existing {SOURCE_PATH} is at {current}; expected {revision}. No files overwritten.')
    print(f'CorridorKey source: {revision}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources-only',action='store_true')
    p.add_argument('--models',nargs='+',choices=['blue','green'],default=['blue'])
    args=p.parse_args()
    print('Optional third-party CorridorKey 1.0. License: CC-BY-NC-SA with additional terms.',flush=True)
    print('https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE',flush=True)
    print('Local school/workstation setup; not bundled for redistribution or resale.',flush=True)
    sources()
    if not args.sources_only:
        for model in args.models:
            print(ensure_weights(model,lambda msg:print(msg,flush=True)),flush=True)


if __name__=='__main__': main()
