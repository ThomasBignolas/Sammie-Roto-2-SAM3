"""Real-model diagnostic using copied frames; no ground-truth accuracy claims."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import uuid
import zipfile
from sammie.aov.cache import atomic_json
from sammie.aov.config import AOVSettings
from sammie.aov.pipeline import run_job
from sammie.aov.export import export_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='SammieRotoFile.sammie')
    p.add_argument('--model', choices=['dvd', 'normalcrafter', 'vda-small', 'both'], default='both')
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--edge', type=int, default=512)
    p.add_argument('--root')
    args = p.parse_args()
    root = Path(args.root or '.test-output/aov-'+uuid.uuid4().hex[:8]).resolve()
    (root/'frames').mkdir(parents=True, exist_ok=True)
    if not list((root/'frames').glob('*.png')):
        with zipfile.ZipFile(args.source) as archive:
            names = sorted(n for n in archive.namelist() if n.startswith('frames/') and n.endswith('.png'))[:args.count]
            if len(names) != args.count:
                raise ValueError('Not enough source frames in the project.')
            for i, name in enumerate(names):
                with archive.open(name) as source, (root/'frames'/f'{i:05d}.png').open('wb') as target:
                    shutil.copyfileobj(source, target)
    settings = AOVSettings(depth=args.model != 'normalcrafter', normals=args.model in ('normalcrafter', 'both'),
                           depth_model='vda-small' if args.model == 'vda-small' else 'dvd', long_edge=args.edge)
    job = dict(root=str(root), first=0, last=args.count-1, extension='png', settings=asdict(settings), fps='24/1')
    atomic_json(root/'request.json', job)
    print('ARTIFACTS', root, flush=True)
    result = run_job(job, lambda done, total, text: print(f'{done}/{total}: {text}', flush=True))
    atomic_json(root/'results.json', result)
    export = dict(job, destination=str(root/('export-'+args.model+'-'+str(args.edge))), start_number=1001)
    result['export'] = export_sequence(export, lambda d,t,s: print(s, flush=True))
    atomic_json(root/'results.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
