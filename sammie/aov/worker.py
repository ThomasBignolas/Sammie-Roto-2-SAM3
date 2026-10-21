"""One isolated process per job. Termination releases CUDA even during a model call."""
import argparse
import json
import os
from pathlib import Path
import traceback
from .cache import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
    request = Path(args.request)
    job = json.loads(request.read_text(encoding='utf-8'))
    def progress(done, total, message):
        print('__AOV__' + json.dumps(dict(done=done, total=total, message=message)), flush=True)
    try:
        if job.get('operation') == 'export':
            from .export import export_sequence
            result = export_sequence(job, progress)
        else:
            from .pipeline import run_job
            result = run_job(job, progress)
        atomic_json(request.with_suffix('.result.json'), result)
    except Exception as error:
        traceback.print_exc()
        atomic_json(request.with_suffix('.result.json'), {'error': str(error)})
        raise SystemExit(1)


if __name__ == '__main__':
    main()
