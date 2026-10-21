"""Diagnostic A/B/C benchmark on cached VideoMaMa alpha; optional ground truth.

Example: .venv/Scripts/python.exe scripts/benchmark_vitmatte.py --session temp --frames 6
Use a COPY of a session: this intentionally replaces its ViTMatte cache, never its
VideoMaMa PNGs. Ground truth: <ground-truth>/<00000>/<object>.png or .npy alpha.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from sammie.refinement.algorithms import RefinementSettings, PRESETS, alpha_float
from sammie.refinement.cache import load_base
from sammie.refinement.pipeline import run_refinement


def extract_project(project, root, start, count):
    """Read selected frames into an isolated benchmark session, never temp/."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    wanted = set(range(start, start+count))
    with zipfile.ZipFile(project) as archive:
        for name in archive.namelist():
            bits = name.split('/')
            select = name in ('session_settings.conf', 'points.json')
            if bits[0] == 'frames' and len(bits) == 2 and Path(name).stem.isdigit():
                select = int(Path(name).stem) in wanted
            if bits[0] in ('matting', 'matting_float', 'masks') and len(bits) == 3 and bits[1].isdigit():
                select = int(bits[1]) in wanted
            if select:
                destination = (root/name).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError('Unsafe project archive member')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, destination.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output)
    return root


def metrics(base, alpha, unknown, previous=None, previous_base=None, truth=None):
    diff = np.abs(alpha-base)
    result = {'alpha_mae_from_baseline': float(diff.mean()),
              'edge_mae_from_baseline': float(diff[unknown].mean()) if unknown.any() else 0,
              'unknown_fraction': float(unknown.mean())}
    if previous is not None:
        result['temporal_variation_unaligned'] = float(np.abs(alpha-previous).mean())
        result['refinement_delta_variation_unaligned'] = float(np.abs((alpha-base)-(previous-previous_base)).mean())
    if truth is not None:
        error = alpha-truth
        result['sad'] = float(np.abs(error).sum()/1000)
        result['mse'] = float(np.square(error).mean())
        def grad(a):
            a = cv2.GaussianBlur(a, (0, 0), 1.4)
            return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
        result['gradient_mae_sobel_sigma_1_4'] = float(np.abs(grad(alpha)-grad(truth)).mean())
    return result


def visual(rgb, stages, path, crop=None):
    panels = []
    for label, a in stages.items():
        composite = (rgb.astype(np.float32)*a[..., None] + np.array([80, 90, 110])*(1-a[..., None])).clip(0, 255).astype(np.uint8)
        if crop:
            x1, y1, x2, y2 = crop
            composite = composite[y1:y2, x1:x2]
        height = max(1, round(composite.shape[0]*512/composite.shape[1]))
        panel = cv2.resize(composite, (512, height))
        cv2.putText(panel, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        panels.append(panel)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))


def benchmark(root, output, start, count, settings, truth_dir=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session_file = root/'session_settings.conf'
    extension = json.loads(session_file.read_text()).get('frame_format', 'png') if session_file.exists() else 'png'
    report = {'kind': 'diagnostic; differences and variation are not accuracy measures',
              'hardware': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
              'torch': torch.__version__, 'runs': {}, 'metrics': {}}
    # Baseline is already cached: inference timing must not be invented.
    report['runs']['VideoMaMa only'] = {'inference_seconds': None,
        'note': 'Existing cached mattes. VideoMaMa was not run by this utility.'}
    for temporal, label in [(False, 'VideoMaMa + ViTMatte'), (True, 'VideoMaMa + ViTMatte + temporal')]:
        print(label, flush=True)
        config = replace(settings, temporal=temporal)
        r = run_refinement(root, range(start, start+count), config, extension=extension,
                           progress=lambda i,n,t: print(f'{i}/{n} {t}', flush=True))
        r['seconds_per_object_frame'] = r['seconds']/len(r['frames'])
        report['runs'][label] = r
        previous, previous_base = {}, {}
        measurements = []
        baseline_measurements = []
        for entry in r['frames']:
            frame, oid = entry['frame'], entry['object']
            source = root/'frames'/f'{frame:05d}.{extension}'
            png = root/'matting'/f'{frame:05d}'/f'{oid}.png'
            base, _ = load_base(root, frame, oid, source, png)
            with np.load(root/'vitmatte'/f'{frame:05d}'/f'{oid}.npz') as data:
                a = data['validated' if temporal else 'refined']
                unknown = data['trimap'] == 128
                truth = None
                if truth_dir:
                    p = Path(truth_dir)/f'{frame:05d}'/f'{oid}.npy'
                    truth = np.load(p) if p.exists() else cv2.imread(str(p.with_suffix('.png')), cv2.IMREAD_UNCHANGED)
                    if truth is None or truth.shape != base.shape:
                        raise ValueError(f'Missing or misaligned ground truth for {frame}/{oid}')
                    truth = alpha_float(truth)
                m = metrics(base, a, unknown, previous.get(oid), previous_base.get(oid), truth)
                measurements.append(dict(frame=frame, object=oid, **m))
                baseline_measurements.append(dict(frame=frame, object=oid,
                    **metrics(base, base, unknown, previous_base.get(oid), previous_base.get(oid), truth)))
                previous[oid], previous_base[oid] = a, base
                if temporal and frame in (start, start+count//2):
                    rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
                    stages = {'A Original VideoMaMa': base, 'B ViTMatte': data['refined'], 'C Validated': a}
                    visual(rgb, stages, output/f'comparison-{frame:05d}-{oid}.jpg')
                    # First occupied boundary gives a reproducible detail crop, no hand-selected winner.
                    ys, xs = np.where(unknown)
                    if len(xs):
                        y = int(np.percentile(ys, 5)); nearby = xs[np.abs(ys-y) < 20]
                        x = int(np.median(nearby)) if len(nearby) else int(np.median(xs))
                        crop = max(0,x-400), max(0,y-250), min(rgb.shape[1],x+400), min(rgb.shape[0],y+350)
                        visual(rgb, stages, output/f'edge-comparison-{frame:05d}-{oid}.jpg', crop)
        report['metrics'][label] = measurements
        report['metrics']['VideoMaMa only'] = baseline_measurements
    if truth_dir:
        report['kind'] = 'Ground-truth SAD, MSE, custom Sobel gradient error; other measures remain diagnostic'
    (output/'benchmark.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--session', help='Existing copied session; its refinement cache is replaced')
    source.add_argument('--project', help='Read-only .sammie input, selected frames extracted under output/input-session')
    p.add_argument('--output', default='.test-output/vitmatte-benchmark')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--preset', choices=PRESETS, default='Balanced')
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--ground-truth')
    args = p.parse_args()
    if args.frames < 1 or args.start < 0:
        p.error('Frame count must be positive and start must be nonnegative')
    if args.project:
        args.session = extract_project(args.project, Path(args.output)/'input-session', args.start, args.frames)
    s = replace(RefinementSettings(), **PRESETS[args.preset], device=args.device, precision=args.precision)
    r = benchmark(args.session, args.output, args.start, args.frames, s, args.ground_truth)
    print(json.dumps({k:{f:v for f,v in run.items() if f not in ('frames','settings')} for k,run in r['runs'].items()}, indent=2))
