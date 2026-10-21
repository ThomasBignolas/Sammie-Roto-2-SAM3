"""Trained-checkpoint acceptance test. Run: .venv/Scripts/python -m scripts.validate_sam3.

Uses the same process bridge, backend, matting and export code as the application.
Requires authorized official checkpoints. This script never substitutes random weights.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import av
import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication, QWidget, QSlider
from sammie import core
from sammie.backends.sam3 import Sam3Backend
from sammie.export_formats import ExportSettings
from sammie.export_workers import SequenceExportWorker
from sammie.matting import MatAnyManager
from sammie.settings_manager import initialize_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['SAM 3', 'SAM 3.1'], default='SAM 3')
    parser.add_argument('--video', default='examples/example_dance.mp4')
    parser.add_argument('--text', default='person')
    parser.add_argument('--frames', type=int, default=6)
    args = parser.parse_args()
    if args.frames < 4:
        parser.error('At least four frames are required to test corrections')
    app = QApplication.instance() or QApplication([])
    root = Path('.test-output').resolve()/('trained-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    core.temp_dir, core.frames_dir = str(root), str(root/'frames')
    core.mask_dir, core.matting_dir = str(root/'masks'), str(root/'matting')
    Path(core.frames_dir).mkdir()
    settings = initialize_settings(str(root))
    settings.set_session_setting('matany_model', 'MatAnyone')
    settings.set_session_setting('matany_res', 720)
    settings.set_session_setting('sequence_start_frame', 1001)
    with av.open(args.video) as container:
        rate = container.streams.video[0].average_rate
        for i, frame in enumerate(container.decode(video=0)):
            if i >= args.frames:
                break
            image = cv2.resize(frame.to_ndarray(format='bgr24'), (1920, 1080))
            if not cv2.imwrite(str(root/'frames'/f'{i:05d}.png'), image):
                raise OSError('Could not write test frame')
    frames = len(list((root/'frames').glob('*.png')))
    if frames < 4:
        raise ValueError('The video must contain at least four frames')
    core.VideoInfo.width, core.VideoInfo.height = 1920, 1080
    core.VideoInfo.total_frames = frames
    core.VideoInfo.fps, core.VideoInfo.fps_rational = float(rate), str(rate)
    core.DeviceManager.setup_device()
    backend = Sam3Backend()
    matting = MatAnyManager()
    results = {'model': args.model, 'trained_checkpoint': True, 'passed': False,
               'fixture': args.video, 'frames': frames, 'source_size': [1920, 1080]}
    started = time.perf_counter()
    try:
        backend.load_segmentation_model(args.model)
        backend.initialize_predictor()
        results['load_seconds'] = time.perf_counter()-started
        # Box requests have an independent fresh session, as in Clear All in the UI.
        boxes = backend.discover(0, box=[480, 200, 1440, 1000])
        if not boxes:
            raise AssertionError('No box detections. Inspect the fixture or choose a suitable video.')
        results['box_detections'] = len(boxes)
        backend.reset_session()
        found = backend.discover(0, text=args.text)
        if not found:
            raise AssertionError('No text detections. Inspect the fixture or change --text.')
        results['text_detections'] = len(found)
        points = backend.accept_candidates(list(found)[:2])
        if len(points) < 2:
            backend.segment_image(0, 1, [[350, 600]], [1])
            points = list(backend.journal)
        started = time.perf_counter()
        seen = []
        for frame, masks in backend.iter_masks(0, frames-1):
            assert set(masks) == {0, 1}
            backend.write_masks(frame, masks)
            seen.append(frame)
        assert seen == list(range(frames))
        results['tracking_seconds'] = time.perf_counter()-started
        results['frames_per_second'] = frames/results['tracking_seconds']
        middle = frames//2
        untouched = (root/'masks'/f'{middle:05d}'/'1.png').read_bytes()
        mask = cv2.imread(str(root/'masks'/f'{middle:05d}'/'0.png'), 0)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise AssertionError('Object 0 is absent on the correction frame; inspect test footage.')
        center = len(xs)//2
        backend.segment_image(middle, 0, [[int(xs[center]), int(ys[center])], [10, 10]], [1, 0])
        assert (root/'masks'/f'{middle:05d}'/'1.png').read_bytes() == untouched
        points = list(backend.journal)
        for frame, masks in backend.iter_masks(middle, frames-1-middle):
            backend.write_masks(frame, masks)
        backward = []
        for frame, masks in backend.iter_masks(middle, middle, reverse=True):
            backend.write_masks(frame, masks)
            backward.append(frame)
        assert backward == list(range(middle-1, -1, -1))
        results['sam_gpu'] = dict(backend.gpu)
        backend.offload_model_to_cpu()
        window = QWidget()
        window.frame_slider = QSlider(window)
        window.frame_slider.setRange(0, frames-1)
        torch.cuda.reset_peak_memory_stats()
        assert matting.load_matting_model(parent_window=window)
        assert matting.run_matting(points, window) == 1
        for frame in range(frames):
            for oid in (0, 1):
                matte = cv2.imread(str(root/'matting'/f'{frame:05d}'/f'{oid}.png'), 0)
                assert matte is not None and matte.shape == (1080, 1920)
        results['matting_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        matting.unload_matting_model()
        for output in ('Segmentation-Alpha', 'Matting-Alpha'):
            destination = root/output
            export = ExportSettings('png', str(destination), 'mask', output,
                                    -1, False, 14, False, None, None)
            SequenceExportWorker(export, points, frames, 'mask',
                                 SimpleNamespace(settings_mgr=settings))._export_png_sequence()
            for frame in range(frames):
                image = cv2.imread(str(destination/f'mask.{1001+frame:04d}.png'), cv2.IMREAD_UNCHANGED)
                assert image is not None and image.shape == (1080, 1920, 4)
        # Restoring after matting must recover the selected identities and all prompts.
        backend.replay_points(points)
        assert backend.journal == points and set(backend.id_map.values()) == {0, 1}
        results['passed'] = True
    except Exception as error:
        results['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        backend.unload_segmentation_model()
        matting.unload_matting_model()
        (root/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(results, indent=2))
        print('Inspect masks and report in:', root)


if __name__ == '__main__':
    main()
