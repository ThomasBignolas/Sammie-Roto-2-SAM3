"""Reproducible real SAM2 + MatAnyone regression. Run from the repository root."""
import json
import os
from pathlib import Path
import time
import uuid
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import av
import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication, QWidget, QSlider
from sammie import core
from sammie.backends.sam2 import Sam2Backend
from sammie.matting import MatAnyManager
from sammie.settings_manager import initialize_settings


def main():
    app = QApplication.instance() or QApplication([])
    root = Path('.test-output') / ('gpu-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    core.temp_dir = str(root)
    core.frames_dir = str(root/'frames')
    core.mask_dir = str(root/'masks')
    core.matting_dir = str(root/'matting')
    Path(core.frames_dir).mkdir()
    settings = initialize_settings(str(root))
    settings.set_session_setting('matany_model', 'MatAnyone')
    settings.set_session_setting('matany_res', 720)
    frames = 6
    with av.open('examples/example_dance.mp4') as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == frames:
                break
            image = cv2.resize(frame.to_ndarray(format='bgr24'), (1920, 1080))
            cv2.imwrite(str(root/'frames'/f'{i:05d}.png'), image)
    core.VideoInfo.width, core.VideoInfo.height = 1920, 1080
    core.VideoInfo.total_frames, core.VideoInfo.fps = frames, 24
    core.DeviceManager.setup_device()
    torch.cuda.reset_peak_memory_stats()
    backend = Sam2Backend()
    started = time.perf_counter()
    assert backend.load_segmentation_model('Base')
    backend.initialize_predictor()
    load_seconds = time.perf_counter()-started
    points = [{'frame': 0, 'object_id': 0, 'positive': True, 'x': 960, 'y': 540},
              {'frame': 0, 'object_id': 1, 'positive': True, 'x': 350, 'y': 600}]
    for point in points:
        backend.segment_image(0, point['object_id'], [[point['x'], point['y']]], [1])
    started = time.perf_counter()
    first = {}
    for index, masks in backend.iter_masks(0, frames-1):
        backend.write_masks(index, masks)
        first[index] = {oid: mask.copy() for oid, mask in masks.items()}
    assert sorted(first) == list(range(frames))
    tracking_seconds = time.perf_counter()-started
    before = (root/'masks/00003/1.png').read_bytes()
    backend.segment_image(3, 0, [[970, 550], [10, 10]], [1, 0])
    assert before == (root/'masks/00003/1.png').read_bytes(), 'Correction overwrote another object'
    backwards = list(backend.iter_masks(3, 3, reverse=True))
    assert [f for f, _ in backwards] == [3, 2, 1, 0]
    for f, masks in backwards:
        backend.write_masks(f, masks)
    preview = backend.preview_point(0, 0, points, 970, 540, True)
    assert preview is not None and preview.shape == (1080, 1920)
    backend.replay_points(points)
    for f, masks in backend.iter_masks(0, frames-1):
        # Exact regression for the same prompt sequence, not just shape checks.
        for oid, mask in masks.items():
            assert np.array_equal(mask, first[f][oid]), f'Replay changed frame {f}, object {oid}'
        backend.write_masks(f, masks)
    sam_peak = torch.cuda.max_memory_allocated() / 2**20
    backend.unload_segmentation_model()
    torch.cuda.reset_peak_memory_stats()
    window = QWidget()
    window.frame_slider = QSlider(window)
    window.frame_slider.setRange(0, frames-1)
    matting = MatAnyManager()
    started = time.perf_counter()
    assert matting.load_matting_model(parent_window=window)
    assert matting.run_matting(points[:1], window) == 1
    matting_seconds = time.perf_counter()-started
    for i in range(frames):
        matte = cv2.imread(str(root/'matting'/f'{i:05d}'/'0.png'), 0)
        assert matte is not None and matte.shape == (1080, 1920)
    results = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'source': 'example_dance.mp4 first six frames resized to 1920x1080',
               'sam2': {'model': 'Base', 'load_seconds': load_seconds, 'tracking_seconds': tracking_seconds,
                        'frames_per_second': frames/tracking_seconds, 'peak_allocated_mib': sam_peak,
                        'objects': 2, 'forward_backward_correction_preview_replay': 'passed'},
               'matanyone': {'working_short_edge': 720, 'seconds_including_load': matting_seconds,
                            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                            'source_size_alpha_frames': frames},
               'sam3_trained_weights': 'NOT TESTED: gated checkpoint access unavailable'}
    matting.unload_matting_model()
    (root/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('Artifacts:', root)


if __name__ == '__main__':
    main()
