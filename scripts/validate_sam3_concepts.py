"""Real multi-concept SAM video validation on a copied project frame sequence.

Run with the application Python: -m scripts.validate_sam3_concepts --model "SAM 3"
User source frames are read only. All outputs stay in an isolated .test-output folder.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import uuid
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.backends.sam3 import Sam3Backend
from sammie.backends.masks import unpack_mask
from sammie.settings_manager import initialize_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['SAM 3', 'SAM 3.1'], default='SAM 3')
    parser.add_argument('--frames', default='.test-output/vitmatte-content/frames')
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    root = Path('.test-output').resolve()/('concepts-'+uuid.uuid4().hex[:8])
    (root/'frames').mkdir(parents=True)
    for i in range(4):
        shutil.copy2(Path(args.frames)/f'{i:05d}.png', root/'frames'/f'{i:05d}.png')
    core.temp_dir, core.frames_dir, core.mask_dir = str(root), str(root/'frames'), str(root/'masks')
    settings = initialize_settings(str(root))
    settings.app_settings_file = str(root/'app.conf')
    settings.set_session_setting('sam3_working_edge', 1920)
    source = cv2.imread(str(root/'frames/00000.png'))
    core.VideoInfo.height, core.VideoInfo.width = source.shape[:2]
    core.VideoInfo.total_frames = 4
    backend = Sam3Backend()
    report = {'model': args.model, 'frames': 4, 'source_dimensions': list(source.shape[:2][::-1]), 'passed': False}
    try:
        backend.load_segmentation_model(args.model)
        backend.initialize_predictor()
        reference = {}
        for phrase in ('person', 'camera'):
            found = backend.discover(0, text=phrase)
            assert found, f'No {phrase} in this fixture'
            reference[phrase] = list(found.values())
        found = backend.discover(0, text='person and camera')
        report['detections'] = dict(backend.candidate_labels)
        records = backend.accept_candidates(list(found))
        assert {r['query']['text'] for r in records} == {'person', 'camera'}
        report['individual_vs_combined_seed_iou'] = []
        for record in records:
            a = unpack_mask(record['seed_mask'])
            iou = max(np.count_nonzero(a & (b > 0))/max(1, np.count_nonzero(a | (b > 0)))
                      for b in reference[record['query']['text']])
            assert iou > .99
            report['individual_vs_combined_seed_iou'].append(iou)
        started = time.perf_counter()
        areas = []
        for frame, masks in backend.iter_masks(0, 3):
            assert set(masks) == {r['object_id'] for r in records}
            assert all(m.shape == source.shape[:2] and m.any() for m in masks.values())
            backend.write_masks(frame, masks)
            areas.append({str(oid): int(np.count_nonzero(m)) for oid, m in masks.items()})
        report['tracking_seconds'] = time.perf_counter()-started
        report['areas_by_frame'] = areas
        person = next(r['object_id'] for r in records if r['query']['text'] == 'person')
        camera = next(r['object_id'] for r in records if r['query']['text'] == 'camera')
        untouched = (root/'masks/00001'/f'{person}.png').read_bytes()
        mask = cv2.imread(str(root/'masks/00001'/f'{camera}.png'), 0)
        _, _, _, point = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 3))
        backend.segment_image(1, camera, [list(point), [1, 1]], [1, 0])
        assert (root/'masks/00001'/f'{person}.png').read_bytes() == untouched
        for frame, masks in backend.iter_masks(1, 2):
            assert all(mask.any() for mask in masks.values())
            backend.write_masks(frame, masks)
        assert [f for f, _ in backend.iter_masks(2, 2, reverse=True)] == [1, 0]
        saved = json.loads(json.dumps(backend.journal))
        report['gpu_before_offload'] = dict(backend.gpu)
        backend.offload_model_to_cpu()
        backend.replay_points(saved)
        assert set(backend.id_map.values()) == {r['object_id'] for r in records}
        assert len(list(backend.iter_masks(1, 1))) == 2
        report['gpu'] = dict(backend.gpu)
        # Additive selection and cancellation use the same committed tracker.
        first = [copy.deepcopy(r) for r in records if r['query']['text'] == 'person']
        backend.replay_points(first)
        old_map, old_journal = dict(backend.id_map), copy.deepcopy(backend.journal)
        backend.discover(0, text='camera')
        backend.accept_candidates([])
        assert backend.id_map == old_map and backend.journal == old_journal
        added = backend.accept_candidates(list(backend.discover(0, text='camera')))
        assert added and len(backend.journal) == len(first)+len(added)
        assert len(list(backend.iter_masks(0, 1))) == 2
        # Comma syntax finds the same two concepts after reset.
        backend.reset_session()
        found = backend.discover(0, text='person, camera')
        assert {value[0]['text'] for value in backend.candidates.values()} == {'person', 'camera'}
        report['comma_detections'] = len(found)
        report['passed'] = True
    finally:
        backend.unload_segmentation_model()
        (root/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        print('Artifacts:', root, flush=True)


if __name__ == '__main__':
    main()
