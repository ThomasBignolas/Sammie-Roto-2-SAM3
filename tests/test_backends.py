import contextlib
import copy
import os
from pathlib import Path
import tempfile
import uuid
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.backends.masks import binary_mask, normalize_box, normalize_points, pack_mask, unpack_mask, match_seeds
from sammie.backends.sam3 import Sam3Backend
from sammie.settings_manager import initialize_settings


class MaskTests(unittest.TestCase):
    def test_xywh_normalization_and_negative_points(self):
        self.assertEqual(normalize_box([90, 45, 10, 5], 100, 50), [.1, .1, .8, .8])
        self.assertEqual(normalize_points([[50, 25]], 100, 50), [[.5, .5]])
        with self.assertRaises(ValueError):
            normalize_box([1, 1, 1, 5], 100, 50)
        with self.assertRaises(ValueError):
            normalize_points([[100, 0]], 100, 50)

    def test_mask_shape_dtype_and_roundtrip(self):
        mask = np.zeros((1, 3, 5), bool)
        mask[0, 1, 2] = True
        out = binary_mask(mask, 20, 12)
        self.assertEqual(out.shape, (12, 20))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(unpack_mask(pack_mask(out)), out > 0)
        self.assertEqual(set(np.unique(out)), {0, 255})

    def test_identity_survives_reordered_detections(self):
        a, b = np.eye(8, dtype=bool), np.fliplr(np.eye(8, dtype=bool))
        seeds = [{'object_id': 3, 'seed_mask': pack_mask(a)}, {'object_id': 8, 'seed_mask': pack_mask(b)}]
        self.assertEqual(match_seeds(seeds, {99: b, 12: a}), {12: 3, 99: 8})
        with self.assertRaises(RuntimeError):
            match_seeds(seeds, {99: np.zeros_like(a)})


class FakeWorker:
    def __init__(self, folder):
        self.folder = folder
        self.calls = []
        self.closed = False
        self.stream_closed = False
        self.masks = {5: np.pad(np.ones((8, 8), np.uint8)*255, ((4, 12), (4, 28))),
                      8: np.pad(np.ones((8, 8), np.uint8)*255, ((4, 12), (20, 12)))}

    def result(self, frame, masks):
        paths = {}
        for oid, mask in masks.items():
            path = self.folder / f'{oid}.png'
            cv2.imwrite(str(path), mask)
            paths[str(oid)] = str(path)
        return {'frame': frame, 'masks': paths, 'gpu': {'peak_mib': 123}}

    def call(self, command, **kwargs):
        self.calls.append((command, copy.deepcopy(kwargs)))
        if command == 'start':
            return {'frames': 4, 'width': 40, 'height': 24}
        request = kwargs['request']
        if request['type'] == 'add_prompt':
            masks = self.masks
            if 'points' in request:
                masks = {request['obj_id']: next(iter(self.masks.values()))}
            return self.result(request['frame_index'], masks)
        return {'is_success': True}

    def stream(self, command, **kwargs):
        self.calls.append((command, kwargs))
        request = kwargs['request']
        start = request['start_frame_index']
        reverse = request['propagation_direction'] == 'backward'
        count = request['max_frame_num_to_track']
        end = (0 if reverse else 3) if count is None else start + (-count if reverse else count)
        try:
            for frame in range(start-1 if reverse else start, end + (-1 if reverse else 1), -1 if reverse else 1):
                # Include disappearance to test stale mask invalidation.
                yield self.result(frame, {} if frame == 2 else self.masks)
        finally:
            self.stream_closed = True

    def close(self):
        self.closed = True


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        Path('.test-output').mkdir(exist_ok=True)
        self.root = Path('.test-output').resolve() / ('test-' + uuid.uuid4().hex)
        self.root.mkdir()
        self.patches = [patch.object(core, 'temp_dir', str(self.root)),
                        patch.object(core, 'frames_dir', str(self.root/'frames')),
                        patch.object(core, 'mask_dir', str(self.root/'masks'))]
        for item in self.patches:
            item.start()
        self.settings = initialize_settings(str(self.root))
        core.VideoInfo.width, core.VideoInfo.height, core.VideoInfo.total_frames = 80, 48, 4
        Path(core.frames_dir).mkdir()
        for i in range(4):
            cv2.imwrite(str(Path(core.frames_dir)/f'{i:05d}.png'), np.full((48, 80, 3), i*40, np.uint8))
        self.backend = Sam3Backend()
        self.worker = FakeWorker(self.root)
        self.backend.predictor = self.worker
        self.backend.loaded_model_name = 'SAM 3'
        self.backend.initialize_predictor()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('test-')
        shutil.rmtree(self.root)

    def select(self):
        self.backend.discover(0, text='person')
        return self.backend.accept_candidates([5, 8])

    def test_text_multiple_instances_selection_and_pruning(self):
        self.backend.discover(0, text='person')
        records = self.backend.accept_candidates([8])
        self.assertEqual(records[0]['object_id'], 0)
        self.assertEqual(self.backend.id_map, {8: 0})
        prune = [x[1]['request'] for x in self.worker.calls if x[0] == 'request' and x[1]['request']['type'] == 'remove_object']
        self.assertEqual(prune[0]['obj_id'], 5)
        self.assertFalse(prune[0]['is_user_action'])
        mask = cv2.imread(str(self.root/'masks/00000/0.png'), 0)
        self.assertEqual(mask.shape, (48, 80))

    def test_box_uses_official_normalized_xywh(self):
        self.backend.discover(0, box=[8, 12, 40, 36])
        req = self.worker.calls[-1][1]['request']
        self.assertEqual(req['bounding_boxes'], [[.1, .25, .4, .5]])

    def test_correction_preserves_other_object_and_restarts_tracking(self):
        self.select()
        untouched = (self.root/'masks/00000/1.png').read_bytes()
        self.backend.segment_image(2, 0, [[20, 10], [30, 10]], [1, 0])
        req = self.worker.calls[-1][1]['request']
        self.assertEqual(req['obj_id'], 5)
        self.assertEqual(req['point_labels'], [1, 0])
        self.assertEqual((self.root/'masks/00000/1.png').read_bytes(), untouched)
        out = list(self.backend.iter_masks(2, 2, reverse=True))
        self.assertEqual([x[0] for x in out], [1, 0])
        self.assertTrue(self.worker.stream_closed)
        absent = list(self.backend.iter_masks(2, 1))[0][1]
        self.assertFalse(absent[0].any())
        self.assertEqual(set(out[0][1]), {0, 1})
        self.assertTrue(self.worker.calls[-1][1]['request']['force_tracker_propagation'])

    def test_replay_uses_mask_matching_and_keeps_points_out_of_seeds(self):
        records = self.select()
        self.backend.journal = []
        self.worker.masks = {27: self.worker.masks[8], 20: self.worker.masks[5]}
        self.backend.replay_points(records)
        self.assertEqual(self.backend.id_map, {20: 0, 27: 1})
        points = core.PointManager()
        points.points = records + [{'frame': 0, 'object_id': 0, 'x': 9, 'y': 9, 'positive': False}]
        self.assertEqual(points.get_sam2_points(0, 0), ([[9, 9]], [0]))

    def test_offload_closes_worker_but_retains_prompts(self):
        records = self.select()
        self.backend.offload_model_to_cpu()
        self.assertTrue(self.worker.closed)
        self.assertIsNone(self.backend.predictor)
        self.assertEqual(self.backend.journal, records)

    def test_cancel_closes_stream_and_keeps_session(self):
        self.select()
        iterator = self.backend.iter_masks(0, 3)
        next(iterator)
        iterator.close()
        self.assertTrue(self.worker.stream_closed)
        self.assertFalse(self.worker.closed)

    def test_missing_frame_is_rejected(self):
        (Path(core.frames_dir)/'00002.png').unlink()
        with self.assertRaises(RuntimeError):
            self.backend._prepare_frames(1920)

    def test_png_export_preserves_dimensions_numbering_and_alpha(self):
        from sammie.export_workers import SequenceExportWorker
        from sammie.export_formats import ExportSettings
        records = self.select()
        for frame, masks in self.backend.iter_masks(0, 3):
            self.backend.write_masks(frame, masks)
        self.settings.set_session_setting('sequence_start_frame', 1001)
        settings = ExportSettings('png', str(self.root/'export'), 'test', 'Segmentation-Alpha',
                                  -1, False, 14, True, 1, 3)
        worker = SequenceExportWorker(settings, records, 4, 'mask', SimpleNamespace(settings_mgr=self.settings))
        worker._export_png_sequence()
        files = sorted((self.root/'export').glob('*.png'))
        self.assertEqual([f.name for f in files], ['mask.1002.png', 'mask.1003.png', 'mask.1004.png'])
        arrays = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in files]
        self.assertTrue(all(a.shape == (48, 80, 4) for a in arrays))
        self.assertFalse(arrays[1][:, :, 3].any())

    def test_exr_keeps_blank_frames_and_identical_channels(self):
        import OpenEXR
        import Imath
        from sammie.export_workers import SequenceExportWorker
        from sammie.export_formats import ExportSettings
        records = self.select()
        for frame, masks in self.backend.iter_masks(0, 3):
            self.backend.write_masks(frame, masks)
        self.settings.set_session_setting('sequence_start_frame', 1001)
        output = self.root/'export'
        settings = ExportSettings('exr', str(output), 'mask', 'Segmentation-Matte',
                                  -1, False, 14, False, None, None)
        output.mkdir()
        SequenceExportWorker(settings, records, 4, 'mask',
                             SimpleNamespace(settings_mgr=self.settings))._export_exr_sequence()
        self.assertEqual(len(list(output.glob('*.exr'))), 4)
        for frame in range(4):
            exr = OpenEXR.InputFile(str(output/f'mask.{1001+frame:04d}.exr'))
            try:
                header = exr.header()
                self.assertEqual(set(header['channels']), {'Object_0.Y', 'Object_1.Y'})
                bounds = header['dataWindow']
                self.assertEqual((bounds.max.x+1, bounds.max.y+1), (80, 48))
                if frame == 2:
                    blank = np.frombuffer(exr.channel('Object_0.Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
                    self.assertFalse(blank.any())
            finally:
                exr.close()

    def test_video_export_preserves_frame_count_and_exact_rate(self):
        import av
        from fractions import Fraction
        from sammie.export_workers import VideoExportWorker
        from sammie.export_formats import ExportSettings
        records = self.select()
        for frame, masks in self.backend.iter_masks(0, 3):
            self.backend.write_masks(frame, masks)
        core.VideoInfo.fps_rational = '24000/1001'
        core.VideoInfo.fps = 24000/1001
        output = self.root/'export.mkv'
        settings = ExportSettings('ffv1', str(self.root), 'mask', 'Segmentation-Alpha',
                                  -1, False, 14, False, None, None)
        worker = VideoExportWorker(settings, records, 4, [str(output)])
        worker._export_single(str(output), -1)
        with av.open(str(output)) as container:
            self.assertEqual(container.streams.video[0].average_rate, Fraction(24000, 1001))
            frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.width, f.height) == (80, 48) for f in frames))
            self.assertFalse(frames[2].to_ndarray(format='rgba')[:, :, 3].any())

    def test_error_releases_worker_and_retains_committed_prompts(self):
        records = self.select()
        with patch.object(self.worker, 'call', side_effect=RuntimeError('CUDA out of memory')):
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                self.backend.segment_image(2, 0, [[20, 10]], [1])
        self.assertTrue(self.worker.closed)
        self.assertIsNone(self.backend.predictor)
        self.assertEqual(self.backend.journal, records)

    def test_reset_requires_prompt_replay(self):
        records = self.select()
        self.backend.reset_session()
        self.assertEqual(self.backend.journal, [])
        self.backend.replay_points(records)
        self.assertEqual(set(self.backend.id_map.values()), {0, 1})


if __name__ == '__main__':
    unittest.main()
