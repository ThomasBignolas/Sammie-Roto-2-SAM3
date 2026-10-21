import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from fractions import Fraction
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import av
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox
from sammie import core, sammie
from sammie.settings_manager import initialize_settings


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path('.test-output').resolve()/('media-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.session = self.root/'session'
        self.settings = initialize_settings(str(self.session))
        self.patches = [patch.object(core, 'temp_dir', str(self.session)),
                        patch.object(core, 'frames_dir', str(self.session/'frames')),
                        patch.object(core, 'mask_dir', str(self.session/'masks')),
                        patch.object(core, 'matting_dir', str(self.session/'matting'))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('media-')
        shutil.rmtree(self.root)

    def test_import_frame_order_count_and_rational_rate(self):
        clip = self.root/'sequence.mkv'
        with av.open(str(clip), 'w') as container:
            stream = container.add_stream('ffv1', rate=Fraction(24000, 1001))
            stream.width, stream.height, stream.pix_fmt = 80, 48, 'yuv420p'
            for i in range(5):
                image = np.full((48, 80, 3), 30+i*35, np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.assertEqual(sammie.load_video(str(clip), None), 5)
        self.assertEqual(Fraction(core.VideoInfo.fps_rational), Fraction(24000, 1001))
        self.assertEqual(self.settings.get_session_setting('video_fps_rational'), '24000/1001')
        files = sorted(Path(core.frames_dir).glob('*.png'))
        self.assertEqual([f.name for f in files], [f'{i:05d}.png' for i in range(5)])
        means = [cv2.imread(str(f)).mean() for f in files]
        self.assertEqual(means, sorted(means))
        self.assertTrue((self.session/'source_timing.json').is_file())

    def test_sam3_controls_hide_for_sam2_and_box_does_not_add_click(self):
        from sammie.sam3_controls import Sam3Controls
        selector = QComboBox()
        selector.addItems(['Base', 'SAM 3', 'SAM 3.1'])
        controls = Sam3Controls(selector)
        self.assertTrue(controls.isHidden())
        selector.setCurrentText('SAM 3')
        self.assertFalse(controls.isHidden())
        controls.box.setChecked(True)
        self.assertIs(controls.take_box_click(10, 20), False)
        self.assertEqual(controls.take_box_click(100, 200), [10, 20, 100, 200])
        self.assertFalse(controls.box.isChecked())
        selector.setCurrentText('Base')
        self.assertTrue(controls.isHidden())
        controls.close()

    def test_seed_points_round_trip_without_losing_metadata(self):
        from sammie.backends.masks import pack_mask
        points = [{'kind': 'sam3_seed', 'frame': 0, 'object_id': 3, 'x': 10, 'y': 12,
                   'positive': True, 'query': {'type': 'add_prompt', 'frame_index': 0, 'text': 'car'},
                   'seed_mask': pack_mask(np.eye(8))}]
        self.settings.save_points(points)
        self.assertEqual(self.settings.load_points(), points)

    def test_source_proxy_and_mask_restore_4k(self):
        from sammie.backends.sam3 import Sam3Backend
        from sammie.backends.masks import binary_mask
        core.VideoInfo.width, core.VideoInfo.height, core.VideoInfo.total_frames = 3840, 2160, 2
        Path(core.frames_dir).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(Path(core.frames_dir)/f'{i:05d}.png'), np.full((2160, 3840, 3), i*80, np.uint8))
        backend = Sam3Backend()
        proxy = backend._prepare_frames(1920)
        self.assertEqual(cv2.imread(str(proxy/'00000.png')).shape, (1080, 1920, 3))
        mask = binary_mask(np.eye(4), 3840, 2160)
        backend.write_masks(1, {0: mask})
        self.assertEqual(cv2.imread(str(Path(core.mask_dir)/'00001/0.png'), 0).shape, (2160, 3840))
        self.assertEqual(cv2.imread(str(Path(core.frames_dir)/'00001.png')).shape, (2160, 3840, 3))
        self.assertEqual(backend._prepare_frames(1920), proxy)


if __name__ == '__main__':
    unittest.main()
