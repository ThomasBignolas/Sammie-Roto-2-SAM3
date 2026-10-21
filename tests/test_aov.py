import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid
import weakref
from unittest.mock import patch
import cv2
import numpy as np
import OpenEXR
import Imath
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QApplication
from sammie import core, sammie
from sammie.settings_manager import initialize_settings
from sammie.aov.config import AOVSettings, resize_rgb, pad_rgb, unit_normals, windows, align_depth
from sammie.aov.cache import active, read_frame
from sammie.aov.pipeline import run_job
from sammie.aov.export import channel_data, export_sequence


class GeometryTests(unittest.TestCase):
    def test_padding_preserves_pixels_and_aspect(self):
        rgb = np.arange(71*113*3, dtype=np.uint8).reshape(71, 113, 3)
        padded = pad_rgb(rgb, 64)
        self.assertEqual(padded.shape, (128, 128, 3))
        np.testing.assert_array_equal(padded[:71, :113], rgb)
        self.assertEqual(resize_rgb(rgb, 56).shape, (35, 56, 3))

    def test_windows_cover_short_exact_and_long_clips(self):
        for length in (1, 5, 14, 17, 18, 194):
            spans = windows(length, 17, 5)
            self.assertEqual(set(i for a,b in spans for i in range(a,b)), set(range(length)))
            for previous, current in zip(spans, spans[1:]):
                self.assertLess(current[0], previous[1])
                self.assertGreater(current[1], previous[1])

    def test_affine_alignment_and_flat_depth(self):
        values = np.linspace(.1, .8, 100).reshape(10, 10)
        scale, shift = align_depth(values, values*1.2+.1)
        np.testing.assert_allclose(values*scale+shift, values*1.2+.1, atol=1e-7)
        self.assertEqual(align_depth(np.ones((3,3)), np.ones((3,3))*2), (1., 1.))

    def test_normal_upsampling_unit_vectors_no_implicit_axis_flip(self):
        vectors = np.array([[[2, 3, -4], [0, 0, 0]]], np.float32)
        n = unit_normals(vectors)
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1, atol=1e-6)
        self.assertLess(n[0,0,2], 0)
        with self.assertRaises(ValueError):
            unit_normals(np.full((1,1,3), np.nan))

    def test_invalid_settings(self):
        for s in (AOVSettings(depth=False,normals=False), AOVSettings(dvd_window=16),
                  AOVSettings(normal_overlap=14), AOVSettings(device='cpu')):
            with self.assertRaises(ValueError):
                s.validate()

    def test_checkerboard_and_alpha_export(self):
        rgb = np.full((24,24,3), 200, np.uint8)
        alpha = np.zeros((24,24), np.uint8)
        alpha[:8] = 255
        checked = sammie._composite_bgcolor(rgb, alpha, (0,255,0), 'checkerboard')
        np.testing.assert_allclose(checked[:8], rgb[:8], atol=1)
        self.assertEqual(set(np.unique(checked[8:])), {104,160})
        rgba = sammie._composite_alpha(rgb, alpha)
        np.testing.assert_array_equal(rgba[...,:3], rgb)
        np.testing.assert_array_equal(rgba[...,3], alpha)


class CacheExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('.test-output').resolve()/('aov-unit-'+uuid.uuid4().hex)
        (self.root/'frames').mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(self.root/'frames'/f'{i:05d}.png'), np.full((19,31,3), i*40, np.uint8))
        self.settings = AOVSettings(device='cpu', depth_model='vda-small')
        self.job = dict(root=str(self.root),first=0,last=2,extension='png',settings=asdict(self.settings),fps='24000/1001')
        self.calls = []
        self.references = []

    def tearDown(self):
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('aov-unit-')
        shutil.rmtree(self.root)

    def factory(self, kind, settings, status):
        self.calls.append(kind)
        class Fake:
            def predict(self, frames, progress):
                h,w = frames[0].shape[:2]
                if kind == 'normals':
                    return np.tile(np.array([.6,0,.8],np.float32), (len(frames),h,w,1))
                return np.stack([np.tile(np.linspace(0,1,w),(h,1)) + i for i in range(len(frames))])
        model = Fake()
        self.references.append(weakref.ref(model))
        return model

    def test_cache_reuse_independent_model_invalidation_and_cleanup(self):
        run_job(self.job, factory=self.factory)
        self.assertEqual(self.calls, ['depth','normals'])
        run_job(self.job, factory=self.factory)
        self.assertEqual(len(self.calls), 2)
        changed = replace(self.settings, normal_axes='flip_yz')
        run_job(dict(self.job, settings=asdict(changed)), factory=self.factory)
        self.assertEqual(self.calls, ['depth','normals','normals'])
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.references))

    def test_source_replacement_invalidates_preview_and_generation(self):
        run_job(self.job, factory=self.factory)
        cv2.imwrite(str(self.root/'frames'/'00001.png'), np.full((19,31,3),240,np.uint8))
        self.assertIsNone(active(self.root, 'depth', self.settings))
        run_job(self.job, factory=self.factory)
        self.assertEqual(len(self.calls),4)

    def test_corrupt_cache_is_recomputed(self):
        run_job(self.job, factory=self.factory)
        folder, _ = active(self.root, 'normals')
        (folder/'00001.npy').write_bytes(b'truncated')
        run_job(self.job, factory=self.factory)
        self.assertEqual(self.calls, ['depth','normals','normals'])
        self.assertEqual(read_frame(self.root, 'normals', 1)[0].shape, (19,31,3))

    def test_failure_does_not_activate_incomplete_pass(self):
        def fail(*args):
            raise RuntimeError('checkpoint incompatible')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint incompatible'):
            run_job(self.job, factory=fail)
        self.assertIsNone(active(self.root, 'depth'))

    def test_nonfinite_model_output_rejected(self):
        class Invalid:
            def predict(self, frames, status):
                return np.full((3,19,31),np.nan)
        with self.assertRaisesRegex(RuntimeError, 'invalid data'):
            run_job(self.job, factory=lambda *a: Invalid())
        self.assertIsNone(active(self.root, 'depth'))

    def test_frame_dimensions_order_channels_precision_and_fps(self):
        run_job(self.job, factory=self.factory)
        job = dict(self.job,destination=str(self.root/'exr'),start_number=1001)
        export_sequence(job)
        names = sorted(p.name for p in (self.root/'exr').glob('*.exr'))
        self.assertEqual(names,['aov.1001.exr','aov.1002.exr','aov.1003.exr'])
        means = []
        for name in names:
            f = OpenEXR.InputFile(str(self.root/'exr'/name))
            try:
                header = f.header()
                self.assertEqual(set(header['channels']), {'Z','depth.Z','Z_raw','N.x','N.y','N.z'})
                self.assertEqual((header['dataWindow'].max.x+1,header['dataWindow'].max.y+1),(31,19))
                self.assertEqual((header['framesPerSecond'].n, header['framesPerSecond'].d), (24000,1001))
                values = np.frombuffer(f.channel('Z_raw', Imath.PixelType(Imath.PixelType.FLOAT)),np.float32)
                means.append(values.mean())
                nx = np.frombuffer(f.channel('N.x', Imath.PixelType(Imath.PixelType.FLOAT)),np.float32)
                np.testing.assert_allclose(nx,.6,atol=1e-6)
            finally:
                f.close()
        self.assertEqual(means, sorted(means))
        with self.assertRaises(FileExistsError):
            export_sequence(job)

    def test_global_depth_range_no_per_frame_normalization(self):
        manifest = dict(source_shape=(2,2),range=[0,10],depth_direction='far_is_larger')
        first = channel_data(np.full((2,2),2),manifest,'depth')['Z']
        last = channel_data(np.full((2,2),8),manifest,'depth')['Z']
        np.testing.assert_allclose(first,.2)
        np.testing.assert_allclose(last,.8)

    def make_mattes(self, ids=(0, 1)):
        for frame in range(3):
            folder = self.root/'matting'/f'{frame:05d}'
            folder.mkdir(parents=True, exist_ok=True)
            for oid in ids:
                cv2.imwrite(str(folder/f'{oid}.png'), np.full((19,31), 64+oid*128, np.uint8))

    def test_multilayer_rgb_alpha_objects_depth_normals(self):
        self.make_mattes()
        for frame in range(3):
            cv2.imwrite(str(self.root/'frames'/f'{frame:05d}.png'), np.full((19,31,3),(10,60,230),np.uint8))
        run_job(self.job, factory=self.factory)
        job = dict(self.job,destination=str(self.root/'exr'),start_number=1001,
                   include_plate=True,include_mattes=True,points=[{'object_id':0},{'object_id':1}],
                   session={'object_names':{'1':'Camera / gear'}})
        report = export_sequence(job)
        self.assertEqual(set(report['channels']), {'R','G','B','A','Object_0.Y','1_Camera_gear.Y',
                                                   'Z','depth.Z','Z_raw','N.x','N.y','N.z'})
        f = OpenEXR.InputFile(str(self.root/'exr/aov.1001.exr'))
        try:
            for key, expected in [('R',230/255),('G',60/255),('B',10/255),('A',192/255),
                                  ('Object_0.Y',64/255),('1_Camera_gear.Y',192/255)]:
                values = np.frombuffer(f.channel(key,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32)
                np.testing.assert_allclose(values, expected)
            meta = json.loads(f.header()['sammie:aov'])
            self.assertIn('unpremultiplied', meta['plate'])
            self.assertEqual(meta['alpha']['source'], 'matting')
        finally:
            f.close()
        # Export-only options never invalidate inference results.
        run_job(job, factory=self.factory)
        self.assertEqual(self.calls,['depth','normals'])

    def test_float_videomama_and_selected_refinement_survive_export(self):
        from sammie.refinement.algorithms import RefinementSettings
        from sammie.refinement.cache import RefinementCache, save_base
        from sammie.aov.matte_layers import MatteLayers
        self.make_mattes(ids=(0,))
        base = np.full((19,31), .12345678, np.float32)
        save_base(self.root, 0, 0, base, self.root/'frames/00000.png',self.root/'matting/00000/0.png')
        job = dict(self.job,points=[{'object_id':0}],session={'matany_model':'VideoMaMa'})
        np.testing.assert_array_equal(MatteLayers(job,(19,31)).channels(0)['A'],base)
        cache = RefinementCache(self.root)
        refined, validated = base+.1, base+.05
        cache.write(0,0,'example',refined=refined,validated=validated,trimap=np.full((19,31),128,np.uint8))
        cache.publish(RefinementSettings(),{'0/0':'example'},{},'test-double')
        run_job(self.job, factory=self.factory)
        for stage, expected in [('original',base),('refined',refined),('validated',validated)]:
            job['session'].update(vitmatte_enabled=True,vitmatte_view=stage)
            export_sequence(dict(job,destination=str(self.root/stage),start_number=1,include_mattes=True))
            f = OpenEXR.InputFile(str(self.root/stage/'aov.0001.exr'))
            try:
                values = np.frombuffer(f.channel('A',Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(19,31)
                np.testing.assert_array_equal(values,expected)
            finally:
                f.close()

    def test_combined_matte_zero_not_original_prompt_ids(self):
        from sammie.aov.matte_layers import MatteLayers
        self.make_mattes(ids=(0,))
        job = dict(self.job,points=[{'object_id':3},{'object_id':7}],
                   session={'matany_model':'VideoMaMa','matany_combined':True,'vitmatte_enabled':True})
        channels = MatteLayers(job,(19,31)).channels(0)
        self.assertEqual(set(channels), {'A','0_Combined.Y'})
        np.testing.assert_allclose(channels['A'],64/255)

    def test_missing_alpha_preflight_and_segmentation_choice(self):
        from sammie.aov.matte_layers import MatteLayers
        run_job(self.job, factory=self.factory)
        job = dict(self.job,destination=str(self.root/'exr'),start_number=1,include_mattes=True,
                   points=[{'object_id':0}])
        with self.assertRaisesRegex(ValueError,'No matting alpha'):
            export_sequence(job)
        self.assertFalse((self.root/'exr').exists())
        self.make_mattes(ids=(0,))
        (self.root/'matting/00001/0.png').unlink()
        with self.assertRaisesRegex(ValueError,'source frame 1'):
            export_sequence(job)
        (self.root/'matting').rename(self.root/'masks')
        job.update(last=0,matte_source='segmentation')
        result = MatteLayers(job,(19,31)).channels(0)
        np.testing.assert_allclose(result['A'],64/255)

    def test_stale_parameters_and_missing_frames_block_export(self):
        run_job(self.job, factory=self.factory)
        job = dict(self.job, destination=str(self.root/'exr'), start_number=1001,
                   settings=asdict(replace(self.settings,long_edge=640)))
        with self.assertRaisesRegex(ValueError,'settings changed'):
            export_sequence(job)
        with self.assertRaises(ValueError):
            run_job(dict(self.job,last=3),factory=self.factory)


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path('.test-output').resolve()/('aov-ui-unit-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.settings = initialize_settings(str(self.root))
        self.settings.app_settings_file = str(self.root/'app.conf')

    def tearDown(self):
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('aov-ui-unit-')
        shutil.rmtree(self.root)

    def test_tab_order_persistence_and_checkerboard_choice(self):
        import sammie_main
        with patch.object(sammie_main,'initialize_settings',return_value=self.settings), \
             patch.object(sammie_main.MainWindow,'_deferred_init'), \
             patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
             patch.object(sammie_main.MainWindow,'_update_current_frame_display'):
            window = sammie_main.MainWindow()
            try:
                tabs = window.sidebar.tab_widget
                self.assertEqual([tabs.tabText(i) for i in range(tabs.count())],
                                 ['Segmentation','Matting','Object Removal','AOVs'])
                tabs.setCurrentIndex(3)
                self.assertEqual(window.view_combo.currentText(),'AOV-Depth')
                window.sidebar.aov_tab.preview.setCurrentText('Normals')
                self.assertEqual(window.view_combo.currentText(),'AOV-Normals')
                for mode in ('Segmentation-BGcolor','Matting-BGcolor'):
                    window.view_combo.setCurrentText(mode)
                    window.background_choice.setCurrentIndex(1)
                    self.assertEqual(self.settings.get_view_options()['background_style'],'checkerboard')
                    self.assertFalse(window.color_picker.isEnabled())
                window.sidebar.aov_tab.controls['long_edge'].setCurrentIndex(2)
                self.assertEqual(self.settings.get_session_setting('aov_settings')['long_edge'],640)
                tab = window.sidebar.aov_tab
                self.assertTrue(tab.include_plate.isChecked())
                self.assertTrue(tab.include_mattes.isChecked())
                tab.matte_source.setCurrentIndex(1)
                tab.include_mattes.setChecked(False)
                tab.load_values_from_settings()
                self.assertFalse(tab.include_mattes.isChecked())
                self.assertFalse(tab.matte_source.isEnabled())
                self.assertEqual(tab.matte_source.currentData(),'segmentation')
            finally:
                window.close()

    def test_cancel_terminates_only_owned_worker(self):
        from sammie.aov.ui import AOVDialog
        dialog = AOVDialog(None, {'root':str(self.root)})
        dialog.process.start(sys.executable, ['-u','-c','import time; time.sleep(30)'])
        self.assertTrue(dialog.process.waitForStarted(5000))
        QTimer.singleShot(100, dialog.reject)
        QTimer.singleShot(5000, dialog.kill_if_running)
        dialog.exec()
        self.assertTrue(dialog.cancelled)
        self.assertEqual(dialog.process.state(),QProcess.NotRunning)
        self.assertIn('Cancelled',dialog.failure)


if __name__ == '__main__':
    unittest.main()
