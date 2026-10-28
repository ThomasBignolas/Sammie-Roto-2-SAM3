import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication, QComboBox
from sammie.refinement.algorithms import (RefinementSettings, make_trimap, merge_alpha,
    scaled_radius, TemporalValidator, PRESETS)
from sammie.refinement.cache import RefinementCache, save_base, load_base
from sammie.refinement.model import MODEL_SPECS, VitMatteModel, predict_tiled, resolve_checkpoint
from sammie.refinement.pipeline import run_refinement
from sammie.settings_manager import initialize_settings
from sammie import core, sammie


class AlgorithmsTests(unittest.TestCase):
    def test_empty_full_and_translucent(self):
        s = RefinementSettings()
        for value, expected in [(0, 0), (1, 255), (.3, 128)]:
            a = np.full((65, 99), value, np.float32)
            self.assertTrue(np.all(make_trimap(a, s) == expected))

    def test_thin_disconnected_boundary_and_low_alpha(self):
        a = np.zeros((1080, 120), np.float32)
        a[:, :35] = 1  # touches border
        a[:, 35:40] = .01  # low-alpha hair adjacent to foreground remains unknown
        a[200:400, 80] = .5  # disconnected thin structure
        tri = make_trimap(a, RefinementSettings())
        self.assertEqual(tri[0, 0], 255)
        self.assertEqual(tri[250, 80], 128)
        self.assertEqual(tri[250, 35], 128)
        self.assertEqual(tri[700, 115], 0)

    def test_resolution_scaling_and_invalid_thresholds(self):
        self.assertEqual(scaled_radius(12, (1080, 1920)), 12)
        self.assertEqual(scaled_radius(12, (2160, 3840)), 24)
        self.assertEqual(scaled_radius(12, (540, 960)), 6)
        with self.assertRaises(ValueError):
            make_trimap(np.zeros((9, 9)), replace(RefinementSettings(), foreground=.01))

    def test_merge_preserves_confident_regions_and_zero_strength(self):
        a = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (70, 1))
        s = RefinementSettings()
        tri = make_trimap(a, s)
        result = merge_alpha(a, 1-a, tri, s)
        np.testing.assert_array_equal(result[tri != 128], a[tri != 128])
        np.testing.assert_array_equal(merge_alpha(a, 1-a, tri, replace(s, strength=0)), a)
        self.assertTrue(np.isfinite(result).all())

    def test_temporal_rejects_new_delta_without_ghosting(self):
        rgb = np.zeros((64, 96, 3), np.uint8)
        a = np.full((64, 96), .5, np.float32)
        validator = TemporalValidator(1)
        validator.apply(rgb, a, a, 0)
        out = validator.apply(rgb, a, a+.3, 1)
        np.testing.assert_allclose(out, a, atol=1e-5)
        # Legitimate baseline motion remains exactly the current baseline.
        moved = np.roll(np.eye(64, 96, dtype=np.float32), 12, axis=1)
        out = validator.apply(rgb, moved, moved, 2)
        np.testing.assert_array_equal(out, moved)

    def test_temporal_cut_disocclusion_and_gap(self):
        a = np.full((64, 96), .5, np.float32)
        validator = TemporalValidator(1)
        validator.apply(np.zeros((64, 96, 3), np.uint8), a, a+.3, 0)
        out = validator.apply(np.full((64, 96, 3), 255, np.uint8), a, a+.3, 1)
        np.testing.assert_allclose(out, a)
        out = validator.apply(np.zeros((64, 96, 3), np.uint8), a, a+.3, 4)
        np.testing.assert_allclose(out, a+.3)

    def test_temporal_stable_refinement_survives(self):
        a = np.full((64, 96), .5, np.float32)
        rgb = np.zeros((64, 96, 3), np.uint8)
        v = TemporalValidator(.65)
        v.apply(rgb, a, a+.1, 0)
        np.testing.assert_allclose(v.apply(rgb, a, a+.1, 1), a+.1)

    def test_overlapping_tiles_cover_unknown_without_seams(self):
        class Constant:
            def predict(self, rgb, tri): return np.full(tri.shape, .37, np.float32)
        s = replace(RefinementSettings(), tile_size=256, overlap=64)
        rgb, tri = np.zeros((377, 601, 3), np.uint8), np.full((377, 601), 128, np.uint8)
        out = predict_tiled(Constant(), rgb, tri, s)
        np.testing.assert_allclose(out, .37, atol=1e-6)
        with self.assertRaises(InterruptedError):
            predict_tiled(Constant(), rgb, tri, s, cancel=lambda: True)

    def test_model_preprocessing_padding_and_original_normalization(self):
        class Capture:
            def __call__(self, pixel_values):
                self.pixels = pixel_values
                return SimpleNamespace(alphas=torch.ones((1, 1, *pixel_values.shape[-2:]))*.25)
        m = VitMatteModel(replace(RefinementSettings(), device='cpu'), Path('.'))
        m.model = Capture()
        out = m.predict(np.zeros((33, 47, 3), np.uint8), np.full((33, 47), 128, np.uint8))
        self.assertEqual(out.shape, (33, 47))
        self.assertEqual(tuple(m.model.pixels.shape), (1, 4, 64, 64))
        self.assertAlmostEqual(m.model.pixels[0, 0, 0, 0].item(), -.485/.229, places=5)
        self.assertAlmostEqual(m.model.pixels[0, 3, 0, 0].item(), 128/255, places=5)
        m.close()


class FakeModel:
    instances = []
    def __init__(self, settings, path):
        self.device, self.closed = torch.device('cpu'), False
        self.calls = 0
        self.instances.append(self)
    def predict(self, rgb, tri):
        self.calls += 1
        return np.full(tri.shape, .4234567, np.float32)
    def close(self): self.closed = True


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = Path('.test-output').resolve()/('refine-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.manager = initialize_settings(str(self.root))
        self.manager.app_settings_file = str(self.root/'app.conf')
        self.s = replace(RefinementSettings(), feather=0)
        self.patches = [patch.object(core, 'temp_dir', str(self.root)),
                        patch.object(core, 'frames_dir', str(self.root/'frames')),
                        patch.object(core, 'matting_dir', str(self.root/'matting'))]
        for p in self.patches: p.start()
        core.VideoInfo.width, core.VideoInfo.height, core.VideoInfo.total_frames = 96, 64, 3
        (self.root/'frames').mkdir()
        for frame in range(3):
            (self.root/'matting'/f'{frame:05d}').mkdir(parents=True)
            cv2.imwrite(str(self.root/'frames'/f'{frame:05d}.png'), np.full((64, 96, 3), 120+frame, np.uint8))
            cv2.imwrite(str(self.root/'matting'/f'{frame:05d}/0.png'), np.full((64, 96), 120, np.uint8))
        FakeModel.instances.clear()

    def tearDown(self):
        for p in self.patches: p.stop()
        assert self.root.parent == Path('.test-output').resolve() and self.root.name.startswith('refine-')
        shutil.rmtree(self.root)

    def run_pipeline(self, **kwargs):
        return run_refinement(self.root, range(3), self.s, model_factory=FakeModel,
                              checkpoint_resolver=lambda *a: (Path('.'), 'test-weight-v1'), **kwargs)

    def test_cache_resume_and_parameter_source_alpha_invalidation(self):
        report = self.run_pipeline()
        self.assertEqual(report['cached_frames'], 0)
        self.assertTrue(FakeModel.instances[-1].closed)
        self.assertEqual(self.run_pipeline()['cached_frames'], 3)
        self.s = replace(self.s, strength=.5)
        self.assertEqual(self.run_pipeline()['cached_frames'], 0)
        cv2.imwrite(str(self.root/'matting/00001/0.png'), np.full((64, 96), 110, np.uint8))
        self.assertEqual(self.run_pipeline()['cached_frames'], 1)  # current + previous-frame dependencies
        cv2.imwrite(str(self.root/'frames/00000.png'), np.zeros((64, 96, 3), np.uint8))
        self.assertEqual(self.run_pipeline()['cached_frames'], 1)  # frame 2 does not depend on frame 0

    def test_cancel_resume_matches_uninterrupted_and_cleans_model(self):
        state = {'cancel': False}
        def progress(i, n, text):
            if i == 1: state['cancel'] = True
        with self.assertRaises(InterruptedError):
            self.run_pipeline(progress=progress, cancel=lambda: state['cancel'])
        self.assertTrue(FakeModel.instances[-1].closed)
        self.assertFalse(RefinementCache(self.root).active.exists())
        self.assertEqual(self.run_pipeline()['cached_frames'], 1)
        first = np.load(self.root/'vitmatte/00002/0.npz')['validated']
        (self.root/'vitmatte/00000/0.json').unlink()
        self.run_pipeline()
        second = np.load(self.root/'vitmatte/00002/0.npz')['validated']
        np.testing.assert_array_equal(first, second)

    def test_model_failure_and_missing_frame_do_not_publish(self):
        with patch.object(FakeModel, 'predict', side_effect=RuntimeError('weights invalid')):
            with self.assertRaisesRegex(RuntimeError, 'weights invalid'): self.run_pipeline()
        self.assertTrue(FakeModel.instances[-1].closed)
        self.assertFalse(RefinementCache(self.root).active.exists())
        (self.root/'frames/00001.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): self.run_pipeline()

    def test_incompatible_local_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Transformers directory'):
            resolve_checkpoint(replace(self.s, checkpoint=str(self.root/'missing')))
        (self.root/'config.json').write_text('{"model_type":"sam"}')
        (self.root/'model.safetensors').write_bytes(b'not weights')
        with self.assertRaisesRegex(ValueError, 'four-channel'):
            resolve_checkpoint(replace(self.s, checkpoint=str(self.root)))

    def test_incomplete_weights_fail_strict_loading_and_cleanup(self):
        from transformers import VitMatteForImageMatting
        m = VitMatteModel(replace(self.s, device='cpu', checkpoint=str(self.root)), self.root)
        with patch.object(VitMatteForImageMatting, 'from_pretrained',
                          return_value=(object(), {'missing_keys':['decoder.matting_head.weight']})):
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'): m.load()
        self.assertIsNone(m.model)

    def test_local_checkpoint_size_must_match_selected_model(self):
        (self.root/'config.json').write_text(json.dumps({'model_type': 'vitmatte',
            'backbone_config': {'num_channels': 4}, 'hidden_size': 384}))
        (self.root/'model.safetensors').write_bytes(b'test weights')
        with self.assertRaisesRegex(ValueError, 'size does not match'):
            resolve_checkpoint(replace(self.s, checkpoint=str(self.root)))
        _, identity = resolve_checkpoint(replace(self.s, checkpoint=str(self.root), model='small-composition-1k'))
        self.assertTrue(identity)

    def test_base_binary_is_pinned_and_loads_weights_only(self):
        from transformers import VitMatteForImageMatting
        spec = MODEL_SPECS['base-composition-1k']
        checkpoint = Path('checkpoints/vitmatte')/spec['revision']
        m = VitMatteModel(replace(self.s, device='cpu'), checkpoint)
        with patch('sammie.refinement.model.digest', return_value=spec['sha256']), \
                patch.object(VitMatteForImageMatting, 'from_pretrained',
                             return_value=(torch.nn.Identity(), {})) as load:
            m.load()
            self.assertFalse(load.call_args.kwargs['use_safetensors'])
            self.assertTrue(load.call_args.kwargs['weights_only'])
            self.assertTrue(load.call_args.kwargs['local_files_only'])
        m.close()
        with patch('sammie.refinement.model.digest', return_value='wrong checksum'):
            with self.assertRaisesRegex(ValueError, 'verified official'): m.load()
        m.checkpoint = self.root
        with self.assertRaisesRegex(ValueError, 'verified official'): m.load()
        self.assertIsNone(m.model)

    def test_switching_model_invalidates_refinement_without_changing_baseline(self):
        self.run_pipeline()
        png = self.root/'matting/00000/0.png'
        original = png.read_bytes()
        self.s = replace(self.s, model='small-composition-1k')
        self.assertIsNone(RefinementCache(self.root).selected(0, 0, self.s, 'validated'))
        self.assertEqual(self.run_pipeline()['cached_frames'], 0)
        self.assertEqual(png.read_bytes(), original)

    def test_worker_cancel_keeps_dialog_alive_until_cleanup(self):
        from sammie.refinement.ui import RefinementDialog
        from PySide6.QtCore import QTimer
        import time
        state = {'cleaned': False}
        def fake_run(*args, cancel, **kwargs):
            try:
                while not cancel(): time.sleep(.001)
                raise InterruptedError('cancelled')
            finally: state['cleaned'] = True
        with patch('sammie.refinement.ui.run_refinement', side_effect=fake_run):
            dialog = RefinementDialog(None, self.s)
            QTimer.singleShot(10, dialog.reject)
            result, failure = dialog.execute()
        self.assertIsNone(result)
        self.assertEqual(failure, 'cancelled')
        self.assertTrue(state['cleaned'])
        self.assertFalse(dialog.worker.isRunning())

    def test_float_base_falls_back_when_png_replaced(self):
        source, png = self.root/'frames/00000.png', self.root/'matting/00000/0.png'
        a = np.full((64, 96), .123456, np.float32)
        save_base(self.root, 0, 0, a, source, png)
        np.testing.assert_array_equal(load_base(self.root, 0, 0, source, png)[0], a)
        cv2.imwrite(str(png), np.zeros((64, 96), np.uint8))
        self.assertFalse(load_base(self.root, 0, 0, source, png)[0].any())

    def test_preview_invalidates_changed_history_but_accepts_project_timestamp_changes(self):
        self.run_pipeline()
        cache = RefinementCache(self.root)
        original = cache.selected(2, 0, self.s, 'validated')
        self.assertIsNotNone(original)
        path = self.root/'frames/00000.png'
        os.utime(path, (path.stat().st_atime+10, path.stat().st_mtime+10))
        np.testing.assert_array_equal(cache.selected(2, 0, self.s, 'validated'), original)
        cv2.imwrite(str(path), np.zeros((64,96,3),np.uint8))
        np.testing.assert_array_equal(cache.selected(2, 0, self.s, 'validated'), original)
        self.assertIsNone(cache.selected(1, 0, self.s, 'validated'))
        cv2.imwrite(str(self.root/'frames/00001.png'), np.zeros((64,96,3),np.uint8))
        self.assertIsNone(cache.selected(2, 0, self.s, 'validated'))

    def test_bounded_invalidation_matches_a_fresh_full_run(self):
        self.run_pipeline()
        cv2.imwrite(str(self.root/'matting/00000/0.png'),np.full((64,96),180,np.uint8))
        self.run_pipeline()
        warm=[]
        for f in range(3):
            with np.load(self.root/f'vitmatte/{f:05d}/0.npz') as data:warm.append(data['validated'].copy())
        shutil.rmtree(self.root/'vitmatte')
        self.run_pipeline()
        for f in range(3):
            with np.load(self.root/f'vitmatte/{f:05d}/0.npz') as data:
                np.testing.assert_allclose(data['validated'],warm[f],atol=1e-7)

    def test_partial_refinement_keeps_unaffected_frames_selectable(self):
        self.run_pipeline()
        before=RefinementCache(self.root).selected(0,0,self.s,'validated').copy()
        run_refinement(self.root,range(1,3),self.s,model_factory=FakeModel,
                       checkpoint_resolver=lambda *args:(Path('.'),'test-weight-v1'))
        np.testing.assert_array_equal(RefinementCache(self.root).selected(0,0,self.s,'validated'),before)

    def test_corrupt_cache_is_recomputed_and_model_version_invalidates(self):
        self.run_pipeline()
        (self.root/'vitmatte/00001/0.npz').write_bytes(b'broken zip archive')
        self.assertEqual(self.run_pipeline()['cached_frames'], 2)
        r = run_refinement(self.root,range(3),self.s,model_factory=FakeModel,
                           checkpoint_resolver=lambda *a:(Path('.'),'new-weight-version'))
        self.assertEqual(r['cached_frames'], 0)

    def test_linear_source_rejected(self):
        (self.root/'session_settings.conf').write_text(json.dumps({'video_file_path':'linear.1001.exr'}))
        with self.assertRaisesRegex(ValueError,'display-referred'): self.run_pipeline()

    def test_disabled_preview_regression_and_float_exr(self):
        from sammie.export_workers import SequenceExportWorker
        from sammie.export_formats import ExportSettings
        import OpenEXR, Imath
        self.run_pipeline()
        points = [{'object_id': 0, 'frame': 0}]
        opts = {'view_mode': 'Matting-Matte', 'float_alpha': True}
        old = sammie.update_image(0, opts, points, return_numpy=True)
        self.assertEqual(old.dtype, np.uint8)
        self.assertTrue(np.all(old == 120))
        self.manager.set_session_setting('matany_model', 'VideoMaMa')
        self.manager.set_session_setting('vitmatte_enabled', True)
        self.manager.set_session_setting('vitmatte_feather', 0)
        a = sammie.update_image(0, opts, points, return_numpy=True)
        self.assertEqual(a.dtype, np.float32)
        self.assertFalse(np.all(a*255 == np.round(a*255)))
        # Draft edits, including temporarily crossed thresholds, must not change
        # either the preview or exported float alpha until refinement is rerun.
        self.manager.set_session_setting('vitmatte_strength', .1)
        self.manager.set_session_setting('vitmatte_background', .99)
        np.testing.assert_array_equal(sammie.update_image(0, opts, points, return_numpy=True), a)
        self.manager.set_session_setting('sequence_start_frame', 1001)
        output = self.root/'export'; output.mkdir()
        settings = ExportSettings('exr', str(output), 'test', 'Matting-Matte', -1, False, 14, False, None, None)
        SequenceExportWorker(settings, points, 3, 'test', SimpleNamespace(settings_mgr=self.manager))._export_exr_sequence()
        f = OpenEXR.InputFile(str(output/'test.1001.exr'))
        try:
            exr = np.frombuffer(f.channel('Object_0.Y', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(64, 96)
        finally:
            f.close()
        np.testing.assert_array_equal(exr, a[..., 0])
        self.assertEqual(len(list(output.glob('*.exr'))), 3)

    def test_draft_settings_keep_completed_result_but_source_changes_invalidate(self):
        from sammie.refinement.display import selected_alpha
        self.run_pipeline()
        self.manager.set_session_setting('matany_model','VideoMaMa')
        self.manager.set_session_setting('vitmatte_enabled',True)
        points=[{'object_id':0}]
        original=selected_alpha(0,points).copy()
        for key,value in [('radius',25),('short_edge',720),('model','small-composition-1k'),
                          ('temporal',False),('strength',.1),('background',.99)]:
            self.manager.set_session_setting('vitmatte_'+key,value)
            np.testing.assert_array_equal(selected_alpha(0,points),original)
        self.assertEqual(RefinementCache(self.root).applied_settings(),self.s)
        self.manager.save_session_settings();self.manager.load_session_settings()
        np.testing.assert_array_equal(selected_alpha(0,points),original)
        self.manager.set_session_setting('vitmatte_view','original')
        np.testing.assert_allclose(selected_alpha(0,points),120/255,atol=1e-7)
        self.manager.set_session_setting('vitmatte_view','validated')
        cv2.imwrite(str(self.root/'matting/00000/0.png'),np.zeros((64,96),np.uint8))
        self.assertFalse(selected_alpha(0,points).any())

    def test_ui_draft_restore_and_wheel_does_not_change_controls(self):
        from sammie.refinement.ui import RefinementControls
        from PySide6.QtCore import QPointF,QPoint,Qt
        from PySide6.QtGui import QWheelEvent
        self.run_pipeline()
        model=QComboBox();model.addItems(['VideoMaMa'])
        controls=RefinementControls(model);controls.setChecked(True)
        try:
            controls.controls['radius'].setValue(35)
            self.assertIn('last completed refinement',controls.status.text())
            controls.restore.click()
            self.assertEqual(RefinementSettings.from_session(self.manager),self.s)
            for widget in (controls.controls['radius'],controls.controls['strength'],controls.controls['model'],controls.view,controls.preset):
                before=widget.currentIndex() if isinstance(widget,QComboBox) else widget.value()
                widget.setFocus()
                event=QWheelEvent(QPointF(10,10),QPointF(10,10),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
                QApplication.sendEvent(widget,event)
                after=widget.currentIndex() if isinstance(widget,QComboBox) else widget.value()
                self.assertEqual(after,before)
        finally:controls.close()

    def test_multiple_object_refinement_and_single_object_combined_setting(self):
        from sammie.refinement.display import selected_alpha
        self.manager.set_session_setting('matany_model', 'VideoMaMa')
        self.manager.set_session_setting('vitmatte_enabled', True)
        self.manager.set_session_setting('vitmatte_feather', 0)
        for frame in range(3):
            cv2.imwrite(str(self.root/'matting'/f'{frame:05d}/7.png'),np.full((64,96),255,np.uint8))
        self.assertEqual(len(self.run_pipeline()['frames']), 6)
        # Known full foreground is unchanged, separate object's uncertainty is refined.
        points = [{'object_id':0},{'object_id':7}]
        self.assertTrue(np.all(selected_alpha(0,points,7) == 1))
        self.assertTrue(np.all(selected_alpha(0,points,0) < .5))
        self.manager.set_session_setting('matany_combined', True)
        self.assertTrue(np.all(selected_alpha(0,[{'object_id':7}]) == 1))

    def test_combined_export_without_zero_prompt_id(self):
        from sammie.export_workers import SequenceExportWorker
        from sammie.export_formats import ExportSettings
        import OpenEXR
        self.manager.set_session_setting('matany_model','VideoMaMa')
        self.manager.set_session_setting('vitmatte_enabled',True)
        self.manager.set_session_setting('matany_combined',True)
        output=self.root/'combined';output.mkdir()
        settings=ExportSettings('exr',str(output),'combined','Matting-Matte',-1,False,14,False,None,None)
        SequenceExportWorker(settings,[{'object_id':4},{'object_id':7}],3,'combined',
                             SimpleNamespace(settings_mgr=self.manager))._export_exr_sequence()
        f=OpenEXR.InputFile(str(output/'combined.0000.exr'))
        try: self.assertEqual(set(f.header()['channels']),{'Object_0.Y'})
        finally: f.close()

    def test_ui_defaults_presets_and_roundtrip(self):
        from sammie.refinement.ui import RefinementControls
        model = QComboBox(); model.addItems(['MatAnyone', 'VideoMaMa'])
        controls = RefinementControls(model)
        self.assertEqual(controls.controls['model'].currentData(), 'base-composition-1k')
        controls.controls['model'].setCurrentIndex(controls.controls['model'].findData('small-composition-1k'))
        self.assertEqual(RefinementSettings.from_session(self.manager).model, 'small-composition-1k')
        controls.load()
        self.assertEqual(controls.controls['model'].currentData(), 'small-composition-1k')
        self.assertFalse(controls.isChecked())
        self.assertTrue(controls.isHidden())
        model.setCurrentText('VideoMaMa')
        self.assertFalse(controls.isHidden())
        controls.setChecked(True)
        controls.preset.setCurrentText('Quality'); controls.apply_preset(2)
        s = RefinementSettings.from_session(self.manager)
        for k, v in PRESETS['Quality'].items(): self.assertEqual(getattr(s, k), v)
        self.manager.save_session_settings()
        self.assertTrue(json.loads((self.root/'session_settings.conf').read_text())['vitmatte_enabled'])
        controls.close()


if __name__ == '__main__': unittest.main()
