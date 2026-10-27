import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch
import cv2
import Imath
import OpenEXR
import numpy as np
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.exr_media import (ImportOptions, inspect, read_rgb, display_rgb, sequence,
                              prepare, commit, project_info, project_rgb, discard)
from sammie.settings_manager import initialize_settings


class ExrImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.root=Path('.test-output').resolve()/('exr-unit-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.rgb=np.full((12,20,3),.18,np.float32)
        self.rgb[1,2]=[-.1,2.,8.]
        self.session=self.root/'session'

    def tearDown(self):
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('exr-unit-')
        shutil.rmtree(self.root)

    def image(self,name='plate.1001.exr',rgb=None,layer='',half=False,header=None):
        rgb=self.rgb if rgb is None else rgb
        h=dict(type=OpenEXR.scanlineimage,framesPerSecond=Fraction(24000,1001))
        h.update(header or {})
        channels={layer+c:rgb[...,i].astype(np.float16 if half else np.float32) for i,c in enumerate('RGB')}
        channels[layer+'A']=np.full(rgb.shape[:2],.25,np.float32)
        with OpenEXR.File(h,channels) as f: f.write(str(self.root/name))
        return self.root/name

    def imported(self,files=None,options=None):
        stage,info=prepare(files or [self.image()],options or ImportOptions(fps='24000/1001'),self.root)
        commit(stage,self.session)
        return info

    def test_half_float_layers_hdr_and_frame_rate(self):
        for half in (False,True):
            p=self.image(f'layer{half}.exr',layer='beauty.',half=half)
            info=inspect(p)
            self.assertEqual(info['layers'],['beauty.'])
            self.assertEqual(info['fps'],'24000/1001')
            rgb,_=read_rgb(p,'beauty.')
            np.testing.assert_array_equal(rgb,self.rgb.astype(np.float16).astype(np.float32) if half else self.rgb)
            with self.assertRaisesRegex(ValueError,'Missing RGB'): read_rgb(p)

    def test_proxy_transfer_exposure_and_native_plate_preservation(self):
        p=self.image()
        info=self.imported([p],ImportOptions(exposure=-1,fps='24000/1001'))
        np.testing.assert_array_equal(project_rgb(self.session,0),self.rgb)
        self.assertEqual((self.session/'source_exr/00000.exr').read_bytes(),p.read_bytes())
        proxy=cv2.imread(str(self.session/'frames/00000.png'))
        expected=display_rgb(self.rgb,ImportOptions(exposure=-1))
        np.testing.assert_array_equal(proxy[...,::-1],expected)
        self.assertEqual(info['source_numbers'],[1001])
        self.assertEqual(info['fps'],'24000/1001')
        self.assertGreater(expected[0,0,0],60)  # linear .09 is not display .09.

    def test_acescg_primaries_and_crop_alignment(self):
        header=dict(displayWindow=(np.array([-2,-1],np.int32),np.array([9,7],np.int32)),
                    dataWindow=(np.array([1,1],np.int32),np.array([4,3],np.int32)),
                    chromaticities=(.713,.293,.165,.830,.128,.044,.32168,.33767))
        p=self.image(rgb=np.full((3,4,3),2.,np.float32),header=header)
        self.assertEqual(inspect(p)['suggested_space'],'acescg')
        rgb,_=read_rgb(p)
        self.assertEqual(rgb.shape,(9,12,3))
        np.testing.assert_array_equal(rgb[2:5,3:7],2.)
        self.assertEqual(np.count_nonzero(rgb),3*4*3)
        gray=display_rgb(np.full((1,1,3),.18,np.float32),ImportOptions(color_space='acescg'))
        self.assertTrue(np.max(gray)-np.min(gray)<=1)

    def test_sequence_sorting_strict_matching_and_gaps(self):
        selected=self.image('plate.1002.exr')
        first=self.image('plate.1001.exr')
        self.image('plateA.1003.exr')
        self.image('plate.0001003.exr')
        files,numbers=sequence(selected)
        self.assertEqual(files,[first,selected])
        self.assertEqual(numbers,[1001,1002])
        gap=self.image('plate.1004.exr')
        with self.assertRaisesRegex(ValueError,'missing or duplicate'):
            prepare([first,gap],ImportOptions(),self.root)

    def test_tiled_rgb_and_project_round_trip_retain_originals(self):
        from sammie import sammie
        tiled=self.image(header={'type':OpenEXR.tiledimage,'tiles':OpenEXR.TileDescription()})
        self.imported([tiled])
        saved=self.root/'portable.sammie'
        overrides=dict(temp_dir=str(self.session),frames_dir=str(self.session/'frames'),
                       mask_dir=str(self.session/'masks'),matting_dir=str(self.session/'matting'))
        with patch.multiple(core,**overrides):
            self.assertEqual(sammie.save_project(str(saved),None),1)
            self.assertTrue(sammie.load_project(str(saved),None))
        np.testing.assert_array_equal(project_rgb(self.session,0),self.rgb)
        self.assertEqual((self.session/'source_exr/00000.exr').read_bytes(),tiled.read_bytes())
        self.assertTrue((self.session/'frames/00000.png').is_file())

    def test_bad_frame_and_cancel_keep_previous_session(self):
        self.session.mkdir()
        old=self.session/'precious.txt';old.write_text('keep')
        files=[self.image(),self.image('plate.1002.exr',rgb=np.zeros((13,20,3),np.float32))]
        with self.assertRaisesRegex(ValueError,'dimensions'):
            prepare(files,ImportOptions(),self.root)
        with self.assertRaises(InterruptedError):
            prepare([files[0]],ImportOptions(),self.root,cancel=lambda:True)
        self.assertEqual(old.read_text(),'keep')
        self.assertFalse(list(self.root.glob('.exr-import-*')))

    def test_nonfinite_missing_rgb_and_multipart_are_rejected(self):
        bad=self.rgb.copy();bad[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'): read_rgb(self.image(rgb=bad))
        with OpenEXR.File({}, {'Z':np.zeros((3,4),np.float32)}) as f: f.write(str(self.root/'depth.exr'))
        with self.assertRaisesRegex(ValueError,'no RGB'): inspect(self.root/'depth.exr')
        parts=[OpenEXR.Part({'name':str(i)}, {'RGB':np.zeros((3,4,3),np.float32)}) for i in range(2)]
        with OpenEXR.File(parts) as f: f.write(str(self.root/'multi.exr'))
        with self.assertRaisesRegex(ValueError,'Multi-part'): inspect(self.root/'multi.exr')

    def test_hdr_export_keeps_values_origin_and_rational_fps(self):
        from sammie.export_workers import SequenceExportWorker
        from sammie.aov.export import write_exr
        p=self.image(header={'dataWindow':(np.array([-2,3],np.int32),np.array([17,14],np.int32)),
                             'displayWindow':(np.array([-2,3],np.int32),np.array([17,14],np.int32))})
        info=self.imported([p])
        channels={c:project_rgb(self.session,0)[...,i] for i,c in enumerate('RGB')}
        channels['A']=np.ones((12,20),np.float32)
        for aov in (False,True):
            dest=self.root/f'export{aov}.exr'
            if aov: write_exr(dest,channels,dict(fps=info['fps'],exr_source=info))
            else: SequenceExportWorker._write_exr_file(str(dest),channels,exr_source=info)
            with OpenEXR.File(str(dest),separate_channels=True) as f:
                np.testing.assert_array_equal(f.channels()['B'].pixels,self.rgb[...,2])
                self.assertEqual(f.header()['framesPerSecond'],Fraction(24000,1001))
                np.testing.assert_array_equal(f.header()['displayWindow'][0],[-2,3])

    def test_foreground_reads_hdr_plate_and_rejects_wrong_input_space(self):
        from sammie.foreground.color import source_path,load_rgb
        from sammie.foreground.config import ForegroundSettings
        from sammie.foreground.pipeline import prepare as prepare_foreground
        self.imported()
        settings=ForegroundSettings(color_space='linear709')
        path=source_path(self.session,0,'png',settings)
        np.testing.assert_array_equal(load_rgb(path),self.rgb)
        with self.assertRaisesRegex(ValueError,'uses linear709'):
            prepare_foreground(dict(root=str(self.session),foreground_settings=asdict(ForegroundSettings())))

    def test_import_dialog_and_mainwindow_preserve_numbering_and_cancel(self):
        import sammie_main
        from sammie.exr_import_ui import ExrImportDialog
        p=self.image()
        d=ExrImportDialog(p)
        self.assertEqual(d.fps.text(),'24000/1001')
        self.assertFalse(d.preview.pixmap().isNull())
        d.close()
        manager=initialize_settings(str(self.session))
        manager.app_settings_file=str(self.root/'app.conf')
        overrides=dict(temp_dir=str(self.session),frames_dir=str(self.session/'frames'),
                       mask_dir=str(self.session/'masks'),matting_dir=str(self.session/'matting'))
        with patch.multiple(core,**overrides),patch.object(sammie_main,'initialize_settings',return_value=manager), \
             patch.object(sammie_main.MainWindow,'_deferred_init'),patch.object(sammie_main.MainWindow,'_setup_console_redirect'):
            window=sammie_main.MainWindow()
            try:
                stage=prepare([p],ImportOptions(fps='24000/1001'),self.root)
                with patch('sammie.exr_import_ui.prepare_dialog',return_value=stage), \
                     patch.object(window.sam_manager,'initialize_predictor'):
                    self.assertTrue(window.load_file(str(p)))
                self.assertEqual(manager.get_session_setting('frame_format'),'png')
                self.assertEqual(manager.get_session_setting('sequence_start_frame'),1001)
                self.assertEqual(manager.get_session_setting('aov_start_number'),1001)
                self.assertEqual(core.VideoInfo.fps_rational,'24000/1001')
                with patch('sammie.exr_import_ui.prepare_dialog',return_value=None):
                    self.assertFalse(window.load_file(str(p)))
                np.testing.assert_array_equal(project_rgb(self.session,0),self.rgb)
            finally: window.close()

    def test_aov_pipeline_uses_proxy_but_delivers_hdr_rgb_and_generated_alpha(self):
        from sammie.aov.pipeline import run_job
        from sammie.aov.config import AOVSettings
        from sammie.aov.export import export_sequence
        self.imported()
        (self.session/'matting/00000').mkdir()
        cv2.imwrite(str(self.session/'matting/00000/0.png'),np.full((12,20),128,np.uint8))
        received=[]
        class Model:
            def predict(self,frames,progress):
                received.append(frames[0].copy())
                return np.zeros((len(frames),*frames[0].shape[:2]),np.float32)
        settings=AOVSettings(normals=False,depth=True,depth_model='vda-small',long_edge=512)
        job=dict(root=str(self.session),first=0,last=0,extension='png',fps='24000/1001',settings=asdict(settings),
                 session={},points=[{'object_id':0}],include_plate=True,include_mattes=True,matte_source='matting',
                 destination=str(self.root/'aov-output'),start_number=1001)
        run_job(job,factory=lambda *args:Model())
        export_sequence(job)
        self.assertTrue(received)
        with OpenEXR.File(str(self.root/'aov-output/aov.1001.exr'),separate_channels=True) as f:
            np.testing.assert_array_equal(f.channels()['B'].pixels,self.rgb[...,2])
            np.testing.assert_allclose(f.channels()['A'].pixels,128/255)

    def test_refinement_accepts_prepared_exr_proxies(self):
        import torch
        from sammie.refinement.pipeline import run_refinement
        from sammie.refinement.algorithms import RefinementSettings
        self.imported()
        (self.session/'session_settings.conf').write_text(json.dumps({'video_file_path':'plate.1001.exr'}))
        (self.session/'matting/00000').mkdir()
        cv2.imwrite(str(self.session/'matting/00000/0.png'),np.full((12,20),128,np.uint8))
        seen=[]
        class Model:
            def __init__(self,*args): self.device=torch.device('cpu')
            def predict(self,rgb,trimap):
                seen.append(rgb.copy())
                return np.full(trimap.shape,.6,np.float32)
            def close(self): pass
        result=run_refinement(self.session,[0],RefinementSettings(short_edge=0),model_factory=Model,
                              checkpoint_resolver=lambda *args:(Path('.'),'test-model'))
        self.assertEqual(len(result['frames']),1)
        self.assertEqual(seen[0].dtype,np.uint8)
        self.assertLessEqual(seen[0].max(),255)


if __name__=='__main__': unittest.main()
