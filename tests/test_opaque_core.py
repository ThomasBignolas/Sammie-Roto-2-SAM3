import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from dataclasses import asdict
import shutil,uuid,unittest
from unittest.mock import patch
import cv2,numpy as np
from sammie.opaque_core import core_weight,protect_core,apply_from_files,_cached_weight


class CoreTests(unittest.TestCase):
    def test_disabled_exact_and_soft_edge_unchanged(self):
        guide=np.zeros((1080,1080),np.uint8);guide[180:900,180:900]=255
        alpha=np.full(guide.shape,.345678,np.float32)
        weight=core_weight(guide,16)
        self.assertIs(protect_core(alpha,weight,0),alpha)
        before=alpha.copy();out=protect_core(alpha,weight)
        self.assertEqual(out[540,540],1)
        np.testing.assert_array_equal(out[weight==0],alpha[weight==0])
        np.testing.assert_array_equal(alpha,before)
        self.assertEqual(weight[195,540],0)
        self.assertTrue(0<weight[200,540]<1)
        self.assertEqual(weight[215,540],1)
        self.assertAlmostEqual(float(protect_core(alpha,weight,.5)[540,540]),(.345678+1)/2,places=6)

    def test_thin_shapes_empty_full_and_scaling(self):
        empty=np.zeros((1080,1080),np.uint8)
        self.assertFalse(core_weight(empty).any())
        self.assertTrue((core_weight(empty+255)==1).all())
        empty[100:900,500:510]=255
        self.assertFalse(core_weight(empty,16).any())
        guide=np.zeros((1080,1080),np.uint8);guide[200:900,200:900]=255
        large=cv2.resize(guide,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        a,b=core_weight(guide),core_weight(large)
        self.assertEqual(a[210,540],b[420,1080])
        self.assertEqual(a[250,540],b[500,1080])

    def test_motion_has_no_history_or_trails(self):
        for offset in (0,40,80,20):
            guide=np.zeros((128,192),np.uint8);guide[20:100,20+offset:60+offset]=255
            alpha=guide.astype(np.float32)/255;alpha[50,40+offset]=.2
            result=protect_core(alpha,core_weight(guide))
            self.assertEqual(result[50,40+offset],1)
            self.assertFalse(result[guide==0].any())

    def test_invalid_parameters_and_alignment(self):
        a=np.ones((10,10),np.float32)
        for inset in (0,-1,float('nan')):
            with self.assertRaises(ValueError):core_weight(a,inset)
        for strength in (-1,2,float('nan')):
            with self.assertRaises(ValueError):protect_core(a,a,strength)
        with self.assertRaises(ValueError):protect_core(a,np.ones((9,10)))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        from sammie import core
        from sammie.settings_manager import initialize_settings
        self.root=Path('.test-output').resolve()/('core-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.settings=initialize_settings(str(self.root));self.settings.app_settings_file=str(self.root/'app.conf')
        for k,v in dict(matany_model='VideoMaMa',matany_combined=True,matany_core_strength=1.,matany_core_inset=16).items():self.settings.set_session_setting(k,v)
        self.patches=[patch.object(core,'temp_dir',str(self.root)),patch.object(core,'mask_dir',str(self.root/'masks')),
                      patch.object(core,'matting_dir',str(self.root/'matting')),patch.object(core,'frames_dir',str(self.root/'frames')),
                      patch.object(core.VideoInfo,'width',192),patch.object(core.VideoInfo,'height',128)]
        for p in self.patches:p.start()
        for f in ('masks/00000','matting/00000','frames'):(self.root/f).mkdir(parents=True)
        self.guide=np.zeros((128,192),np.uint8);self.guide[20:110,20:172]=255
        left=self.guide.copy();left[:,100:]=0
        right=self.guide.copy();right[:,:90]=0
        cv2.imwrite(str(self.root/'masks/00000/3.png'),left);cv2.imwrite(str(self.root/'masks/00000/7.png'),right)
        self.alpha=self.guide.copy();self.alpha[50:60,40:50]=90;self.alpha[50:60,140:150]=90
        cv2.imwrite(str(self.root/'matting/00000/0.png'),self.alpha)
        cv2.imwrite(str(self.root/'frames/00000.png'),np.zeros((128,192,3),np.uint8))
        self.points=[dict(object_id=3),dict(object_id=7)]

    def tearDown(self):
        _cached_weight.cache_clear()
        for p in reversed(self.patches):p.stop()
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('core-')
        shutil.rmtree(self.root)

    def test_preview_float_aov_export_and_inspector_match(self):
        from sammie import core,sammie
        from sammie.aov.matte_layers import MatteLayers
        from sammie.matte_quality import load_inspection
        for vit in (False,True):
            self.settings.set_session_setting('vitmatte_enabled',vit)
            preview=sammie._get_display_mask(0,self.points,None,core.matting_dir,core.apply_matany_postprocessing,True)
            job=dict(root=str(self.root),session=asdict(self.settings.session_settings),points=self.points,
                     first=0,last=0,matte_source='matting')
            exported=MatteLayers(job,(128,192)).channels(0)['A']
            np.testing.assert_array_equal(preview,exported)
            np.testing.assert_array_equal(preview,load_inspection(0,self.points)[1])
            self.assertEqual(preview[55,45],1);self.assertEqual(preview[55,145],1)
            self.assertEqual(preview[0,0],0)
        np.testing.assert_array_equal(cv2.imread(str(self.root/'matting/00000/0.png'),0),self.alpha)

    def test_guides_and_settings_invalidate_cached_core(self):
        args=(self.alpha,self.root,0,[3,7])
        settings=asdict(self.settings.session_settings)
        first=apply_from_files(*args,settings);self.assertEqual(first[55,45],1)
        path=self.root/'masks/00000/3.png';cv2.imwrite(str(path),np.zeros_like(self.guide))
        second=apply_from_files(*args,settings);self.assertAlmostEqual(float(second[55,45]),90/255,places=6)
        settings['matany_core_strength']=0
        self.assertIs(apply_from_files(*args,settings),self.alpha)

    def test_standard_float_exr_uses_combined_core_without_vitmatte(self):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QApplication
        from sammie.export_workers import SequenceExportWorker
        from sammie.export_formats import ExportSettings
        from sammie import core,sammie
        import OpenEXR,Imath
        app=QApplication.instance() or QApplication([])
        self.settings.set_session_setting('vitmatte_enabled',False)
        self.settings.set_session_setting('sequence_start_frame',1001)
        out=self.root/'export';out.mkdir()
        config=ExportSettings('exr',str(out),'test','Matting-Matte',-1,False,14,False,None,None)
        SequenceExportWorker(config,self.points,1,'test',SimpleNamespace(settings_mgr=self.settings))._export_exr_sequence()
        f=OpenEXR.InputFile(str(out/'test.1001.exr'))
        try:
            self.assertEqual(set(f.header()['channels']),{'Object_0.Y'})
            actual=np.frombuffer(f.channel('Object_0.Y',Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(128,192)
        finally:f.close()
        expected=sammie._get_display_mask(0,self.points,None,core.matting_dir,core.apply_matany_postprocessing,True)
        np.testing.assert_array_equal(actual,expected)

    def test_missing_matte_never_becomes_manufactured_foreground(self):
        from sammie.opaque_core import apply_to_preview
        a=np.zeros_like(self.alpha);(self.root/'matting/00000/0.png').unlink()
        np.testing.assert_array_equal(apply_to_preview(a,0,self.points),a)

    def test_missing_individual_object_is_not_filled_from_its_guide(self):
        from sammie.opaque_core import apply_to_preview
        self.settings.set_session_setting('matany_combined',False)
        left=cv2.imread(str(self.root/'masks/00000/3.png'),0)
        cv2.imwrite(str(self.root/'matting/00000/3.png'),left)
        result=apply_to_preview(left,0,self.points)
        self.assertEqual(result[55,145],0)

    def test_ui_controls_roundtrip_and_default_off(self):
        from PySide6.QtWidgets import QApplication
        from sammie_main import MattingTab
        from sammie.settings_manager import SessionSettings
        app=QApplication.instance() or QApplication([])
        self.assertEqual(SessionSettings().matany_core_strength,0)
        tab=MattingTab();self.assertEqual(tab.core_strength_slider.value(),100)
        tab.core_strength_slider.setValue(75);tab.core_inset_slider.setValue(24)
        self.assertEqual(self.settings.get_session_setting('matany_core_strength'),.75)
        self.assertTrue(self.settings.save_session_settings());self.assertTrue(self.settings.load_session_settings())
        self.assertEqual(self.settings.get_session_setting('matany_core_inset'),24)
        tab.close()

if __name__=='__main__':unittest.main()
