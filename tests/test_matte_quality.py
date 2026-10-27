import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch
import cv2
import numpy as np
from sammie.matte_quality import inspect_alpha, load_inspection


class QualityTests(unittest.TestCase):
    def test_opaque_interior_and_background_are_not_flagged(self):
        guide = np.zeros((1080,1080),np.uint8); guide[200:800,200:800]=255
        result = inspect_alpha(guide,guide)
        self.assertFalse(result.interior.any()); self.assertFalse(result.exterior.any())

    def test_hole_and_leak_are_flagged_without_modifying_input(self):
        guide = np.zeros((1080,1080),np.uint8); guide[200:800,200:800]=255
        alpha = guide.astype(np.float32)/255; alpha[500,500]=.4; alpha[100,100]=.2
        saved=alpha.copy(); saved_guide=guide.copy()
        result=inspect_alpha(alpha,guide)
        self.assertEqual(np.count_nonzero(result.interior),1)
        self.assertEqual(np.count_nonzero(result.exterior),1)
        self.assertTrue(result.interior[500,500]); self.assertTrue(result.exterior[100,100])
        np.testing.assert_array_equal(alpha,saved); np.testing.assert_array_equal(guide,saved_guide)

    def test_edge_uncertainty_is_excluded_and_radius_scales(self):
        for scale in (1,2):
            guide=np.zeros((1080*scale,1080*scale),np.uint8)
            guide[200*scale:800*scale,200*scale:800*scale]=255
            alpha=guide.astype(np.float32)/255
            alpha[195*scale:205*scale,400*scale:500*scale]=.5
            result=inspect_alpha(alpha,guide,margin=16)
            self.assertEqual(result.margin_pixels,16*scale)
            self.assertFalse(result.interior.any()); self.assertFalse(result.exterior.any())

    def test_empty_full_and_frame_boundaries(self):
        empty=np.zeros((100,100),np.float32); full=np.ones_like(empty)
        self.assertTrue(inspect_alpha(empty,full).interior.all())
        self.assertTrue(inspect_alpha(full,empty).exterior.all())
        self.assertFalse(inspect_alpha(full,full).interior.any())
        self.assertFalse(inspect_alpha(empty,empty).exterior.any())

    def test_invalid_inputs_fail_clearly(self):
        good=np.zeros((10,10),np.float32)
        for bad in (np.zeros((11,10)),np.zeros((10,10,3)),np.zeros((0,0)),
                    good+np.nan, good+np.inf, good+1.1, good-.1, good.astype(np.uint16)):
            with self.assertRaises(ValueError):inspect_alpha(bad,good)
        for kwargs in (dict(margin=-1),dict(margin=float('nan')),dict(foreground=.2,background=.3)):
            with self.assertRaises(ValueError):inspect_alpha(good,good,**kwargs)


class QualityIntegrationTests(unittest.TestCase):
    def setUp(self):
        from sammie import core
        from sammie.settings_manager import initialize_settings
        self.root=Path('.test-output').resolve()/('quality-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.settings=initialize_settings(str(self.root))
        self.settings.set_session_setting('matany_combined',True)
        self.patches=[patch.object(core,'temp_dir',str(self.root)),
                      patch.object(core,'matting_dir',str(self.root/'matting')),
                      patch.object(core,'mask_dir',str(self.root/'masks')),
                      patch.object(core,'frames_dir',str(self.root/'frames')),
                      patch.object(core.VideoInfo,'width',128),patch.object(core.VideoInfo,'height',96)]
        for p in self.patches:p.start()
        for name in ('matting/00000','masks/00000','frames'):(self.root/name).mkdir(parents=True)
        self.rgb=np.full((96,128,3),100,np.uint8)
        cv2.imwrite(str(self.root/'frames/00000.png'),self.rgb)
        self.alpha=np.zeros((96,128),np.uint8);self.alpha[20:80,20:100]=255;self.alpha[40,40]=100
        cv2.imwrite(str(self.root/'matting/00000/0.png'),self.alpha)
        guide=(self.alpha>0).astype(np.uint8)*255
        for oid in (3,7):cv2.imwrite(str(self.root/f'masks/00000/{oid}.png'),guide)
        self.points=[dict(object_id=3),dict(object_id=7)]

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('quality-')
        shutil.rmtree(self.root)

    def test_combined_matte_object_zero_and_missing_frame(self):
        rgb,alpha,guide=load_inspection(0,self.points)
        np.testing.assert_array_equal(rgb,self.rgb)
        np.testing.assert_array_equal(alpha,self.alpha.astype(np.float32)/255)
        self.assertTrue(inspect_alpha(alpha,guide).interior[40,40])
        with self.assertRaisesRegex(ValueError,'No matte exists'):load_inspection(1,self.points)
        with self.assertRaisesRegex(ValueError,'No matte exists'):load_inspection(0,[])

    def test_float_selected_view_retained(self):
        self.settings.set_session_setting('vitmatte_enabled',True)
        alpha=self.alpha.astype(np.float32)/255;alpha[40,40]=.123456
        with patch('sammie.refinement.display.selected_alpha',return_value=alpha):
            _,loaded,_=load_inspection(0,self.points)
        np.testing.assert_array_equal(alpha,loaded)

    def test_dialog_views_and_invalid_threshold_preserve_alpha(self):
        from PySide6.QtWidgets import QApplication
        from sammie.matte_quality_ui import MatteQualityDialog
        from sammie_main import MattingTab
        app=QApplication.instance() or QApplication([])
        rgb,alpha,guide=load_inspection(0,self.points);saved=alpha.copy()
        dialog=MatteQualityDialog(rgb,alpha,guide,0)
        dialog.show();app.processEvents()
        self.assertTrue(dialog.flags.interior[40,40])
        for mode in range(4):
            dialog.view.setCurrentIndex(mode);dialog.zoom.setCurrentText('100%')
            self.assertEqual(dialog.image.pixmap().width(),128)
            self.assertEqual(dialog.image.pixmap().height(),96)
        previous=dialog.flags
        dialog.foreground.setValue(.01);dialog.background.setValue(.5);dialog.analyze()
        self.assertIs(dialog.flags,previous)
        np.testing.assert_array_equal(alpha,saved)
        tab=MattingTab();self.assertEqual(tab.inspect_matting_btn.text(),'Inspect Matte Quality')
        tab.close();dialog.close()


if __name__=='__main__':unittest.main()
