import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import unittest
import uuid
import weakref
from unittest.mock import patch
import cv2
import numpy as np
import OpenEXR
import Imath
from sammie.foreground.config import ForegroundSettings
from sammie.foreground.color import to_linear709, from_linear709, merge_foreground, source_path, load_rgb
from sammie.foreground.cache import read_result, manifest
from sammie.foreground.pipeline import prepare, run_job
from sammie.aov.config import AOVSettings
from sammie.aov.export import export_sequence, write_exr


class ColorTests(unittest.TestCase):
    def test_acescg_hdr_roundtrip_and_srgb_transfer(self):
        rgb=np.array([[[.18,.32,.5],[3.,8.,18.]]],np.float32)
        for space in ('acescg','srgb','linear709'):
            np.testing.assert_allclose(from_linear709(to_linear709(rgb,space),space),rgb,rtol=2e-6,atol=2e-6)

    def test_merge_preserves_alpha_opaque_hdr_and_bypass(self):
        rgb=np.full((3,4,3),4.,np.float32)
        alpha=np.array([[0,.2,.8,1]]*3,np.float32)
        saved=alpha.copy()
        raw=np.full((8,8,3),.5,np.float32)
        settings=ForegroundSettings(color_space='acescg',plate_path='plate.0000.exr')
        out=merge_foreground(rgb,alpha,raw,settings)
        np.testing.assert_array_equal(alpha,saved)
        np.testing.assert_array_equal(out[alpha==1],rgb[alpha==1])
        np.testing.assert_array_equal(merge_foreground(rgb,alpha,raw,replace(settings,strength=0)),rgb)
        self.assertTrue(np.isfinite(out).all())

    def test_explicit_exr_alignment(self):
        settings=ForegroundSettings(plate_path='D:/shot/plate.0993.exr',plate_first=993,color_space='acescg')
        self.assertEqual(source_path('unused',58,'png',settings).name,'plate.1051.exr')
        with self.assertRaises(ValueError): source_path('.',0,'png',replace(settings,plate_path='plate.exr'))

    def test_invalid_settings(self):
        for changes in [dict(screen='auto'),dict(resolution=4096),dict(exposure_ev=float('nan')),
                        dict(strength=-1),dict(despill=2),dict(color_space='unknown'),dict(plate_path='a.mp4')]:
            with self.assertRaises(ValueError): replace(ForegroundSettings(),**changes).validate()

    def test_linear_despill_preserves_luminance(self):
        source=np.zeros((3,4,3),np.float32)
        alpha=np.full((3,4),.2,np.float32)
        raw=np.tile(np.array([.2,.3,.8],np.float32),(8,8,1))
        base=ForegroundSettings(color_space='linear709',plate_path='x.0000.exr',exposure_ev=0)
        a=merge_foreground(source,alpha,raw,base)
        b=merge_foreground(source,alpha,raw,replace(base,despill=1))
        np.testing.assert_allclose(a@np.array([.2126,.7152,.0722]),b@np.array([.2126,.7152,.0722]),atol=1e-6)
        self.assertLess(b[...,2].mean(),a[...,2].mean())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.root=Path('.test-output').resolve()/('foreground-unit-'+uuid.uuid4().hex)
        (self.root/'frames').mkdir(parents=True)
        for frame in range(3):
            cv2.imwrite(str(self.root/'frames'/f'{frame:05d}.png'),np.full((19,31,3),(20,90,130),np.uint8))
            folder=self.root/'matting'/f'{frame:05d}'
            folder.mkdir(parents=True)
            alpha=np.tile(np.linspace(0,255,31,dtype=np.uint8),(19,1))
            cv2.imwrite(str(folder/'0.png'),alpha)
        self.options=ForegroundSettings(resolution=1024,device='cpu',precision='fp32')
        self.job=dict(root=str(self.root),first=0,last=2,extension='png',fps='24000/1001',points=[{'object_id':0}],
            session={'matany_model':'VideoMaMa'},matte_source='matting',foreground_settings=asdict(self.options),
            settings=asdict(AOVSettings(depth=False,normals=False,foreground=True)))
        self.calls=[]
        self.refs=[]

    def tearDown(self):
        assert self.root.parent==Path('.test-output').resolve() and self.root.name.startswith('foreground-unit-')
        shutil.rmtree(self.root)

    def factory(self,settings,status):
        owner=self
        class Fake:
            def predict(self,rgb,alpha):
                owner.calls.append(alpha.copy())
                return np.full((settings.resolution,settings.resolution,3),.4,np.float16)
        model=Fake()
        self.refs.append(weakref.ref(model))
        return model

    def generate(self,job=None,factory=None):
        job=job or self.job
        prepared=prepare(job)
        return run_job(dict(job,**prepared),factory=factory or self.factory)

    def test_cache_reuse_artist_settings_and_model_invalidation(self):
        self.generate()
        # Identical image+alpha can share a prediction across frame indices.
        self.assertEqual(len(self.calls),1)
        changed=replace(self.options,strength=.4,edge_power=2,despill=.2)
        job=dict(self.job,foreground_settings=asdict(changed))
        report=self.generate(job)
        self.assertTrue(report['passes']['foreground']['cached'])
        self.assertEqual(len(self.calls),1)
        read_result(job,0,changed)
        exposure=replace(self.options,exposure_ev=-1)
        with self.assertRaisesRegex(ValueError,'settings changed'): read_result(self.job,0,exposure)
        self.generate(dict(self.job,foreground_settings=asdict(exposure)))
        self.assertEqual(len(self.calls),2)
        self.assertTrue(all(r() is None for r in self.refs))

    def test_source_and_matte_changes_reject_old_result(self):
        self.generate()
        cv2.imwrite(str(self.root/'matting/00001/0.png'),np.full((19,31),100,np.uint8))
        with self.assertRaisesRegex(ValueError,'alpha changed'): read_result(self.job,1,self.options)
        self.generate()
        cv2.imwrite(str(self.root/'frames/00000.png'),np.full((19,31,3),210,np.uint8))
        with self.assertRaisesRegex(ValueError,'plate changed'): read_result(self.job,0,self.options)

    def test_vitmatte_original_refined_validated_inputs_exact(self):
        from sammie.refinement.algorithms import RefinementSettings
        from sammie.refinement.cache import RefinementCache, save_base
        base=np.full((19,31),.12345678,np.float32)
        save_base(self.root,0,0,base,self.root/'frames/00000.png',self.root/'matting/00000/0.png')
        cache=RefinementCache(self.root)
        refined,validated=base+.1,base+.05
        cache.write(0,0,'example',refined=refined,validated=validated,trimap=np.full((19,31),128,np.uint8))
        cache.publish(RefinementSettings(),{'0/0':'example'},{},'test-double')
        for view,expected in [('original',base),('refined',refined),('validated',validated)]:
            job=dict(self.job,last=0,session=dict(self.job['session'],vitmatte_enabled=True,vitmatte_view=view))
            self.generate(job)
            np.testing.assert_array_equal(self.calls[-1],expected)
            _,alpha,_,_=read_result(job,0,self.options)
            np.testing.assert_array_equal(alpha,expected)
        old=dict(job,session=dict(job['session'],vitmatte_view='refined'))
        with self.assertRaisesRegex(ValueError,'alpha changed'): read_result(old,0,self.options)

    def test_cancel_cleanup_resume_and_corrupt_prediction(self):
        self.job['last']=1
        cv2.imwrite(str(self.root/'matting/00001/0.png'),np.full((19,31),190,np.uint8))
        calls=[]
        owner=self
        def factory(settings,status):
            class Interrupting:
                def predict(self,rgb,alpha):
                    calls.append(1)
                    if len(calls)==2: raise InterruptedError('cancelled')
                    return np.full((settings.resolution,settings.resolution,3),.5,np.float16)
            value=Interrupting()
            owner.refs.append(weakref.ref(value))
            return value
        with self.assertRaises(InterruptedError): self.generate(factory=factory)
        self.assertIsNone(manifest(self.root,self.options))
        self.assertTrue(all(r() is None for r in self.refs))
        report=self.generate()
        self.assertEqual(report['passes']['foreground']['reused'],1)
        key=report['passes']['foreground']['records']['0']['key']
        (self.root/'foreground/predictions'/f'{key}.npz').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError,'corrupt'): read_result(self.job,0,self.options)
        self.generate()
        read_result(self.job,0,self.options)

    def test_failed_model_does_not_activate(self):
        def fail(*args): raise RuntimeError('incompatible checkpoint')
        with self.assertRaisesRegex(RuntimeError,'checkpoint'): self.generate(factory=fail)
        self.assertIsNone(manifest(self.root,self.options))

    def test_missing_selected_refinement_does_not_silently_use_base(self):
        job=dict(self.job,session=dict(self.job['session'],vitmatte_enabled=True,vitmatte_view='validated'))
        with self.assertRaisesRegex(ValueError,'ViTMatte result'): prepare(job)

    def test_linear_source_requires_explicit_color(self):
        with self.assertRaisesRegex(ValueError,'actual color space'):
            prepare(dict(self.job,session=dict(self.job['session'],video_file_path='plate.0000.exr')))

    def test_official_refinement_cache_survives_another_pc_model_path(self):
        from sammie.refinement.algorithms import RefinementSettings
        from sammie.refinement.cache import RefinementCache
        cache=RefinementCache(self.root)
        alpha=np.full((19,31),.12345,np.float32)
        cache.write(0,0,'portable',refined=alpha,validated=alpha,trimap=np.full((19,31),128,np.uint8))
        cache.publish(RefinementSettings(),{'0/0':'portable'},{},'pinned-model',
                      model_files={str(self.root/'other-PC'/'model.safetensors'):[123,456]})
        np.testing.assert_array_equal(cache.selected(0,0,RefinementSettings(),'validated'),alpha)

    def test_single_frame_test_preserves_other_cached_frames(self):
        self.generate()
        self.generate(dict(self.job,first=1,last=1))
        for frame in (0,1,2): read_result(self.job,frame,self.options)

    def test_multilayer_export_exact_alpha_rgb_channels_and_numbers(self):
        self.generate()
        report=export_sequence(dict(self.job,destination=str(self.root/'exr'),start_number=993,include_plate=True,include_mattes=True))
        self.assertEqual(report['exported'],3)
        f=OpenEXR.InputFile(str(self.root/'exr/aov.0993.exr'))
        try:
            self.assertEqual(set(f.header()['channels']),{'R','G','B','A','Object_0.Y','foreground.R','foreground.G','foreground.B','foreground.A'})
            self.assertEqual((f.header()['framesPerSecond'].n,f.header()['framesPerSecond'].d),(24000,1001))
            np.testing.assert_array_equal(np.frombuffer(f.channel('A'),np.float32),np.frombuffer(f.channel('foreground.A'),np.float32))
            np.testing.assert_allclose(np.frombuffer(f.channel('R'),np.float32),130/255)
        finally: f.close()
        self.assertTrue((self.root/'exr/aov.0995.exr').is_file())

    def test_acescg_exr_override_preserves_hdr_original_plate(self):
        self.job['last']=0
        rgb=np.full((19,31,3),2.5,np.float32)
        path=self.root/'original.1051.exr'
        write_exr(path,{c:rgb[...,i] for i,c in enumerate('RGB')},{'fps':'24/1'})
        options=replace(self.options,plate_path=str(path),plate_first=1051,color_space='acescg')
        job=dict(self.job,foreground_settings=asdict(options))
        self.generate(job)
        export_sequence(dict(job,destination=str(self.root/'exr'),start_number=1051,include_plate=True,include_mattes=True))
        f=OpenEXR.InputFile(str(self.root/'exr/aov.1051.exr'))
        try: np.testing.assert_array_equal(np.frombuffer(f.channel('R'),np.float32),np.full(19*31,2.5,np.float32))
        finally: f.close()

    def test_depth_normals_and_foreground_share_one_multilayer_exr(self):
        from sammie.aov.pipeline import run_job as generate_geometry
        self.generate()
        aov=AOVSettings(foreground=True,device='cpu',depth_model='vda-small')
        job=dict(self.job,settings=asdict(aov))
        def factory(kind,settings,status):
            class Geometry:
                def predict(self,frames,progress):
                    shape=(len(frames),*frames[0].shape[:2])
                    if kind=='normals': return np.tile(np.array([0,0,1],np.float32),(*shape,1))
                    return np.ones(shape,np.float32)
            return Geometry()
        generate_geometry(job,factory=factory)
        report=export_sequence(dict(job,destination=str(self.root/'exr'),start_number=1,include_plate=True,include_mattes=True))
        self.assertTrue({'foreground.R','foreground.A','N.x','N.y','N.z','Z','depth.Z','Z_raw','A','R'}.issubset(report['channels']))

    def test_stale_alpha_preflight_writes_no_export(self):
        self.generate()
        cv2.imwrite(str(self.root/'matting/00002/0.png'),np.full((19,31),30,np.uint8))
        with self.assertRaisesRegex(ValueError,'alpha changed'):
            export_sequence(dict(self.job,destination=str(self.root/'exr'),start_number=1))
        self.assertFalse((self.root/'exr').exists())


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_controls_presets_and_selection(self):
        from sammie.foreground.ui import ForegroundControls
        w=ForegroundControls()
        self.assertFalse(w.enabled.isChecked())
        self.assertFalse(w.parameters.isEnabled())
        w.enabled.setChecked(True)
        self.assertTrue(w.parameters.isEnabled())
        w.preset.setCurrentText('Green screen / neutral exposure')
        w.apply_preset()
        self.assertEqual(w.settings().screen,'green')
        self.assertEqual(w.settings().exposure_ev,0)
        w.preset.setCurrentText('Blue sky / hair (tested)')
        w.apply_preset()
        self.assertEqual(w.settings().exposure_ev,-2)
        w.close()


if __name__=='__main__': unittest.main()
