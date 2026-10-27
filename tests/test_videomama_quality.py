import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
import cv2
import numpy as np
import torch
from videomama.pipeline_svd_mask_numpy import VideoInferencePipeline


class Distribution:
    def __init__(self, x): self.x = x
    def sample(self, generator=None):
        return self.x[:, :1]*.2 + torch.randn(self.x[:, :1].shape, generator=generator)*.1


class TinyVAE:
    config = SimpleNamespace(scaling_factor=1.)
    def encode(self, x): return SimpleNamespace(latent_dist=Distribution(x))
    def decode(self, x, num_frames): return SimpleNamespace(sample=x.repeat(1,3,1,1))


class TinyUNet:
    config = SimpleNamespace(cross_attention_dim=4, addition_time_embed_dim=1)
    add_embedding = SimpleNamespace(linear_1=SimpleNamespace(in_features=3))
    def __call__(self, x, *args, **kwargs):
        return SimpleNamespace(sample=x[:, :, :1]*.1+x[:, :, 1:2]+x[:, :, 2:3])


def tiny_pipeline():
    model = VideoInferencePipeline.__new__(VideoInferencePipeline)
    model.device=torch.device('cpu'); model.weight_dtype=torch.float32
    model.enable_model_cpu_offload=False; model.vae_encode_chunk_size=1
    model.vae=TinyVAE(); model.unet=TinyUNet()
    return model


class VideoPrecisionTests(unittest.TestCase):
    def test_mixed_float_and_byte_frames_keep_sub_byte_values(self):
        byte = np.full((4,6), 128, np.uint8)
        alpha = np.full((4,6), .123456, np.float32)
        result = tiny_pipeline()._cv2_to_tensor([byte, alpha])
        self.assertEqual(tuple(result.shape), (1,2,3,4,6))
        self.assertAlmostEqual(float(result[0,0,0,0,0]), 128/255*2-1, places=6)
        self.assertAlmostEqual(float(result[0,1,0,0,0]), .123456*2-1, places=7)

    def test_uint8_preprocessing_is_exactly_unchanged(self):
        frames=[np.arange(72,dtype=np.uint8).reshape(4,6,3)]
        old=torch.stack([torch.from_numpy(x.copy()).permute(2,0,1).float()/255 for x in frames]).unsqueeze(0)*2-1
        torch.testing.assert_close(tiny_pipeline()._cv2_to_tensor(frames), old, rtol=0, atol=0)

    def test_invalid_float_scales_and_nonfinite_values_are_rejected(self):
        for value in (-.1, 1.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tiny_pipeline()._cv2_to_tensor([np.full((2,2),value,np.float32)])
        with self.assertRaises(ValueError):
            tiny_pipeline()._cv2_to_tensor([np.zeros((2,2),np.uint16)])

    def test_full_inference_seed_covers_vae_and_preserves_global_rng(self):
        model=tiny_pipeline()
        rgb=[np.full((8,8,3),80,np.uint8) for _ in range(3)]
        masks=[np.full((8,8),.555555,np.float32) for _ in range(3)]
        torch.manual_seed(123);before=torch.get_rng_state().clone()
        first=model.run(rgb,masks,seed=42,return_float=True)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        torch.manual_seed(999)
        second=model.run(rgb,masks,seed=42,return_float=True)
        np.testing.assert_array_equal(first,second)
        third=model.run(rgb,masks,seed=43,return_float=True)
        self.assertFalse(np.array_equal(first,third))


class VideoHandoffTests(unittest.TestCase):
    def test_real_chunk_loop_passes_float_feedback_and_keeps_frame_order(self):
        from PySide6.QtWidgets import QApplication, QWidget, QSlider
        from sammie import core
        from sammie.matting import VideoMaMaManager
        from sammie.settings_manager import initialize_settings
        app=QApplication.instance() or QApplication([])
        root=Path('.test-output').resolve()/('handoff-'+uuid.uuid4().hex);root.mkdir(parents=True)
        patches=[patch.object(core,'temp_dir',str(root)),patch.object(core,'frames_dir',str(root/'frames')),
                 patch.object(core,'mask_dir',str(root/'masks')),patch.object(core,'matting_dir',str(root/'matting')),
                 patch.object(core.VideoInfo,'width',64),patch.object(core.VideoInfo,'height',64),
                 patch.object(core.VideoInfo,'total_frames',20),
                 patch.object(core.DeviceManager,'get_device',return_value=torch.device('cpu')),
                 patch.object(core.DeviceManager,'clear_cache')]
        for p in patches:p.start()
        parent=QWidget();parent.frame_slider=QSlider()
        try:
            settings=initialize_settings(str(root))
            for k,v in dict(vitmatte_enabled=True,matany_res=64,matany_chunk=16,matany_overlap=4).items():settings.set_session_setting(k,v)
            (root/'frames').mkdir()
            for i in range(20):
                (root/'masks'/f'{i:05d}').mkdir(parents=True)
                cv2.imwrite(str(root/'frames'/f'{i:05d}.png'),np.full((64,64,3),i,np.uint8))
                cv2.imwrite(str(root/'masks'/f'{i:05d}/0.png'),np.full((64,64),255,np.uint8))
            calls=[]
            def run(cond_frames,mask_frames,**kwargs):
                calls.append([x.copy() for x in mask_frames])
                return [np.full((64,64,3),.123456+int(x[0,0,0])*.001,np.float32) for x in cond_frames]
            manager=VideoMaMaManager();manager.pipeline=SimpleNamespace(run=run)
            self.assertEqual(manager.run_matting([{'object_id':0}],parent,combined=False),1)
            self.assertEqual([len(x) for x in calls],[16,8])
            for i in range(4):
                self.assertEqual(calls[1][i].dtype,np.float32)
                self.assertAlmostEqual(float(calls[1][i][0,0]),.123456+(12+i)*.001,places=6)
            self.assertEqual(calls[1][4].dtype,np.uint8)
            for i in range(20):
                with np.load(root/'matting_float'/f'{i:05d}/0.npz') as a:
                    self.assertAlmostEqual(float(a['alpha'][32,32]),.123456+i*.001,places=6)
        finally:
            parent.close()
            for p in reversed(patches):p.stop()
            assert root.parent==Path('.test-output').resolve() and root.name.startswith('handoff-')
            shutil.rmtree(root)


if __name__=='__main__':unittest.main()
