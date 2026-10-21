"""Real VideoMaMa -> unload -> ViTMatte validation on isolated example frames.

No user session is modified. Requires the existing VideoMaMa/SVD weights; the
normal downloader opens if absent. Reports measured timings, not accuracy.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import cv2
import numpy as np
import torch
from PySide6.QtWidgets import QApplication, QWidget, QSlider
from sammie import core
from sammie.matting import VideoMaMaManager
from sammie.settings_manager import initialize_settings
from sammie.refinement.algorithms import RefinementSettings
from sammie.refinement.model import VitMatteModel, resolve_checkpoint
from sammie.refinement.pipeline import run_refinement


def main():
    app = QApplication.instance() or QApplication([])
    root = Path('.test-output').resolve()/('vitmatte-gpu-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    for attr, name in [('temp_dir',''),('frames_dir','frames'),('mask_dir','masks'),('matting_dir','matting')]:
        setattr(core, attr, str(root/name))
    manager = initialize_settings(str(root))
    manager.app_settings_file = str(root/'app.conf')
    manager.set_session_setting('matany_model','VideoMaMa')
    manager.set_session_setting('matany_res',480)
    manager.set_session_setting('matany_chunk',3)
    manager.set_session_setting('matany_overlap',1)
    manager.set_session_setting('vitmatte_enabled',True)
    manager.set_session_setting('vitmatte_short_edge',720)
    (root/'frames').mkdir()
    count = 4
    with av.open('examples/example_dance.mp4') as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == count: break
            rgb = cv2.resize(frame.to_ndarray(format='bgr24'), (1920,1080))
            cv2.imwrite(str(root/'frames'/f'{i:05d}.png'), rgb)
            (root/'masks'/f'{i:05d}').mkdir(parents=True)
            mask = np.zeros((1080,1920),np.uint8)
            cv2.ellipse(mask,(960,600),(240,420),0,0,360,255,-1)
            cv2.imwrite(str(root/'masks'/f'{i:05d}/0.png'),mask)
    core.VideoInfo.width,core.VideoInfo.height,core.VideoInfo.total_frames = 1920,1080,count
    core.DeviceManager.setup_device()
    parent=QWidget(); parent.frame_slider=QSlider(parent); parent.frame_slider.setRange(0,count-1)
    points=[{'frame':0,'object_id':0,'positive':True,'x':960,'y':600}]
    model=VideoMaMaManager()
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    assert model.load_matting_model(parent_window=parent)
    assert model.run_matting(points,parent)==1
    baseline={'seconds_including_load':time.perf_counter()-start,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
              'frames':count,'working_short_edge':480,'chunk':3,'overlap':1}
    # Verify opt-in float return leaves the original uint8 conversion unchanged.
    image = cv2.cvtColor(cv2.imread(str(root/'frames/00000.png')),cv2.COLOR_BGR2RGB)
    inputs = [cv2.resize(image,(256,256))]*2
    masks = [np.full((256,256),128,np.uint8)]*2
    with torch.autocast('cuda',enabled=False):
        torch.manual_seed(123)
        legacy = model.pipeline.run(inputs,masks,return_float=False)
        torch.manual_seed(123)
        floating = model.pipeline.run(inputs,masks,return_float=True)
    assert all(np.array_equal(a,(b*255).clip(0,255).astype(np.uint8)) for a,b in zip(legacy,floating))
    baseline['default_uint8_vs_float_conversion_exact'] = True
    model.unload_matting_model()
    baseline['allocated_after_unload_mib']=torch.cuda.memory_allocated()/1048576
    for f in range(count):
        with np.load(root/'matting_float'/f'{f:05d}/0.npz') as a:
            assert a['alpha'].shape==(1080,1920) and a['alpha'].dtype==np.float32
    config=RefinementSettings.from_session(manager)
    result=run_refinement(root,range(count),config,progress=lambda i,n,t: print(t,flush=True))
    assert result['legacy_alpha_frames']==0
    assert torch.cuda.memory_allocated()/1048576 < 50
    # CPU fallback and BF16 are compared with FP32 on the same actual pretrained model.
    rgb=cv2.cvtColor(cv2.imread(str(root/'frames/00000.png')),cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,(256,144));tri=np.full((144,256),128,np.uint8)
    checkpoint,_=resolve_checkpoint(config)
    comparisons={}
    reference=None
    for device,precision in [('cuda','fp32'),('cuda','bf16'),('cpu','fp32')]:
        m=VitMatteModel(replace(config,device=device,precision=precision),checkpoint)
        start=time.perf_counter()
        try: a=m.predict(rgb,tri)
        finally: m.close()
        if reference is None: reference=a
        comparisons[device+'-'+precision]={'seconds_including_load':time.perf_counter()-start,
            'mae_vs_cuda_fp32':float(np.abs(a-reference).mean()),'max_error':float(np.abs(a-reference).max())}
    report={'hardware':torch.cuda.get_device_name(),'videomama':baseline,'refinement':result,'precision':comparisons}
    (root/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Artifacts:',root)
    return root


if __name__=='__main__': main()
