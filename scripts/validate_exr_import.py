"""Real ViTMatte Base CUDA smoke test on imported EXR plates, not an accuracy benchmark.

Uses three frames from the upstream example clip converted to ACEScg EXRs and
synthetic elliptical input alpha. No user project or footage is modified.
"""
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import uuid
import av
import cv2
import numpy as np
import OpenEXR
import torch
from sammie.exr_media import ImportOptions, prepare, commit, project_rgb
from sammie.foreground.color import srgb_to_linear, from_linear709
from sammie.refinement.algorithms import RefinementSettings
from sammie.refinement.pipeline import run_refinement
from sammie.aov.export import write_exr


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This validation requires CUDA; no GPU result was produced.')
    root=Path('.test-output').resolve()/('exr-gpu-'+uuid.uuid4().hex[:8])
    sources=root/'source';sources.mkdir(parents=True)
    files=[]
    with av.open('examples/example_dance.mp4') as video:
        for i,frame in enumerate(video.decode(video=0)):
            if i==3: break
            rgb=cv2.resize(frame.to_ndarray(format='rgb24'),(640,360)).astype(np.float32)/255
            linear=from_linear709(srgb_to_linear(rgb),'acescg')
            linear[0,0]=[.18,2.,8.]  # Verify unclipped HDR plate delivery separately.
            path=sources/f'plate.{1001+i:04d}.exr'
            with OpenEXR.File({'type':OpenEXR.scanlineimage,'framesPerSecond':Fraction(24000,1001)},
                              {c:linear[...,j] for j,c in enumerate('RGB')}) as f:
                f.write(str(path))
            files.append(path)
    session=root/'session'
    stage,info=prepare(files,ImportOptions(color_space='acescg',fps='24000/1001'),root)
    commit(stage,session)
    (session/'session_settings.conf').write_text(json.dumps({'video_file_path':str(files[0])}))
    for frame in range(len(files)):
        folder=session/'matting'/f'{frame:05d}';folder.mkdir()
        alpha=np.zeros((360,640),np.uint8)
        cv2.ellipse(alpha,(320+frame*2,200),(85,140),0,0,360,255,-1)
        alpha=cv2.GaussianBlur(alpha,(9,9),2)
        cv2.imwrite(str(folder/'0.png'),alpha)
    settings=RefinementSettings(model='base-composition-1k',device='cuda',precision='fp32',
                                short_edge=0,tile_size=384,overlap=64)
    result=run_refinement(session,range(len(files)),settings,progress=lambda i,n,t:print(t,flush=True))
    for frame in range(len(files)):
        rgb=project_rgb(session,frame)
        with np.load(session/'vitmatte'/f'{frame:05d}/0.npz') as data:
            alpha=data['validated']
        assert alpha.shape==(360,640) and np.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1
        channels={c:rgb[...,j] for j,c in enumerate('RGB')};channels['A']=alpha
        destination=root/f'validated.{1001+frame}.exr'
        write_exr(destination,channels,dict(fps='24000/1001',exr_source=info))
        with OpenEXR.File(str(destination),separate_channels=True) as f:
            assert f.channels()['B'].pixels[0,0]==8
            assert f.header()['framesPerSecond']==Fraction(24000,1001)
    report=dict(gpu=torch.cuda.get_device_name(),source_shape=[360,640],frames=len(files),
                settings=asdict(settings),refinement=result,hdr_export_verified=True,
                limitation='Synthetic baseline alpha, not VideoMaMa; execution/format validation only, no matte-accuracy claim.')
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(artifacts=str(root),seconds=result['seconds'],peak_allocated_mib=result['peak_allocated_mib'],frames=len(files))))


if __name__=='__main__': main()
