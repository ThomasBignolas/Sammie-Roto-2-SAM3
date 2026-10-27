"""Exercise working-project saves, real ViTMatte CUDA workers and Nuke delivery.

Uses a tiny synthetic sequence; this is an execution test, not an accuracy test.
"""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import cv2
import numpy as np
from sammie.projects import create_project,read_project,save_metadata,pack_project,atomic_json
from sammie.settings_manager import SessionSettings
from sammie.workflow.jobs import make_job


def main():
    root=Path('.test-output/workflow').resolve()/('validation-'+uuid.uuid4().hex[:8])
    source=root/'session';(source/'frames').mkdir(parents=True)
    session=asdict(SessionSettings(video_width=384,video_height=216,total_frames=3,video_fps=24,
                    video_fps_rational='24/1',sequence_start_frame=1001,matany_model='VideoMaMa',
                    vitmatte_enabled=True,vitmatte_model='base-composition-1k',vitmatte_short_edge=0,
                    vitmatte_tile_size=384,vitmatte_overlap=64,vitmatte_device='cuda'))
    points=[dict(frame=0,object_id=0,x=190,y=110,positive=True)]
    for frame in range(3):
        rgb=np.full((216,384,3),[80,110,140],np.uint8)
        cv2.ellipse(rgb,(190+frame,110),(65,85),0,0,360,[190,130,80],-1)
        alpha=np.zeros((216,384),np.uint8)
        cv2.ellipse(alpha,(190+frame,110),(65,85),0,0,360,255,-1)
        alpha=cv2.GaussianBlur(alpha,(9,9),2)
        cv2.imwrite(str(source/'frames'/f'{frame:05d}.png'),rgb)
        folder=source/'matting'/f'{frame:05d}';folder.mkdir(parents=True)
        cv2.imwrite(str(folder/'0.png'),alpha)
    (source/'save-benchmark.bin').write_bytes(bytes(range(256))*262144)  # 64 MiB payload.
    save_metadata(source,session,points)
    descriptor=create_project(source,root/'shot.sammieproj')
    data,_=read_project(descriptor)
    measurements=[]
    for _ in range(3):
        began=time.perf_counter();save_metadata(data,session,points,descriptor);measurements.append(time.perf_counter()-began)
    began=time.perf_counter();pack_project(data,root/'portable.sammie');pack_seconds=time.perf_counter()-began
    job=make_job(descriptor,['vitmatte','nuke'],context=0)
    job['delivery_parent']=str(root/'nuke')
    reports={}
    for stage in job['stages']:
        request=root/f'{stage}.json';atomic_json(request,dict(job,stage=stage))
        with (root/f'{stage}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u','-m','sammie.workflow.worker',str(request)],
                                  stdout=log,stderr=subprocess.STDOUT)
        response=json.loads(request.with_suffix('.result.json').read_text())
        if result.returncode:raise RuntimeError(response)
        reports[stage]=response
        job['completed'][stage]=response['receipt']
    # Reuse a completed stage through the same actual worker.
    request=root/'resume.json';atomic_json(request,dict(job,stage='vitmatte'))
    result=subprocess.run([sys.executable,'-u','-m','sammie.workflow.worker',str(request)],check=True,capture_output=True)
    assert json.loads(request.with_suffix('.result.json').read_text())['cached']
    report=dict(root=str(root),save_seconds=measurements,pack_seconds=pack_seconds,
                payload_mib=64,reports=reports,limitation='Synthetic 384x216 three-frame input and synthetic baseline alpha; no accuracy claim. Cached local storage benchmark, not a large-shot timing estimate.')
    atomic_json(root/'result.json',report)
    atomic_json(root.parent/'latest-validation.json',dict(root=str(root)))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
