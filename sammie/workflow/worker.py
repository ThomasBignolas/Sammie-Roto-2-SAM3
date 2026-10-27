"""One stage per process: GPU resources cannot accumulate between queue stages."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import traceback
from sammie.projects import ProjectLock, atomic_json, bind_root, read_project
from .jobs import frame_range, RangeTransaction, recover, signature, project_revision


def output_signature(job,stage):
    from sammie.aov.cache import fingerprint
    import hashlib
    root=Path(job['root']); first,last=frame_range(job,stage)
    folders={'tracking':['masks'],'matting':['matting','matting_float'],'vitmatte':['vitmatte']}.get(stage,[])
    values=[]
    for folder in folders:
        for frame in range(first,last+1):
            for path in sorted((root/folder/f'{frame:05d}').glob('*')):
                if path.is_file(): values.append([str(path.relative_to(root)),fingerprint(path)['sha256']])
    pointers={'depth':'aov/depth.json','normals':'aov/normals.json','foreground':'foreground/active.json'}
    if stage in pointers:
        p=root/pointers[stage]
        if p.exists():
            values.append([pointers[stage],fingerprint(p)['sha256']])
            data=json.loads(p.read_text())
            if stage in ('depth','normals'):
                folder=root/'aov'/data['key']
                files=[folder/'manifest.json',*sorted(folder.glob('*.npy'))]
            else:
                files=[root/'foreground/predictions'/f"{r['key']}.npz" for r in data['records'].values()]
            for path in files:
                values.append([str(path.relative_to(root)),fingerprint(path)['sha256'] if path.exists() else None])
    if stage=='vitmatte':
        p=root/'vitmatte/active.json'
        values.append(['active',fingerprint(p)['sha256'] if p.exists() else None])
    if stage=='nuke':
        receipt=root/'.workflow'/f"{job['id']}-nuke.json"
        if receipt.exists():
            info=json.loads(receipt.read_text())
            for path in sorted(Path(info['directory']).rglob('*')):
                if path.is_file(): values.append([str(path),fingerprint(path)['sha256']])
    return hashlib.sha256(json.dumps(values).encode()).hexdigest() if values else None


def setup(job,stage):
    from sammie import core,settings_manager as sm
    manager=sm.initialize_settings(job['root'])
    keys={f.name for f in fields(sm.SessionSettings)}
    manager.session_settings=sm.SessionSettings(**{k:v for k,v in job['session'].items() if k in keys})
    bind_root(job['root'],manager)
    first,last=frame_range(job,stage)
    manager.session_settings.in_point=first; manager.session_settings.out_point=last
    for field,key in [('width','video_width'),('height','video_height'),('total_frames','total_frames'),
                      ('fps','video_fps'),('fps_rational','video_fps_rational'),('color_space','color_space')]:
        setattr(core.VideoInfo,field,job['session'].get(key,getattr(core.VideoInfo,field)))
    return manager


def run_stage(job,stage,progress):
    from sammie import core
    manager=setup(job,stage)
    root=Path(job['root']); first,last=frame_range(job,stage)
    if stage in ('tracking','matting'):
        from PySide6.QtWidgets import QApplication
        app=QApplication.instance() or QApplication([])
        core.DeviceManager.setup_device()
    if stage=='tracking':
        import shutil
        from sammie.backends.manager import SamManager
        model=SamManager()
        original=core.mask_dir
        scratch=root/'.workflow/replay-masks'
        scratch.mkdir(parents=True,exist_ok=True)
        try:
            if not model.load_segmentation_model(): raise RuntimeError('SAM model could not be loaded.')
            model.initialize_predictor()
            core.mask_dir=str(scratch)  # Replaying saved prompts must not overwrite outside the repair.
            model.replay_points(job['points'])
            core.mask_dir=original
            anchor=job['anchor']
            seen=set()
            for start,count,reverse in ((anchor,last-anchor,False),(anchor,anchor-first,True)):
                if reverse and not count: continue
                iterator=model.iter_masks(start_frame_idx=start,max_frame_num_to_track=count,reverse=reverse)
                try:
                    for frame,masks in iterator:
                        if first<=frame<=last:
                            model.write_masks(frame,masks);seen.add(frame)
                            progress(len(seen),last-first+1,f'Tracking frame {frame}')
                finally: iterator.close()
            if seen!=set(range(first,last+1)): raise RuntimeError('Tracking returned an incomplete range.')
        finally:
            core.mask_dir=original
            model.unload_segmentation_model()
            if scratch.exists(): shutil.rmtree(scratch)
    elif stage=='matting':
        from sammie import matting
        model=matting.create_matting_manager()
        from tqdm import tqdm
        class Progress:
            def wasCanceled(self): return False
            def close(self): pass
            def setValue(self,value): progress(value,100,'Matting')
            def setLabelText(self,text): progress(0,0,text)
        model._make_progress_dialog=lambda parent,total,unit='frame': (Progress(),tqdm(total=total,unit=unit))
        try:
            if not model.load_matting_model(): raise RuntimeError('Matting model could not be loaded.')
            if not model.run_matting(job['points'],None,combined=manager.session_settings.matany_combined):
                raise RuntimeError('Matting did not complete.')
            ids={p['object_id'] for p in job['points']}
            if manager.session_settings.matany_combined and len(ids)>1: ids={0}
            for frame in range(first,last+1):
                for oid in ids:
                    if not (root/'matting'/f'{frame:05d}'/f'{oid}.png').is_file():
                        raise RuntimeError(f'Matting omitted frame {frame}, object {oid}.')
        finally: model.unload_matting_model()
    elif stage=='vitmatte':
        from sammie.refinement.pipeline import run_refinement
        from sammie.refinement.algorithms import RefinementSettings
        return run_refinement(root,range(first,last+1),RefinementSettings.from_session(manager),
                              core.get_frame_extension(),progress)
    else:
        from sammie.aov.config import AOVSettings
        from dataclasses import asdict, replace
        options=AOVSettings.from_dict(job['session'].get('aov_settings',{}))
        options=replace(options,depth=stage=='depth',normals=stage=='normals',foreground=stage=='foreground')
        request=dict(root=str(root),first=first,last=last,extension=core.get_frame_extension(),
                     session=job['session'],points=job['points'],settings=asdict(options),
                     foreground_settings=job['session'].get('foreground_settings',{}),
                     fps=job['session'].get('video_fps_rational') or str(core.VideoInfo.fps or 24),
                     sequence_start=job['session'].get('sequence_start_frame',0),
                     matte_source=job['session'].get('aov_matte_source','matting'))
        if stage in ('depth','normals'):
            from sammie.aov.pipeline import run_job
            return run_job(request,progress)
        if stage=='foreground':
            from sammie.foreground.pipeline import prepare,run_job
            return run_job(prepare(request,progress),progress)
        if stage=='nuke':
            from .nuke import deliver
            result=deliver(job,progress)
            atomic_json(root/'.workflow'/f"{job['id']}-nuke.json",result)
            return result
    return {'completed':True}


def execute(job,stage,runner=run_stage,progress=lambda *args:None):
    root,_=read_project(job['project'])
    if root!=Path(job['root']).resolve(): raise ValueError('Queued project moved; remove and add it again.')
    with ProjectLock(root):
        recover(root)
        if job.get('revision') and project_revision(root)!=job['revision']:
            raise ValueError('Project settings or prompts changed after queueing. Remove this queued item and add the updated project again.')
        before=signature(job,stage)
        prior=job.get('completed',{}).get(stage)
        if prior and prior['input']==before and prior.get('output')==output_signature(job,stage):
            return dict(receipt=prior,cached=True)
        first,last=frame_range(job,stage)
        with RangeTransaction(root,stage,first,last):
            result=runner(job,stage,progress)
            after=output_signature(job,stage)
            if after is None: raise RuntimeError(f'{stage} produced no outputs.')
        return dict(receipt=dict(input=before,output=after),result=result,cached=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('request');args=p.parse_args()
    request=Path(args.request)
    job=json.loads(request.read_text())
    def progress(n,total,text):
        print('__WORKFLOW__'+json.dumps(dict(done=n,total=total,message=text)),flush=True)
    try:
        result=execute(job,job['stage'],progress=progress)
        atomic_json(request.with_suffix('.result.json'),result)
    except BaseException as error:
        traceback.print_exc()
        atomic_json(request.with_suffix('.result.json'),{'error':str(error)})
        raise SystemExit(1)


if __name__=='__main__':
    os.environ['SAMMIE_QUEUE_WORKER']='1'
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    main()
