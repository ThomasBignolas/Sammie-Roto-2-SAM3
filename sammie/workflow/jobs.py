"""Durable queue and stage signatures; no GUI or neural-model imports."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid
from sammie.projects import atomic_json, read_project

STAGES = ('tracking','matting','vitmatte','depth','normals','foreground','nuke')
LABELS = dict(tracking='SAM tracking',matting='Matting',vitmatte='ViTMatte',depth='Depth',
              normals='Normals',foreground='Foreground RGB',nuke='Nuke delivery')


def project_revision(root):
    root=Path(root)
    session=json.loads((root/'session_settings.conf').read_text())
    settings={k:v for k,v in session.items() if k.startswith(('sam','matany_','vitmatte_','aov_','foreground_','video_'))
              or k in ('total_frames','frame_format','sequence_start_frame','holes','dots','grow','border_fix','contact_seam_width','object_names')}
    points=json.loads((root/'points.json').read_text()) if (root/'points.json').exists() else []
    return hashlib.sha256(json.dumps([settings,points],sort_keys=True).encode()).hexdigest()


def make_job(descriptor, stages, first=None, last=None, context=16, anchor=None):
    root,_=read_project(descriptor)
    session=json.loads((root/'session_settings.conf').read_text())
    points=json.loads((root/'points.json').read_text()) if (root/'points.json').exists() else []
    total=int(session.get('total_frames',0))
    first = 0 if first is None else int(first)
    last = total-1 if last is None else int(last)
    if not 0 <= first <= last < total: raise ValueError('Choose a valid inclusive timeline range.')
    if not stages or any(s not in STAGES for s in stages): raise ValueError('Choose processing stages.')
    if not 0 <= context <= 256: raise ValueError('Context handles must be between 0 and 256 frames.')
    if ('tracking' in stages or 'matting' in stages) and not points:
        raise ValueError('Select objects and save the project before queueing tracking/matting.')
    if 'vitmatte' in stages and session.get('matany_model') != 'VideoMaMa':
        raise ValueError('ViTMatte refinement requires VideoMaMa mattes.')
    effective=[max(0,first-context),min(total-1,last+context)]
    return dict(id=uuid.uuid4().hex,project=str(Path(descriptor).resolve()),root=str(root),revision=project_revision(root),
                session=session,points=points,stages=[s for s in STAGES if s in stages],
                requested=[first,last],effective=effective,context=context,
                anchor=max(effective[0],min(effective[1],first if anchor is None else anchor)),
                status='pending',completed={},error='',progress='Ready')


def frame_range(job, stage):
    if stage in ('depth','normals'): return 0,int(job['session']['total_frames'])-1
    return tuple(job['requested'] if stage=='nuke' else job['effective'])


def signature(job, stage):
    """Content hashes bind stage reuse to actual inputs, settings and model revisions."""
    from sammie.aov.cache import fingerprint
    root=Path(job['root'])
    first,last=frame_range(job,stage)
    session=job['session']
    fields = {'tracking':('sam_model','sam3_working_edge'),
              'matting':('sam_model','holes','dots','border_fix','grow','contact_seam_width'),
              'vitmatte':(), 'depth':(), 'normals':(), 'foreground':(), 'nuke':()}
    settings={k:v for k,v in session.items() if k in fields[stage] or
              (stage in ('matting','foreground','nuke') and k.startswith('matany_')) or
              (stage in ('vitmatte','foreground','nuke') and k.startswith('vitmatte_')) or
              (stage in ('depth','normals','foreground','nuke') and k.startswith(('aov_','foreground_')))}
    directories=['frames']
    if stage in ('matting','foreground','nuke'): directories+=['masks']
    if stage in ('vitmatte','foreground','nuke'): directories+=['matting','matting_float']
    if stage in ('foreground','nuke'): directories+=['vitmatte','source_exr']
    inputs=[]
    for directory in directories:
        folder=root/directory
        if not folder.exists(): continue
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or p.suffix in ('.tmp','.part'): continue
            relative=p.relative_to(folder)
            number=relative.parts[0].split('.')[0]
            if number.isdecimal() and not first <= int(number) <= last: continue
            if directory=='vitmatte' and len(relative.parts)==1 and p.name!='active.json': continue
            f=fingerprint(p)
            inputs.append([directory+'/'+relative.as_posix(),f['sha256']])
    checkpoint=session.get('vitmatte_checkpoint')
    if checkpoint and stage in ('vitmatte','foreground','nuke'):
        for path in sorted(Path(checkpoint).glob('*')):
            if path.is_file() and path.suffix in ('.json','.bin','.safetensors'):
                inputs.append(['checkpoint/'+path.name,fingerprint(path)['sha256']])
    if stage=='nuke':
        from .worker import output_signature
        for upstream in ('depth','normals','foreground'):
            inputs.append([upstream,output_signature(job,upstream)])
    # Hash implementation and pinned runtime/model configs, avoiding unrelated docs/UI edits.
    app=Path(__file__).resolve().parents[2]
    versions=[]
    for p in sorted((app/'sammie').rglob('*.py')):
        if p.name.endswith('_ui.py') or p.name=='ui.py': continue
        versions.append([p.relative_to(app).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest()])
    value=dict(version=1,stage=stage,range=[first,last],settings=settings,inputs=inputs,
               points=job['points'] if stage in ('tracking','matting','foreground','nuke') else [],code=versions)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class QueueStore:
    def __init__(self,path):
        self.path=Path(path)
        self.jobs=json.loads(self.path.read_text())['jobs'] if self.path.exists() else []
        for job in self.jobs:
            if job['status']=='running':
                job.update(status='interrupted',progress='Interrupted; resume will validate completed stages.')

    def save(self): atomic_json(self.path,dict(version=1,jobs=self.jobs))
    def add(self,job): self.jobs.append(job); self.save()

    def resume(self):
        for job in self.jobs:
            if job['status'] in ('failed','cancelled','interrupted'):
                job.update(status='pending',error='')
        self.save()


class RangeTransaction:
    """Back up changed output frames, with recovery after a killed worker/crash."""
    def __init__(self,root,stage,first,last):
        self.root=Path(root).resolve()
        self.folder=self.root/'.workflow/transaction'
        if self.folder.exists(): recover(self.root)
        directories={'tracking':['masks'],'matting':['matting','matting_float'],
                     'vitmatte':['vitmatte']}.get(stage,[])
        self.names=[f'{d}/{f:05d}' for d in directories for f in range(first,last+1)]
        self.names += {'vitmatte':['vitmatte/active.json'],'foreground':['foreground/active.json'],
                       'depth':['aov/depth.json'],'normals':['aov/normals.json']}.get(stage,[])

    def __enter__(self):
        import shutil
        present=[]
        self.folder.mkdir(parents=True)
        try:
            for name in self.names:
                path=self.root/name
                if path.exists():
                    target=self.folder/'before'/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if path.is_dir(): shutil.copytree(path,target)
                    else: shutil.copy2(path,target)
                    present.append(name)
            atomic_json(self.folder/'journal.json',dict(names=self.names,present=present))
        except BaseException:
            shutil.rmtree(self.folder)
            raise
        return self

    def __exit__(self,kind,*args):
        import shutil
        if kind: recover(self.root)
        else:
            (self.folder/'journal.json').unlink(missing_ok=True)
            shutil.rmtree(self.folder)


def recover(root):
    import shutil
    root=Path(root).resolve()
    from .nuke import recover_delivery
    recover_delivery(root)
    folder=root/'.workflow/transaction'
    journal=folder/'journal.json'
    if journal.exists():
        data=json.loads(journal.read_text())
        for name in data['names']:
            path=root/name
            if not path.resolve().is_relative_to(root) or Path(name).parts[0] not in ('masks','matting','matting_float','vitmatte','foreground','aov'):
                raise ValueError('Invalid interrupted-job recovery path.')
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink(missing_ok=True)
            if name in data['present']:
                backup=folder/'before'/name
                path.parent.mkdir(parents=True,exist_ok=True)
                if backup.is_dir(): shutil.copytree(backup,path)
                else: shutil.copy2(backup,path)
    if folder.exists():
        assert folder.resolve().parent==root/'.workflow'
        shutil.rmtree(folder)
