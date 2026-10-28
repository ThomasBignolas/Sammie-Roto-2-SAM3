"""A self-contained Nuke delivery: aligned float EXRs and a ready-to-open graph."""
from dataclasses import asdict
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import cv2
import numpy as np
from sammie.projects import atomic_json


def script_text(first,last,fps,width,height,passes):
    """Fixed node names and relative paths avoid executable user text in .nk files."""
    lines=['version 13.2 v1', 'add_layer {N N.x N.y N.z}',
           'add_layer {foreground foreground.red foreground.green foreground.blue foreground.alpha}',
           f'Root {{\n first_frame {first}\n last_frame {last}\n fps {float(Fraction(str(fps))):.12g}\n colorManagement Nuke\n format "{width} {height} 0 0 {width} {height} 1 Sammie"\n}}',
           'Read {\n inputs 0\n file "[file dirname [value root.name]]/frames/sammie.%04d.exr"\n'+
           f' first {first}\n last {last}\n origfirst {first}\n origlast {last}\n raw true\n name Sammie_Passes\n label "Linear Rec709 RGB; data channels untouched"\n xpos 0\n ypos 0\n}}',
           'set SOURCE [stack 0]',
           'Shuffle {\n in rgba\n alpha white\n name Original_Plate\n xpos -300\n ypos 120\n}',
           'set PLATE [stack 0]', 'push $SOURCE',
           'Shuffle {\n in rgba\n red alpha\n green alpha\n blue alpha\n alpha white\n name Generated_Alpha\n xpos 200\n ypos 120\n}',
           'set ALPHA [stack 0]', 'push $SOURCE',
           'Premult {\n name Plate_With_Alpha\n xpos 0\n ypos 180\n}',
           'set CUTOUT [stack 0]',
           f'CheckerBoard2 {{\n inputs 0\n format "{width} {height} 0 0 {width} {height} 1 Sammie"\n name Checkerboard\n xpos -180\n ypos 260\n}}',
           'set CHECKER [stack 0]', 'push $CUTOUT',
           'Merge2 {\n inputs 2\n operation over\n name Plate_On_Checker\n xpos 0\n ypos 340\n}',
           'set COMP [stack 0]']
    if 'foreground' in passes:
        lines += ['push $SOURCE','Shuffle {\n in foreground\n name Recovered_Foreground\n xpos 420\n ypos 120\n}',
                  'Premult {\n name Premult_Foreground\n xpos 420\n ypos 180\n}', 'set FG [stack 0]',
                  'push $CHECKER','push $FG','Merge2 {\n inputs 2\n operation over\n name Foreground_On_Checker\n xpos 420\n ypos 340\n}',
                  'set FGCOMP [stack 0]']
    for kind,layer,channels,x in [('normals','N',('red','green','blue'),680),('depth','depth',('red','red','red'),900)]:
        if kind in passes:
            lines += ['push $SOURCE',f'Shuffle {{\n in {layer}\n red {channels[0]}\n green {channels[1]}\n blue {channels[2]}\n alpha white\n name {kind.title()}_Data\n label "Raw data; depth is relative, not metres"\n xpos {x}\n ypos 120\n}}']
    lines += ['push $ALPHA', 'push $FGCOMP' if 'foreground' in passes else 'push $PLATE',
              'push $COMP','push $PLATE','Viewer {\n inputs 4\n name Viewer1\n xpos 0\n ypos 460\n}']
    return '\n'.join(lines)+'\n'


def available_passes(job):
    from sammie.aov.cache import active
    from sammie.aov.config import AOVSettings
    from sammie.foreground.cache import manifest
    from sammie.foreground.config import ForegroundSettings
    settings=AOVSettings.from_dict(job['session'].get('aov_settings',{}))
    first,last=job['requested']; indices=set(range(first,last+1)); found=[]
    for kind in ('depth','normals'):
        cached=active(job['root'],kind,settings)
        if cached and indices.issubset(cached[1]['frames']): found.append(kind)
    fg=manifest(job['root'],ForegroundSettings.from_dict(job['session'].get('foreground_settings',{})))
    if fg and set(map(str,indices)).issubset(fg['records']): found.append('foreground')
    return found


def recover_delivery(root):
    """Remove only a staging directory recorded by this project's exporter."""
    journal=Path(root)/'.workflow/delivery-staging.json'
    if not journal.exists(): return
    data=json.loads(journal.read_text())
    target=Path(data['parent']).resolve()
    token=data['token']
    if len(token)!=32 or any(c not in '0123456789abcdef' for c in token):
        raise ValueError('Invalid Nuke staging recovery token.')
    stage=target/('.sammie-delivery-'+token)
    if stage.exists():
        marker=stage/'.sammie-delivery-owner.json'
        if stage.resolve().parent!=target or not marker.is_file() or json.loads(marker.read_text()).get('project')!=str(Path(root).resolve()):
            raise ValueError('Nuke staging recovery ownership could not be verified.')
        shutil.rmtree(stage)
    journal.unlink(missing_ok=True)


def deliver(job,progress=lambda *args:None):
    from sammie.aov.export import write_exr,channel_data
    from sammie.aov.cache import active
    from sammie.aov.config import AOVSettings
    from sammie.aov.matte_layers import MatteLayers
    from sammie.exr_media import project_info,project_rgb
    from sammie.foreground.color import to_linear709
    from sammie.foreground.cache import read_result
    from sammie.foreground.config import ForegroundSettings
    root=Path(job['root']); session=job['session']; first,last=job['requested']
    width,height=session['video_width'],session['video_height']
    fps=session.get('video_fps_rational') or str(session.get('video_fps') or 24)
    start=session.get('sequence_start_frame',0)+first
    target=Path(job.get('delivery_parent') or Path(job['project']).parent/'nuke').resolve()
    target.mkdir(parents=True,exist_ok=True)
    name=Path(job['project']).stem+'-'+uuid.uuid4().hex[:10]
    destination=target/name
    recover_delivery(root)
    token=uuid.uuid4().hex
    stage=target/('.sammie-delivery-'+token)
    stage.mkdir()
    atomic_json(stage/'.sammie-delivery-owner.json',{'project':str(root.resolve())})
    atomic_json(root/'.workflow/delivery-staging.json',dict(parent=str(target),token=token))
    (stage/'frames').mkdir()
    passes=job.get('nuke_passes',available_passes(job))
    available=available_passes(job)
    if any(p not in available for p in passes):
        recover_delivery(root)
        raise ValueError('A selected Nuke pass is missing or stale. Generate it first.')
    request=dict(job,first=first,last=last,extension=session.get('frame_format','png'),fps=fps,
                 matte_source=job.get('matte_source',session.get('aov_matte_source','matting')))
    metadata=dict(fps=fps,source_frames=list(range(first,last+1)),start_number=start,
                  plate='Straight linear Rec709, converted from imported RGB for Nuke. Original project plate unchanged.',
                  depth='Relative shot-normalized depth; not metres',passes=passes)
    try:
        mattes=MatteLayers(request,(height,width))
        # Never silently substitute original alpha when the selected refined view is stale.
        if request['matte_source']=='matting' and session.get('vitmatte_enabled') and session.get('vitmatte_view')!='original':
            from sammie.refinement.cache import RefinementCache
            cache=RefinementCache(root)
            settings=cache.applied_settings()
            for frame in range(first,last+1):
                for oid in mattes.ids:
                    if settings is None or cache.selected(frame,oid,settings,session.get('vitmatte_view','validated')) is None:
                        raise ValueError(f'ViTMatte is missing/stale at timeline frame {frame}; refine it or choose Original VideoMaMa.')
        info=project_info(root)
        if info: metadata['exr_source']=dict(info,options=dict(info['options'],color_space='linear709'))
        aov=AOVSettings.from_dict(session.get('aov_settings',{}))
        cached={p:active(root,p,aov) for p in passes if p!='foreground'}
        fgsettings=ForegroundSettings.from_dict(session.get('foreground_settings',{}))
        for frame in range(first,last+1):
            progress(frame-first,last-first+1,f'Nuke delivery: frame {frame}')
            if info:
                plate=to_linear709(project_rgb(root,frame),info['options']['color_space'])
            else:
                rgb=cv2.imread(str(root/'frames'/f"{frame:05d}.{request['extension']}"))
                if rgb is None: raise ValueError(f'Missing plate frame {frame}')
                plate=to_linear709(rgb[...,::-1].astype(np.float32)/255,'srgb')
            channels=mattes.channels(frame)
            channels.update({c:plate[...,i] for i,c in enumerate('RGB')})
            for kind,(folder,manifest) in cached.items():
                channels.update(channel_data(np.load(folder/f'{frame:05d}.npy',allow_pickle=False),manifest,kind))
            if 'foreground' in passes:
                foreground,alpha,_,_=read_result(request,frame,fgsettings,mattes,channels['A'])
                foreground=to_linear709(foreground,fgsettings.color_space)
                channels.update({f'foreground.{c}':foreground[...,i] for i,c in enumerate('RGB')})
                channels['foreground.A']=alpha
            write_exr(stage/'frames'/f'sammie.{start+frame-first:04d}.exr',channels,metadata)
        (stage/'sammie.nk').write_text(script_text(start,start+last-first,fps,width,height,passes),encoding='utf-8')
        atomic_json(stage/'delivery.json',metadata)
        stage.rename(destination)
        return dict(directory=str(destination.resolve()),script=str((destination/'sammie.nk').resolve()),passes=passes)
    finally:
        recover_delivery(root)


def find_nuke(saved=''):
    if saved and Path(saved).is_file(): return saved
    candidates=list(Path(os.environ.get('ProgramFiles','C:/Program Files')).glob('Nuke*/Nuke*.exe'))
    candidates=[p for p in candidates if p.stem.lower().startswith('nuke') and p.stem[4:5].isdigit()]
    return str(max(candidates,key=lambda p:p.stat().st_mtime)) if candidates else ''


def launch(executable,script):
    if not Path(executable).is_file() or not Path(script).is_file(): raise ValueError('Nuke executable or script is missing.')
    return subprocess.Popen([executable,str(script)],cwd=str(Path(script).parent))
