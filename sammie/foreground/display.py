"""Display only: recovered RGB is composited in linear Rec.709, then encoded sRGB."""
from dataclasses import asdict
import cv2
import numpy as np
from .config import ForegroundSettings
from .cache import read_result
from .color import to_linear709, linear_to_srgb, srgb_to_linear


def preview(frame, points, view_options):
    from sammie import core
    from sammie.settings_manager import get_settings_manager
    manager=get_settings_manager()
    settings=ForegroundSettings.from_dict(manager.get_session_setting('foreground_settings',{}))
    job=dict(root=core.temp_dir,extension=core.get_frame_extension(),points=points,
             matte_source=manager.get_session_setting('aov_matte_source','matting'),session=asdict(manager.session_settings))
    try:
        rgb,alpha,_,_=read_result(job,frame,settings)
        linear=to_linear709(rgb,settings.color_space)
        if view_options['view_mode']=='AOV-Foreground-BGcolor':
            h,w=alpha.shape
            if view_options.get('background_style')=='checkerboard':
                cell=max(8,round(min(h,w)/40))
                rows=((np.arange(h)//cell)%2).astype(np.uint8)
                cols=((np.arange(w)//cell)%2).astype(np.uint8)
                grid=rows[:,None]^cols[None,:]
                background=np.repeat(np.where(grid[...,None],160/255,104/255).astype(np.float32),3,axis=2)
            else:
                background=np.broadcast_to(np.array(view_options.get('bgcolor',(0,255,0)),np.float32)/255,linear.shape)
            linear=linear*alpha[...,None]+srgb_to_linear(background)*(1-alpha[...,None])
        return np.round(np.clip(linear_to_srgb(linear),0,1)*255).astype(np.uint8)
    except (OSError,ValueError,KeyError) as error:
        image=core.load_base_frame(frame)
        if image is None: return None
        image=image.copy()
        message='CorridorKey unavailable - Generate for this alpha/settings'
        cv2.putText(image,message,(20,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,200,80),2,cv2.LINE_AA)
        return image
