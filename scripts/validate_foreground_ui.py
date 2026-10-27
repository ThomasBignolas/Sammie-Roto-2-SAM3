"""Exercise actual UI -> CPU alpha preparation -> GPU worker -> multilayer EXR.

Run only against a copied session beneath .test-output, never the live project.
"""
import argparse
from dataclasses import asdict
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import uuid
from unittest.mock import patch
import cv2
import numpy as np
import OpenEXR
import Imath
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.settings_manager import initialize_settings
from sammie.aov.config import AOVSettings
from sammie.aov.matte_layers import MatteLayers
from sammie.foreground.config import ForegroundSettings
from sammie.foreground.cache import manifest, read_result
from sammie.foreground.color import load_rgb, source_path
import sammie_main


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    parser.add_argument('--first',type=int,default=0)
    parser.add_argument('--last',type=int,default=2)
    parser.add_argument('--plate',default='')
    parser.add_argument('--plate-first',type=int,default=0)
    parser.add_argument('--color',choices=['srgb','linear709','acescg'],default='srgb')
    parser.add_argument('--resolution',type=int,default=2048)
    args=parser.parse_args()
    root=Path(args.session).resolve()
    if not root.is_relative_to(Path('.test-output').resolve()):
        raise ValueError('Use a copied .test-output session.')
    app=QApplication.instance() or QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    if font>=0: app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],9))
    for attr,sub in [('temp_dir',''),('frames_dir','frames'),('mask_dir','masks'),('matting_dir','matting'),('backup_dir','backup'),('removal_dir','removal')]:
        setattr(core,attr,str(root/sub))
    manager=initialize_settings(str(root))
    manager.app_settings_file=str(root/'test-app.conf')
    image=cv2.imread(str(root/'frames'/f'{args.first:05d}.png'))
    core.VideoInfo.height,core.VideoInfo.width=image.shape[:2]
    core.VideoInfo.total_frames=max(args.last+1,len(list((root/'frames').glob('*.png'))))
    core.VideoInfo.fps=24
    core.VideoInfo.fps_rational='24/1'
    options=ForegroundSettings(plate_path=args.plate,plate_first=args.plate_first,color_space=args.color,resolution=args.resolution)
    for key,value in dict(aov_settings=asdict(AOVSettings(depth=False,normals=False,foreground=True)),
        foreground_settings=asdict(options),in_point=args.first,out_point=args.last,
        aov_start_number=993+args.first,video_fps_rational='24/1',aov_matte_source='matting',
        aov_include_plate=True,aov_include_mattes=True,background_style='checkerboard').items():
        manager.set_session_setting(key,value)
    manager.save_session_settings()
    errors=[]
    destination=root/('export-'+uuid.uuid4().hex[:6])
    destination.mkdir()
    with patch.object(sammie_main,'initialize_settings',return_value=manager), \
         patch.object(sammie_main.MainWindow,'_deferred_init'), \
         patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
         patch.object(sammie_main,'show_message_dialog',side_effect=lambda *a,**k:errors.append(k)), \
         patch.object(sammie_main.QFileDialog,'getExistingDirectory',return_value=str(destination)):
        window=sammie_main.MainWindow()
        try:
            window.resize(1500,1000)
            window.show()
            window.point_manager.points=[dict(frame=0,object_id=0,x=10,y=10,positive=True)]
            window.frame_slider.setRange(0,core.VideoInfo.total_frames-1)
            window.frame_slider.setValue(args.first)
            window.sidebar.tab_widget.setCurrentIndex(3)
            app.processEvents()
            tab=window.sidebar.aov_tab
            tab.generate.click()
            assert not errors,errors
            cached=manifest(root,options)
            assert cached and cached['gpu'],cached
            assert all(str(i) in cached['records'] for i in range(args.first,args.last+1))
            predictions=list((root/'foreground/predictions').glob('*.npz'))
            stamps={p.name:p.stat().st_mtime_ns for p in predictions}
            tab.preview.setCurrentText('Foreground on background')
            window.fit_to_screen()
            app.processEvents()
            scroll=window.sidebar.tab_widget.widget(3)
            scroll.verticalScrollBar().setValue(450)
            app.processEvents()
            window.grab().save(str(destination/'foreground-ui.png'))
            # Live artist adjustments must reuse model predictions.
            tab.foreground.controls['strength'].setValue(.75)
            app.processEvents()
            assert stamps=={p.name:p.stat().st_mtime_ns for p in predictions}
            tab.foreground.controls['strength'].setValue(1.)
            tab.foreground.test.click()  # Reuses current frame and retains other frames.
            assert not errors,errors
            assert 'cache reused' in tab.status.text(),tab.status.text()
            tab.export.click()
            assert not errors,errors
            expected_job=tab.make_job()
            expected_job.update(points=window.point_manager.points,session=asdict(manager.session_settings))
            mattes=MatteLayers(expected_job,image.shape[:2])
            for frame in range(args.first,args.last+1):
                path=destination/f'aov.{993+frame:04d}.exr'
                f=OpenEXR.InputFile(str(path))
                try:
                    channels=set(f.header()['channels'])
                    assert channels=={'R','G','B','A','Object_0.Y','foreground.R','foreground.G','foreground.B','foreground.A'},channels
                    shape=image.shape[:2]
                    expected=mattes.channels(frame)['A']
                    for name in ('A','foreground.A'):
                        np.testing.assert_array_equal(np.frombuffer(f.channel(name,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(shape),expected)
                    original=load_rgb(source_path(root,frame,'png',options))
                    fg,_,_,_=read_result(expected_job,frame,options,mattes)
                    for i,c in enumerate('RGB'):
                        rgb=np.frombuffer(f.channel(c,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(shape)
                        recovered=np.frombuffer(f.channel('foreground.'+c,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(shape)
                        np.testing.assert_array_equal(rgb,original[...,i])
                        np.testing.assert_array_equal(recovered[expected==1],original[...,i][expected==1])
                        np.testing.assert_array_equal(recovered,fg[...,i])
                finally: f.close()
            report=dict(passed=True,frames=list(range(args.first,args.last+1)),dimensions=[image.shape[1],image.shape[0]],
                settings=asdict(options),alpha_source=manager.get_session_setting('vitmatte_model'),
                alpha_view=manager.get_session_setting('vitmatte_view'),alpha_max_error=0.,opaque_rgb_max_error=0.,
                cache_tuning=True,current_frame_reuse=True,channels=sorted(channels),gpu=cached['gpu'],
                total_seconds=cached['seconds'],inference_seconds=cached['inference_seconds'],errors=errors)
            (destination/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2),flush=True)
            print('ARTIFACTS',destination,flush=True)
        finally: window.close()


if __name__=='__main__': main()
