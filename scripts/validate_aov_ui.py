"""Exercise the real MainWindow -> QProcess -> model/cache -> preview -> EXR path."""
import argparse
from dataclasses import asdict
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import uuid
import zipfile
from unittest.mock import patch
import cv2
import numpy as np
import OpenEXR
import Imath
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from sammie import core, sammie
from sammie.settings_manager import initialize_settings
from sammie.aov.config import AOVSettings
from sammie.aov.cache import active
from sammie.aov.export import channel_data
import sammie_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--edge', type=int, default=512)
    parser.add_argument('--last', type=int, default=2)
    parser.add_argument('--matte-project', help='Optional original .sammie: copy and verify existing VideoMaMa PNG alpha, without changing the archive.')
    args = parser.parse_args()
    root = Path(args.session).resolve()
    if not root.is_relative_to(Path('.test-output').resolve()):
        raise ValueError('Use a copied .test-output session.')
    app = QApplication.instance() or QApplication([])
    font = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    if font >= 0:
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 9))
    for attr, sub in [('temp_dir',''),('frames_dir','frames'),('mask_dir','masks'),
                      ('matting_dir','matting'),('backup_dir','backup'),('removal_dir','removal')]:
        setattr(core, attr, str(root/sub))
    settings = initialize_settings(str(root))
    settings.app_settings_file = str(root/'ui-application.conf')
    image = cv2.imread(str(root/'frames/00000.png'))
    core.VideoInfo.height, core.VideoInfo.width = image.shape[:2]
    core.VideoInfo.total_frames = len(list((root/'frames').glob('*.png')))
    core.VideoInfo.fps = 24
    core.VideoInfo.fps_rational = '24/1'
    options = AOVSettings(long_edge=args.edge)
    settings.set_session_setting('aov_settings', asdict(options))
    settings.set_session_setting('in_point', 0)
    settings.set_session_setting('out_point', args.last)
    settings.set_session_setting('video_fps_rational', '24/1')
    settings.set_session_setting('aov_start_number', 1001)
    settings.set_session_setting('aov_include_plate', True)
    settings.set_session_setting('aov_include_mattes', True)
    settings.set_session_setting('aov_matte_source', 'matting')
    if args.matte_project:
        import hashlib
        with zipfile.ZipFile(args.matte_project) as archive:
            session = json.loads(archive.read('session_settings.conf'))
            ids = sorted({p['object_id'] for p in json.loads(archive.read('points.json'))})
            if len(ids) != 1 or session.get('vitmatte_enabled'):
                raise ValueError('This validation mode expects one existing unrefined matte. Float/multi-object paths have separate automated tests.')
            for i in range(args.last+1):
                source = f'frames/{i:05d}.png'
                assert hashlib.sha256(archive.read(source)).digest() == hashlib.sha256((root/source).read_bytes()).digest()
                name = f'matting/{i:05d}/{ids[0]}.png'
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.read(name))
            for k,v in session.items():
                if k.startswith('matany_'):
                    settings.set_session_setting(k,v)
        oid = ids[0]
    else:
        oid = 0
        for i in range(args.last+1):
            path = root/'matting'/f'{i:05d}'/'0.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path),np.full(image.shape[:2],128,np.uint8))
    settings.set_session_setting('vitmatte_enabled',False)
    settings.set_session_setting('matany_grow',0)
    settings.set_session_setting('matany_gamma',1.)
    settings.set_session_setting('object_names',{})
    settings.save_session_settings()
    destination = root/('ui-export-'+uuid.uuid4().hex[:6])
    destination.mkdir()
    errors = []
    with patch.object(sammie_main,'initialize_settings',return_value=settings), \
         patch.object(sammie_main.MainWindow,'_deferred_init'), \
         patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
         patch.object(sammie_main,'show_message_dialog',side_effect=lambda *a,**kw: errors.append(kw)), \
         patch.object(sammie_main.QFileDialog,'getExistingDirectory',return_value=str(destination)):
        window = sammie_main.MainWindow()
        try:
            window.resize(1400,900)
            window.show()
            app.processEvents()
            tab = window.sidebar.aov_tab
            window.point_manager.points=[dict(frame=0,object_id=oid,x=10,y=10,positive=True)]
            window.frame_slider.setRange(0,core.VideoInfo.total_frames-1)
            window.sidebar.tab_widget.setCurrentIndex(3)
            tab.generate.click()  # actual model inference through QProcess
            assert not errors, errors
            for kind, label in [('depth','Depth'),('normals','Normals')]:
                assert active(root,kind,options)
                tab.preview.setCurrentText(label)
                app.processEvents()
                window.fit_to_screen()
                app.processEvents()
                window.grab().save(str(destination/f'ui-{kind}.png'))
            tab.generate.click()  # actual cache reuse through QProcess
            assert 'cache reused' in tab.status.text(), tab.status.text()
            scroll = window.sidebar.tab_widget.widget(3)
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            app.processEvents()
            window.grab().save(str(destination/'ui-export-controls.png'))
            tab.export.click()
            assert not errors, errors
            files = sorted(destination.glob('*.exr'))
            assert len(files) == args.last+1
            for i, path in enumerate(files):
                f = OpenEXR.InputFile(str(path))
                try:
                    assert set(f.header()['channels']) == {'R','G','B','A',f'Object_{oid}.Y','Z','depth.Z','Z_raw','N.x','N.y','N.z'}
                    for kind in ('depth','normals'):
                        folder, manifest = active(root,kind,options)
                        expected = channel_data(np.load(folder/f'{i:05d}.npy'),manifest,kind)
                        for channel, values in expected.items():
                            actual = np.frombuffer(f.channel(channel,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(values.shape)
                            # UI and isolated worker use different OpenCV/NumPy
                            # releases; interpolation may differ by a few float32 ULPs.
                            np.testing.assert_allclose(actual, values, rtol=1e-6, atol=1e-7)
                    plate = cv2.cvtColor(cv2.imread(str(root/'frames'/f'{i:05d}.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
                    alpha = cv2.imread(str(root/'matting'/f'{i:05d}'/f'{oid}.png'),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
                    for channel, values in dict(R=plate[...,0],G=plate[...,1],B=plate[...,2],A=alpha,**{f'Object_{oid}.Y':alpha}).items():
                        actual = np.frombuffer(f.channel(channel,Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(values.shape)
                        np.testing.assert_array_equal(actual,values)
                finally:
                    f.close()
            # Validate checkerboard on a synthetic half-transparent matte over real RGB.
            (root/'masks/00000').mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(root/'masks/00000/0.png'),np.full(image.shape[:2],128,np.uint8))
            window.point_manager.points=[dict(frame=0,object_id=0,x=10,y=10,positive=True)]
            window.view_combo.setCurrentText('Segmentation-BGcolor')
            window.background_choice.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(destination/'ui-checkerboard.png'))
            report = dict(passed=True, export_frames=len(files), source_shape=list(image.shape[:2]),
                          channels=['R','G','B','A',f'Object_{oid}.Y','Z','depth.Z','Z_raw','N.x','N.y','N.z'],
                          alpha_source='existing VideoMaMa project' if args.matte_project else 'synthetic alpha',errors=errors)
            (destination/'ui-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2),flush=True)
            print('UI ARTIFACTS',destination,flush=True)
        finally:
            window.close()


if __name__ == '__main__':
    main()
