"""Exercise the actual MainWindow refinement button/worker/preview/export path.

Run against an isolated session copy: --session .test-output/vitmatte-content
Does not load SAM or rerun VideoMaMa. No model inference is mocked.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import OpenEXR, Imath
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from sammie import core, sammie
from sammie.settings_manager import initialize_settings
from sammie.export_workers import SequenceExportWorker
from sammie.export_formats import ExportSettings
from sammie.refinement.model import MODEL_SPECS
import sammie_main


def main(root, run_matting=False, model=None):
    root=Path(root).resolve()
    if not root.is_relative_to(Path('.test-output').resolve()):
        raise ValueError('Use an isolated .test-output session copy for this validation.')
    app=QApplication.instance() or QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    if font>=0: app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],9))
    for attr,name in [('temp_dir',''),('frames_dir','frames'),('mask_dir','masks'),
                      ('matting_dir','matting'),('backup_dir','backup'),('removal_dir','removal')]:
        setattr(core,attr,str(root/name))
    settings=initialize_settings(str(root));settings.app_settings_file=str(root/'app.conf')
    settings.load_session_settings()
    frames=sorted((root/'frames').glob('*.png')); first=cv2.imread(str(frames[0]))
    core.VideoInfo.height,core.VideoInfo.width=first.shape[:2]
    core.VideoInfo.total_frames=len(frames)
    settings.set_session_setting('in_point',0);settings.set_session_setting('out_point',len(frames)-1)
    settings.set_session_setting('matany_model','VideoMaMa')
    settings.set_session_setting('vitmatte_enabled',True)
    if model:
        settings.set_session_setting('vitmatte_model',model)
        settings.set_session_setting('vitmatte_checkpoint','')
    settings.set_session_setting('sequence_start_frame',1001)
    if run_matting:
        settings.set_session_setting('matany_res',480)
        settings.set_session_setting('matany_chunk',3)
        settings.set_session_setting('matany_overlap',1)
    settings.save_session_settings()
    core.DeviceManager.setup_device()
    with patch.object(sammie_main,'initialize_settings',return_value=settings), \
         patch.object(sammie_main.MainWindow,'_deferred_init'), \
         patch.object(sammie_main.MainWindow,'_setup_console_redirect'), \
         patch.object(sammie_main,'show_message_dialog',side_effect=lambda *a,**k: (_ for _ in ()).throw(AssertionError(k))):
        window=sammie_main.MainWindow()
        window.point_manager.points=[{'frame':0,'object_id':0,'x':10,'y':10,'positive':True}]
        window.frame_slider.setRange(0,len(frames)-1)
        window.sidebar.load_values_from_settings()
        window.sidebar.tab_widget.setCurrentIndex(1)
        app.processEvents()
        assert window.view_combo.currentText()=='Matting-Matte'
        controls=window.matting_tab.refinement
        assert controls.isVisible() and controls.isChecked()
        assert controls.controls['model'].currentData()==settings.get_session_setting('vitmatte_model')
        if run_matting:
            window.matting_tab.run_matting_btn.click()  # actual automatic two-model handoff
            assert window.matany_manager.pipeline is None
        else:
            controls.run.click()  # actual modal worker + actual model inference
        assert (root/'vitmatte/active.json').exists()
        values=[]
        for stage in ('original','refined','validated'):
            controls.view.setCurrentIndex(controls.view.findData(stage))
            values.append(sammie.update_image(1,{'view_mode':'Matting-Matte','float_alpha':True},window.point_manager.points,True)[...,0])
        assert not np.array_equal(values[0],values[1])
        export=root/'ui-export';export.mkdir(exist_ok=True)
        options=ExportSettings('exr',str(export),'test','Matting-Matte',-1,False,14,False,None,None)
        SequenceExportWorker(options,window.point_manager.points,len(frames),'test',window)._export_exr_sequence()
        f=OpenEXR.InputFile(str(export/'test.1002.exr'))
        try: a=np.frombuffer(f.channel('Object_0.Y',Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(first.shape[:2])
        finally: f.close()
        np.testing.assert_array_equal(a,values[2])
        window.resize(1500,1000);app.processEvents()
        window.sidebar.tab_widget.widget(1).ensureWidgetVisible(controls)
        window.grab().save(str(root/'vitmatte-ui.png'))
        window.close()
    print('PASS: actual MainWindow worker, A/B/C preview, exact float EXR channels and numbering; GPU allocated MiB:',torch.cuda.memory_allocated()/1048576)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True)
    p.add_argument('--run-matting',action='store_true',help='Also execute Run Matting and its automatic refinement handoff')
    p.add_argument('--model',choices=MODEL_SPECS,help='Select official model in the copied session')
    args=p.parse_args();main(args.session,args.run_matting,args.model)
