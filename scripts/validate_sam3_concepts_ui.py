"""Actual MainWindow text entry, selection dialog and combined-mask/export check."""
import argparse
import json
import os
import shutil
from pathlib import Path
import uuid
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
import OpenEXR, Imath
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QListWidget, QPushButton
from sammie import core
from sammie.settings_manager import initialize_settings
from sammie.export_workers import SequenceExportWorker
from sammie.export_formats import ExportSettings
import sammie_main


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['SAM 3', 'SAM 3.1'], default='SAM 3')
    p.add_argument('--image', default='.test-output/vitmatte-content/frames/00000.png')
    args = p.parse_args()
    root = Path('.test-output').resolve()/('concept-ui-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    # Use an isolated, unnumbered still so the import doesn't open the sequence-FPS dialog.
    shutil.copy2(args.image, root/'input.png')
    for attr, sub in [('temp_dir','session'),('frames_dir','session/frames'),
                      ('mask_dir','session/masks'),('matting_dir','session/matting'),
                      ('backup_dir','session/masks_backup'),('removal_dir','session/removal')]:
        setattr(core, attr, str(root/sub))
    settings = initialize_settings(core.temp_dir)
    settings.app_settings_file = str(root/'application.conf')
    app = QApplication.instance() or QApplication([])
    font = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    if font >= 0: app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 9))
    core.DeviceManager.setup_device()
    labels = []
    def choose():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QDialog) and dialog.windowTitle().startswith('Choose objects to add'):
            items = dialog.findChild(QListWidget)
            labels.append([items.item(i).text() for i in range(items.count())])
            dialog.grab().save(str(root/f'selection-{len(labels)}.png'))
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Select all').click()
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
    timer = QTimer(); timer.timeout.connect(choose); timer.start(100)
    report = {'model': args.model, 'passed': False}
    with patch.object(sammie_main, 'initialize_settings', return_value=settings), \
            patch.object(sammie_main.MainWindow, '_deferred_init'), \
            patch.object(sammie_main.MainWindow, '_setup_console_redirect'), \
            patch.object(sammie_main.QMessageBox, 'question', return_value=sammie_main.QMessageBox.Yes), \
            patch.object(sammie_main.QMessageBox, 'warning', side_effect=lambda *a: (_ for _ in ()).throw(AssertionError(str(a[-1])))):
        window = sammie_main.MainWindow()
        try:
            assert window.load_file(str(root/'input.png'))
            window.segmentation_tab.sam_model_combo.setCurrentText(args.model)
            window.segmentation_tab.sam_model_btn.click()
            controls = window.segmentation_tab.sam3_controls
            controls.text.setText('person and camera'); controls.find.click()
            assert len(window.point_manager.points) == 2
            assert all(any(phrase in label for label in labels[0]) for phrase in ('person', 'camera'))
            window.track_objects()
            assert window.sam_manager.propagated
            masks = [cv2.imread(str(Path(core.mask_dir)/f'00000/{oid}.png'), 0) for oid in (0, 1)]
            combined, _ = core.load_combined_mask(0, [0, 1])
            np.testing.assert_array_equal(combined, np.maximum(*masks))
            destination = root/'export'; destination.mkdir()
            settings.set_session_setting('sequence_start_frame', 1001)
            options = ExportSettings('exr', str(destination), 'mask', 'Segmentation-Matte', -1, False, 14, False, None, None)
            SequenceExportWorker(options,window.point_manager.points,1,'mask',window)._export_exr_sequence()
            exr = OpenEXR.InputFile(str(destination/'mask.1001.exr'))
            try:
                assert set(exr.header()['channels']) == {'Object_0.Y', 'Object_1.Y'}
                for oid, mask in enumerate(masks):
                    alpha = np.frombuffer(exr.channel(f'Object_{oid}.Y', Imath.PixelType(Imath.PixelType.FLOAT)),np.float32).reshape(mask.shape)
                    np.testing.assert_array_equal(alpha, mask.astype(np.float32)/255)
            finally: exr.close()
            window.clear_all_points()
            for phrase in ('person', 'camera'):
                controls.text.setText(phrase); controls.find.click()
            assert len(window.point_manager.points) == 2
            assert len(settings.load_points()) == 2
            window._update_current_frame_display(); app.processEvents()
            window.grab().save(str(root/'combined-selection.png'))
            report.update(passed=True, labels=labels, export_channels=['Object_0.Y','Object_1.Y'],
                          dimensions=[core.VideoInfo.width,core.VideoInfo.height], gpu=window.sam_manager.gpu)
        finally:
            timer.stop()
            window.close()
            (root/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2)); print('Artifacts:',root)


if __name__ == '__main__': main()
