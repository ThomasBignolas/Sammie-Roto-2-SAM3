"""Real trained image inference through MainWindow controls, in an isolated test session."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import uuid
from unittest.mock import patch
import av
import cv2
import numpy as np
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from sammie import core
from sammie.settings_manager import initialize_settings
import sammie_main


def main():
    root = Path('.test-output').resolve()/('ui-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    for attr, sub in [('temp_dir', 'session'), ('frames_dir', 'session/frames'),
                      ('mask_dir', 'session/masks'), ('matting_dir', 'session/matting'),
                      ('backup_dir', 'session/masks_backup'), ('removal_dir', 'session/removal')]:
        setattr(core, attr, str(root/sub))
    settings = initialize_settings(core.temp_dir)
    settings.app_settings_file = str(root/'application.conf')
    with av.open('examples/example_dance.mp4') as container:
        first = next(container.decode(video=0)).to_ndarray(format='bgr24')
    cv2.imwrite(str(root/'input.png'), first)
    app = QApplication.instance() or QApplication([])
    if Path('C:/Windows/Fonts/segoeui.ttf').is_file():
        font_id = QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    core.DeviceManager.setup_device()
    with patch.object(sammie_main, 'initialize_settings', return_value=settings), \
            patch.object(sammie_main.MainWindow, '_deferred_init'), \
            patch.object(sammie_main.MainWindow, '_setup_console_redirect'), \
            patch.object(sammie_main, 'select_candidates', side_effect=lambda _, masks, frame: list(masks)[:2]), \
            patch.object(sammie_main.QMessageBox, 'question', return_value=sammie_main.QMessageBox.Yes), \
            patch.object(sammie_main.QMessageBox, 'warning', side_effect=AssertionError('Unexpected UI error dialog')):
        window = sammie_main.MainWindow()
        window.resize(1400, 1000)
        app.processEvents()
        try:
            assert window.load_file(str(root/'input.png'))
            assert core.VideoInfo.total_frames == 1
            for name in ('SAM 3', 'SAM 3.1'):
                window.clear_all_points()
                window.segmentation_tab.sam_model_combo.setCurrentText(name)
                window.segmentation_tab.sam_model_btn.click()
                assert window.sam_manager.loaded_model_name == name
                controls = window.segmentation_tab.sam3_controls
                assert controls.isVisible()
                controls.text.setText('person')
                controls.find.click()
                assert len(window.point_manager.points) == 2
                mask = cv2.imread(str(Path(core.mask_dir)/'00000/0.png'), 0)
                assert mask is not None and mask.any()
                ys, xs = np.nonzero(mask)
                window.segmentation_tab.object_spinbox.setValue(0)
                window.add_point_from_click(int(xs[len(xs)//2]), int(ys[len(ys)//2]), True)
                assert len(window.point_manager.points) == 3
                window.track_objects()
                assert window.sam_manager.propagated
                window._update_current_frame_display()
                app.processEvents()
                window.grab().save(str(root/(name.replace(' ', '')+'.png')))
                print(name, 'MainWindow single-image selection, correction and tracking: PASSED')
                window.clear_all_points()
                for oid, x in ((0, .8), (1, .27)):
                    window.segmentation_tab.object_spinbox.setValue(oid)
                    window.add_point_from_click(int(core.VideoInfo.width*x), core.VideoInfo.height//2, True)
                window.track_objects()
                for oid in (0, 1):
                    mask = cv2.imread(str(Path(core.mask_dir)/f'00000/{oid}.png'), 0)
                    assert mask is not None and np.count_nonzero(mask) > 100
                window.clear_all_points()
                controls.text.clear()
                controls.box.setChecked(True)
                window.add_point_from_click(core.VideoInfo.width//4, core.VideoInfo.height//5, True)
                window.add_point_from_click(core.VideoInfo.width*3//4, core.VideoInfo.height*9//10, True)
                assert window.point_manager.points and all(p.get('kind') == 'sam3_seed' for p in window.point_manager.points)
                print(name, 'independent manual objects and UI box selection: PASSED')
            window.clear_all_points()
            window.segmentation_tab.sam_model_combo.setCurrentText('Base')
            window.segmentation_tab.sam_model_btn.click()
            assert window.sam_manager.loaded_model_name == 'Base'
            assert window.segmentation_tab.sam3_controls.isHidden()
            window.add_point_from_click(core.VideoInfo.width//2, core.VideoInfo.height//2, True)
            assert len(window.point_manager.points) == 1
            print('Switch back to SAM 2 and point segmentation: PASSED')
        finally:
            window.close()
    print('UI screenshots:', root)


if __name__ == '__main__':
    main()
