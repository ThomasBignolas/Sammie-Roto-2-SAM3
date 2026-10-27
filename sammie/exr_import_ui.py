"""EXR import controls and cancellable, transactional proxy preparation."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QComboBox,
    QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QProgressDialog)
from .exr_media import ImportOptions, inspect, sequence, read_rgb, display_rgb, prepare


class ExrImportDialog(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Import EXR plate / sequence')
        self.path = Path(path).resolve()
        self.files, self.numbers = sequence(path)
        info = inspect(path)
        layout = QFormLayout(self)
        self.range = QComboBox()
        if len(self.files) > 1:
            self.range.addItem(f'Whole sequence: {len(self.files)} frames ({self.numbers[0]} to {self.numbers[-1]})', True)
        self.range.addItem(f'Single image: {self.path.name}', False)
        layout.addRow('Load:', self.range)
        self.layer = QComboBox()
        for layer in info['layers']:
            self.layer.addItem(layer.rstrip('.') or 'R / G / B', layer)
        layout.addRow('RGB layer:', self.layer)
        self.color = QComboBox()
        for label, value in [('Linear Rec.709 / sRGB primaries', 'linear709'), ('ACEScg (linear AP1)', 'acescg'), ('sRGB (already display encoded)', 'srgb')]:
            self.color.addItem(label, value)
        self.color.setCurrentIndex(max(0,self.color.findData(info['suggested_space'])))
        layout.addRow('Input color space:', self.color)
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-12,12)
        self.exposure.setSingleStep(.25)
        layout.addRow('Preview / model exposure (stops):', self.exposure)
        self.fps = QLineEdit(info['fps'])
        layout.addRow('Frame rate (e.g. 24000/1001):', self.fps)
        note = QLabel('Confirm the color space; an EXR does not always identify it. The original HDR files are kept. '
                      'Models use full-resolution sRGB PNG proxies; exposure affects only these proxies. '
                      'RGB is read as stored, ignoring input alpha: use unpremultiplied plates. '
                      'The display window is the canvas; overscan outside it is excluded. No OCIO view/LUT is applied.')
        note.setWordWrap(True)
        note.setMaximumWidth(520)
        layout.addRow(note)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        layout.addRow(self.preview)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addRow(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)
        self._loaded_layer, self._rgb = None, None
        for widget in (self.color, self.layer):
            widget.currentIndexChanged.connect(self.update_preview)
        self.exposure.valueChanged.connect(self.update_preview)
        self.update_preview()

    def options(self):
        return ImportOptions(self.layer.currentData(), self.color.currentData(), self.exposure.value(), self.fps.text().strip())

    def update_preview(self, *_):
        try:
            if self._loaded_layer != self.layer.currentData():
                import cv2
                rgb, _ = read_rgb(self.path, self.layer.currentData())
                ratio = min(1., 500/rgb.shape[1], 260/rgb.shape[0])
                self._rgb = cv2.resize(rgb, (max(1,round(rgb.shape[1]*ratio)), max(1,round(rgb.shape[0]*ratio))))
                self._loaded_layer = self.layer.currentData()
            rgb = display_rgb(self._rgb, self.options())
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image))
            self.error.clear()
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except Exception as error:
            self.error.setText(str(error))
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def validate_and_accept(self):
        try:
            self.options().validate()
            files = self.files if self.range.currentData() else [self.path]
            if len(files) > 1 and self.numbers != list(range(self.numbers[0], self.numbers[0]+len(files))):
                raise ValueError('Missing or duplicate frame numbers. Fill the sequence gaps or select Single image.')
            self.accept()
        except ValueError as error:
            self.error.setText(str(error))


def prepare_dialog(path, session_root, parent=None):
    dialog = ExrImportDialog(path, parent)
    if dialog.exec() != QDialog.Accepted:
        return None
    files = dialog.files if dialog.range.currentData() else [dialog.path]
    progress = QProgressDialog('Importing EXRs...', 'Cancel', 0, len(files), parent)
    progress.setWindowTitle('EXR import')
    progress.setWindowModality(Qt.WindowModal)
    progress.setAutoClose(False)
    progress.show()
    def update(done,total,message):
        progress.setLabelText(message)
        progress.setValue(done)
        QApplication.processEvents()
    try:
        return prepare(files, dialog.options(), Path(session_root).resolve().parent,
                       progress=update, cancel=progress.wasCanceled)
    except InterruptedError:
        return None
    finally:
        progress.close()
