"""SAM3 controls embedded in the existing segmentation tab."""
from functools import wraps
import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                              QPushButton, QComboBox, QDialog, QDialogButtonBox,
                              QListWidget, QListWidgetItem, QMessageBox)
from sammie import core
from sammie.settings_manager import get_settings_manager


def segmentation_action(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            if method.__name__ != 'load_segmentation_model':
                selected = self.segmentation_tab.sam_model_combo.currentText()
                if selected != self.sam_manager.loaded_model_name:
                    raise RuntimeError(f'Click Load Model to activate {selected} before editing or tracking.')
            return method(self, *args, **kwargs)
        except Exception as error:
            self.sam_manager.propagated = False
            self.update_status_bar(str(error))
            QMessageBox.warning(self, 'Segmentation failed', str(error))
    return guarded


class Sam3Controls(QWidget):
    def __init__(self, selector):
        super().__init__()
        self.corner = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text = QLineEdit()
        self.text.setPlaceholderText('Describe objects, e.g. person in a red coat')
        self.find = QPushButton('Find objects')
        self.box = QPushButton('Box: click two corners')
        self.box.setCheckable(True)
        self.box.toggled.connect(lambda checked: setattr(self, 'corner', None))
        self.resolution = QComboBox()
        for label, edge in [('1920 px (recommended)', 1920), ('1280 px', 1280),
                            ('2560 px', 2560), ('Source resolution', 0)]:
            self.resolution.addItem(label, edge)
        self.resolution.setToolTip('Working image long edge; masks are restored to source dimensions. '
                                   'The official network still encodes at 1008 px. Reload model to apply.')
        self.resolution.currentIndexChanged.connect(self._save_resolution)
        layout.addWidget(self.text)
        row = QHBoxLayout()
        row.addWidget(self.find)
        row.addWidget(self.box)
        layout.addLayout(row)
        layout.addWidget(QLabel('Working resolution (long edge):'))
        layout.addWidget(self.resolution)
        self.status = QLabel('Load the model to begin. Up to 16 selected objects.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        selector.currentTextChanged.connect(self.update_model)
        self.update_model(selector.currentText())

    def _save_resolution(self):
        get_settings_manager().set_session_setting('sam3_working_edge', self.resolution.currentData())

    def update_model(self, model):
        self.setVisible(model in ('SAM 3', 'SAM 3.1'))
        self.box.setChecked(False)

    def restore(self):
        edge = get_settings_manager().get_session_setting('sam3_working_edge', 1920)
        self.resolution.setCurrentIndex(max(0, self.resolution.findData(edge)))

    def take_box_click(self, x, y):
        if not self.box.isChecked():
            return None
        if self.corner is None:
            self.corner = (x, y)
            self.status.setText('Click the opposite corner of the bounding box.')
            return False
        rectangle = [*self.corner, x, y]
        self.box.setChecked(False)
        return rectangle


def select_candidates(parent, masks, frame):
    dialog = QDialog(parent)
    dialog.setWindowTitle('Choose objects to track (maximum 16)')
    dialog.resize(640, 480)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel('Check the instances to keep. Each selected instance gets its own object ID.'))
    items = QListWidget()
    items.setIconSize(QSize(192, 128))
    source = core.load_base_frame(frame)
    for oid, mask in masks.items():
        preview = cv2.resize(source, (192, 128), interpolation=cv2.INTER_AREA)
        selected = cv2.resize(mask, (192, 128), interpolation=cv2.INTER_NEAREST) > 0
        preview = (preview * 0.35).astype(np.uint8)
        preview[selected] = (preview[selected] * 0.5 + np.array([0, 160, 70])).clip(0, 255)
        preview = np.ascontiguousarray(preview)
        qimage = QImage(preview.data, 192, 128, 192*3, QImage.Format_RGB888).copy()
        item = QListWidgetItem(QIcon(QPixmap.fromImage(qimage)), f'Detected instance {oid}')
        item.setData(Qt.UserRole, oid)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        items.addItem(item)
    layout.addWidget(items)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.Accepted:
        return []
    return [items.item(i).data(Qt.UserRole) for i in range(items.count())
            if items.item(i).checkState() == Qt.Checked]
