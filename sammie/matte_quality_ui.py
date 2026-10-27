"""Read-only current-frame quality review; overlays never enter the export path."""
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QScrollArea,
    QDialogButtonBox)
from .matte_quality import inspect_alpha


class MatteQualityDialog(QDialog):
    def __init__(self, rgb, alpha, guide, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Matte quality inspection - frame {frame} (all objects)')
        self.resize(1050, 800)
        self.rgb, self.alpha, self.guide = rgb, alpha, guide
        layout = QVBoxLayout(self)
        note = QLabel('Review candidates only: magenta = low alpha inside the guide; cyan = alpha outside it. '
                      'Real transparency, hair, motion blur and inaccurate SAM guides can also be flagged. '
                      'This does not change or export your matte. Uses the current Matting preview alpha, '
                      'including its ViTMatte A/B/C selection, postprocessing and any base-alpha fallback.')
        note.setWordWrap(True); layout.addWidget(note)
        controls = QHBoxLayout(); form = QFormLayout()
        self.margin = QSpinBox(); self.margin.setRange(0, 128); self.margin.setValue(16)
        self.foreground = QDoubleSpinBox(); self.foreground.setRange(.01, 1); self.foreground.setSingleStep(.01); self.foreground.setValue(.95)
        self.background = QDoubleSpinBox(); self.background.setRange(0, .99); self.background.setSingleStep(.01); self.background.setValue(.05)
        form.addRow('Ignore edge band (px at 1080):', self.margin)
        form.addRow('Interior alpha below:', self.foreground)
        form.addRow('Exterior alpha above:', self.background)
        controls.addLayout(form)
        views = QFormLayout()
        self.view = QComboBox(); self.view.addItems(['RGB + flags', 'Alpha + flags', 'Alpha only', 'Guide only'])
        self.zoom = QComboBox(); self.zoom.addItems(['Fit', '25%', '50%', '100%'])
        views.addRow('View:', self.view); views.addRow('Zoom:', self.zoom)
        self.apply = QPushButton('Inspect with these thresholds'); views.addRow(self.apply)
        controls.addLayout(views); layout.addLayout(controls)
        self.stats = QLabel(); self.stats.setWordWrap(True); layout.addWidget(self.stats)
        self.image = QLabel(); self.image.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea(); self.scroll.setAlignment(Qt.AlignCenter); self.scroll.setWidget(self.image)
        layout.addWidget(self.scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.apply.clicked.connect(self.analyze)
        self.view.currentIndexChanged.connect(self.render)
        self.zoom.currentIndexChanged.connect(self.render)
        self.flags = None
        self.analyze()

    def analyze(self):
        try:
            flags = inspect_alpha(self.alpha, self.guide, self.margin.value(),
                                  self.foreground.value(), self.background.value())
        except ValueError as exc:
            self.stats.setText(str(exc) + ' Previous inspection retained.'); return
        self.flags = flags
        inside, outside = np.count_nonzero(flags.interior), np.count_nonzero(flags.exterior)
        self.stats.setText(f'Magenta: {inside:,} pixels. Cyan: {outside:,} pixels. '
                           f'Excluded edge band: {flags.margin_pixels}px at source resolution. '
                           f'{self.alpha.shape[1]} x {self.alpha.shape[0]}. Use 100% to inspect fine detail.')
        self.render()

    def render(self):
        if self.flags is None: return
        mode = self.view.currentIndex()
        if mode == 0:
            image = self.rgb.copy()
        else:
            values = self.guide if mode == 3 else np.round(self.alpha*255).astype(np.uint8)
            image = cv2.cvtColor(values, cv2.COLOR_GRAY2RGB)
        if mode < 2:
            for mask, color in ((self.flags.interior, (255, 48, 200)), (self.flags.exterior, (0, 235, 255))):
                image[mask] = np.round(image[mask]*.25 + np.array(color)*.75).astype(np.uint8)
        h, w = image.shape[:2]
        zoom = self.zoom.currentText()
        scale = min(1., max(1,self.scroll.viewport().width())/w,
                    max(1,self.scroll.viewport().height())/h) if zoom == 'Fit' else float(zoom[:-1])/100
        if scale != 1:
            image = cv2.resize(image, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image)
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.image.setPixmap(pixmap); self.image.resize(pixmap.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'flags') and self.zoom.currentText() == 'Fit': self.render()
