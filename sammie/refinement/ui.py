"""Optional controls and a cancellable refinement worker in the existing Matting tab."""
from dataclasses import asdict
import threading
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QGroupBox, QFormLayout, QVBoxLayout, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QLineEdit, QFileDialog,
    QDialog, QProgressBar)
from sammie import core
from sammie.settings_manager import get_settings_manager
from .algorithms import RefinementSettings, PRESETS
from .pipeline import run_refinement
from .model import MODEL_SPECS


class RefinementControls(QGroupBox):
    changed = Signal()

    def __init__(self, model_combo):
        super().__init__('ViTMatte Edge Refinement')
        self.setCheckable(True)
        self.setChecked(False)
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.controls = {}
        self.preset = QComboBox()
        self.preset.addItems(['Balanced', 'Fast', 'Quality'])
        form.addRow('Preset:', self.preset)
        model = QComboBox()
        for key, spec in MODEL_SPECS.items():
            model.addItem(spec['label'], key)
        model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        model.setMinimumContentsLength(16)
        model.setToolTip('Official hustvl weights; downloaded automatically on first use.')
        self.controls['model'] = model
        model.currentIndexChanged.connect(self.save)
        form.addRow('Model:', model)
        checkpoint = QLineEdit()
        checkpoint.setPlaceholderText('Automatic official download')
        checkpoint.setToolTip('Optional local Transformers directory: config.json + model.safetensors')
        self.controls['checkpoint'] = checkpoint
        form.addRow('Checkpoint folder:', checkpoint)
        browse = QPushButton('Choose local checkpoint…')
        browse.clicked.connect(self.browse)
        form.addRow(browse)
        for key, title, low, high, step in [
            ('strength', 'Refinement strength:', 0, 1, .05),
            ('radius', 'Unknown radius at 1080p:', 0, 100, 1),
            ('expansion', 'Extra edge expansion:', 0, 100, 1),
            ('foreground', 'Foreground confidence:', .01, 1, .01),
            ('background', 'Background confidence:', 0, .99, .01),
            ('temporal_strength', 'Temporal strength:', 0, 1, .05),
        ]:
            spin = QSpinBox() if isinstance(step, int) else QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            self.controls[key] = spin
            form.addRow(title, spin)
            spin.valueChanged.connect(self.save)
        for key, title in [('temporal', 'Validate temporal changes'), ('debug', 'Export debug trimaps')]:
            box = QCheckBox(title)
            self.controls[key] = box
            form.addRow(box)
            box.toggled.connect(self.save)
        for key, title, choices in [
            ('short_edge', 'Processing short edge:', [('720', 720), ('1080', 1080), ('1440', 1440), ('2160', 2160), ('Source', 0)]),
            ('tile_size', 'Tile size:', [('512', 512), ('768', 768), ('1024', 1024)]),
            ('precision', 'Precision:', [('FP32', 'fp32'), ('BF16 (CUDA)', 'bf16')]),
            ('device', 'Device:', [('Auto', 'auto'), ('CUDA', 'cuda'), ('CPU', 'cpu')]),
        ]:
            combo = QComboBox()
            for text, value in choices: combo.addItem(text, value)
            if key == 'precision':
                combo.setToolTip('FP32 is recommended. Base BF16 showed substantial alpha differences in validation; compare before using it for final mattes.')
            self.controls[key] = combo
            form.addRow(title, combo)
            combo.currentIndexChanged.connect(self.save)
        self.view = QComboBox()
        for label, value in [('A · Original VideoMaMa', 'original'), ('B · ViTMatte merged', 'refined'),
                             ('C · Temporally validated', 'validated')]:
            self.view.addItem(label, value)
        self.view.setToolTip('Selects the matte used by the existing preview AND exports. Switch while scrubbing to compare.')
        form.addRow('Preview / export:', self.view)
        self.run = QPushButton('Refine cached mattes / Resume')
        form.addRow(self.run)
        self.status = QLabel('Optional. Original mattes remain available. Model loads after VideoMaMa unloads.')
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self._loading = False
        checkpoint.editingFinished.connect(self.save)
        self.toggled.connect(self.save)
        self.view.currentIndexChanged.connect(self.save)
        self.preset.activated.connect(self.apply_preset)
        model_combo.currentTextChanged.connect(lambda v: self.setVisible(v == 'VideoMaMa'))
        self.setVisible(model_combo.currentText() == 'VideoMaMa')
        self.load()

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose ViTMatte Transformers checkpoint directory')
        if path:
            self.controls['checkpoint'].setText(path)
            self.save()

    def load(self):
        self._loading = True
        manager = get_settings_manager()
        self.setChecked(manager.get_session_setting('vitmatte_enabled', False))
        for k, widget in self.controls.items():
            value = manager.get_session_setting('vitmatte_'+k, asdict(RefinementSettings())[k])
            if isinstance(widget, QComboBox): widget.setCurrentIndex(widget.findData(value))
            elif isinstance(widget, QCheckBox): widget.setChecked(value)
            elif isinstance(widget, QLineEdit): widget.setText(value)
            else: widget.setValue(value)
        self.view.setCurrentIndex(self.view.findData(manager.get_session_setting('vitmatte_view', 'validated')))
        self._loading = False

    def save(self, *_):
        if self._loading:
            return
        manager = get_settings_manager()
        before = {k: manager.get_session_setting('vitmatte_'+k) for k in self.controls}
        manager.set_session_setting('vitmatte_enabled', self.isChecked())
        manager.set_session_setting('vitmatte_view', self.view.currentData())
        for k, widget in self.controls.items():
            if isinstance(widget, QComboBox): value = widget.currentData()
            elif isinstance(widget, QCheckBox): value = widget.isChecked()
            elif isinstance(widget, QLineEdit): value = widget.text().strip()
            else: value = widget.value()
            manager.set_session_setting('vitmatte_'+k, value)
        if before != {k: manager.get_session_setting('vitmatte_'+k) for k in self.controls}:
            self.status.setText('Changed settings require Refine / Resume. Stale or incomplete results show original mattes.')
        manager.save_session_settings()
        self.changed.emit()

    def apply_preset(self, _):
        manager = get_settings_manager()
        for k, value in PRESETS[self.preset.currentText()].items():
            manager.set_session_setting('vitmatte_'+k, value)
        self.load()
        self.save()


class RefinementWorker(QThread):
    progress = Signal(int, int, str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, root, frames, settings, extension):
        super().__init__()
        self.args = root, frames, settings, extension
        self.cancelled = threading.Event()

    def run(self):
        try:
            self.result.emit(run_refinement(*self.args, progress=self.progress.emit, cancel=self.cancelled.is_set))
        except Exception as e:
            self.error.emit(str(e))


class RefinementDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle('ViTMatte Edge Refinement')
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        self.label = QLabel('Preparing ViTMatte…')
        self.label.setWordWrap(True)
        self.bar = QProgressBar()
        self.button = QPushButton('Cancel')
        for w in (self.label, self.bar, self.button): layout.addWidget(w)
        start, end, _ = core.get_frame_range()
        self.worker = RefinementWorker(core.temp_dir, range(start, end+1), settings, core.get_frame_extension())
        self.report, self.failure = None, None
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.got_result)
        self.worker.error.connect(self.got_error)
        self.worker.finished.connect(self.accept)
        self.button.clicked.connect(self.reject)

    def update_progress(self, done, total, message):
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.label.setText(message)

    def got_result(self, report): self.report = report
    def got_error(self, error): self.failure = error

    def reject(self):
        # Keep dialog and worker alive until cancellation has released model tensors.
        self.worker.cancelled.set()
        self.button.setEnabled(False)
        self.label.setText('Cancelling after the current model operation…')

    def execute(self):
        self.worker.start()
        self.exec()
        self.worker.wait()
        return self.report, self.failure
