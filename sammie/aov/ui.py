"""Fourth sidebar tab and a responsive, cancellable subprocess progress dialog."""
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path
import uuid
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QCheckBox, QComboBox, QSpinBox, QPushButton, QDialog, QProgressBar)
from sammie import core
from sammie.settings_manager import get_settings_manager
from .config import AOVSettings
from .versions import ROOT
from .cache import atomic_json


class AOVTab(QWidget):
    generate_requested = Signal()
    export_requested = Signal()
    preview_requested = Signal(str)
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.loading = True
        layout = QVBoxLayout(self)
        title = QLabel('AOVs · Depth and Surface Normals')
        title.setWordWrap(True)
        layout.addWidget(title)
        group = QGroupBox('Generate from the original plate')
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.controls = {}
        for key, title in [('depth', 'Depth AOV'), ('normals', 'Normal AOV')]:
            widget = QCheckBox(title)
            self.controls[key] = widget
            widget.toggled.connect(self.save)
            form.addRow(widget)
        choices = {
            'depth_model': ('Depth model:', [('DVD v1.1 (non-commercial)', 'dvd'), ('Video Depth Anything Small', 'vda-small')]),
            'long_edge': ('Processing long edge:', [(str(n), n) for n in (384, 512, 640, 768, 1024)]),
            'offload': ('GPU memory:', [('Model CPU offload', 'model'), ('Sequential CPU offload (slower)', 'sequential')]),
            'device': ('Device:', [('NVIDIA CUDA', 'cuda'), ('CPU (slow; VDA / normals)', 'cpu')]),
            'normal_axes': ('Normal convention:', [('Model native (recommended)', 'native'), ('Flip Y and Z', 'flip_yz')]),
        }
        for key, (title, values) in choices.items():
            widget = QComboBox()
            widget.setMinimumContentsLength(15)
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            for label, value in values:
                widget.addItem(label, value)
            self.controls[key] = widget
            widget.currentIndexChanged.connect(self.save)
            form.addRow(title, widget)
        form.addRow('Normals model:', QLabel('NormalCrafter · LiveActionAOV'))
        self.controls['long_edge'].setToolTip('512 is the starting point for 16 GB. Source-sized EXRs are restored after inference. 1024 can exceed 16 GB.')
        for key, title, low, high, step in [
            ('dvd_window', 'DVD window (frames):', 5, 81, 4),
            ('dvd_overlap', 'DVD overlap:', 1, 40, 1),
            ('normal_window', 'Normal window:', 2, 32, 1),
            ('normal_overlap', 'Normal overlap:', 1, 16, 1),
            ('decode_chunk', 'Normal VAE decode batch:', 1, 7, 1),
        ]:
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.valueChanged.connect(self.save)
            self.controls[key] = spin
            form.addRow(title, spin)
        layout.addWidget(group)
        notice = QLabel('DVD: relative depth, not measured distance; weights are non-commercial. '
                        'NormalCrafter has SVD ancestry: check upstream terms for paid work. '
                        'Models download on first use. Run install_aov.bat once.')
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.generate = QPushButton('Generate / Reuse cached AOVs')
        self.generate.clicked.connect(self.generate_requested)
        layout.addWidget(self.generate)
        self.preview = QComboBox()
        self.preview.addItems(['Depth', 'Normals', 'Original plate'])
        self.preview.currentTextChanged.connect(self.preview_requested)
        layout.addWidget(QLabel('Timeline preview:'))
        layout.addWidget(self.preview)
        export_group = QGroupBox('Multilayer EXR contents')
        export_form = QFormLayout(export_group)
        export_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.include_plate = QCheckBox('Include original RGB plate (straight)')
        self.include_mattes = QCheckBox('Include generated alpha + object layers')
        self.matte_source = QComboBox()
        self.matte_source.addItem('Matting (selected ViTMatte view)', 'matting')
        self.matte_source.addItem('Segmentation masks', 'segmentation')
        for widget in (self.include_plate, self.include_mattes):
            widget.toggled.connect(self.save)
            export_form.addRow(widget)
        self.matte_source.currentIndexChanged.connect(self.save)
        self.include_mattes.toggled.connect(self.matte_source.setEnabled)
        export_form.addRow('Alpha source:', self.matte_source)
        self.include_mattes.setToolTip('A is the union of existing object mattes. Named layers retain individual mattes. Generate mattes first or disable this option.')
        layout.addWidget(export_group)
        self.start_number = QSpinBox()
        self.start_number.setRange(0, 99999999)
        self.start_number.setValue(1001)
        layout.addWidget(QLabel('First exported frame number:'))
        layout.addWidget(self.start_number)
        self.start_number.valueChanged.connect(self.save)
        self.export = QPushButton('Export multilayer AOV EXR sequence...')
        self.export.clicked.connect(self.export_requested)
        layout.addWidget(self.export)
        self.status = QLabel('Uses the timeline In/Out range. No segmentation or matting required.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        self.load_values_from_settings()

    def load_values_from_settings(self):
        self.loading = True
        manager = get_settings_manager()
        values = asdict(AOVSettings.from_dict(manager.get_session_setting('aov_settings', {})))
        for key, widget in self.controls.items():
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(values[key])))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(values[key])
            else:
                widget.setValue(values[key])
        source_start = manager.get_session_setting('sequence_start_frame', 0)
        self.start_number.setValue(manager.get_session_setting('aov_start_number', 0) or source_start or 1001)
        self.include_plate.setChecked(manager.get_session_setting('aov_include_plate', True))
        self.include_mattes.setChecked(manager.get_session_setting('aov_include_mattes', True))
        self.matte_source.setCurrentIndex(max(0, self.matte_source.findData(manager.get_session_setting('aov_matte_source', 'matting'))))
        self.matte_source.setEnabled(self.include_mattes.isChecked())
        self.loading = False

    def settings(self):
        values = {}
        for key, widget in self.controls.items():
            values[key] = (widget.currentData() if isinstance(widget, QComboBox) else
                           widget.isChecked() if isinstance(widget, QCheckBox) else widget.value())
        return AOVSettings.from_dict(values)

    def save(self, *_):
        if self.loading:
            return
        manager = get_settings_manager()
        manager.set_session_setting('aov_settings', asdict(self.settings()))
        manager.set_session_setting('aov_start_number', self.start_number.value())
        manager.set_session_setting('aov_include_plate', self.include_plate.isChecked())
        manager.set_session_setting('aov_include_mattes', self.include_mattes.isChecked())
        manager.set_session_setting('aov_matte_source', self.matte_source.currentData())
        manager.save_session_settings()
        self.status.setText('Changed model settings require Generate. Export contents and numbering use existing results.')
        self.changed.emit()

    def make_job(self):
        manager = get_settings_manager()
        first, last, _ = core.get_frame_range()
        return dict(root=str(Path(core.temp_dir).resolve()), first=first, last=last,
                    extension=core.get_frame_extension(), settings=asdict(self.settings()),
                    fps=manager.get_session_setting('video_fps_rational', '') or str(core.VideoInfo.fps or 24),
                    sequence_start=manager.get_session_setting('sequence_start_frame', 0))


class AOVDialog(QDialog):
    def __init__(self, parent, job):
        super().__init__(parent)
        self.setWindowTitle('AOV Processing')
        self.setMinimumWidth(490)
        layout = QVBoxLayout(self)
        self.label = QLabel('Starting isolated AOV worker...')
        self.label.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.button = QPushButton('Cancel')
        for widget in (self.label, self.bar, self.button):
            layout.addWidget(widget)
        self.result, self.failure, self.cancelled = None, None, False
        self.tail, self.buffer = [], ''
        self.request = Path(job['root'])/'aov'/'jobs'/(uuid.uuid4().hex+'.json')
        atomic_json(self.request, job)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.error)
        self.button.clicked.connect(self.reject)
        self.job = job

    def execute(self):
        # CPU exports use the UI environment to share its exact matte processing.
        python = (Path(sys.executable) if self.job.get('operation') == 'export' else
                  ROOT/'.runtime'/'aov-env'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
        if not python.is_file():
            return None, 'AOV runtime is missing. Run install_aov.bat, then try again.'
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('PYTHONPATH', str(ROOT))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start(str(python), ['-u', '-m', 'sammie.aov.worker', str(self.request)])
        self.exec()
        return self.result, self.failure

    def read_output(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if '__AOV__' in line:
                try:
                    item = json.loads(line.split('__AOV__', 1)[1])
                    self.bar.setRange(0, item['total'])
                    self.bar.setValue(item['done'])
                    self.label.setText(item['message'])
                except (ValueError, KeyError):
                    pass
            else:
                self.tail = (self.tail + [line[-1500:]])[-15:]
        self.buffer = self.buffer[-10000:]

    def on_process_finished(self, code, *_):
        self.read_output()
        path = self.request.with_suffix('.result.json')
        if self.cancelled:
            self.failure = 'Cancelled. Completed passes remain cached; the interrupted pass can be regenerated.'
        elif path.exists():
            self.result = json.loads(path.read_text(encoding='utf-8'))
            self.failure = self.result.get('error')
        elif code != 0:
            self.failure = 'AOV worker failed:\n' + '\n'.join(self.tail[-8:])
        else:
            self.failure = 'Worker exited without a result.'
        super().accept()

    def error(self, error):
        if error == QProcess.FailedToStart:
            self.failure = 'Cannot start AOV worker: ' + self.process.errorString()
            super().accept()

    def reject(self):
        if self.process.state() != QProcess.NotRunning:
            self.cancelled = True
            self.button.setEnabled(False)
            self.label.setText('Cancelling and releasing GPU resources...')
            self.process.terminate()
            QTimer.singleShot(2000, self.kill_if_running)
        else:
            super().reject()

    def kill_if_running(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
