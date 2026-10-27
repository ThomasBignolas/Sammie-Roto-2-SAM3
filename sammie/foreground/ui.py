"""Artist controls for an optional foreground-color AOV, inside the existing tab."""
from dataclasses import asdict
from pathlib import Path
import re
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QGroupBox,QVBoxLayout,QFormLayout,QCheckBox,QComboBox,
    QDoubleSpinBox,QSpinBox,QLabel,QPushButton,QLineEdit,QFileDialog,QWidget,QHBoxLayout)
from .config import ForegroundSettings, PRESETS
from .versions import ROOT, SOURCE_PATH, WEIGHTS


class ForegroundControls(QGroupBox):
    changed = Signal()
    test_requested = Signal()

    def __init__(self):
        super().__init__('CorridorKey Foreground RGB (optional)')
        self.loading = True
        outer = QVBoxLayout(self)
        self.enabled = QCheckBox('Enable recovered foreground AOV')
        outer.addWidget(self.enabled)
        self.parameters = QWidget()
        form = QFormLayout(self.parameters)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.preset = QComboBox()
        self.preset.setMinimumContentsLength(15)
        self.preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset.addItem('Custom settings')
        self.preset.addItems(list(PRESETS))
        form.addRow('Preset:',self.preset)
        self.controls = {}
        choices = {
            'screen': ('Checkpoint:', [('Blue / CorridorKeyBlue 1.0','blue'),('Green / CorridorKey 1.0','green')]),
            'resolution': ('Model resolution:',[(str(v),v) for v in (1024,1536,2048)]),
            'device': ('Device:',[('NVIDIA CUDA','cuda'),('CPU (slow)','cpu')]),
            'precision': ('Inference precision:',[('FP16 mixed (CUDA)','fp16'),('FP32','fp32')]),
            'color_space': ('Plate color space:',[('sRGB (imported display frames)','srgb'),('Linear Rec.709','linear709'),('ACEScg (linear)','acescg')]),
        }
        for key,(label,values) in choices.items():
            widget=QComboBox()
            widget.setMinimumContentsLength(15)
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            for name,value in values: widget.addItem(name,value)
            widget.currentIndexChanged.connect(self.on_change)
            self.controls[key]=widget
            form.addRow(label,widget)
        for key,label,low,high,step in [('exposure_ev','Model input exposure (stops):',-6,4,.25),
                ('strength','RGB cleanup strength:',0,1,.05),('edge_power','Edge coverage:',1,8,.25),
                ('despill','Additional edge despill:',0,1,.05)]:
            spin=QDoubleSpinBox()
            spin.setRange(low,high)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.valueChanged.connect(self.on_change)
            self.controls[key]=spin
            form.addRow(label,spin)
        self.controls['exposure_ev'].setToolTip('Exposure is restored after inference. Changing exposure requires Generate. -2 was tested on the blue-sky shot.')
        self.controls['strength'].setToolTip('0 returns original RGB. Changing strength, edge coverage or despill reuses the cached prediction.')
        self.controls['edge_power'].setToolTip('Blend weight = strength × (1 − alpha^coverage). Higher values affect more near-opaque edge pixels; alpha=1 remains exact.')
        self.controls['despill'].setToolTip('Optional luminance-preserving blue/green suppression in recovered edge colors. Start at 0; natural blue or green edge colors may also be affected.')
        path = QLineEdit()
        path.setReadOnly(True)
        path.setPlaceholderText('Use the imported display RGB frames')
        path.textChanged.connect(self.on_change)
        self.controls['plate_path']=path
        form.addRow('Optional original EXR plate:',path)
        row=QWidget()
        buttons=QHBoxLayout(row)
        buttons.setContentsMargins(0,0,0,0)
        choose=QPushButton('Choose first EXR...')
        clear=QPushButton('Use imported frames')
        choose.clicked.connect(self.browse)
        clear.clicked.connect(self.clear_plate)
        buttons.addWidget(choose)
        buttons.addWidget(clear)
        form.addRow(row)
        first=QSpinBox()
        first.setRange(0,99999999)
        first.valueChanged.connect(self.on_change)
        first.setToolTip('EXR number corresponding to project frame 0, not the current In point. Dimensions must match the imported frames.')
        self.controls['plate_first']=first
        form.addRow('EXR number at project frame 0:',first)
        notice=QLabel('Uses the selected alpha below, including the selected ViTMatte view. Alpha is preserved exactly. CK1.0 processes frames independently; this is not CK2.0 time awareness. EXR override and foreground export share the selected plate color space; viewer uses an sRGB preview.')
        notice.setWordWrap(True)
        form.addRow(notice)
        self.test=QPushButton('Test CorridorKey on current frame')
        self.test.clicked.connect(self.test_requested)
        form.addRow(self.test)
        self.status=QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        outer.addWidget(self.parameters)
        self.enabled.toggled.connect(self.parameters.setEnabled)
        self.parameters.setEnabled(False)
        self.preset.activated.connect(self.apply_preset)
        self.load({})

    def settings(self):
        values={}
        for key,w in self.controls.items():
            values[key]=w.currentData() if isinstance(w,QComboBox) else w.text() if isinstance(w,QLineEdit) else w.value()
        return ForegroundSettings.from_dict(values)

    def load(self, options):
        self.loading=True
        for key,value in asdict(ForegroundSettings.from_dict(options)).items():
            w=self.controls[key]
            if isinstance(w,QComboBox): w.setCurrentIndex(max(0,w.findData(value)))
            elif isinstance(w,QLineEdit): w.setText(value)
            else: w.setValue(value)
        self.loading=False
        self.update_status()

    def update_status(self):
        screen=self.controls['screen'].currentData()
        _,revision,filename,_=WEIGHTS[screen]
        installed=(SOURCE_PATH/'CorridorKeyModule'/'inference_engine.py').is_file()
        model=(ROOT/'.runtime'/'corridorkey-models'/screen/revision/filename).is_file()
        self.status.setText('Runtime ready. '+('Selected model is cached.' if model else 'Selected model downloads on first Generate.') if installed else 'Run install_corridorkey.bat once on this PC. Models run locally.')

    def on_change(self,*_):
        if not self.loading:
            self.preset.setCurrentIndex(0)
            self.update_status()
            self.changed.emit()

    def apply_preset(self,*_):
        name=self.preset.currentText()
        if name in PRESETS:
            self.load(dict(asdict(self.settings()),**PRESETS[name]))
            self.changed.emit()

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'First original RGB EXR','','OpenEXR (*.exr)')
        if path:
            values=asdict(self.settings())
            match=re.search(r'(\d+)(?=\.exr$)',Path(path).name,re.I)
            values.update(plate_path=path,plate_first=int(match[0]) if match else 0,color_space='acescg')
            self.load(values)
            self.changed.emit()

    def clear_plate(self):
        self.load(dict(asdict(self.settings()),plate_path='',plate_first=0,color_space='srgb'))
        self.changed.emit()
