"""Range repairs and a persisted sequential shot queue, using isolated workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,
    QSpinBox,QCheckBox,QDialogButtonBox,QFileDialog,QTableWidget,QTableWidgetItem,QProgressBar,QMessageBox)
from sammie import core
from sammie.projects import atomic_json
from .jobs import QueueStore,make_job,STAGES,LABELS,recover

ROOT=Path(__file__).resolve().parents[2]


class JobDialog(QDialog):
    def __init__(self,descriptor,parent=None,repair=False):
        super().__init__(parent)
        self.setWindowTitle('Repair selected range' if repair else 'Add shot to queue')
        from sammie.projects import read_project
        root,_=read_project(descriptor)
        session=json.loads((root/'session_settings.conf').read_text())
        self.descriptor=descriptor
        self.root=root
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        layout.addWidget(QLabel(Path(descriptor).name))
        total=session['total_frames']
        self.first=QSpinBox();self.last=QSpinBox();self.context=QSpinBox()
        for box in (self.first,self.last): box.setRange(0,max(0,total-1))
        self.first.setValue((session.get('in_point') or 0) if repair else 0)
        self.last.setValue((session.get('out_point') if session.get('out_point') is not None else total-1) if repair else total-1)
        self.context.setRange(0,256);self.context.setValue(16)
        form.addRow('First timeline frame (inclusive):',self.first)
        form.addRow('Last timeline frame (inclusive):',self.last)
        form.addRow('Context handles (each side):',self.context)
        self.controls={}
        defaults={'tracking':False,'matting':True,'vitmatte':session.get('vitmatte_enabled',False),
                  'depth':False,'normals':False,'foreground':False,'nuke':False}
        for stage in STAGES:
            box=QCheckBox(LABELS[stage]);box.setChecked(defaults[stage]);layout.addWidget(box)
            self.controls[stage]=box
            box.toggled.connect(self.describe)
        self.note=QLabel();self.note.setWordWrap(True);self.note.setMinimumWidth(530);layout.addWidget(self.note)
        self.cached=QLabel();self.cached.setWordWrap(True);layout.addWidget(self.cached)
        for box in (self.first,self.last,self.context):box.valueChanged.connect(self.describe)
        self.error=QLabel();self.error.setWordWrap(True);layout.addWidget(self.error)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.finish);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.job=None
        self.anchor=session.get('current_frame',0)
        self.describe()

    def describe(self,*_):
        if not hasattr(self,'note'): return
        selected=[s for s,c in self.controls.items() if c.isChecked()]
        messages=['Selected steps run in order. Handles are processed AND saved around the selected range; inspect the boundaries.']
        if 'tracking' in selected: messages.append('New tracking changes the matting guide: regenerate matting, then ViTMatte/foreground where needed.')
        if 'matting' in selected: messages.append('New matting makes affected ViTMatte/foreground results stale; unchecked steps are not run automatically.')
        if 'vitmatte' in selected: messages.append('ViTMatte reuses cached VideoMaMa. Changed alpha needs fresh foreground recovery.')
        if any(s in selected for s in ('depth','normals')): messages.append('Depth/normals use the FULL shot to preserve temporal context and depth normalization.')
        messages.append('Queue settings are captured now. Resume validates input/output hashes before skipping a completed stage.')
        self.note.setText('\n\n'.join(messages))
        if hasattr(self,'cached'):
            start=max(0,self.first.value()-self.context.value())
            end=min(self.last.maximum(),self.last.value()+self.context.value())
            counts=[]
            for directory,label,extension in [('masks','SAM','png'),('matting','Matting','png'),('vitmatte','ViTMatte','npz')]:
                n=sum(any((self.root/directory/f'{f:05d}').glob('*.'+extension)) for f in range(start,end+1))
                counts.append(f'{label}: {n}/{end-start+1} frames present')
            self.cached.setText('Effective range: '+str(start)+'–'+str(end)+'\n'+'; '.join(counts)+
                '\nFile presence is not freshness. Changed prompts/settings require a new queue item; resume verifies hashes.')

    def finish(self):
        try:
            selected=[s for s,c in self.controls.items() if c.isChecked()]
            self.job=make_job(self.descriptor,selected,self.first.value(),self.last.value(),self.context.value(),self.anchor)
            self.accept()
        except Exception as error: self.error.setText(str(error))


class QueueDialog(QDialog):
    def __init__(self,parent,store):
        super().__init__(parent)
        self.setWindowTitle('Sammie shot queue');self.resize(850,520)
        self.window=parent;self.store=store
        self.auto_close=False
        self.killer=None
        self.process=None;self.active=None;self.stage_index=0;self.stopping=False;self.running=False
        layout=QVBoxLayout(self)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Shot / range','Stages','Status','Progress / error'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        row=QHBoxLayout();layout.addLayout(row)
        self.add=QPushButton('Add working projects...');self.remove=QPushButton('Remove selected')
        self.start=QPushButton('Start / Resume');self.stop=QPushButton('Stop and keep completed stages');self.stop.setEnabled(False)
        for b in (self.add,self.remove,self.start,self.stop):row.addWidget(b)
        self.add.clicked.connect(self.add_projects);self.remove.clicked.connect(self.remove_selected)
        self.start.clicked.connect(self.begin);self.stop.clicked.connect(self.cancel)
        self.bar=QProgressBar();layout.addWidget(self.bar)
        self.label=QLabel('Each shot needs saved object selections. Models run sequentially; close other GPU-heavy apps for large shots.')
        self.label.setWordWrap(True);layout.addWidget(self.label)
        from sammie.background_io import add_activity_status
        add_activity_status(self,layout,self.label)
        self.close_button=QPushButton('Close');self.close_button.clicked.connect(self.reject);layout.addWidget(self.close_button)
        self.refresh()

    def refresh(self):
        self.table.setRowCount(len(self.store.jobs))
        for row,job in enumerate(self.store.jobs):
            values=[f"{Path(job['project']).stem}: {job['requested'][0]}–{job['requested'][1]}",
                    ', '.join(LABELS[s] for s in job['stages']),job['status'],job.get('error') or job.get('progress','')]
            for col,value in enumerate(values):self.table.setItem(row,col,QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def add_projects(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Add prepared shots',self.window.projects.location('scenes'),'Working projects (*.sammieproj)')
        for path in paths:
            try:
                d=JobDialog(path,self)
                if d.exec()==QDialog.Accepted:
                    d.job['delivery_parent']=self.window.projects.location('nuke')
                    self.store.add(d.job)
            except Exception as error: QMessageBox.warning(self,'Queue',str(error))
        self.refresh()

    def remove_selected(self):
        rows=sorted({i.row() for i in self.table.selectedIndexes()},reverse=True)
        for row in rows:self.store.jobs.pop(row)
        self.store.save();self.refresh()

    def begin(self):
        if self.running:return
        self.store.resume()
        self.stopping=False;self.running=True
        for b in (self.add,self.remove,self.start):b.setEnabled(False)
        self.stop.setEnabled(True)
        self.next_job()

    def next_job(self):
        self.active=next((j for j in self.store.jobs if j['status']=='pending'),None) if not self.stopping else None
        if self.active is None:
            self.running=False;self.stop.setEnabled(False)
            for b in (self.add,self.remove,self.start):b.setEnabled(True)
            self.label.setText('Queue stopped. Completed stages remain reusable.' if self.stopping else 'Queue finished. Review each shot status above.')
            self.store.save();self.refresh()
            if self.auto_close and not self.stopping and all(j['status']=='complete' for j in self.store.jobs):
                self.accept()
            return
        self.active.update(status='running',error='')
        self.stage_index=0
        self.next_stage()

    def next_stage(self):
        job=self.active
        if self.stopping:
            job['status']='cancelled';self.next_job();return
        if self.stage_index>=len(job['stages']):
            job.update(status='complete',progress='All selected stages complete')
            self.store.save();self.refresh();QTimer.singleShot(0,self.next_job);return
        stage=job['stages'][self.stage_index]
        runtime='aov-env' if stage in ('depth','normals') else ('corridorkey-env' if stage=='foreground' else None)
        python=ROOT/'.runtime'/runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python') if runtime else Path(sys.executable)
        if not python.is_file():self.fail(f'Missing {runtime}. Run install_school.bat -Models OnDemand.');return
        self.request=Path(job['root'])/'.workflow/jobs'/f"{job['id']}-{stage}.json"
        atomic_json(self.request,dict(job,stage=stage))
        self.request.with_suffix('.result.json').unlink(missing_ok=True)
        self.buffer='';self.tail=[]
        self.process=QProcess(self);self.process.setProcessChannelMode(QProcess.MergedChannels)
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONPATH',str(ROOT));env.insert('PYTHONUNBUFFERED','1')
        env.insert('QT_QPA_PLATFORM','offscreen');env.insert('SAMMIE_QUEUE_WORKER','1')
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished_stage)
        self.process.errorOccurred.connect(self.process_error)
        job['progress']='Starting '+LABELS[stage];self.store.save();self.refresh()
        self.bar.setRange(0,0);self.label.setText(job['progress'])
        self.process.start(str(python),['-u','-m','sammie.workflow.worker',str(self.request)])

    def process_error(self,error):
        if error==QProcess.FailedToStart:self.fail('Could not start worker: '+self.process.errorString())

    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if '__WORKFLOW__' in line:
                try:
                    info=json.loads(line.split('__WORKFLOW__',1)[1]);self.bar.setRange(0,info['total']);self.bar.setValue(info['done'])
                    self.label.setText(info['message']);self.active['progress']=info['message']
                except (ValueError,KeyError):pass
            else:
                self.tail=(self.tail+[line[-1500:]])[-20:]
                if line.strip():self.label.setText(line[-400:])
        self.buffer=self.buffer[-20000:]

    def finished_stage(self,code,*_):
        if self.killer is not None:
            self.killer.wait(timeout=10)
            self.killer=None
        self.read_output()
        job=self.active
        if self.stopping:
            try:
                from sammie.projects import ProjectLock
                with ProjectLock(job['root']):recover(job['root'])
                job.update(status='cancelled',progress='Cancelled; interrupted output range restored.')
            except Exception as error:job.update(status='failed',error='Recovery required: '+str(error))
            self.store.save();self.next_job();return
        path=self.request.with_suffix('.result.json')
        result=json.loads(path.read_text()) if path.exists() else {}
        if code or result.get('error') or 'receipt' not in result:
            self.fail(result.get('error') or '\n'.join(self.tail[-8:]) or f'Worker exited {code} without a result.');return
        stage=job['stages'][self.stage_index]
        job['completed'][stage]=result['receipt']
        if stage=='nuke' and result.get('result'):job['delivery']=result['result']
        job['progress']=LABELS[stage]+(' reused' if result.get('cached') else ' complete')
        self.store.save();self.refresh();self.stage_index+=1
        QTimer.singleShot(0,self.next_stage)

    def fail(self,message):
        self.active.update(status='failed',error=message)
        self.store.save();self.refresh();QTimer.singleShot(0,self.next_job)

    def cancel(self):
        self.stopping=True;self.stop.setEnabled(False)
        self.label.setText('Stopping the worker tree and restoring the interrupted range...')
        if self.process and self.process.state()!=QProcess.NotRunning:
            if os.name=='nt':
                self.killer=subprocess.Popen(['taskkill.exe','/PID',str(self.process.processId()),'/T','/F'],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            else:self.process.terminate()
        else:self.next_job()

    def reject(self):
        if self.running:self.cancel()
        else:super().reject()


def queue_store(window):
    workspace=window.settings_mgr.get_app_setting('workspace_path','')
    path=(Path(workspace)/'.sammie/queue.json') if workspace else ROOT/'.runtime/shot-queue.json'
    return QueueStore(path)


def run_dialog(window,store,start=False,auto_close=False):
    if window.is_playing:window.toggle_play_pause()
    if hasattr(window,'preview'):window.preview.close()
    window.projects.flush()
    window.sam_manager.unload_segmentation_model()
    window.matany_manager.unload_matting_model()
    window.removal_manager.unload_minimax_model();window.removal_manager.unload_propainterx_model()
    window.projects.release()
    dialog=QueueDialog(window,store)
    dialog.auto_close=auto_close
    try:
        if start:QTimer.singleShot(0,dialog.begin)
        dialog.exec()
    finally:
        window.projects.reacquire()
        if window.projects.descriptor:recover(core.temp_dir)
        for job in store.jobs:
            if Path(job['root']).resolve()!=Path(core.temp_dir).resolve(): continue
            if 'tracking' in job.get('completed',{}):
                window.sam_manager.propagated=True
                window.settings_mgr.set_session_setting('is_propagated',True)
            if 'matting' in job.get('completed',{}):
                window.matany_manager.propagated=True
                window.settings_mgr.set_session_setting('is_matted',True)
        window._update_current_frame_display()
        window.update_tracking_status()
        window.update_matting_status()
        window.update_status_bar('Queue closed. Load Model to resume interactive segmentation.')


def show_queue(window):
    run_dialog(window,queue_store(window))


def repair_range(window):
    if not window.projects.descriptor and not window.projects.save():return
    window.projects.flush()
    dialog=JobDialog(window.projects.descriptor,window,repair=True)
    if dialog.exec()!=QDialog.Accepted:return
    dialog.job['delivery_parent']=window.projects.location('nuke')
    # Repair is a deliberate single-shot operation; do not start other pending shots.
    store=QueueStore(Path(dialog.job['root'])/'.workflow/repair-queue.json')
    store.jobs=[dialog.job];store.save()
    run_dialog(window,store,start=True)


def send_to_nuke(window):
    from .nuke import find_nuke,launch,available_passes
    if not window.projects.descriptor and not window.projects.save():return
    window.projects.flush()
    first,last,_=core.get_frame_range()
    job=make_job(window.projects.descriptor,['nuke'],first,last,0)
    job['delivery_parent']=window.projects.location('nuke')
    job['nuke_passes']=available_passes(job)
    # A delivery-only queue has the same cancellation, isolation and error handling.
    path=Path(job['root'])/'.workflow/nuke-queue.json'
    store=QueueStore(path);store.jobs=[job];store.save()
    run_dialog(window,store,start=True,auto_close=True)
    if job['status']!='complete':return
    result=job.get('delivery')
    if not result:
        receipt=Path(job['root'])/'.workflow'/f"{job['id']}-nuke.json"
        result=json.loads(receipt.read_text())
    executable=find_nuke(window.settings_mgr.get_app_setting('nuke_executable',''))
    if not executable:
        executable,_=QFileDialog.getOpenFileName(window,'Choose Nuke executable (Cancel keeps the .nk delivery)','', 'Executable (*.exe);;All files (*)')
    if executable:
        window.settings_mgr.set_app_setting('nuke_executable',executable);window.settings_mgr.save_app_settings()
        launch(executable,result['script'])
    window.update_status_bar('Nuke delivery: '+result['script'])
