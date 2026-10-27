"""In-app release information and a deferred handoff to the verified updater."""
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import uuid
from PySide6.QtCore import QThread,Signal
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QTextEdit,QPushButton

ROOT=Path(__file__).resolve().parents[1]
API='https://api.github.com/repos/ThomasBignolas/Sammie-Roto-2-SAM3/releases/latest'


def installed_version(root=ROOT):
    root=Path(root)
    if (root/'.git').exists():
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,timeout=10).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True,timeout=10).strip())
        return commit,dirty
    path=root/'SCHOOL_BUILD.json'
    if path.exists():return json.loads(path.read_text())['commit'],False
    return 'unknown',False


def latest_release(opener=urllib.request.urlopen):
    request=urllib.request.Request(API,headers={'Accept':'application/vnd.github+json','User-Agent':'Sammie-Roto-School'})
    with opener(request,timeout=20) as response:
        value=json.loads(response.read(2*1024*1024))
    if (value.get('draft') or value.get('prerelease') or
        not value.get('html_url','').startswith('https://github.com/ThomasBignolas/Sammie-Roto-2-SAM3/releases/tag/')):
        raise ValueError('Unexpected school release response.')
    return value


def start_update(root=ROOT,pid=None):
    root=Path(root).resolve()
    if os.name!='nt':raise RuntimeError('The school updater currently supports Windows. Use the documented Git update on other systems.')
    script=root/'scripts/update_school.ps1'
    if not script.is_file():raise RuntimeError('School updater is missing; use the online CMD updater.')
    folder=root/'.runtime';folder.mkdir(exist_ok=True)
    log=folder/('ui-update-'+uuid.uuid4().hex[:10]+'.log')
    with log.open('w') as output:
        process=subprocess.Popen(['powershell.exe','-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',
            str(script),'-Directory',str(root),'-WaitForPid',str(pid or os.getpid()),'-Restart'],
            cwd=root,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return log,process


class ReleaseWorker(QThread):
    result=Signal(object);failure=Signal(str)
    def run(self):
        try:self.result.emit(latest_release())
        except Exception as error:self.failure.emit(str(error))


class UpdateDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent)
        self.window=parent;self.release=None;self.closing=False
        self.setWindowTitle('School updates');self.resize(620,470)
        layout=QVBoxLayout(self)
        self.current,self.dirty=installed_version()
        self.label=QLabel('Installed: '+self.current[:12]+(' (local source changes)' if self.dirty else ''))
        self.label.setWordWrap(True);layout.addWidget(self.label)
        self.notes=QTextEdit();self.notes.setReadOnly(True);layout.addWidget(self.notes)
        self.check=QPushButton('Check GitHub for updates');self.check.clicked.connect(self.fetch);layout.addWidget(self.check)
        self.update=QPushButton('Save, close and update');self.update.setEnabled(False);self.update.clicked.connect(self.install);layout.addWidget(self.update)
        self.close_button=QPushButton('Close');self.close_button.clicked.connect(self.reject);layout.addWidget(self.close_button)
        self.worker=ReleaseWorker(self);self.worker.result.connect(self.loaded);self.worker.failure.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.fetch()

    def fetch(self):
        self.check.setEnabled(False);self.notes.setPlainText('Checking the school repository...');self.worker.start()

    def loaded(self,release):
        self.release=release
        same=release.get('target_commitish')==self.current
        self.notes.setPlainText(release['tag_name']+'\n\n'+release.get('body',''))
        message='You have this release.' if same else 'A published release is available.'
        if self.dirty:message+=' Local source edits must be committed or preserved before updating.'
        self.label.setText('Installed: '+self.current[:12]+'\n'+message)
        self.update.setEnabled(not same and not self.dirty and os.name=='nt')

    def failed(self,message):self.notes.setPlainText('Could not check updates. '+message+'\nThe CMD updater remains available.')

    def finished(self):
        self.check.setEnabled(True)
        if self.closing:super().reject()

    def reject(self):
        if self.worker.isRunning():
            self.closing=True;self.notes.setPlainText('Finishing the release check (20-second network timeout)...')
        else:super().reject()

    def install(self):
        try:
            self.window.projects.flush()
            log,_=start_update()
            self.window.update_status_bar('Update log: '+str(log))
            self.accept()
            self.window.close()
        except Exception as error:self.notes.setPlainText(str(error))
