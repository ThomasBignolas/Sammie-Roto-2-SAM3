"""Working/portable project actions, separate from model and timeline UI."""
from dataclasses import asdict
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QProgressDialog
from . import core, projects
from .settings_manager import SessionSettings


class ProjectController:
    def __init__(self, window):
        self.window = window
        self.scratch = Path(core.temp_dir).resolve()
        self.descriptor = None
        self.lock = None

    def flush(self):
        w = self.window
        w._save_current_ui_state()
        projects.save_metadata(core.temp_dir,asdict(w.settings_mgr.session_settings),w.point_manager.get_all_points(),self.descriptor)

    def location(self,kind):
        value=self.window.settings_mgr.get_app_setting('workspace_path','')
        return str(Path(value)/kind) if value else ''

    def set_workspace(self):
        path=QFileDialog.getExistingDirectory(self.window,'Set Project — choose workspace',
                self.window.settings_mgr.get_app_setting('workspace_path',''))
        if not path: return
        root=projects.set_workspace(path)
        manager=self.window.settings_mgr
        manager.set_app_setting('workspace_path',str(root))
        manager.set_app_setting('export_folder_path',str(root/'exports'))
        manager.set_app_setting('export_use_input_folder',False)
        manager.save_app_settings()
        self.window.update_status_bar(f'Project workspace: {root} — scenes, footage, exports, nuke')

    def _progress(self, title, action):
        d = QProgressDialog(title,'Cancel',0,0,self.window)
        d.setWindowModality(Qt.WindowModal)
        d.setAutoClose(False)
        d.show()
        def update(n,total,message):
            d.setMaximum(total); d.setValue(n); d.setLabelText(message)
            QApplication.processEvents()
        try: return action(update,d.wasCanceled)
        finally: d.close()

    def activate(self, descriptor, resume=True):
        root,_ = projects.read_project(descriptor)
        if self.descriptor and Path(descriptor).resolve()==self.descriptor:
            return
        lock = projects.ProjectLock(root).acquire()
        try:
            from .workflow.jobs import recover
            recover(root)
            if self.descriptor: self.flush()
            self.window.sam_manager.unload_segmentation_model()
            if self.lock: self.lock.release()
            self.descriptor, self.lock = Path(descriptor).resolve(), lock
            projects.bind_root(root,self.window.settings_mgr)
            self.window.settings_mgr.session_settings = SessionSettings()
            self.window.settings_mgr.load_session_settings()
            self.window.point_manager.points = []
            self.window.settings_mgr.set_app_setting('last_working_project',str(self.descriptor))
            self.window.settings_mgr.save_app_settings()
            self.window.setWindowTitle(f'Sammie-Roto — {self.descriptor.stem} (working project)')
            if resume: self.window.resume_prev_session()
        except BaseException:
            if self.lock is not lock: lock.release()
            raise

    def detach(self):
        if not self.descriptor: return
        self.flush()
        self.window.sam_manager.unload_segmentation_model()
        if self.lock: self.lock.release()
        self.lock, self.descriptor = None,None
        projects.bind_root(self.scratch,self.window.settings_mgr)
        self.window.settings_mgr.set_app_setting('last_working_project','')
        self.window.settings_mgr.save_app_settings()
        self.window.setWindowTitle('Sammie-Roto — unsaved session')

    def save(self, save_as=False):
        if core.VideoInfo.total_frames < 1:
            raise ValueError('Import footage or open a project first.')
        self.flush()
        if self.descriptor and not save_as:
            self.window.update_status_bar('Project saved — media and cached results stayed in place.')
            return True
        path,_=QFileDialog.getSaveFileName(self.window,'Save working project',self.location('scenes'), 'Working project (*.sammieproj)')
        if not path: return False
        descriptor=self._progress('Preparing working project',lambda progress,cancel:
            projects.create_project(core.temp_dir,path,progress,cancel))
        self.activate(descriptor,resume=bool(self.window.sam_manager.loaded_model_name))
        self.window.point_manager.points = self.window.settings_mgr.load_points() or []
        self.window.update_status_bar('Working project ready. Subsequent saves write settings and points only.')
        return True

    def pack(self):
        if core.VideoInfo.total_frames < 1: raise ValueError('Import footage first.')
        self.flush()
        path,_=QFileDialog.getSaveFileName(self.window,'Pack portable project',self.location('exports'), 'Portable project (*.sammie)')
        if not path: return
        if not path.lower().endswith('.sammie'): path += '.sammie'
        self._progress('Packing portable project',lambda progress,cancel:
                       projects.pack_project(core.temp_dir,path,progress,cancel))
        self.window.update_status_bar(f'Portable project saved: {path}')

    def open(self, path=None):
        if not path:
            path,_=QFileDialog.getOpenFileName(self.window,'Open project',self.location('scenes'), 'Sammie projects (*.sammieproj *.sammie)')
        if not path: return False
        if Path(path).suffix.lower()=='.sammieproj':
            self.activate(path)
        else:
            # Unpack into a fresh temporary session; never remove a working project.
            self._progress('Opening portable project',lambda progress,cancel:
                           projects.unpack_project(path,self.scratch,progress,cancel))
            self.detach()
            self.window.sam_manager.unload_segmentation_model()
            self.window.settings_mgr.session_settings = SessionSettings()
            self.window.settings_mgr.load_session_settings()
            self.window.point_manager.points=[]
            self.window.resume_prev_session()
            self.window.update_status_bar('Portable project opened. Save Project creates a fast working copy.')
        return True

    def restore(self):
        path=self.window.settings_mgr.get_app_setting('last_working_project','')
        if path: self.activate(path,resume=False)

    def release(self):
        if self.lock: self.lock.release()

    def reacquire(self):
        if self.lock: self.lock.acquire()
