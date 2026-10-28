"""Latest-request-only preview controller. QPixmap construction stays on Qt's UI thread."""
from dataclasses import asdict
import json
from multiprocessing import shared_memory
from pathlib import Path
import sys
import time
import numpy as np
from PySide6.QtCore import QObject, QProcess, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap


class PreviewController(QObject):
    ready = Signal(int, QPixmap)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.pending = None
        self.inflight = None
        self.serial = 0
        self.generation = 0
        self.buffer = bytearray()
        self.errors = ''
        self.last_cache_hit = False
        self.cache_bytes = 0
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self._dispatch)
        self.started_at = 0
        self.last_requested_frame = 0
        self.watchdog = QTimer(self)
        self.watchdog.timeout.connect(self._heartbeat)
        self.watchdog.start(500)

    @property
    def busy(self):
        return self.pending is not None or self.inflight is not None

    def invalidate(self):
        self.generation += 1
        self.serial += 1
        self.pending = None

    def request(self, frame, view, points, manager):
        from . import core
        self.serial += 1
        self.last_requested_frame = frame
        self.pending = dict(id=self.serial, generation=self.generation,
            root=str(Path(core.temp_dir).resolve()), frame=int(frame), view=view,
            points=[{k:v for k,v in p.items() if k in ('object_id','frame','x','y','positive','kind')} for p in points],
            session=asdict(manager.session_settings),
            video={k: getattr(core.VideoInfo, k) for k in ('width','height','total_frames','fps','color_space')},
            threads=manager.get_app_setting('cpu_threads', 0),
            cache_mib=manager.get_app_setting('preview_cache_mib', 256))
        # Collapse slider/setting signals in the same short burst.
        self.debounce.start(12)

    def _dispatch(self):
        if self.pending is None or self.inflight is not None:
            return
        if self.process is None:
            self.started_at = time.monotonic()
            process = self.process = QProcess(self)
            process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
            process.setProgram(sys.executable)
            process.setArguments(['-u', '-m', 'sammie.preview_worker'])
            process.readyReadStandardOutput.connect(self._receive)
            process.readyReadStandardError.connect(self._stderr)
            process.started.connect(self._dispatch)
            process.finished.connect(self._exited)
            process.errorOccurred.connect(self._process_error)
            process.start()
            return
        if self.process.state() != QProcess.Running:
            return
        request, self.pending = self.pending, None
        self.started_at = time.monotonic()
        self.inflight = request['id']
        self.process.write((json.dumps(request)+'\n').encode())

    def _heartbeat(self):
        if not self.busy:
            return
        elapsed = time.monotonic() - self.started_at if self.started_at else 0
        self.status.emit(f'Rendering frame {self.last_requested_frame} — {elapsed:.0f}s')
        if elapsed > 120:
            self.close()
            self.failed.emit('Preview timed out after 120 seconds. The worker was stopped; seek again to retry.')

    def _stderr(self):
        self.errors = (self.errors + bytes(self.process.readAllStandardError()).decode(errors='replace'))[-4000:]

    def _receive(self):
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b'\n' in self.buffer:
            line, _, rest = self.buffer.partition(b'\n')
            self.buffer = bytearray(rest)
            try:
                response = json.loads(line)
                if response['id'] == self.serial:
                    self.last_cache_hit = response.get('hit', False)
                    self.cache_bytes = response.get('bytes', 0)
                    if 'error' in response:
                        self.failed.emit(response['error'])
                    elif 'memory' in response:
                        memory = shared_memory.SharedMemory(name=response['memory'])
                        try:
                            array = np.ndarray(response['shape'], np.uint8, buffer=memory.buf)
                            height, width = array.shape[:2]
                            fmt = QImage.Format_RGBA8888 if array.shape[2] == 4 else QImage.Format_RGB888
                            # Detach before asking the child to recycle its transfer buffer.
                            image = QImage(array.data, width, height, array.strides[0], fmt).copy()
                            pixmap = QPixmap.fromImage(image)
                            del array
                        finally:
                            memory.close()
                        self.ready.emit(response['frame'], pixmap)
                        self.status.emit(f"Frame {response['frame']} — {'cached' if self.last_cache_hit else 'rendered'}")
                    else:
                        self.ready.emit(response['frame'], QPixmap())
            except Exception as error:
                self.failed.emit(f'Preview failed: {error}')
            self.inflight = None
        self._dispatch()

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self.failed.emit(f'Cannot start preview process: {self.process.errorString()}')
            self.pending = self.inflight = None
            self.process.deleteLater()
            self.process = None

    def _exited(self, code, status):
        if self.process is not None:
            self.failed.emit(f'Preview process stopped ({code}). {self.errors[-800:]}')
            self.process.deleteLater()
            self.process = None
            self.pending = self.inflight = None
            self.buffer.clear()

    def close(self):
        self.invalidate()
        self.debounce.stop()
        process, self.process = self.process, None
        self.inflight = None
        self.buffer.clear()
        self.status.emit('')
        if process is not None:
            process.blockSignals(True)
            process.closeWriteChannel()
            if process.state() != QProcess.NotRunning and not process.waitForFinished(100):
                from .processes import stop_worker_tree
                if process.processId(): stop_worker_tree(int(process.processId()))
                else: process.kill()
                process.waitForFinished(2000)
            process.deleteLater()
