"""Cancellable IO dialog. The worker never touches Qt widgets or session state."""
import threading
import time
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import QProgressDialog

_active = threading.local()


def report(message, done=0, total=0):
    callback = getattr(_active, 'progress', None)
    if callback: callback(done, total, message)


def check_cancelled():
    callback = getattr(_active, 'cancel', None)
    if callback and callback():
        raise InterruptedError('Operation cancelled; model resources are being released.')


class ProgressSink:
    """Legacy model progress adapter; no QWidget may be created in a worker."""
    def __init__(self, message, _cancel=None, minimum=0, maximum=100, parent=None):
        self.message, self.value, self.maximum = message, minimum, maximum
        report(message, minimum, maximum)
    def setLabelText(self, text):
        self.message = text
        report(text, self.value, self.maximum)
    def setValue(self, value):
        self.value = value
        report(self.message, value, self.maximum)
    def setRange(self, minimum, maximum): self.maximum = maximum
    def setMaximum(self, maximum): self.maximum = maximum
    def wasCanceled(self):
        callback = getattr(_active, 'cancel', None)
        return bool(callback and callback())
    def close(self): pass
    def show(self): pass
    def setWindowTitle(self, *_): pass
    def setWindowModality(self, *_): pass
    def setAutoClose(self, *_): pass
    def setAutoReset(self, *_): pass
    def setMinimumDuration(self, *_): pass


def progress_dialog(*args, **kwargs):
    return (ProgressSink if getattr(_active, 'progress', None) else QProgressDialog)(*args, **kwargs)


class IOThread(QThread):
    progress = Signal(int, int, str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.cancelled = threading.Event()
        self.result = None
        self.error = None

    def run(self):
        _active.progress = self.progress.emit
        _active.cancel = self.cancelled.is_set
        try:
            self.result = self.action(self.progress.emit, self.cancelled.is_set)
        except BaseException as error:
            self.error = error
        finally:
            _active.progress = _active.cancel = None


def run_io(parent, title, action, on_progress=None):
    if getattr(_active, 'progress', None) is not None:
        return action(_active.progress, _active.cancel)
    worker = IOThread(action)
    dialog = QProgressDialog(title, 'Cancel', 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    started = time.monotonic()
    state = {'message': title, 'updated': started}
    timer = QTimer(dialog)
    def heartbeat():
        elapsed = time.monotonic() - started
        stalled = time.monotonic() - state['updated']
        suffix = '\nThis stage has no finer progress updates yet.' if stalled > 15 else ''
        if worker.cancelled.is_set():
            suffix = '\nCancel requested; waiting for the active model/file operation to return.'
        dialog.setLabelText(f"{state['message']}\nElapsed: {int(elapsed)//60}:{int(elapsed)%60:02d}{suffix}")
    timer.timeout.connect(heartbeat)
    timer.start(500)
    # Cancellation requests cleanup; keep the nested event loop alive until the
    # worker has stopped. Closing a dialog must never destroy a running QThread.
    def cancel():
        worker.cancelled.set()
        dialog.setLabelText('Cancelling — waiting for the current file to finish…')
        dialog.setCancelButton(None)
        dialog.show()
    def progress(n, total, message):
        if not worker.cancelled.is_set():
            state.update(message=message, updated=time.monotonic())
            dialog.setMaximum(total)
            dialog.setValue(n)
            heartbeat()
            if on_progress: on_progress(n, total, message)
    dialog.canceled.connect(cancel)
    worker.progress.connect(progress)
    worker.finished.connect(dialog.accept)
    worker.start()
    # exec() can return on Escape/close before a worker finishes; re-enter rather
    # than join on the GUI thread and freeze the application.
    while worker.isRunning():
        dialog.exec()
        if worker.isRunning():
            cancel()
    worker.wait()
    timer.stop()
    dialog.close()
    print(f'{title}: {time.monotonic()-started:.2f}s' + (f' — {worker.error}' if worker.error else ' — complete'))
    if worker.error is not None:
        raise worker.error
    return worker.result


def run_compute(parent, title, action, on_progress=None):
    """Run one exclusive inference operation with the same device/autocast policy.

    Qt widgets, downloads and candidate selection must be handled by the caller
    on the main thread. Cancellation occurs at model-safe checkpoints.
    """
    import torch
    from contextlib import nullcontext
    from .core import DeviceManager
    device = DeviceManager.get_device()
    dtype = DeviceManager.get_dtype()
    def compute(progress, cancel):
        context = torch.autocast(device.type, dtype=dtype) if device.type in ('cuda','mps','xpu') else nullcontext()
        with torch.inference_mode(), context:
            from .performance import available_memory
            resources = f'{available_memory()/1024**3:.1f} GiB RAM available'
            if device.type == 'cuda':
                free, total = torch.cuda.mem_get_info(device)
                resources += f'; GPU {free/1024**3:.1f}/{total/1024**3:.1f} GiB free'
            print(f'{title}: {resources}')
            check_cancelled()
            result = action(progress, cancel)
            return result
    if parent is not None and hasattr(parent, 'viewer'):
        if getattr(parent, '_compute_busy', False):
            raise RuntimeError('Another model operation is already running.')
        parent._compute_busy = True
        parent.viewer._preview_timer.stop()
        if parent.is_playing: parent.toggle_play_pause()
    try:
        return run_io(parent, title, compute, on_progress)
    finally:
        if parent is not None and hasattr(parent, 'viewer'):
            parent._compute_busy = False


def add_activity_status(dialog, layout, stage_label):
    """Elapsed/waiting status for existing model subprocess/thread dialogs."""
    from PySide6.QtWidgets import QLabel
    label = QLabel()
    label.setWordWrap(True)
    layout.addWidget(label)
    started = time.monotonic()
    state = [stage_label.text(), started]
    timer = QTimer(dialog)
    def tick():
        now = time.monotonic()
        if stage_label.text() != state[0]: state[:] = [stage_label.text(), now]
        seconds = int(now-started)
        text = f'Elapsed: {seconds//60}:{seconds%60:02d}'
        if now-state[1] > 15: text += ' — waiting for the current model stage; no finer progress available.'
        label.setText(text)
    timer.timeout.connect(tick)
    dialog.finished.connect(timer.stop)
    timer.start(500)
    tick()
