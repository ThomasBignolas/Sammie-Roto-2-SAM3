"""Local JSON-lines RPC with file-backed masks and backpressure per video frame."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


class WorkerClient:
    def __init__(self, status=None, python=None):
        self.status = status or print
        executable = Path(python or os.environ.get('SAMMIE_SAM3_PYTHON', '') or
                          ROOT / '.runtime/sam3-env/Scripts/python.exe')
        if not executable.is_file():
            raise RuntimeError('SAM 3 runtime is missing. Run install_sam3.bat first.')
        self.messages = queue.Queue()
        self.busy = False
        self.process = subprocess.Popen(
            [str(executable), '-u', '-m', 'sammie.backends.worker'], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.stderr_tail = []
        self.readers = [threading.Thread(target=self._read, daemon=True),
                        threading.Thread(target=self._read_errors, daemon=True)]
        for reader in self.readers:
            reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    self.messages.put({'error': 'Invalid response from SAM 3 worker'})
        finally:
            self.messages.put({'error': 'SAM 3 worker exited. ' + '\n'.join(self.stderr_tail[-8:])})

    def _read_errors(self):
        for line in self.process.stderr:
            self.stderr_tail.append(line.rstrip())
            self.stderr_tail = self.stderr_tail[-40:]

    def _send(self, data):
        self.process.stdin.write(json.dumps(data) + '\n')
        self.process.stdin.flush()

    def _receive(self, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = self.messages.get(timeout=0.05)
            except queue.Empty:
                # Keep Qt painting during model loading without allowing a reentrant edit.
                from PySide6.QtCore import QEventLoop
                from PySide6.QtWidgets import QApplication
                if QApplication.instance():
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                continue
            if 'status' in result:
                self.status(result['status'])
                continue
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result
        self.close()
        raise TimeoutError('SAM 3 did not respond within 15 minutes; the worker was stopped.')

    def call(self, command, **params):
        if self.busy:
            raise RuntimeError('SAM 3 is already processing a request')
        self.busy = True
        try:
            self._send({'command': command, **params})
            return self._receive().get('result')
        finally:
            self.busy = False

    def stream(self, command, **params):
        if self.busy:
            raise RuntimeError('SAM 3 is already processing a request')
        self.busy = True
        pending_ack = False
        try:
            self._send({'command': command, **params})
            while True:
                response = self._receive()
                if response.get('done'):
                    return
                pending_ack = True
                yield response['result']
                self._send({'continue': True})
                pending_ack = False
        finally:
            try:
                if pending_ack and self.process.poll() is None:
                    self._send({'continue': False})
                    self._receive()
            finally:
                self.busy = False

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        for reader in self.readers:
            reader.join(timeout=2)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()
