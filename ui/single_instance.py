"""Reuse the running macOS app with an OS lock and a small local open request."""

import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

from PySide6.QtCore import QObject, QTimer, Signal


class SingleInstance(QObject):
    open_requested = Signal()

    def __init__(self, parent=None, *, name=None, directory=None):
        super().__init__(parent)
        root = Path(__file__).resolve().parents[1]
        identity = f"{os.getuid()}:{root}"
        self.name = name or "airis-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if Path(self.name).name != self.name:
            raise ValueError("The application lock name must not contain directories.")
        self.directory = Path(directory) if directory is not None else root / "logs" / "runtime"
        self.lock_path = self.directory / (self.name + ".lock")
        self.request_path = self.directory / (self.name + ".request")
        self._descriptor = None
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll_requests)

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            # Replace atomically so the running UI never reads a partial request.
            with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as request:
                request.write(b"show\n")
                temporary = Path(request.name)
            try:
                os.replace(temporary, self.request_path)
            finally:
                temporary.unlink(missing_ok=True)
            return False
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self.request_path.unlink(missing_ok=True)
        self.timer.start()
        return True

    def poll_requests(self):
        if self._descriptor is None:
            return
        try:
            message = self.request_path.read_bytes()
            self.request_path.unlink(missing_ok=True)
        except FileNotFoundError:
            return
        if message == b"show\n":
            self.open_requested.emit()

    def close(self):
        self.timer.stop()
        if self._descriptor is not None:
            self.request_path.unlink(missing_ok=True)
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None
