"""Coordinate PortAudio streams and refresh macOS device defaults while idle."""

from contextlib import contextmanager
import platform
from threading import Lock


_lock = Lock()
_active_streams = 0


@contextmanager
def audio_session(sounddevice, *, follow_system_default=False):
    """Lease the device list until a stream closes.

    PortAudio snapshots macOS devices and defaults when initialized. Refresh
    that snapshot before automatic device selection, while no Airis stream is
    open. Explicit device IDs retain their existing enumeration. Concurrent
    capture and playback keep their active streams intact.
    """
    global _active_streams
    with _lock:
        if follow_system_default and platform.system() == "Darwin" and _active_streams == 0:
            # These sounddevice wrappers balance PortAudio's initialization count.
            if getattr(sounddevice, "_initialized", 1):
                sounddevice._terminate()
            sounddevice._initialize()
        _active_streams += 1
    try:
        yield
    finally:
        with _lock:
            _active_streams -= 1
