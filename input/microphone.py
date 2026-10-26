"""Bounded push-to-talk recording without writing audio files."""

from threading import Lock

from audio_runtime import audio_session
from .audio import RecordedAudio


class MicrophoneRecorder:
    def __init__(self, *, device: int | str | None = None, max_duration: float = 30.0):
        if max_duration <= 0 or max_duration > 30:
            raise ValueError("max_duration must be greater than zero and at most 30 seconds.")
        self.device = device
        self.max_duration = max_duration
        self._stream = None
        self._session = None
        self._chunks = []
        self._frames = 0
        self._sample_rate = 0
        self._limit = 0
        self._lock = Lock()
        self._overflow = False

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def limit_reached(self) -> bool:
        with self._lock:
            return self.is_recording and self._frames >= self._limit

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frames

    def snapshot(self) -> RecordedAudio:
        """Copy the current recording without consuming data or stopping capture."""
        import numpy as np

        with self._lock:
            data = b"".join(self._chunks)
            rate = self._sample_rate
        return RecordedAudio(np.frombuffer(data, dtype=np.float32).copy(), rate)

    def start(self) -> None:
        if self.is_recording:
            raise RuntimeError("Microphone recording is already active.")
        import sounddevice as sd

        session = audio_session(sd, follow_system_default=self.device is None)
        session.__enter__()
        try:
            self._start_stream(sd)
        except BaseException:
            session.__exit__(None, None, None)
            raise
        self._session = session

    def _start_stream(self, sd) -> None:
        device = sd.query_devices(self.device, "input")
        if device["max_input_channels"] < 1:
            raise RuntimeError("The selected device has no microphone input.")
        self._sample_rate = int(device["default_samplerate"])
        self._limit = int(self.max_duration * self._sample_rate)
        self._chunks = []
        self._frames = 0
        self._overflow = False

        def capture(data, frames, _time, status):
            with self._lock:
                self._overflow |= bool(status.input_overflow)
                keep = min(frames, self._limit - self._frames)
                if keep > 0:
                    self._chunks.append(bytes(data)[:keep * 4])
                    self._frames += keep

        stream = sd.RawInputStream(
            samplerate=self._sample_rate, channels=1, dtype="float32",
            device=self.device, callback=capture,
        )
        try:
            stream.start()
        except BaseException:
            stream.close()
            raise
        self._stream = stream

    def stop(self) -> RecordedAudio:
        if not self.is_recording:
            raise RuntimeError("Microphone recording is not active.")
        import numpy as np

        stream, self._stream = self._stream, None
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                self._session.__exit__(None, None, None)
                self._session = None
        with self._lock:
            data = b"".join(self._chunks)
            self._chunks.clear()
        if self._overflow:
            raise RuntimeError("The microphone lost audio frames. Please record again.")
        return RecordedAudio(np.frombuffer(data, dtype=np.float32).copy(), self._sample_rate)

    def cancel(self) -> None:
        """Stop recording and discard captured audio."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
            finally:
                try:
                    stream.close()
                finally:
                    self._session.__exit__(None, None, None)
                    self._session = None
        with self._lock:
            self._chunks.clear()
