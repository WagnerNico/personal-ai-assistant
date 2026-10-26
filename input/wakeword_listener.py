"""Optional CPU wake listening and bounded, in-memory voice turn capture."""

from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np

from audio_runtime import audio_session
from .audio_devices import standby_input_device
from .audio import RecordedAudio
from .wakeword import WakeWordDetector
from .wakeword_assets import DEFAULT_MODEL_DIRECTORY
from .wakeword_features import FRAME_SAMPLES, SAMPLE_RATE
from output.telemetry import logger


DEFAULT_WAKE_MODEL = DEFAULT_MODEL_DIRECTORY / "airis_wakeword.onnx"
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class WakeCapture:
    """Keep a rolling prefix, then finish a voice turn after trailing silence.

    In activation-only mode, detection discards the audio and returns an event.
    Optional combined capture uses adaptive audio energy for endpointing. The
    desktop opens a fresh CommandCapture after acknowledging the activation.
    """

    def __init__(self, detector, *, silence_seconds=1.2, pre_roll_seconds=1.6,
                 initial_grace_seconds=3.0, max_seconds=30.0, activation_only=False, candidate_only=False):
        if (not 0.4 <= silence_seconds <= 3 or not FRAME_SECONDS <= max_seconds <= 30
                or not FRAME_SECONDS <= pre_roll_seconds < max_seconds
                or not 0 <= initial_grace_seconds <= 30):
            raise ValueError("Invalid wake recording duration settings.")
        self.detector = detector
        self.activation_only = activation_only
        self.candidate_only = candidate_only
        self._candidate_tail = 0
        self.silence_seconds = silence_seconds
        self.initial_grace_seconds = initial_grace_seconds
        self.max_frames = int(max_seconds / FRAME_SECONDS)
        self.prefix = deque(maxlen=max(1, int(pre_roll_seconds / FRAME_SECONDS)))
        self.recording = False
        self.finished = False
        self.chunks = []
        self.elapsed = 0.0
        self.last_voice = 0.0
        self.followup = False
        self.noise = 0.002
        self.endpoint_threshold = 0.008
        self.frames = 0
        self.peak_score = 0.0
        self.confirmed_peak = 0.0
        self.previous_score = 0.0
        self.interval_score = 0.0
        self.interval_rms = 0.0
        self.interval_peak = 0.0
        self.detector.reset()

    def feed(self, samples):
        """Return events for exactly one normalized 80 ms frame."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (FRAME_SAMPLES,) or not np.isfinite(samples).all():
            raise ValueError("Wake capture requires a finite mono 80 ms frame at 16 kHz.")
        samples = np.clip(samples, -1, 1)
        if self.finished:
            return []
        if self._candidate_tail:
            self.prefix.append(samples.copy())
            self._candidate_tail -= 1
            if not self._candidate_tail:
                self.finished = True
                audio = RecordedAudio(np.concatenate(list(self.prefix)), SAMPLE_RATE)
                self.prefix.clear()
                return [("candidate", audio)]
            return []
        rms = float(np.sqrt(np.mean(samples * samples)))
        events = []
        if not self.recording:
            self.prefix.append(samples.copy())
            scored = self.detector.feed(samples)
            self.frames += 1
            self.interval_rms = max(self.interval_rms, rms)
            self.interval_peak = max(self.interval_peak, float(np.max(np.abs(samples))))
            for frame in scored:
                self.interval_score = max(self.interval_score, frame.score)
                self.peak_score = max(self.peak_score, frame.score)
                self.confirmed_peak = max(self.confirmed_peak, min(self.previous_score, frame.score))
                self.previous_score = frame.score
            if self.frames % 3 == 0 or any(frame.activated for frame in scored):
                events.append(("diagnostics", {
                    "score": self.interval_score, "peak_score": self.peak_score,
                    "confirmed_peak": self.confirmed_peak,
                    "rms": self.interval_rms, "audio_peak": self.interval_peak,
                    "frames": self.frames,
                }))
                self.interval_score = self.interval_rms = self.interval_peak = 0.0
            if self.frames % 60 == 0:
                logger.info("Wake signal: audio_s=%.2f rms=%.6f highest_score=%.6f confirmed_peak=%.6f",
                            self.frames * FRAME_SECONDS, rms, self.peak_score, self.confirmed_peak)
            if any(frame.activated for frame in scored):
                if self.candidate_only:
                    # Include the final consonant before verification. This is
                    # wake audio only, never a buffered question or history entry.
                    self._candidate_tail = 3
                    return events
                if self.activation_only:
                    self.finished = True
                    self.prefix.clear()
                    events.append(("activated", None))
                    return events
                self.recording = True
                self.chunks = list(self.prefix)
                self.prefix.clear()
                self.endpoint_threshold = max(0.008, self.noise * 2.5)
                events.append(("state", "Recording"))
            elif rms < max(0.015, self.noise * 3):
                self.noise = 0.98 * self.noise + 0.02 * rms
            return events
        self.chunks.append(samples.copy())
        self.elapsed += FRAME_SECONDS
        if rms >= self.endpoint_threshold:
            self.last_voice = self.elapsed
            if self.elapsed >= 0.24:
                self.followup = True
        quiet = self.elapsed - self.last_voice >= self.silence_seconds
        finished = (quiet and (self.followup or self.elapsed >= self.initial_grace_seconds))
        if finished or len(self.chunks) >= self.max_frames:
            audio = RecordedAudio(np.concatenate(self.chunks), SAMPLE_RATE)
            self.finished = True
            self.chunks.clear()
            events.append(("audio", audio))
        return events


class CommandCapture:
    """Capture one question; wait_seconds=None keeps idle capture active."""

    def __init__(self, *, silence_seconds=1.2, wait_seconds=10.0, max_seconds=30.0):
        if (not 0.4 <= silence_seconds <= 3 or
                (wait_seconds is not None and not 0 < wait_seconds <= 30) or not 1 <= max_seconds <= 30):
            raise ValueError("Invalid command capture duration settings.")
        self.silence_seconds = silence_seconds
        self.wait_frames = None if wait_seconds is None else int(wait_seconds / FRAME_SECONDS)
        self.max_frames = int(max_seconds / FRAME_SECONDS)
        self.prefix = deque(maxlen=3)
        self.chunks = []
        self.recording = self.finished = False
        self.waited = self.quiet = 0
        self.noise = 0.002
        self.threshold = 0.008

    def feed(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (FRAME_SAMPLES,) or not np.isfinite(samples).all():
            raise ValueError("Command capture requires a finite mono 80 ms frame at 16 kHz.")
        if self.finished:
            return []
        samples = np.clip(samples, -1, 1)
        rms = float(np.sqrt(np.mean(samples * samples)))
        events = []
        if not self.recording:
            self.prefix.append(samples.copy())
            self.waited += 1
            self.threshold = max(0.008, self.noise * 2.5)
            if rms >= self.threshold:
                self.recording = True
                self.chunks = list(self.prefix)
                self.prefix.clear()
                events.append(("state", "Recording"))
            else:
                self.noise = 0.98 * self.noise + 0.02 * rms
                if self.wait_frames is not None and self.waited >= self.wait_frames:
                    self.finished = True
                    self.prefix.clear()
                    events.append(("timeout", None))
            return events
        self.chunks.append(samples.copy())
        self.quiet = self.quiet + 1 if rms < self.threshold else 0
        if self.quiet * FRAME_SECONDS >= self.silence_seconds or len(self.chunks) >= self.max_frames:
            self.finished = True
            audio = RecordedAudio(np.concatenate(self.chunks), SAMPLE_RATE)
            self.chunks.clear()
            events.append(("audio", audio))
        return events


class WakeWordListener:
    """One reusable detector with a fresh microphone stream for each voice turn.

    Loading and inference run outside Qt and the GPU inference executor. Streams
    close before delivering audio, allowing default output devices to refresh.
    """

    def __init__(self, model_path=DEFAULT_WAKE_MODEL, *, device=None,
                 detector_factory=WakeWordDetector, silence_seconds=1.2, threshold=None, candidate_threshold=None):
        if candidate_threshold is not None and not 0 < candidate_threshold <= 1:
            raise ValueError("Wake candidate threshold must be between zero and one.")
        self.model_path = Path(model_path)
        self.device = device
        self.detector_factory = detector_factory
        self.silence_seconds = silence_seconds
        self.threshold = threshold
        self.candidate_threshold = candidate_threshold
        self.detector = None
        self._stop = Event()
        self._thread = None

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, emit):
        self._start(emit, command=False)

    def start_command(self, emit, *, wait_seconds=10.0):
        """Open a question stream after playback, optionally without an idle timeout."""
        self._start(emit, command=True, wait_seconds=wait_seconds)

    def _start(self, emit, *, command, wait_seconds=10.0):
        if self.is_running:
            raise RuntimeError("Wake listening is already active.")
        self._stop.clear()
        self._thread = Thread(target=self._run, args=(emit, command, wait_seconds), name="airis-wake-listener", daemon=True)
        self._thread.start()

    def stop(self):
        """Request cancellation; Qt polls is_running instead of blocking."""
        self._stop.set()

    def wait_stopped(self, timeout=5.0):
        """Wait on a background worker before opening another audio stream."""
        if self._thread is not None:
            self._thread.join(timeout)
        if self.is_running:
            raise RuntimeError("The wake listener is still stopping. Please try again.")

    def _run(self, emit, command=False, wait_seconds=10.0):
        audio = None
        candidate = None
        activated = timeout = False
        error = None
        try:
            if not command:
                emit(("state", "Loading"))
            if not command and self.detector is None:
                if self.detector_factory is WakeWordDetector and not self.model_path.is_file():
                    raise RuntimeError("No wake-word model is installed in input/models/wakeword.")
                self.detector = self.detector_factory(self.model_path)
            if self._stop.is_set():
                return
            if not command and isinstance(self.detector, WakeWordDetector):
                threshold = self.threshold
                if self.candidate_threshold is not None:
                    threshold = min(self.candidate_threshold, threshold if threshold is not None else self.detector.calibrated_threshold)
                self.detector.set_threshold(threshold)
            capture = (CommandCapture(silence_seconds=self.silence_seconds, wait_seconds=wait_seconds) if command else
                       WakeCapture(self.detector, silence_seconds=self.silence_seconds, activation_only=True,
                                   candidate_only=self.candidate_threshold is not None,
                                   pre_roll_seconds=2.4 if self.candidate_threshold is not None else 1.6))
            import sounddevice as sd

            pending = Queue(maxsize=25)
            overflow = Event()
            def receive(data, frames, _time, status):
                if status or frames != FRAME_SAMPLES:
                    overflow.set()
                    return
                try:
                    pending.put_nowait(bytes(data))
                except Full:
                    overflow.set()

            with audio_session(sd, follow_system_default=self.device is None):
                selected = self.device if command else standby_input_device(sd, self.device)
                device = sd.query_devices(selected, "input")
                if device["max_input_channels"] < 1:
                    raise RuntimeError("The selected device has no microphone input.")
                threshold = getattr(getattr(self.detector, "trigger", None), "threshold", None)
                emit(("microphone", {"name": device["name"], "sample_rate": SAMPLE_RATE,
                                     "threshold": threshold, "standby": not command}))
                logger.info("Wake listener: microphone=%s capture_hz=%d threshold=%s",
                            device["name"], SAMPLE_RATE, threshold)
                # CoreAudio performs native device rate conversion when needed.
                with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                       blocksize=FRAME_SAMPLES, device=selected, callback=receive):
                    emit(("state", "Awaiting question" if command else "Listening"))
                    last_audio = monotonic()
                    while not self._stop.is_set():
                        if overflow.is_set():
                            raise RuntimeError("Wake microphone lost audio frames. Disable and enable wake listening to retry.")
                        try:
                            block = pending.get(timeout=0.1)
                        except Empty:
                            if monotonic() - last_audio > 3:
                                raise RuntimeError("No audio is arriving from the microphone. Check macOS microphone permission and the selected input device.")
                            continue
                        last_audio = monotonic()
                        for kind, value in capture.feed(np.frombuffer(block, dtype=np.float32)):
                            if kind == "audio":
                                audio = value
                            elif kind == "candidate":
                                candidate = value
                            elif kind == "activated":
                                activated = True
                            elif kind == "timeout":
                                timeout = True
                            else:
                                emit((kind, value))
                        if capture.finished:
                            break
        except Exception as exception:
            error = str(exception)
            logger.error("Wake listener failed: %s", error)
        finally:
            # The audio lease has ended before the UI may start playback.
            if not self._stop.is_set():
                if error is not None:
                    emit(("error", error))
                elif audio is not None:
                    emit(("audio", audio))
                elif candidate is not None:
                    emit(("candidate", candidate))
                elif activated:
                    emit(("activated", None))
                elif timeout:
                    emit(("timeout", None))
            emit(("stopped", None))
