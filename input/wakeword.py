"""Persistent CPU wake-word detection; does not open or record a microphone."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .wakeword_features import FRAME_SAMPLES, SAMPLE_RATE, StreamingMel


@dataclass(frozen=True)
class WakeFrame:
    time_seconds: float
    score: float
    activated: bool


class Trigger:
    """Require consecutive positive frames and rearm after quiet plus cooldown."""

    def __init__(self, threshold, consecutive=2, cooldown_seconds=1.5):
        if not 0 < threshold <= 1 or consecutive < 1 or cooldown_seconds < 0:
            raise ValueError("Invalid wake trigger settings.")
        self.threshold, self.consecutive = threshold, consecutive
        self.cooldown_seconds = cooldown_seconds
        self.reset()

    def reset(self):
        self.above = 0
        self.armed = True
        self.last_activation = float("-inf")

    def update(self, score, time_seconds):
        if not self.armed:
            if score < self.threshold * 0.5 and time_seconds - self.last_activation >= self.cooldown_seconds:
                self.armed = True
            self.above = 0
            return False
        self.above = self.above + 1 if score >= self.threshold else 0
        if self.above >= self.consecutive:
            self.armed, self.last_activation, self.above = False, time_seconds, 0
            return True
        return False


class WakeWordDetector:
    """Feed mono PCM16 or normalized float audio and receive scored 80 ms frames."""

    def __init__(self, model_path, *, threshold=None):
        import onnxruntime as ort

        model_path = Path(model_path)
        self.model_path = model_path
        config = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
        if config.get("frontend") not in ("airis-mel-v1", "google-speech-embedding-v1") or config.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("The model uses an unsupported audio frontend.")
        if config.get("sha256") and hashlib.sha256(model_path.read_bytes()).hexdigest() != config["sha256"]:
            raise ValueError("The wake-word model does not match its recorded file hash.")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.calibrated_threshold = float(config["threshold"])
        self.set_threshold(threshold)
        if config["frontend"] == "google-speech-embedding-v1":
            from .google_wakeword_features import GoogleEmbedding
            directory = (model_path.parent / config["embedding_directory"]).resolve()
            for filename, expected in config["embedding_hashes"].items():
                if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Google base artifact hash mismatch: {filename}")
            self.frontend = GoogleEmbedding(directory)
        else:
            self.frontend = StreamingMel()
        self.reset()

    def set_threshold(self, threshold=None):
        """Apply a runtime override without modifying the calibrated artifact."""
        self.trigger = Trigger(self.calibrated_threshold if threshold is None else threshold)

    def reset(self):
        self.frontend.reset()
        self.trigger.reset()
        self.pending = np.empty(0, np.float32)
        self.samples_processed = 0

    def score_frame(self, samples):
        features = self.frontend.push(samples)
        score = float(self.session.run(None, {self.input_name: features[None]})[0].reshape(-1)[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("The wake-word classifier produced an invalid probability.")
        self.samples_processed += FRAME_SAMPLES
        return score

    def feed(self, samples):
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError("Wake detection requires mono audio.")
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768
        else:
            samples = samples.astype(np.float32)
        if not np.isfinite(samples).all() or (samples.size and np.max(np.abs(samples)) > 1.01):
            raise ValueError("Supply PCM16 or normalized finite float audio at 16 kHz.")
        pending = np.concatenate((self.pending, samples))
        frames = []
        for start in range(0, len(pending) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            score = self.score_frame(pending[start:start + FRAME_SAMPLES])
            time_seconds = self.samples_processed / SAMPLE_RATE
            frames.append(WakeFrame(time_seconds, score, self.trigger.update(score, time_seconds)))
        self.pending = pending[len(frames) * FRAME_SAMPLES:].copy()
        return frames
