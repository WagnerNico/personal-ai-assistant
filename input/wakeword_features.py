"""Deterministic mel features shared by offline training and CPU detection."""

import numpy as np
from scipy.fft import rfft


SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000
FRAME_SAMPLES = 1280
FFT_SIZE = 512
HOP = 160
FRAME_LENGTH = 400
MEL_BANDS = 40
FEATURE_FRAMES = 1 + (WINDOW_SAMPLES - FRAME_LENGTH) // HOP


class MelFeatures:
    """Use a fixed normalization with no pretrained weights or per-clip statistics."""

    def __init__(self):
        self.window = np.hanning(FRAME_LENGTH).astype(np.float32)
        frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        limits = 2595 * np.log10(1 + np.array([125, 7500]) / 700)
        centers = 700 * (10 ** (np.linspace(*limits, MEL_BANDS + 2) / 2595) - 1)
        left = (frequencies[:, None] - centers[:-2]) / (centers[1:-1] - centers[:-2])
        right = (centers[2:] - frequencies[:, None]) / (centers[2:] - centers[1:-1])
        self.filters = np.maximum(0, np.minimum(left, right)).astype(np.float32)

    def frames(self, samples):
        framed = np.lib.stride_tricks.sliding_window_view(samples, FRAME_LENGTH, axis=-1)[..., ::HOP, :]
        spectrum = rfft(framed * self.window, n=FFT_SIZE, axis=-1, workers=1)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        mel = power @ self.filters
        return ((np.log(np.maximum(mel, 1e-8)) + 8) / 8).astype(np.float32)

    def __call__(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim not in (1, 2) or samples.shape[-1] != WINDOW_SAMPLES or not np.isfinite(samples).all():
            raise ValueError("Mel extraction requires finite two-second 16 kHz waveforms.")
        return self.frames(samples)


class StreamingMel:
    """Update only the eight new mel frames for each 80 ms audio block."""

    def __init__(self):
        self.extractor = MelFeatures()
        self.reset()

    def reset(self):
        self.audio = np.zeros(WINDOW_SAMPLES, np.float32)
        self.features = self.extractor(self.audio)

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (FRAME_SAMPLES,) or not np.isfinite(samples).all():
            raise ValueError("Supply exactly 1,280 finite mono samples per feature update.")
        self.audio[:-FRAME_SAMPLES] = self.audio[FRAME_SAMPLES:]
        self.audio[-FRAME_SAMPLES:] = samples
        count = FRAME_SAMPLES // HOP
        self.features[:-count] = self.features[count:]
        start = (FEATURE_FRAMES - count) * HOP
        end = (FEATURE_FRAMES - 1) * HOP + FRAME_LENGTH
        self.features[-count:] = self.extractor.frames(self.audio[start:end])
        return self.features
