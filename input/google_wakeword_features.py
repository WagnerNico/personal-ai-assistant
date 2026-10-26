"""Original Google mel preprocessing and a persistent CPU embedding frontend."""

from pathlib import Path

import numpy as np
from scipy.fft import rfft


class GoogleMel:
    def __init__(self, assets_path):
        assets = np.load(assets_path, allow_pickle=False)
        self.window = assets["window"]
        self.filters = assets["filters"]
        self.epsilon = float(assets["epsilon"])
        if (self.window.shape != (400,) or self.filters.shape != (257, 32)
                or not np.isfinite(self.window).all() or not np.isfinite(self.filters).all()
                or not np.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError("Unsupported Google frontend constants.")

    def frames(self, samples):
        samples = np.asarray(samples, np.float32)
        framed = np.lib.stride_tricks.sliding_window_view(samples, 400, axis=-1)[..., ::160, :]
        magnitude = np.abs(rfft(framed * self.window, n=512, axis=-1, workers=1))
        return np.log(magnitude @ self.filters + self.epsilon).astype(np.float32)


class GoogleEmbedding:
    """Share the same converted CNN and mel implementation offline and online."""

    def __init__(self, directory, *, threads=1):
        import onnxruntime as ort
        directory = Path(directory)
        self.mel = GoogleMel(directory / "frontend_assets.npz")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        ort.disable_telemetry_events()
        self.session = ort.InferenceSession(str(directory / "google_speech_embedding.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.zero_embedding = self.embeddings(self.mel.frames(np.zeros((1, 12400), np.float32)))[0]
        self.reset()

    def embeddings(self, mel):
        embeddings = self.session.run(None, {self.input_name: np.asarray(mel, np.float32)})[0].reshape(-1, 96)
        if not np.isfinite(embeddings).all():
            raise ValueError("The Google base produced nonfinite embeddings.")
        return embeddings

    def reset(self):
        self.audio = np.zeros(12400, np.float32)
        self.spectrogram = self.mel.frames(self.audio)
        self.features = np.tile(self.zero_embedding, (16, 1))

    def push(self, samples):
        samples = np.asarray(samples, np.float32)
        if samples.shape != (1280,) or not np.isfinite(samples).all():
            raise ValueError("Google embedding updates require 1,280 finite mono samples.")
        self.audio[:-1280] = self.audio[1280:]
        self.audio[-1280:] = samples
        self.spectrogram[:-8] = self.spectrogram[8:]
        self.spectrogram[-8:] = self.mel.frames(self.audio[-1520:])
        self.features[:-1] = self.features[1:]
        self.features[-1] = self.embeddings(self.spectrogram[None])[0]
        return self.features

    def clips(self, samples):
        """Vectorize the last sixteen 775 ms contexts of each two-second clip."""
        samples = np.asarray(samples, np.float32)
        if samples.ndim != 2 or samples.shape[1] != 32000:
            raise ValueError("Supply batches of two-second waveforms for Google features.")
        contexts = np.lib.stride_tricks.sliding_window_view(samples, 12400, axis=-1)[:, 400::1280]
        mel = self.mel.frames(contexts.reshape(-1, 12400))
        return self.embeddings(mel).reshape(len(samples), 16, 96)
