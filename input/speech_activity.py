"""Check short wake candidates for speech using a pinned CPU-only Silero model."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models/vad/silero_vad.onnx"
MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def install_speech_activity_model(path=MODEL_PATH):
    """Explicitly download a verified model; inference never accesses the network."""
    path = Path(path)
    with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
        data = response.read(4 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("The downloaded Silero model does not match its recorded hash.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


class SpeechActivity:
    """Use independent recurrent state for every candidate; no PyTorch or GPU."""

    def __init__(self, model_path=MODEL_PATH, *, session=None):
        if session is None:
            import onnxruntime as ort
            path = Path(model_path)
            if not path.is_file():
                raise RuntimeError("Install the speech check with: .venv/bin/python -m input.speech_activity --install")
            if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError("The Silero speech-check model does not match its recorded hash.")
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.session = session

    def probabilities(self, samples):
        samples = np.asarray(samples, np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) > 16000 * 4:
            raise ValueError("Speech checking requires at most four seconds of finite mono 16 kHz audio.")
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros(64, np.float32)
        scores = []
        for start in range(0, len(samples), 512):
            chunk = np.pad(samples[start:start + 512], (0, max(0, start + 512 - len(samples))))
            output, state = self.session.run(None, {
                "input": np.concatenate([context, chunk])[None], "state": state,
                "sr": np.array(16000, np.int64),
            })
            probability = float(np.asarray(output).reshape(-1)[0])
            if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(state).all():
                raise ValueError("The speech-check model produced an invalid result.")
            scores.append(probability)
            context = chunk[-64:].copy()
        return scores

    def has_speech(self, samples):
        # A brief unvoiced onset must not consume a Whisper run. Keep the minimum
        # short enough for the two syllables in the standalone name.
        probabilities = self.probabilities(samples)
        return sum(score >= 0.5 for score in probabilities) >= 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", required=True)
    parser.parse_args()
    print(install_speech_activity_model())
