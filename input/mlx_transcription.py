"""Whisper Turbo transcription through MLX using the original FP16 checkpoint."""

from dataclasses import dataclass, field
from typing import Any

from .audio import RecordedAudio


MODEL_ID = "openai/whisper-large-v3-turbo"


def _language_code(language):
    if language is None or language.lower() == "auto":
        return None
    return {"english": "en", "german": "de"}.get(language.lower(), language.lower())


@dataclass
class MLXWhisperBackend:
    model: Any
    _prepared: bool = field(default=False, init=False)

    def prepare(self):
        """Warm feature extraction, the encoder, and greedy text decoding."""
        if self._prepared:
            return
        import numpy as np

        self.model.generate(
            np.zeros(16000, dtype=np.float32), language="en", task="transcribe",
            return_timestamps=False, temperature=0.0, sample_len=2, verbose=None,
        )
        self._prepared = True
        self._release_temporary_buffers()

    def transcribe(self, audio: RecordedAudio, *, language=None, max_new_tokens=256):
        if audio.sample_rate != 16000:
            raise ValueError("MLX Whisper expects 16 kHz mono audio.")
        result = self.model.generate(
            audio.samples, language=_language_code(language), task="transcribe",
            return_timestamps=False, temperature=0.0, sample_len=max_new_tokens, verbose=None,
        )
        text = result.text.strip()
        self._release_temporary_buffers()
        return text

    def _release_temporary_buffers(self):
        """Release unused encoder allocations before Qwen needs GPU memory."""
        if callable(getattr(type(self.model), "parameters", None)):
            import mlx.core as mx
            mx.clear_cache()


def load_whisper_mlx_backend(*, model_id=MODEL_ID):
    """Load Whisper Turbo with MLX's native Hugging Face weight conversion."""
    from mlx_audio.stt.utils import load

    return MLXWhisperBackend(load(model_id))
