"""Text transcription with openai/whisper-large-v3-turbo."""

from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from typing import Any, Protocol
import platform

from .audio import RecordedAudio


MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_SAMPLE_RATE = 16000


class TranscriptionBackend(Protocol):
    def transcribe(self, audio: RecordedAudio, *, language: str | None = None, max_new_tokens: int = 256) -> str:
        ...


@dataclass
class WhisperBackend:
    model: Any
    processor: Any
    device: str
    dtype: Any

    def transcribe(self, audio: RecordedAudio, *, language: str | None = None, max_new_tokens: int = 256) -> str:
        import torch

        inputs = self.processor(
            audio.samples, sampling_rate=audio.sample_rate,
            return_tensors="pt", return_attention_mask=True,
        ).to(self.device, dtype=self.dtype)
        options = {"task": "transcribe", "max_new_tokens": max_new_tokens, "num_beams": 1}
        if language is not None:
            options["language"] = language
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, **options)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def load_whisper_backend(
    *, device: str | None = None, dtype: Any = None,
    model_id: str = MODEL_ID, local_files_only: bool = False,
) -> WhisperBackend:
    """Load once and reuse; load on CPU before transferring to an accelerator."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16
    processor = WhisperProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="cpu", local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return WhisperBackend(model=model, processor=processor, device=device, dtype=dtype)


@lru_cache(maxsize=1)
def _get_default_backend() -> TranscriptionBackend:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        from .mlx_transcription import load_whisper_mlx_backend
        return load_whisper_mlx_backend()
    return load_whisper_backend()


def prepare_transcription():
    """Load and warm the shared transcription backend before a voice request."""
    backend = _get_default_backend()
    if callable(getattr(type(backend), "prepare", None)):
        backend.prepare()
    return backend


def transcribe_audio(
    audio: RecordedAudio, *, backend: TranscriptionBackend | None = None,
    language: str | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Validate and resample a short recording, returning text in its source language.

    Supports up to 30 seconds. Empty or near-silent recordings return an empty
    string without loading Whisper. No recording is saved or added to history.
    """
    import numpy as np

    if max_new_tokens is not None and (type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256):
        raise ValueError("Transcription max_new_tokens must be an integer between 1 and 256.")
    if not isinstance(audio.sample_rate, int) or isinstance(audio.sample_rate, bool) or audio.sample_rate <= 0:
        raise ValueError("Audio sample_rate must be a positive integer.")
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite, one-dimensional mono samples.")
    if samples.size > audio.sample_rate * 30:
        raise ValueError("Recordings must be at most 30 seconds. Split longer audio into shorter recordings.")
    if not samples.size or float(np.max(np.abs(samples))) < 0.001:
        return ""
    if audio.sample_rate != WHISPER_SAMPLE_RATE:
        from scipy.signal import resample_poly

        factor = gcd(audio.sample_rate, WHISPER_SAMPLE_RATE)
        samples = resample_poly(samples, WHISPER_SAMPLE_RATE // factor, audio.sample_rate // factor).astype(np.float32)
    if backend is None:
        backend = _get_default_backend()
    options = {"language": language}
    if max_new_tokens is not None:
        options["max_new_tokens"] = max_new_tokens
    result = backend.transcribe(RecordedAudio(samples, WHISPER_SAMPLE_RATE), **options)
    if not isinstance(result, str):
        raise TypeError("The transcription backend must return a string.")
    return result.strip()
