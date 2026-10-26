"""Product entry point: turn an LLM's response text into Airis speech."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import platform

from .voice_clone import create_voice_prompt, generate_speech, load_clone_model
from ._shared import load_prompts


@lru_cache(maxsize=1)
def _get_default_voice() -> tuple[Any, Any]:
    """Prepare the default model and voice lazily, then reuse them."""
    model = load_clone_model()
    return model, create_voice_prompt(model)


def text_to_speech(
    text: str,
    *,
    language: str | None = None,
    output_path: str | Path | None = None,
    model: Any = None,
    voice_prompt: Any = None,
) -> tuple[Any, int]:
    """Convert complete LLM response text into (audio, sample_rate).

    Apple Silicon defaults to MLX Audio in the shared project environment.
    The first default call loads the model and reference voice; later calls
    reuse both. No model is loaded on import. Pass model and voice_prompt
    together to use a voice prepared by your application. Saving a WAV file
    is optional; audio playback is handled by the integrating application.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string containing the LLM response.")
    if not text.strip():
        raise ValueError("text must not be empty or contain only whitespace.")
    if (model is None) != (voice_prompt is None):
        raise ValueError("Provide model and voice_prompt together.")

    if model is None and platform.system() == "Darwin" and platform.machine() == "arm64":
        import io
        import soundfile as sf
        from .mlx_speech import synthesize_speech
        audio = synthesize_speech(text, language=language or load_prompts()["language"])
        samples, sample_rate = sf.read(io.BytesIO(audio.wav_bytes), dtype="float32")
        if output_path is not None:
            audio.save(output_path)
        return samples, sample_rate

    if model is None:
        model, voice_prompt = _get_default_voice()

    return generate_speech(
        model,
        text,
        voice_prompt,
        language=language,
        output_path=output_path,
    )
