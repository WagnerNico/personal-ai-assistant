"""Apple Silicon Qwen TTS adapter with incremental PCM and cached voice profiles."""

import io
from functools import lru_cache

from .audio import SpeechAudio, combine_audio
from .voices import DEFAULT_VOICE, VOICE_PROFILES, voice_reference


MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


@lru_cache(maxsize=1)
def _get_model():
    from mlx_audio.tts.utils import load_model

    return load_model(MODEL_ID)


@lru_cache(maxsize=len(VOICE_PROFILES))
def _get_voice(voice=DEFAULT_VOICE):
    path, transcript = voice_reference(voice)
    from mlx_audio.utils import load_audio

    model = _get_model()
    reference = load_audio(str(path), sample_rate=model.sample_rate)
    return model, reference, transcript


def prepare(*, voice=DEFAULT_VOICE):
    """Warm the model and reference conditioning without playing or saving audio."""
    model, reference, transcript = _get_voice(voice)
    for _ in model.generate(
        text="Ready.", ref_audio=reference, ref_text=transcript,
        lang_code="English", max_tokens=8, verbose=False,
        stream=True, streaming_interval=0.32,
    ):
        pass


def stream_speech(text, *, language=None, voice=DEFAULT_VOICE):
    """Yield independent WAV chunks, each containing only newly generated samples."""
    yield from _generate_speech(text, language=language, stream=True, voice=voice)


def synthesize_speech(text, *, language=None, voice=DEFAULT_VOICE):
    """Use complete waveform decoding for a fully buffered spoken sentence."""
    return combine_audio(_generate_speech(text, language=language, stream=False, voice=voice))


def _generate_speech(text, *, language, stream, voice):
    import numpy as np
    import soundfile as sf

    model, reference, transcript = _get_voice(voice)
    for result in model.generate(
        text=text, ref_audio=reference, ref_text=transcript,
        lang_code=language or "auto", stream=stream,
        streaming_interval=0.32, verbose=False,
    ):
        samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if not samples.size:
            continue
        buffer = io.BytesIO()
        sf.write(buffer, samples, result.sample_rate, format="WAV", subtype="PCM_16")
        yield SpeechAudio(buffer.getvalue(), int(result.sample_rate))
