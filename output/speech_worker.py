"""Optional persistent PyTorch TTS worker for custom runtimes."""

from contextlib import redirect_stdout
import base64
import io
import json
import sys
from functools import lru_cache

from .voices import DEFAULT_VOICE, VOICE_PROFILES, validate_voice, voice_reference


@lru_cache(maxsize=len(VOICE_PROFILES))
def _get_voice(voice):
    """Share the legacy model while caching each reference's conditioning."""
    from .speech_output import _get_default_voice
    from .voice_clone import create_voice_prompt

    model, prompt = _get_default_voice()
    if voice != DEFAULT_VOICE:
        path, transcript = voice_reference(voice)
        prompt = create_voice_prompt(model, path, transcript)
    return model, prompt


def synthesize(text: str, language: str | None = None, voice: str = DEFAULT_VOICE) -> dict:
    validate_voice(voice)
    # Keep all model output away from the stdout protocol stream.
    with redirect_stdout(sys.stderr):
        import soundfile as sf
        from .speech_output import text_to_speech

        if voice == DEFAULT_VOICE:
            audio, sample_rate = text_to_speech(text, language=language)
        else:
            model, prompt = _get_voice(voice)
            audio, sample_rate = text_to_speech(text, language=language, model=model, voice_prompt=prompt)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "ok": True,
        "wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "sample_rate": int(sample_rate),
    }


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            voice = validate_voice(request.get("voice", DEFAULT_VOICE))
            if request.get("action") == "prepare":
                with redirect_stdout(sys.stderr):
                    _get_voice(voice)
                response = {"ok": True}
            elif request.get("action") == "synthesize":
                response = synthesize(request["text"], request.get("language"), voice)
            else:
                raise ValueError("Unknown speech worker action.")
        except Exception as error:
            response = {"ok": False, "error": str(error)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
