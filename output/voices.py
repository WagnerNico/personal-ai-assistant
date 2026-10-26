"""Built-in voice references and their preserved matching transcripts."""

from pathlib import Path

if __package__:
    from ._shared import load_prompts
else:
    from _shared import load_prompts


VOICE_PROFILES = ("low", "mid", "high")
DEFAULT_VOICE = "mid"
VOICE_PROFILES_DIR = Path(__file__).resolve().parent / "voice_profiles"


def validate_voice(voice: str) -> str:
    if voice not in VOICE_PROFILES:
        raise ValueError("Voice must be 'low', 'mid', or 'high'.")
    return voice


def voice_reference(voice: str = DEFAULT_VOICE) -> tuple[Path, str]:
    """Resolve a built-in recording without importing any model dependencies."""
    validate_voice(voice)
    prompts = load_prompts()
    transcript = (
        prompts["voice_clone"]["reference_text"] if voice == "low"
        else prompts["voice_design"]["text"]
    )
    return VOICE_PROFILES_DIR / voice / "reference.wav", transcript
