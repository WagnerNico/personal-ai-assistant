"""Compose per-turn response instructions without changing conversation history."""

from pathlib import Path


SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
SPEECH_SYSTEM_PROMPT_PATH = Path(__file__).with_name("speech_system_prompt.txt")


def build_system_prompt(base_prompt: str | None = None, *, mode: str = "text") -> str:
    """Keep the base instructions and append spoken-response guidance in speech mode."""
    if mode not in ("text", "speech"):
        raise ValueError("mode must be 'text' or 'speech'.")
    if base_prompt is None:
        base_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not isinstance(base_prompt, str) or not base_prompt.strip():
        raise ValueError("system_prompt must be a nonempty string.")
    if mode == "speech":
        guidance = SPEECH_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return f"{base_prompt}\n\n{guidance}"
    return base_prompt
