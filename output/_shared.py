"""Small shared helpers without importing the speech model at module import."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"


def load_prompts() -> dict[str, Any]:
    """Load the preserved texts and voice instructions from prompts.json."""
    with PROMPTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def save_audio(
    output_path: str | Path,
    audio: Any,
    sample_rate: int,
    *,
    subtype: str | None = None,
) -> None:
    import soundfile as sf

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype=subtype)
