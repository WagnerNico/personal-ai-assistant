"""Design a reference voice for Airis, as a function or command-line script."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

if __package__:
    from ._shared import load_prompts, save_audio
else:
    from _shared import load_prompts, save_audio


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def load_design_model(device: str = "mps", dtype: Any = None) -> Any:
    """Load once and reuse the returned model for multiple requests."""
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.bfloat16 if dtype is None else dtype,
    )


def generate_reference(
    model: Any = None,
    *,
    text: str | None = None,
    instruct: str | None = None,
    language: str | None = None,
    seed: int | None = 7,
    output_path: str | Path | None = None,
    device: str = "mps",
) -> tuple[Any, int]:
    """Return (audio, sample_rate); optionally also save the reference audio.

    Omitted texts use prompts.json. Pass a loaded model to avoid loading it
    again. The original seed is preserved; use seed=None to leave the process's
    random generators unchanged (for example, in a concurrent application).
    """
    prompts = load_prompts()
    reference = prompts["voice_design"]

    if seed is not None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if model is None:
        model = load_design_model(device=device)

    wavs, sample_rate = model.generate_voice_design(
        text=reference["text"] if text is None else text,
        language=prompts["language"] if language is None else language,
        instruct=reference["instruct"] if instruct is None else instruct,
    )
    audio = wavs[0]
    if output_path is not None:
        save_audio(output_path, audio, sample_rate)
    return audio, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "assistant_reference.wav",
    )
    args = parser.parse_args()
    generate_reference(device=args.device, output_path=args.output)


if __name__ == "__main__":
    main()
