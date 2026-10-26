"""Clone a reference voice for speech generation, or run an Airis demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

if __package__:
    from ._shared import load_prompts, save_audio
    from .voices import VOICE_PROFILES_DIR, VOICE_PROFILES, voice_reference
else:
    from _shared import load_prompts, save_audio
    from voices import VOICE_PROFILES_DIR, VOICE_PROFILES, voice_reference


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REFERENCE_AUDIO = (
    VOICE_PROFILES_DIR / "mid" / "reference.wav"
)


def load_clone_model(device: str = "mps", dtype: Any = None) -> Any:
    """Load once and reuse the returned model for multiple requests."""
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.bfloat16 if dtype is None else dtype,
    )


def create_voice_prompt(
    model: Any,
    reference_audio: str | Path = DEFAULT_REFERENCE_AUDIO,
    reference_text: str | None = None,
) -> Any:
    """Prepare once per reference recording and reuse the voice prompt.

    reference_text must exactly match the speech in reference_audio. The
    mid and high profiles use the original voice_design.text transcript. The
    low profile uses voice_clone.reference_text. Other recordings need an explicit
    matching reference_text.
    """
    reference_path = Path(reference_audio).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_path}")

    if reference_text is None:
        for profile in VOICE_PROFILES:
            path, transcript = voice_reference(profile)
            if reference_path == path.resolve():
                reference_text = transcript
                break
        if reference_text is None:
            raise ValueError("Provide the exact reference_text for this reference recording.")

    return model.create_voice_clone_prompt(
        ref_audio=str(reference_path),
        ref_text=reference_text,
    )


def generate_speech(
    model: Any,
    text: str,
    voice_prompt: Any,
    *,
    language: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, int]:
    """Speak arbitrary text and return (audio, sample_rate).

    Reuse model and voice_prompt across calls. If output_path is provided,
    also save a PCM_16 WAV file, as in the original demo.
    """
    if language is None:
        language = load_prompts()["language"]

    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
    )
    audio = wavs[0]
    if output_path is not None:
        save_audio(output_path, audio, sample_rate, subtype="PCM_16")
    return audio, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sample", choices=("1", "2"), default="2")
    parser.add_argument("--text", help="Speak this text instead of a demo text.")
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument(
        "--reference-text", help="Exact transcript of the reference recording.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = args.text if args.text is not None else load_prompts()["demo_texts"][args.sample]
    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"airis_test_{args.sample}.wav"

    model = load_clone_model(device=args.device)
    voice_prompt = create_voice_prompt(model, args.reference_audio, args.reference_text)
    generate_speech(model, text, voice_prompt, output_path=output_path)


if __name__ == "__main__":
    main()
