# Local voice reference setup

Airis supports `low`, `mid`, and `high` profiles (default: `mid`), but **no WAV reference recordings are included** in this public-facing source snapshot.

The default Apple Silicon MLX-TTS implementation in `output/mlx_speech.py` loads the selected profile's `reference.wav`. Consequently, speech preparation or voice replies will fail until you supply a suitable local reference recording.

Place your own appropriately licensed or consented WAV recordings here:

```text
output/voice_profiles/
├── low/reference.wav
├── mid/reference.wav
└── high/reference.wav
```

You only need the recording(s) for profiles you intend to use. `output/prompts.json` contains the expected transcripts; the supplied recording must match the reference text exactly. To use a different transcript in custom voice-cloning experiments, provide the corresponding `reference_text` through the relevant API.

**Do not commit real voice recordings or personal audio.** The repository's `.gitignore` excludes WAV files in these profile directories. The text-chat workflow does not need reference audio.
