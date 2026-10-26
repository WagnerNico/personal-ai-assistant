# Local voice profiles

The application supports `low`, `mid`, and `high` voice profiles; `mid` is the default.

**Reference recordings are not part of the showcase source tree.** This is intentional: voice-cloning reference audio may be private or subject to third-party voice rights.

For voice-cloning experiments, supply your own **consented, correctly licensed** audio at:

```text
output/voice_profiles/
├── low/reference.wav
├── mid/reference.wav
└── high/reference.wav
```

Each sample must match the spoken transcript expected by `output/prompts.json`. If you use a different recording, pass the correct `reference_text` to `create_voice_prompt`; it must exactly match the reference recording. Sample output audio should likewise be kept local unless you have permission to distribute it.

Voice features relying on absent recordings will not work without local setup. See `output/voices.py` and `output/voice_clone.py` for the current implementation.

Do not commit personal recordings to this repository. Audio that existed in the *private development repository's previous commits* is not removed from history by deleting files in a new branch.
