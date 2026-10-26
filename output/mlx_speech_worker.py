"""Persistent MLX TTS worker: one request yields audio chunks followed by done."""

import base64
from contextlib import redirect_stdout
import json
import sys

from .voices import DEFAULT_VOICE, validate_voice


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            voice = validate_voice(request.get("voice", DEFAULT_VOICE))
            if request.get("action") == "prepare":
                with redirect_stdout(sys.stderr):
                    from .mlx_speech import prepare
                    prepare(voice=voice)
                print(json.dumps({"ok": True}), flush=True)
            elif request.get("action") == "stream":
                with redirect_stdout(sys.stderr):
                    from .mlx_speech import stream_speech
                    chunks = stream_speech(request["text"], language=request.get("language"), voice=voice)
                while True:
                    with redirect_stdout(sys.stderr):
                        audio = next(chunks, None)
                    if audio is None:
                        break
                    print(json.dumps({
                        "ok": True, "event": "audio", "sample_rate": audio.sample_rate,
                        "wav_base64": base64.b64encode(audio.wav_bytes).decode("ascii"),
                    }), flush=True)
                print(json.dumps({"ok": True, "event": "done"}), flush=True)
            elif request.get("action") == "synthesize":
                with redirect_stdout(sys.stderr):
                    from .mlx_speech import synthesize_speech
                    audio = synthesize_speech(request["text"], language=request.get("language"), voice=voice)
                print(json.dumps({
                    "ok": True, "sample_rate": audio.sample_rate,
                    "wav_base64": base64.b64encode(audio.wav_bytes).decode("ascii"),
                }), flush=True)
            else:
                raise ValueError("Unknown MLX speech worker action.")
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
