"""Opt-in silent streaming check using the full MLX speech model on Apple Silicon."""

import io
import os
import platform
import time
import unittest
import wave

from output.speech_backend import SubprocessSpeechBackend


@unittest.skipUnless(
    os.environ.get("AIRIS_RUN_MLX_SPEECH_TESTS") == "1"
    and platform.system() == "Darwin" and platform.machine() == "arm64",
    "Set AIRIS_RUN_MLX_SPEECH_TESTS=1 to download/run the full speech model on Apple Silicon.",
)
class MLXSpeechRuntimeTests(unittest.TestCase):
    def test_full_model_streams_valid_audio_without_playing_or_saving(self):
        backend = SubprocessSpeechBackend(engine="mlx")
        try:
            backend.prepare()
            complete = backend.synthesize("Hello, welcome back.", language="English")
            with wave.open(io.BytesIO(complete.wav_bytes), "rb") as recording:
                self.assertEqual(recording.getframerate(), 24000)
                self.assertGreater(recording.getnframes(), 0)
            started = time.perf_counter()
            chunks, duration = 0, 0.0
            for audio in backend.stream("Hello, welcome back.", language="English"):
                if chunks == 0:
                    print(f"\nFirst audio chunk: {time.perf_counter() - started:.2f}s", flush=True)
                with wave.open(io.BytesIO(audio.wav_bytes), "rb") as recording:
                    self.assertEqual(recording.getframerate(), 24000)
                    self.assertEqual(recording.getsampwidth(), 2)
                    self.assertGreater(recording.getnframes(), 0)
                    duration += recording.getnframes() / recording.getframerate()
                chunks += 1
            self.assertGreater(chunks, 1)
            print(f"Audio: {duration:.2f}s in {chunks} chunks; generation: {time.perf_counter() - started:.2f}s", flush=True)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
