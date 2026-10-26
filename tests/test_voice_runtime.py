"""Opt-in, silent full-model checks for Whisper, cache equivalence, and voice latency."""

import json
import os
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter, sleep
import unittest


ENABLED = os.environ.get("AIRIS_RUN_VOICE_RUNTIME_TESTS") == "1" and platform.system() == "Darwin" and platform.machine() == "arm64"


@unittest.skipUnless(ENABLED, "Set AIRIS_RUN_VOICE_RUNTIME_TESTS=1 on Apple Silicon to run the full voice models.")
class VoiceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        processes = subprocess.check_output(["ps", "-A", "-o", "command="], text=True)
        if any(" -m ui.desktop" in line or " -m output.mlx_speech_worker" in line for line in processes.splitlines()):
            raise unittest.SkipTest("Quit Airis before running full-model tests to avoid duplicate model memory.")
        from input.transcription import prepare_transcription
        cls.whisper = prepare_transcription()
        cls.qwen = None

    def test_01_mlx_whisper_matches_the_preserved_reference_transcript(self):
        import soundfile as sf
        from input.audio import RecordedAudio
        from input.transcription import transcribe_audio
        from output._shared import load_prompts
        from output.voice_clone import DEFAULT_REFERENCE_AUDIO

        samples, rate = sf.read(DEFAULT_REFERENCE_AUDIO, dtype="float32")
        started = perf_counter()
        text = transcribe_audio(RecordedAudio(samples, rate), backend=self.whisper, language="English")
        normalize = lambda value: re.findall(r"[a-z0-9']+", value.lower())
        expected, actual = normalize(load_prompts()["voice_design"]["text"]), normalize(text)
        # Word edit distance tolerates punctuation and small decoding differences.
        row = list(range(len(actual) + 1))
        for index, word in enumerate(expected, 1):
            updated = [index]
            for other, candidate in enumerate(actual, 1):
                updated.append(min(updated[-1] + 1, row[other] + 1, row[other - 1] + (word != candidate)))
            row = updated
        error_rate = row[-1] / len(expected)
        print(f"\nMLX Whisper: {perf_counter() - started:.2f}s; reference word error rate: {error_rate:.3f}", flush=True)
        self.assertLess(error_rate, 0.15)

    def test_02_real_qwen_cache_reuses_history_without_changing_greedy_output(self):
        from processing.mlx_model import MLXBackend, load_mlx_backend

        type(self).qwen = load_mlx_backend()
        backend = type(self).qwen
        backend.temperature = 0.0
        messages = [{"role": "system", "content": "Answer briefly and accurately."},
                    {"role": "user", "content": "Name the capital of France."}]
        first = "".join(backend.stream(messages, max_new_tokens=24)).strip()
        messages += [{"role": "assistant", "content": first},
                     {"role": "user", "content": "Name the capital of Germany."}]
        cached = "".join(backend.stream(messages, max_new_tokens=24)).strip()
        reused = backend.last_reused_tokens
        uncached = MLXBackend(backend.model, backend.tokenizer, temperature=0.0).respond(messages, max_new_tokens=24).strip()
        self.assertGreater(reused, 0)
        self.assertEqual(cached, uncached)
        print(f"\nQwen cache: reused {reused} tokens; cached and uncached replies match.", flush=True)
        backend.reset_cache()

    def test_03_recorded_voice_to_first_response_audio_is_measured_without_playback(self):
        import numpy as np
        import soundfile as sf
        from assistant import Assistant
        from input.audio import RecordedAudio
        from input.live_transcription import LiveTranscriptionSession
        from output.speech_backend import SubprocessSpeechBackend
        from output.voice_clone import DEFAULT_REFERENCE_AUDIO
        from processing.mlx_model import MODEL_ID, load_mlx_backend
        from ui.workflow import ConversationWorkflow

        samples, rate = sf.read(DEFAULT_REFERENCE_AUDIO, dtype="float32")
        live = LiveTranscriptionSession(language="English")
        live.update(RecordedAudio(samples, rate))
        final = RecordedAudio(np.concatenate([samples, np.zeros(int(rate * 0.1), np.float32)]), rate)
        backend = type(self).qwen or load_mlx_backend()
        speech = SubprocessSpeechBackend(engine="mlx")
        first_audio = []
        try:
            speech.prepare()
            backend.warm_up()
            released = perf_counter()
            def consume(audio):
                if not first_audio:
                    first_audio.append(perf_counter() - released)
                sleep(audio.duration_seconds)
                return False
            with Assistant(backend=backend, speech_backend=speech) as assistant:
                ConversationWorkflow(assistant, player=consume).execute(
                    audio=final, transcription_session=live, released_at=released,
                    mode="speech", language="English", max_new_tokens=128,
                )
            self.assertTrue(live.final_reused)
            self.assertTrue(first_audio)
            result = {"release_to_first_audio_s": round(first_audio[0], 2),
                      "preview_reused": live.final_reused, "playback": "simulated", "model": MODEL_ID}
            destination = Path(__file__).resolve().parents[1] / "logs" / "voice_latency.json"
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(f"\nRecorded voice latency: {json.dumps(result)}", flush=True)
        finally:
            speech.close()


if __name__ == "__main__":
    unittest.main()
