"""Verify provisional snapshots, final verification, cancellation, and MLX routing."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from input.audio import RecordedAudio
from input.live_transcription import LiveTranscriptionSession
from input.mlx_transcription import MLXWhisperBackend, load_whisper_mlx_backend
from input.transcription import _get_default_backend, transcribe_audio


class LiveTranscriptionTests(unittest.TestCase):
    def test_provisional_text_is_revised_and_final_silence_reuses_verified_samples(self):
        transcriber = Mock(side_effect=["An unfinished", "A complete sentence."])
        session = LiveTranscriptionSession(transcriber=transcriber, prepare=Mock(), language="English")
        first = RecordedAudio(np.full(16000, 0.25, np.float32), 16000)
        second = RecordedAudio(np.full(32000, 0.25, np.float32), 16000)
        self.assertEqual(session.update(first), "An unfinished")
        self.assertEqual(session.update(second), "A complete sentence.")
        final = RecordedAudio(np.concatenate([second.samples, np.zeros(3200, np.float32)]), 16000)
        self.assertEqual(session.finish(final), "A complete sentence.")
        self.assertTrue(session.final_reused)
        self.assertEqual(transcriber.call_count, 2)

    def test_new_speech_or_changed_samples_require_complete_final_transcription(self):
        for changed_prefix in (False, True):
            with self.subTest(changed_prefix=changed_prefix):
                transcriber = Mock(side_effect=["Provisional words.", "Final complete words."])
                session = LiveTranscriptionSession(transcriber=transcriber, prepare=Mock())
                previous = RecordedAudio(np.full(16000, 0.25, np.float32), 16000)
                session.update(previous)
                samples = previous.samples.copy()
                if changed_prefix:
                    samples[0] = 0.5
                else:
                    samples = np.concatenate([samples, np.full(3200, 0.25, np.float32)])
                final = RecordedAudio(samples, 16000)
                self.assertEqual(session.finish(final), "Final complete words.")
                self.assertFalse(session.final_reused)
                self.assertIs(transcriber.call_args.args[0], final)

    def test_cancel_discards_a_preview_that_finishes_late(self):
        started, released = Event(), Event()
        def transcribe(audio, *, language):
            started.set()
            if not released.wait(2):
                raise TimeoutError("Test did not release transcription.")
            return "Cancelled words."
        session = LiveTranscriptionSession(transcriber=transcribe, prepare=Mock())
        audio = RecordedAudio(np.full(16000, 0.25, np.float32), 16000)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(session.update, audio)
            try:
                self.assertTrue(started.wait(2))
                session.cancel()
            finally:
                released.set()
            self.assertEqual(future.result(timeout=3), "")
        self.assertIsNone(session.last_audio)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            session.finish(audio)

    def test_mlx_loader_and_transcription_keep_turbo_fp16_and_native_language(self):
        module = ModuleType("mlx_audio.stt.utils")
        module.load = Mock()
        module.load.return_value.generate.return_value = SimpleNamespace(text=" Hallo Airis. ")
        with patch.dict(sys.modules, {"mlx_audio.stt.utils": module}):
            backend = load_whisper_mlx_backend()
        module.load.assert_called_once_with("openai/whisper-large-v3-turbo")
        audio = RecordedAudio(np.full(16000, 0.25, np.float32), 16000)
        self.assertEqual(transcribe_audio(audio, backend=backend, language="German"), "Hallo Airis.")
        arguments = backend.model.generate.call_args.kwargs
        self.assertEqual(arguments["language"], "de")
        self.assertEqual(arguments["task"], "transcribe")
        self.assertFalse(arguments["return_timestamps"])
        backend.prepare()
        self.assertEqual(backend.model.generate.call_args.kwargs["sample_len"], 2)
        calls = backend.model.generate.call_count
        backend.prepare()
        self.assertEqual(backend.model.generate.call_count, calls)

    def test_default_apple_silicon_backend_loads_mlx_once(self):
        _get_default_backend.cache_clear()
        try:
            with patch("input.transcription.platform.system", return_value="Darwin"), patch(
                "input.transcription.platform.machine", return_value="arm64",
            ), patch("input.mlx_transcription.load_whisper_mlx_backend", return_value=Mock()) as loader:
                self.assertIs(_get_default_backend(), _get_default_backend())
                loader.assert_called_once()
        finally:
            _get_default_backend.cache_clear()


if __name__ == "__main__":
    unittest.main()
