"""Test microphone buffering and Whisper boundaries without microphone access."""

import importlib.util
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from input.audio import RecordedAudio
from input.microphone import MicrophoneRecorder
from input.transcription import WhisperBackend, load_whisper_backend, transcribe_audio


class MicrophoneTests(unittest.TestCase):
    def test_native_sample_rate_capture_is_bounded_and_stays_in_memory(self):
        module = ModuleType("sounddevice")
        module.query_devices = Mock(return_value={"default_samplerate": 48000, "max_input_channels": 1})
        module.RawInputStream = Mock()
        module._terminate = Mock()
        module._initialize = Mock()
        recorder = MicrophoneRecorder(max_duration=0.1)
        with patch.dict(sys.modules, {"sounddevice": module}):
            recorder.start()
            callback = module.RawInputStream.call_args.kwargs["callback"]
            samples = np.full(6000, 0.25, dtype=np.float32)
            callback(samples.tobytes(), 6000, None, SimpleNamespace(input_overflow=False))
            self.assertTrue(recorder.limit_reached)
            snapshot = recorder.snapshot()
            self.assertTrue(recorder.is_recording)
            self.assertEqual(recorder.frame_count, 4800)
            snapshot.samples[0] = 0.5
            audio = recorder.stop()
        self.assertEqual(audio.sample_rate, 48000)
        self.assertEqual(audio.samples.shape, (4800,))
        self.assertFalse(recorder.is_recording)
        self.assertEqual(float(audio.samples[0]), 0.25)
        module.RawInputStream.return_value.stop.assert_called_once()
        module.RawInputStream.return_value.close.assert_called_once()

    def test_cancel_discards_recording_and_closes_stream(self):
        module = ModuleType("sounddevice")
        module.query_devices = Mock(return_value={"default_samplerate": 48000, "max_input_channels": 1})
        module.RawInputStream = Mock()
        module._terminate = Mock()
        module._initialize = Mock()
        recorder = MicrophoneRecorder()
        with patch.dict(sys.modules, {"sounddevice": module}):
            recorder.start()
            recorder.cancel()
        self.assertFalse(recorder.is_recording)
        module.RawInputStream.return_value.abort.assert_called_once()
        module.RawInputStream.return_value.close.assert_called_once()


class TranscriptionTests(unittest.TestCase):
    def test_input_import_does_not_load_models_or_open_devices(self):
        subprocess.run([sys.executable, "-B", "-c", (
            "import sys; import input.microphone; import input.transcription; "
            "assert not {'torch', 'transformers', 'sounddevice', 'numpy'}.intersection(sys.modules)"
        )], check=True)

    def test_silent_and_invalid_audio_never_load_whisper(self):
        with patch("input.transcription._get_default_backend") as loader:
            for samples in (np.array([], dtype=np.float32), np.zeros(16000, dtype=np.float32)):
                self.assertEqual(transcribe_audio(RecordedAudio(samples, 16000)), "")
            for audio in (
                RecordedAudio(np.ones(10), 0), RecordedAudio(np.ones((5, 2)), 16000),
                RecordedAudio(np.array([np.nan]), 16000), RecordedAudio(np.ones(480001), 16000),
            ):
                with self.assertRaises(ValueError):
                    transcribe_audio(audio)
            loader.assert_not_called()

    def test_native_microphone_audio_is_resampled_to_whisper_rate(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("Run in .venv with input/requirements.txt installed.")
        backend = Mock()
        backend.transcribe.return_value = " Hello, Airis. "
        samples = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
        text = transcribe_audio(RecordedAudio(samples, 48000), backend=backend, language="English")
        self.assertEqual(text, "Hello, Airis.")
        converted = backend.transcribe.call_args.args[0]
        self.assertEqual(converted.sample_rate, 16000)
        self.assertEqual(converted.samples.shape, (16000,))
        self.assertEqual(converted.samples.dtype, np.float32)
        self.assertEqual(backend.transcribe.call_args.kwargs["language"], "English")

    def test_whisper_generation_uses_real_features_and_transcribes_source_language(self):
        import torch
        from transformers import WhisperFeatureExtractor

        extractor = WhisperFeatureExtractor(feature_size=128)
        processor = Mock(side_effect=extractor)
        processor.batch_decode.return_value = ["Hello, Airis."]
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2]])
        backend = WhisperBackend(model, processor, "cpu", torch.float32)
        samples = np.full(16000, 0.25, dtype=np.float32)
        result = transcribe_audio(RecordedAudio(samples, 16000), backend=backend, language="English")
        self.assertEqual(result, "Hello, Airis.")
        arguments = model.generate.call_args.kwargs
        self.assertEqual(arguments["input_features"].shape, (1, 128, 3000))
        self.assertIn("attention_mask", arguments)
        self.assertEqual(arguments["task"], "transcribe")
        self.assertEqual(arguments["language"], "English")

    def test_loader_uses_requested_checkpoint_and_serial_device_transfer(self):
        # Import native extensions outside patch.dict: restoring sys.modules
        # must not unload and later reinitialize PyTorch's native module.
        import torch

        module = ModuleType("transformers")
        module.WhisperProcessor = Mock()
        module.WhisperForConditionalGeneration = Mock()
        with patch.dict(sys.modules, {"transformers": module}):
            backend = load_whisper_backend(device="cpu", local_files_only=True)
        self.assertEqual(module.WhisperForConditionalGeneration.from_pretrained.call_args.args[0], "openai/whisper-large-v3-turbo")
        self.assertEqual(module.WhisperForConditionalGeneration.from_pretrained.call_args.kwargs["device_map"], "cpu")
        backend.model.to.assert_called_once_with("cpu")
        backend.model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
