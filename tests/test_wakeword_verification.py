"""Confirm names without treating noise, ordinary speech or spelling as chat."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from input.audio import RecordedAudio
from input.speech_activity import SpeechActivity
from input.wakeword_verification import WakePhraseVerifier, is_wake_phrase
from input.transcription import transcribe_audio


class WakePhraseVerificationTests(unittest.TestCase):
    def test_name_and_observed_spellings_are_whole_utterances(self):
        for phrase in ("Airis", "Hey, Airis!", "Ares.", "Eris.", "Hey errors.", "Aris", "Errors.", "Arrows.", "Hey Aerith!"):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_wake_phrase(phrase))
        for phrase in ("", "Hey", "There are errors.", "The air is clear.", "Hey Alice.", "Hey Iris.",
                       "Open Airis.", "I spoke to Ares.", "Hey Airis, open Outlook.", "Airis 123"):
            with self.subTest(phrase=phrase):
                self.assertFalse(is_wake_phrase(phrase))

    def test_non_speech_never_calls_whisper_even_if_loud(self):
        activity = Mock()
        activity.has_speech.return_value = False
        transcriber = Mock(return_value="Airis")
        verifier = WakePhraseVerifier(transcriber=transcriber, speech_activity=activity)
        result = verifier.inspect(RecordedAudio(np.full(16000, 0.5, np.float32), 16000))
        self.assertFalse(result.accepted)
        self.assertFalse(result.speech)
        self.assertEqual(result.transcript, "")
        transcriber.assert_not_called()

    def test_confirmation_normalizes_quiet_speech_without_changing_original_audio(self):
        activity = Mock()
        activity.has_speech.return_value = True
        transcriber = Mock(return_value="Hey, Ares.")
        verifier = WakePhraseVerifier(transcriber=transcriber, speech_activity=activity)
        audio = RecordedAudio(np.full(16000, 0.005, np.float32), 16000)
        result = verifier.inspect(audio)
        self.assertTrue(result.accepted)
        self.assertEqual(result.transcript, "Hey, Ares.")
        transcriber.assert_called_once()
        self.assertEqual(transcriber.call_args.kwargs, {"language": "English"})
        self.assertLessEqual(float(np.max(transcriber.call_args.args[0].samples)), 0.1)
        np.testing.assert_array_equal(audio.samples, np.full(16000, 0.005, np.float32))

    def test_ordinary_speech_is_rejected_without_logging_its_words(self):
        activity = Mock()
        activity.has_speech.return_value = True
        transcriber = Mock(return_value="A private ordinary conversation.")
        verifier = WakePhraseVerifier(transcriber=transcriber, speech_activity=activity)
        with self.assertLogs("airis.speech", level="INFO") as log:
            self.assertFalse(verifier.verify(RecordedAudio(np.full(16000, 0.2, np.float32), 16000)))
        self.assertNotIn("private ordinary", str(log.output))

    def test_wrong_audio_rate_is_rejected_before_loading_models(self):
        verifier = WakePhraseVerifier(transcriber=Mock(), speech_activity=Mock())
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            verifier.verify(RecordedAudio(np.zeros(48000, np.float32), 48000))
        verifier.activity.has_speech.assert_not_called()

    def test_inference_never_downloads_a_missing_model(self):
        verifier = WakePhraseVerifier(transcriber=Mock())
        with patch("input.wakeword_verification.SpeechActivity", side_effect=RuntimeError("Model missing")), \
                patch("input.wakeword_verification.install_speech_activity_model") as download:
            with self.assertRaisesRegex(RuntimeError, "Model missing"):
                verifier.verify(RecordedAudio(np.zeros(16000, np.float32), 16000))
        download.assert_not_called()

    def test_ui_preparation_can_install_model_once_and_then_stay_offline(self):
        verifier = WakePhraseVerifier(transcriber=Mock())
        with patch("input.wakeword_verification.MODEL_PATH") as path, \
                patch("input.wakeword_verification.SpeechActivity", return_value=Mock()) as activity, \
                patch("input.wakeword_verification.install_speech_activity_model") as download:
            path.is_file.return_value = False
            verifier.prepare(download=True)
            verifier.prepare(download=True)
        download.assert_called_once()
        activity.assert_called_once()

    def test_short_transcription_budget_is_forwarded_without_changing_default(self):
        backend = Mock()
        backend.transcribe.return_value = "Airis"
        audio = RecordedAudio(np.full(16000, 0.1, np.float32), 16000)
        transcribe_audio(audio, backend=backend, language="English", max_new_tokens=24)
        self.assertEqual(backend.transcribe.call_args.kwargs, {"language": "English", "max_new_tokens": 24})
        transcribe_audio(audio, backend=backend)
        self.assertEqual(backend.transcribe.call_args.kwargs, {"language": None})
        for limit in (0, 257, True, 2.5):
            with self.assertRaises(ValueError):
                transcribe_audio(audio, backend=backend, max_new_tokens=limit)


class SpeechActivityTests(unittest.TestCase):
    def test_context_and_recurrent_state_reset_for_each_candidate(self):
        session = Mock()
        inputs = []
        def run(_, data):
            inputs.append({key: value.copy() for key, value in data.items()})
            return np.array([[0.9]], np.float32), np.ones((2, 1, 128), np.float32)
        session.run.side_effect = run
        activity = SpeechActivity(session=session)
        samples = np.full(2048, 0.1, np.float32)
        self.assertTrue(activity.has_speech(samples))
        self.assertTrue(activity.has_speech(samples))
        for index in (0, 4):
            self.assertEqual(inputs[index]["input"].shape, (1, 576))
            self.assertFalse(inputs[index]["input"][0, :64].any())
            self.assertFalse(inputs[index]["state"].any())
        self.assertTrue(inputs[1]["state"].all())
        np.testing.assert_array_equal(inputs[1]["input"][0, :64], samples[:64])

    def test_short_noise_spike_is_insufficient_and_partial_frame_is_padded(self):
        session = Mock()
        session.run.return_value = np.array([[0.99]]), np.zeros((2, 1, 128), np.float32)
        activity = SpeechActivity(session=session)
        self.assertFalse(activity.has_speech(np.ones(100, np.float32)))
        self.assertEqual(session.run.call_args.args[1]["input"].shape, (1, 576))
        self.assertFalse(session.run.call_args.args[1]["input"][0, 164:].any())

    def test_invalid_audio_or_model_values_do_not_confirm_speech(self):
        session = Mock()
        activity = SpeechActivity(session=session)
        for samples in (np.array([np.nan]), np.zeros((1, 10)), np.zeros(64001)):
            with self.assertRaises(ValueError):
                activity.has_speech(samples)
        session.run.assert_not_called()
        session.run.return_value = np.array([[np.nan]]), np.zeros((2, 1, 128), np.float32)
        with self.assertRaisesRegex(ValueError, "invalid result"):
            activity.has_speech(np.zeros(512))


if __name__ == "__main__":
    unittest.main()
