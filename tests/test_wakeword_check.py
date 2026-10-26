"""Verify microphone diagnosis stays separate from conversation and training."""

from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

import numpy as np

from input.audio import RecordedAudio
from input.wakeword_check import check_wake_phrase
from input.wakeword_verification import WakeVerification


class WakePhraseCheckTests(unittest.TestCase):
    def test_diagnostic_uses_standby_microphone_instead_of_bluetooth_default(self):
        sd = Mock()
        sd.query_devices.return_value = {"name": "Built-in microphone"}
        factory = Mock(return_value=self.recorder)
        with patch.dict(sys.modules, {"sounddevice": sd}), \
                patch("audio_runtime.platform.system", return_value="Linux"), \
                patch("input.wakeword_check.standby_input_device", return_value=2):
            result = check_wake_phrase(standby=True, recorder_factory=factory,
                                      detector_factory=Mock(return_value=self.detector))
        factory.assert_called_once_with(device=2, max_duration=5.0)
        self.assertEqual(result["microphone"], "Built-in microphone")

    def test_classifier_candidate_requires_same_verification_as_live_detection(self):
        self.detector.feed.return_value[1].activated = True
        verifier = Mock()
        verifier.inspect.return_value = WakeVerification(False, False)
        result = self.run_check(verifier=verifier)
        self.assertTrue(result["classifier_activated"])
        self.assertFalse(result["activated"])
        self.assertTrue(result["verification_enabled"])
        self.transcriber.assert_not_called()
        self.assertLessEqual(len(verifier.inspect.call_args.args[0].samples), 38400)
        verifier.inspect.return_value = WakeVerification(True, True, "Ares.")
        result = self.run_check(verifier=verifier)
        self.assertTrue(result["activated"])
        self.assertEqual(result["transcript"], "Ares.")

    def setUp(self):
        self.recorder = Mock()
        self.recorder.is_recording = False
        self.recorder.limit_reached = True
        self.recorder.start.side_effect = lambda: setattr(self.recorder, "is_recording", True)
        self.audio = RecordedAudio(np.full(48000, 0.02, np.float32), 48000)
        def stop():
            self.recorder.is_recording = False
            return self.audio
        self.recorder.stop.side_effect = stop
        self.detector = Mock()
        self.detector.trigger.threshold = 0.993588
        self.detector.feed.return_value = [SimpleNamespace(score=value, activated=False)
                                           for value in (0.1, 0.98, 0.97, 0.1)]
        self.transcriber = Mock(return_value="Hey Airis.")

    def run_check(self, **kwargs):
        return check_wake_phrase(recorder_factory=Mock(return_value=self.recorder),
                                 detector_factory=Mock(return_value=self.detector),
                                 transcriber=self.transcriber, **kwargs)

    def test_same_audio_is_transcribed_and_resampled_for_diagnostic_model_scores(self):
        result = self.run_check(threshold=0.9, device="Headset", language="English")
        self.assertEqual(result["transcript"], "Hey Airis.")
        self.assertEqual(result["peak_score"], 0.98)
        self.assertEqual(result["confirmed_peak"], 0.97)
        self.assertFalse(result["activated"])
        self.assertFalse(result["audio_saved"])
        self.assertFalse(result["used_for_training"])
        converted = self.detector.feed.call_args.args[0]
        self.assertEqual(len(converted), 16000 + 7680)
        self.transcriber.assert_called_once_with(self.audio, language="English")
        self.recorder.stop.assert_called_once()
        self.detector.reset.assert_called_once()

    def test_cancelled_check_discards_capture_and_skips_recognition(self):
        cancelled = Event()
        self.recorder.limit_reached = False
        self.recorder.start.side_effect = lambda: (setattr(self.recorder, "is_recording", True), cancelled.set())
        self.assertIsNone(self.run_check(cancelled=cancelled))
        self.recorder.cancel.assert_called_once()
        self.transcriber.assert_not_called()
        self.detector.feed.assert_not_called()

    def test_empty_audio_reports_capture_failure(self):
        self.audio = RecordedAudio(np.empty(0, np.float32), 16000)
        with self.assertRaisesRegex(RuntimeError, "delivered no audio"):
            self.run_check()
        self.transcriber.assert_not_called()

    def test_zero_audio_reports_no_signal_without_whisper_hallucination(self):
        self.audio = RecordedAudio(np.zeros(16000, np.float32), 16000)
        result = self.run_check()
        self.assertEqual(result["rms"], 0)
        self.assertEqual(result["transcript"], "")
        self.transcriber.assert_not_called()


if __name__ == "__main__":
    unittest.main()
