"""Exercise buffered wake capture and stream ownership without real devices."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from input.wakeword_assets import install_wake_model
from input.wakeword_listener import CommandCapture, WakeCapture, WakeWordListener
from input.wakeword import WakeWordDetector


def frame(value=0):
    return np.full(1280, value, np.float32)


def detector(activation_frame=1):
    fake = Mock()
    fake.trigger = SimpleNamespace(threshold=0.993588)
    calls = [0]
    def feed(samples):
        calls[0] += 1
        return [SimpleNamespace(score=0.999, activated=calls[0] == activation_frame)]
    fake.feed.side_effect = feed
    return fake


class WakeCaptureTests(unittest.TestCase):
    def test_candidate_contains_phrase_tail_without_direct_activation(self):
        capture = WakeCapture(detector(), candidate_only=True, activation_only=True, pre_roll_seconds=2.4)
        events = capture.feed(frame(0.1))
        self.assertFalse(capture.finished)
        self.assertFalse(any(kind in ("activated", "candidate", "audio") for kind, _ in events))
        self.assertEqual(capture.feed(frame(0.2)), [])
        self.assertEqual(capture.feed(frame(0.3)), [])
        events = capture.feed(frame(0.4))
        self.assertEqual([kind for kind, _ in events], ["candidate"])
        np.testing.assert_array_equal(events[0][1].samples[-1280:], frame(0.4))
        self.assertEqual(len(events[0][1].samples), 5120)
        self.assertTrue(capture.finished)
        self.assertFalse(capture.chunks)

    def test_activation_only_discards_wake_audio_without_recording_a_question(self):
        capture = WakeCapture(detector(3), activation_only=True)
        events = []
        for value in (0.1, 0.2, 0.3):
            events.extend(capture.feed(frame(value)))
        self.assertIn(("activated", None), events)
        self.assertFalse(any(kind == "audio" for kind, _ in events))
        self.assertTrue(capture.finished)
        self.assertEqual(list(capture.prefix), [])
        self.assertEqual(capture.chunks, [])
        self.assertEqual(capture.feed(frame(0.4)), [])

    def test_diagnostics_keep_brief_peaks_between_display_updates(self):
        fake = detector(9999)
        fake.feed.side_effect = [[SimpleNamespace(score=score, activated=False)]
                                for score in (0.1, 0.98, 0.97, 0.05, 0.03, 0.01)]
        capture = WakeCapture(fake)
        readings = []
        for _ in range(6):
            readings.extend(value for kind, value in capture.feed(frame(0.03)) if kind == "diagnostics")
        self.assertEqual(len(readings), 2)
        self.assertEqual(readings[0]["score"], 0.98)
        self.assertEqual(readings[1]["score"], 0.05)
        self.assertEqual(readings[1]["peak_score"], 0.98)
        self.assertEqual(readings[1]["confirmed_peak"], 0.97)
        self.assertEqual(readings[1]["frames"], 6)
        self.assertGreater(readings[1]["rms"], 0)

    def test_runtime_threshold_override_is_reversible_and_leaves_calibration_intact(self):
        instance = WakeWordDetector.__new__(WakeWordDetector)
        instance.calibrated_threshold = 0.993588
        instance.set_threshold(0.8)
        self.assertFalse(instance.trigger.update(0.9, 0.08))
        self.assertTrue(instance.trigger.update(0.9, 0.16))
        self.assertEqual(instance.calibrated_threshold, 0.993588)
        instance.set_threshold()
        self.assertFalse(instance.trigger.update(0.9, 0.08))
        self.assertFalse(instance.trigger.update(0.9, 0.16))
        with self.assertRaises(ValueError):
            instance.set_threshold(1.1)
        self.assertEqual(instance.trigger.threshold, 0.993588)

    def test_prefix_and_first_question_samples_survive_silence_endpoint(self):
        fake = detector(4)
        capture = WakeCapture(fake, silence_seconds=0.4)
        events = []
        for value in (0.01, 0.02, 0.03, 0.04, 0.1, 0.11, 0.12, 0.13, 0, 0, 0, 0, 0, 0):
            events.extend(capture.feed(frame(value)))
            if any(kind == "audio" for kind, _ in events):
                break
        self.assertIn(("state", "Recording"), events)
        audio = next(value for kind, value in events if kind == "audio")
        self.assertEqual(audio.sample_rate, 16000)
        np.testing.assert_array_equal(audio.samples[:1280], frame(0.01))
        np.testing.assert_array_equal(audio.samples[4 * 1280:5 * 1280], frame(0.1))
        self.assertEqual(fake.feed.call_count, 4)

    def test_initial_pause_allows_question_after_wake_phrase(self):
        capture = WakeCapture(detector(), silence_seconds=0.4)
        capture.feed(frame(0.1))
        for _ in range(20):
            self.assertFalse(any(kind == "audio" for kind, _ in capture.feed(frame())))
        capture.feed(frame(0.1))
        events = []
        for _ in range(6):
            events.extend(capture.feed(frame()))
        self.assertTrue(any(kind == "audio" for kind, _ in events))

    def test_recording_is_bounded_even_without_silence(self):
        capture = WakeCapture(detector(), max_seconds=2)
        events = []
        for _ in range(100):
            events = capture.feed(frame(0.1))
            if any(kind == "audio" for kind, _ in events):
                break
        audio = next(value for kind, value in events if kind == "audio")
        self.assertLessEqual(len(audio.samples), 32000)

    def test_idle_audio_is_bounded_and_no_turn_is_created(self):
        capture = WakeCapture(detector(9999))
        for _ in range(1000):
            self.assertFalse(any(kind == "audio" for kind, _ in capture.feed(frame())))
        self.assertLessEqual(len(capture.prefix), 20)
        self.assertEqual(capture.chunks, [])


class WakeListenerTests(unittest.TestCase):
    def test_candidate_is_delivered_only_after_microphone_and_lease_close(self):
        import audio_runtime
        ordering, events = [], []
        sd = self.sounddevice([0.1] * 5 + [0] * 6, ordering)
        listener = WakeWordListener("unused.onnx", detector_factory=lambda _: detector(), candidate_threshold=0.85)
        def emit(event):
            events.append(event)
            if event[0] == "candidate":
                self.assertEqual(ordering[-1], "closed")
                self.assertEqual(audio_runtime._active_streams, 0)
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"):
            listener.start(emit)
            listener.wait_stopped()
        self.assertEqual(sum(kind == "candidate" for kind, _ in events), 1)
        self.assertFalse(any(kind in ("activated", "audio") for kind, _ in events))

    def test_standby_uses_safe_microphone_but_question_keeps_system_selection(self):
        sd = self.sounddevice([0.1] * 5 + [0] * 6, [])
        listener = WakeWordListener("unused.onnx", detector_factory=lambda _: detector(), silence_seconds=0.4)
        with patch.dict(sys.modules, {"sounddevice": sd}), \
                patch("audio_runtime.platform.system", return_value="Linux"), \
                patch("input.wakeword_listener.standby_input_device", return_value=2) as select:
            listener.start(lambda _: None)
            listener.wait_stopped()
            listener.start_command(lambda _: None)
            listener.wait_stopped()
        select.assert_called_once_with(sd, None)
        self.assertEqual([call.kwargs["device"] for call in sd.RawInputStream.call_args_list], [2, None])

    def sounddevice(self, values, ordering):
        module = ModuleType("sounddevice")
        module.query_devices = Mock(return_value={"name": "Test microphone", "max_input_channels": 1})
        class Stream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                ordering.append("opened")
                for value in values:
                    self.kwargs["callback"](frame(value).tobytes(), 1280, None, False)
                return self
            def __exit__(self, *_):
                ordering.append("closed")
        module.RawInputStream = Mock(side_effect=Stream)
        return module

    def test_stream_and_audio_lease_close_before_activation_and_detector_reuses(self):
        import audio_runtime
        ordering, events = [], []
        sd = self.sounddevice([0.1] * 5 + [0] * 6, ordering)
        factory = Mock(return_value=detector())
        listener = WakeWordListener("unused.onnx", detector_factory=factory, silence_seconds=0.4)
        def emit(event):
            events.append(event)
            if event[0] == "activated":
                self.assertEqual(ordering[-1], "closed")
                self.assertEqual(audio_runtime._active_streams, 0)
        # Reset the fake detector's activation counter for each new session.
        factory.return_value.reset.side_effect = lambda: setattr(factory.return_value, "feed", detector().feed)
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"):
            for _ in range(2):
                listener.start(emit)
                listener.wait_stopped()
        factory.assert_called_once()
        self.assertEqual(sum(kind == "activated" for kind, _ in events), 2)
        self.assertFalse(any(kind == "audio" for kind, _ in events))
        self.assertEqual(ordering, ["opened", "closed", "opened", "closed"])

    def test_cancelled_listener_closes_stream_without_sending_audio(self):
        ordering, events = [], []
        sd = self.sounddevice([], ordering)
        listener = WakeWordListener("unused.onnx", detector_factory=lambda _: detector())
        def emit(event):
            events.append(event)
            if event == ("state", "Listening"):
                listener.stop()
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"):
            listener.start(emit)
            listener.wait_stopped()
        self.assertEqual(ordering, ["opened", "closed"])
        self.assertFalse(any(kind == "audio" for kind, _ in events))
        self.assertEqual(events[-1], ("stopped", None))

    def test_overflow_reports_error_and_releases_stream(self):
        ordering, events = [], []
        sd = self.sounddevice([0] * 26, ordering)
        listener = WakeWordListener("unused.onnx", detector_factory=lambda _: detector())
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"):
            listener.start(events.append)
            listener.wait_stopped()
        self.assertEqual(ordering, ["opened", "closed"])
        self.assertTrue(any(kind == "error" and "lost audio" in value for kind, value in events))
        self.assertFalse(any(kind == "audio" for kind, _ in events))

    def test_failed_model_load_does_not_open_microphone(self):
        events = []
        listener = WakeWordListener("unused.onnx", detector_factory=Mock(side_effect=ValueError("Invalid model")))
        with patch.dict(sys.modules, {"sounddevice": Mock()}) as modules:
            listener.start(events.append)
            listener.wait_stopped()
            modules["sounddevice"].RawInputStream.assert_not_called()
        self.assertIn(("error", "Invalid model"), events)

    def test_microphone_without_callbacks_reports_error_instead_of_listening_forever(self):
        ordering, events = [], []
        sd = self.sounddevice([], ordering)
        listener = WakeWordListener("unused.onnx", detector_factory=lambda _: detector())
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"), \
                patch("input.wakeword_listener.monotonic", side_effect=[0, 4]):
            listener.start(events.append)
            listener.wait_stopped()
        self.assertEqual(ordering, ["opened", "closed"])
        self.assertTrue(any(kind == "error" and "No audio is arriving" in value for kind, value in events))

    def test_fresh_command_stream_returns_question_audio_without_loading_detector(self):
        ordering, events = [], []
        sd = self.sounddevice([0.1] * 5 + [0] * 6, ordering)
        factory = Mock()
        listener = WakeWordListener("unused.onnx", detector_factory=factory, silence_seconds=0.4)
        with patch.dict(sys.modules, {"sounddevice": sd}), patch("audio_runtime.platform.system", return_value="Linux"):
            listener.start_command(events.append)
            listener.wait_stopped()
        factory.assert_not_called()
        self.assertIn(("state", "Awaiting question"), events)
        self.assertEqual(sum(kind == "audio" for kind, _ in events), 1)
        self.assertFalse(any(kind == "activated" for kind, _ in events))
        self.assertEqual(ordering, ["opened", "closed"])

    def test_persistent_question_stream_passes_unlimited_idle_wait_to_capture(self):
        sd = self.sounddevice([0.1] * 5 + [0] * 6, [])
        factory = Mock()
        listener = WakeWordListener("unused.onnx", detector_factory=factory, silence_seconds=0.4)
        with patch.dict(sys.modules, {"sounddevice": sd}), \
                patch("audio_runtime.platform.system", return_value="Linux"), \
                patch("input.wakeword_listener.CommandCapture", wraps=CommandCapture) as capture:
            listener.start_command(lambda _: None, wait_seconds=None)
            listener.wait_stopped()
        capture.assert_called_once_with(silence_seconds=0.4, wait_seconds=None)
        factory.assert_not_called()


class CommandCaptureTests(unittest.TestCase):
    def test_persistent_wait_keeps_only_a_short_prefix_and_accepts_a_late_question(self):
        capture = CommandCapture(wait_seconds=None, silence_seconds=0.4)
        for _ in range(1000):
            self.assertEqual(capture.feed(frame()), [])
        self.assertFalse(capture.finished)
        self.assertEqual(len(capture.prefix), 3)
        self.assertEqual(capture.chunks, [])
        events = capture.feed(frame(0.2))
        self.assertIn(("state", "Recording"), events)
        for _ in range(6):
            events.extend(capture.feed(frame()))
        self.assertEqual(sum(kind == "audio" for kind, _ in events), 1)
        self.assertFalse(any(kind == "timeout" for kind, _ in events))

    def test_waits_for_question_then_preserves_its_first_frame(self):
        capture = CommandCapture(silence_seconds=0.4)
        for _ in range(20):
            self.assertEqual(capture.feed(frame()), [])
        events = capture.feed(frame(0.2))
        self.assertIn(("state", "Recording"), events)
        capture.feed(frame(0.1))
        for _ in range(6):
            events.extend(capture.feed(frame()))
        audio = next(value for kind, value in events if kind == "audio")
        np.testing.assert_array_equal(audio.samples[2 * 1280:3 * 1280], frame(0.2))
        self.assertEqual(sum(kind == "audio" for kind, _ in events), 1)

    def test_silent_timeout_does_not_create_audio(self):
        capture = CommandCapture(wait_seconds=0.4)
        events = []
        for _ in range(10):
            events.extend(capture.feed(frame()))
        self.assertEqual(events, [("timeout", None)])
        self.assertTrue(capture.finished)

    def test_question_duration_remains_bounded_without_silence(self):
        capture = CommandCapture(max_seconds=1)
        events = []
        for _ in range(100):
            events.extend(capture.feed(frame(0.1)))
        audio = next(value for kind, value in events if kind == "audio")
        self.assertLessEqual(len(audio.samples), 16000)
        self.assertEqual(sum(kind == "audio" for kind, _ in events), 1)


class WakeAssetTests(unittest.TestCase):
    def test_deployment_preserves_hashes_threshold_and_failed_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training" / "head.onnx"
            source.parent.mkdir()
            source.write_bytes(b"classifier")
            base = root / "base"
            base.mkdir()
            hashes = {}
            for filename in ("google_speech_embedding.onnx", "frontend_assets.npz"):
                (base / filename).write_bytes(filename.encode())
                hashes[filename] = hashlib.sha256(filename.encode()).hexdigest()
            metadata = {"frontend": "google-speech-embedding-v1", "embedding_directory": "../base",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "embedding_hashes": hashes,
                        "threshold": 0.993, "ready_for_ui": False, "synthetic_acceptance_passed": False}
            source.with_suffix(".json").write_text(json.dumps(metadata))
            result = install_wake_model(source, root / "installed")
            config = json.loads(result.with_suffix(".json").read_text())
            self.assertEqual(result.read_bytes(), source.read_bytes())
            self.assertFalse(config["ready_for_ui"])
            self.assertFalse(config["synthetic_acceptance_passed"])
            self.assertEqual(config["threshold"], 0.993)
            self.assertEqual(config["embedding_directory"], "google_base")
            (base / "frontend_assets.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                install_wake_model(source, root / "rejected")
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
