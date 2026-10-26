"""Verify macOS output changes without playing audio or resetting active capture."""

import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from input.microphone import MicrophoneRecorder
from output.playback import AudioStreamPlayer, play_audio
from tests.test_playback import test_audio


def sound_device():
    module = ModuleType("sounddevice")
    module.system_output = "Laptop speakers"
    module.snapshot_output = module.system_output
    module._initialized = 1
    module._terminate = Mock()
    def initialize():
        module.snapshot_output = module.system_output
    module._initialize = Mock(side_effect=initialize)
    module.query_devices = Mock(return_value={"default_samplerate": 48000, "max_input_channels": 1})
    module.RawInputStream = Mock()
    module.outputs = []
    def open_output(**kwargs):
        stream = Mock()
        stream.write.return_value = False
        stream.selected_device = kwargs["device"] or module.snapshot_output
        module.outputs.append(stream)
        return stream
    module.RawOutputStream = Mock(side_effect=open_output)
    return module


class AudioRoutingTests(unittest.TestCase):
    def setUp(self):
        self.sd = sound_device()
        self.platform = patch("audio_runtime.platform.system", return_value="Darwin")
        self.platform.start()
        self.modules = patch.dict(sys.modules, {"sounddevice": self.sd})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.platform.stop)

    def test_headphones_and_speakers_follow_system_selection_between_replies(self):
        with AudioStreamPlayer() as player:
            player.write(test_audio())
            player.write(test_audio())
        self.sd.system_output = "Headphones"
        play_audio(test_audio())
        self.sd.system_output = "Laptop speakers"
        play_audio(test_audio())
        self.assertEqual([stream.selected_device for stream in self.sd.outputs], [
            "Laptop speakers", "Headphones", "Laptop speakers",
        ])
        self.assertEqual(self.sd._initialize.call_count, 3)
        self.assertEqual(self.sd.outputs[0].write.call_count, 2)

    def test_active_microphone_is_not_reset_by_playback(self):
        recorder = MicrophoneRecorder()
        recorder.start()
        try:
            play_audio(test_audio())
            self.assertTrue(recorder.is_recording)
            self.sd._terminate.assert_called_once()
            self.sd.RawInputStream.return_value.close.assert_not_called()
        finally:
            recorder.cancel()
        self.sd.system_output = "Headphones"
        play_audio(test_audio())
        self.assertEqual(self.sd._initialize.call_count, 2)
        self.assertEqual(self.sd.outputs[-1].selected_device, "Headphones")

    def test_explicit_device_is_preserved_without_reenumeration(self):
        play_audio(test_audio(), device="External speaker")
        self.assertEqual(self.sd.outputs[0].selected_device, "External speaker")
        self.sd._terminate.assert_not_called()
        self.sd._initialize.assert_not_called()

    def test_failed_stream_start_releases_the_lease_for_the_next_reply(self):
        first = Mock()
        first.start.side_effect = RuntimeError("Device unavailable.")
        self.sd.RawOutputStream.side_effect = [first, Mock()]
        with self.assertRaisesRegex(RuntimeError, "Device unavailable"):
            play_audio(test_audio())
        first.close.assert_called_once()
        play_audio(test_audio())
        self.assertEqual(self.sd._initialize.call_count, 2)

    def test_other_platforms_do_not_reset_portaudio(self):
        with patch("audio_runtime.platform.system", return_value="Linux"):
            play_audio(test_audio())
        self.sd._terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
