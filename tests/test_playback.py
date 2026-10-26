"""Verify in-memory audio playback and the CLI's explicit saving behavior."""

import io
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch
import wave

from assistant import AssistantResponse, main
from output.audio import SpeechAudio
from output.playback import play_audio


def test_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(b"\x00\x00" * 3)
    return SpeechAudio(buffer.getvalue(), 24000)


class PlaybackTests(unittest.TestCase):
    def test_playback_sends_pcm_from_memory_to_the_output_device(self):
        module = ModuleType("sounddevice")
        module.RawOutputStream = MagicMock()
        stream = module.RawOutputStream.return_value
        with patch.dict(sys.modules, {"sounddevice": module}):
            play_audio(test_audio(), device="Test speaker")
        module.RawOutputStream.assert_called_once_with(
            samplerate=24000, channels=1, dtype="int16", device="Test speaker",
        )
        stream.write.assert_called_once_with(b"\x00\x00" * 3)

    def test_cli_playback_never_saves_without_an_output_argument(self):
        audio = test_audio()
        with patch.object(sys, "argv", ["assistant", "Hello.", "--mode", "speech", "--play"]):
            with patch("assistant.Assistant") as assistant, patch("output.playback.play_audio") as play:
                assistant.return_value.__enter__.return_value.respond.return_value = AssistantResponse("Hello.", audio)
                with patch.object(SpeechAudio, "save", side_effect=AssertionError("Unexpected file write.")):
                    with patch("sys.stdout", new_callable=io.StringIO):
                        main()
                play.assert_called_once_with(audio)

    def test_playback_flag_is_rejected_in_text_mode(self):
        with patch.object(sys, "argv", ["assistant", "Hello.", "--mode", "text", "--play"]):
            with patch("assistant.Assistant") as assistant, patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    main()
                assistant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
