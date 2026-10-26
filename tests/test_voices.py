"""Check voice switching across sessions, reference caches, and worker requests."""

import io
import json
import sys
import unittest
from unittest.mock import Mock, patch

from assistant import Assistant
from output.voices import voice_reference


class VoiceTests(unittest.TestCase):
    def test_profiles_use_preserved_transcripts_and_share_one_mlx_model(self):
        from output import mlx_speech
        from output._shared import load_prompts

        mlx_speech._get_voice.cache_clear()
        mlx_speech._get_model.cache_clear()
        loader = Mock(return_value=Mock(sample_rate=24000))
        audio_loader = Mock(side_effect=[object(), object(), object()])
        try:
            with patch.dict(sys.modules, {
                "mlx_audio.tts.utils": Mock(load_model=loader),
                "mlx_audio.utils": Mock(load_audio=audio_loader),
            }):
                voices = [mlx_speech._get_voice(profile) for profile in ("mid", "low", "high")]
                self.assertIs(mlx_speech._get_voice("mid"), voices[0])
            loader.assert_called_once_with(mlx_speech.MODEL_ID)
            self.assertEqual(audio_loader.call_count, 3)
            self.assertEqual([call.args[0] for call in audio_loader.call_args_list], [
                str(voice_reference(profile)[0]) for profile in ("mid", "low", "high")
            ])
            self.assertTrue(all(voice[0] is voices[0][0] for voice in voices))
            prompts = load_prompts()
            self.assertEqual(voices[0][2], prompts["voice_design"]["text"])
            self.assertEqual(voices[1][2], prompts["voice_clone"]["reference_text"])
            self.assertEqual(voices[2][2], prompts["voice_design"]["text"])
        finally:
            mlx_speech._get_voice.cache_clear()
            mlx_speech._get_model.cache_clear()

    def test_selected_voice_reaches_all_worker_actions(self):
        from output.mlx_speech_worker import main
        from output.speech_backend import SubprocessSpeechBackend
        from tests.test_streaming_speech import audio_chunk

        worker = Mock()
        worker.poll.return_value = None
        worker.stdout.readline.side_effect = [
            '{"ok":true}\n',
            '{"ok":true,"event":"done"}\n',
            '{"ok":false,"error":"Expected test error."}\n',
        ]
        backend = SubprocessSpeechBackend(engine="mlx")
        with patch("output.speech_backend.subprocess.Popen", return_value=worker) as launch:
            try:
                backend.set_voice("low")
                launch.assert_not_called()
                backend.prepare()
                list(backend.stream("Hello."))
                backend.set_voice("high")
                with self.assertRaisesRegex(RuntimeError, "Expected test error"):
                    backend.synthesize("Hello.")
                launch.assert_called_once()
                requests = [json.loads(call.args[0]) for call in worker.stdin.write.call_args_list]
            finally:
                backend.close()
        self.assertEqual([request["voice"] for request in requests], ["low", "low", "high"])
        self.assertEqual([request["action"] for request in requests], ["prepare", "stream", "synthesize"])
        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO("".join(json.dumps(request) + "\n" for request in requests))), patch(
            "sys.stdout", stdout,
        ), patch("output.mlx_speech.prepare") as prepare, patch(
            "output.mlx_speech.stream_speech", return_value=iter([]),
        ) as stream, patch("output.mlx_speech.synthesize_speech", return_value=audio_chunk()) as synthesize:
            main()
        prepare.assert_called_once_with(voice="low")
        stream.assert_called_once_with("Hello.", language=None, voice="low")
        synthesize.assert_called_once_with("Hello.", language=None, voice="high")

    def test_failed_voice_change_keeps_session_choice_and_history(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        speech = Mock()
        speech.set_voice.side_effect = RuntimeError("Voice unavailable.")
        assistant = Assistant(backend=backend, speech_backend=speech)
        assistant.respond("Hello.")
        history = assistant.history
        with self.assertRaisesRegex(RuntimeError, "Voice unavailable"):
            assistant.set_voice("high")
        self.assertEqual(assistant.voice, "mid")
        self.assertEqual(assistant.history, history)
        with self.assertRaisesRegex(ValueError, "Voice must"):
            assistant.set_voice("../missing")
        self.assertEqual(speech.set_voice.call_count, 1)


if __name__ == "__main__":
    unittest.main()
