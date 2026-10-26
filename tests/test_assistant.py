"""Verify mode switching, session boundaries, and the persistent TTS bridge."""

import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from assistant import Assistant
from output.audio import SpeechAudio
from output.speech_backend import SubprocessSpeechBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(b"\x00\x00" * 3)
    return buffer.getvalue()


class AssistantTests(unittest.TestCase):
    def test_interface_speech_never_calls_llm_or_enters_history(self):
        backend, speech = Mock(), Mock()
        audio = SpeechAudio(wav_bytes(), 24000)
        speech.synthesize.return_value = audio
        assistant = Assistant(backend=backend, speech_backend=speech)
        self.assertIs(assistant.synthesize_speech("I'm listening.", language="English"), audio)
        speech.synthesize.assert_called_once_with("I'm listening.", language="English")
        backend.respond.assert_not_called()
        self.assertEqual(assistant.history, [])

    def test_speech_guidance_is_per_turn_and_preserves_custom_base_and_history(self):
        backend = Mock()
        backend.respond.return_value = "A natural reply."
        speech = Mock()
        speech.synthesize.return_value = SpeechAudio(wav_bytes(), 24000)
        base = "You are Airis. Respect my custom instructions."
        assistant = Assistant(backend=backend, speech_backend=speech, system_prompt=base)
        assistant.respond("Hello.", mode="text")
        self.assertEqual(backend.respond.call_args.args[0][0]["content"], base)
        assistant.respond("Speak naturally.", mode="speech")
        messages = backend.respond.call_args.args[0]
        self.assertTrue(messages[0]["content"].startswith(base + "\n\n"))
        self.assertIn("Do not use headings, Markdown", messages[0]["content"])
        self.assertIn("short, complete sentence", messages[0]["content"])
        self.assertEqual(messages[1:3], assistant.history[:2])
        assistant.respond("Now use text.", mode="text")
        self.assertEqual(backend.respond.call_args.args[0][0]["content"], base)
        self.assertEqual(len(assistant.history), 6)

    def test_import_and_text_mode_do_not_load_tts_or_model_dependencies(self):
        subprocess.run([
            sys.executable, "-B", "-c", (
                "import sys; from assistant import Assistant; "
                "from unittest.mock import Mock; "
                "backend = Mock(); backend.respond.return_value = 'Hello.'; "
                "result = Assistant(backend=backend).respond('Hello.', mode='text'); "
                "assert result.text == 'Hello.' and result.audio is None; "
                "assert not {'torch', 'qwen_tts', 'soundfile', 'numpy', 'mlx', 'mlx_lm'}.intersection(sys.modules)"
            ),
        ], cwd=PROJECT_ROOT, check=True)

    def test_mode_switching_keeps_text_and_only_synthesizes_speech_turns(self):
        backend = Mock()
        backend.respond.side_effect = ["First reply.", "Second reply.", "Third reply."]
        speech = Mock()
        speech.synthesize.return_value = SpeechAudio(wav_bytes(), 24000)
        with Assistant(backend=backend, speech_backend=speech) as assistant:
            self.assertIsNone(assistant.respond("First.", mode="text").audio)
            response = assistant.respond("Second.", mode="speech", language="English")
            self.assertEqual(response.text, "Second reply.")
            self.assertEqual(response.audio.sample_rate, 24000)
            self.assertIsNone(assistant.respond("Third.", mode="text").audio)
            speech.synthesize.assert_called_once_with("Second reply.", language="English")
            messages = backend.respond.call_args.args[0]
            self.assertEqual(messages[2]["content"], "First reply.")
            self.assertEqual(messages[4]["content"], "Second reply.")
        # Injected services can be shared and belong to the caller.
        speech.close.assert_not_called()

    def test_sessions_have_independent_history_and_return_copies(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        first = Assistant(backend=backend)
        second = Assistant(backend=backend)
        first.respond("My name is Alex.")
        history = first.history
        history[0]["content"] = "Edited by the UI."
        self.assertEqual(first.history[0]["content"], "My name is Alex.")
        self.assertEqual(second.history, [])
        first.reset_history()
        self.assertEqual(first.history, [])

    def test_invalid_modes_and_speech_errors_do_not_commit_a_turn(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        speech = Mock()
        speech.synthesize.side_effect = RuntimeError("TTS unavailable.")
        assistant = Assistant(backend=backend, speech_backend=speech)
        with self.assertRaises(ValueError):
            assistant.respond("Hello.", mode="invalid")
        backend.respond.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "TTS unavailable"):
            assistant.respond("Hello.", mode="speech")
        self.assertEqual(assistant.history, [])

    def test_default_worker_is_created_lazily_reused_and_closed(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        with patch("assistant.SubprocessSpeechBackend") as worker:
            worker.return_value.synthesize.return_value = SpeechAudio(wav_bytes(), 24000)
            with Assistant(backend=backend) as assistant:
                assistant.respond("Hello.")
                worker.assert_not_called()
                assistant.respond("Hello.", mode="speech")
                assistant.respond("Hello again.", mode="speech")
                worker.assert_called_once()
            worker.return_value.close.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                assistant.respond("Hello.")


class SpeechBridgeTests(unittest.TestCase):
    def test_worker_can_prepare_without_audio_and_reuse_the_process(self):
        script = (
            "import base64, io, json, sys, wave\n"
            "prepared = False\n"
            "for line in sys.stdin:\n"
            " request = json.loads(line)\n"
            " if request['action'] == 'prepare':\n"
            "  prepared = True\n"
            "  print(json.dumps({'ok': True}), flush=True)\n"
            " else:\n"
            "  assert prepared\n"
            "  buffer = io.BytesIO()\n"
            "  with wave.open(buffer, 'wb') as audio:\n"
            "   audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000); audio.writeframes(b'\\x00\\x00')\n"
            "  print(json.dumps({'ok': True, 'sample_rate': 24000, 'wav_base64': base64.b64encode(buffer.getvalue()).decode()}), flush=True)\n"
        )
        backend = SubprocessSpeechBackend(command=[sys.executable, "-u", "-c", script])
        try:
            backend.prepare()
            process = backend._process
            audio = backend.synthesize("Hello.")
            self.assertIs(backend._process, process)
            self.assertEqual(audio.sample_rate, 24000)
        finally:
            backend.close()

    def test_real_child_process_returns_wav_and_is_reused_between_requests(self):
        # A small protocol worker substitutes for the expensive TTS model.
        script = (
            "import base64, io, json, os, sys, wave\n"
            "for line in sys.stdin:\n"
            " request = json.loads(line)\n"
            " buffer = io.BytesIO()\n"
            " with wave.open(buffer, 'wb') as audio:\n"
            "  audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)\n"
            "  audio.writeframes(b'\\x00\\x00' * len(request['text']))\n"
            " print(json.dumps({'ok': True, 'sample_rate': 24000, 'wav_base64': base64.b64encode(buffer.getvalue()).decode()}), flush=True)\n"
        )
        backend = SubprocessSpeechBackend(command=[sys.executable, "-u", "-c", script])
        try:
            first = backend.synthesize("Hello.")
            process = backend._process
            second = backend.synthesize("Hello again.", language="English")
            self.assertIs(backend._process, process)
            for audio, frames in ((first, 6), (second, 12)):
                with wave.open(io.BytesIO(audio.wav_bytes), "rb") as recording:
                    self.assertEqual(recording.getframerate(), audio.sample_rate)
                    self.assertEqual(recording.getnframes(), frames)
            with tempfile.TemporaryDirectory() as directory:
                path = first.save(Path(directory) / "nested" / "reply.wav")
                self.assertEqual(path.read_bytes(), first.wav_bytes)
        finally:
            backend.close()
        self.assertIsNotNone(process.poll())
        backend.close()

    def test_worker_error_is_reported_and_process_is_closed(self):
        script = "import sys; [print('{\"ok\":false,\"error\":\"Test failure.\"}', flush=True) for line in sys.stdin]"
        backend = SubprocessSpeechBackend(command=[sys.executable, "-u", "-c", script])
        try:
            with self.assertRaisesRegex(RuntimeError, "Test failure"):
                backend.synthesize("Hello.")
        finally:
            backend.close()

    def test_actual_worker_encodes_the_existing_tts_function_as_pcm_wav(self):
        from output.speech_worker import synthesize
        import base64

        with patch("output.speech_output.text_to_speech", return_value=([0.0, 0.25, -0.25], 24000)):
            response = synthesize("Hello.", "English")
        with wave.open(io.BytesIO(base64.b64decode(response["wav_base64"])), "rb") as recording:
            self.assertEqual(recording.getframerate(), 24000)
            self.assertEqual(recording.getsampwidth(), 2)
            self.assertEqual(recording.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
