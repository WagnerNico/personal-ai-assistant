"""Verify the complete interaction loop independently of native UI and devices."""

import unittest
import io
import wave
from unittest.mock import Mock, patch
from threading import Event

from assistant import Assistant, AssistantResponse
from actions.media_control import SpeechMediaSession
from input.audio import RecordedAudio
from output.audio import SpeechAudio
from ui.workflow import ConversationWorkflow, VoiceTurnCancelled


class ConversationWorkflowTests(unittest.TestCase):
    def test_session_owner_can_keep_media_paused_across_voice_turns(self):
        control = Mock(last_error="")
        control.pause.return_value = {"bundle": "test.player"}
        media = SpeechMediaSession(control)
        assistant = Mock()
        assistant.respond.return_value = AssistantResponse("Hello.")
        workflow = ConversationWorkflow(assistant, player=Mock(), media_session=media)
        for _ in range(2):
            workflow.execute(text="Hello", mode="speech", keep_media_paused=True)
        control.pause.assert_called_once()
        control.resume.assert_not_called()
        self.assertTrue(media.active)
        media.end()
        control.resume.assert_called_once()

    def test_cancelled_turn_never_pauses_media_or_transcribes(self):
        cancelled = Event()
        cancelled.set()
        media, transcriber, assistant = Mock(), Mock(), Mock()
        workflow = ConversationWorkflow(assistant, transcriber=transcriber, media_session=media)
        with self.assertRaises(VoiceTurnCancelled):
            workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech", cancelled=cancelled)
        media.begin.assert_not_called()
        transcriber.assert_not_called()
        assistant.respond.assert_not_called()

    def test_cancellation_during_transcription_prevents_tools_or_model_calls(self):
        cancelled = Event()
        def transcribe(*args, **kwargs):
            cancelled.set()
            return "Open Spotify"
        media, assistant = Mock(), Mock()
        workflow = ConversationWorkflow(assistant, transcriber=transcribe, media_session=media)
        with self.assertRaises(VoiceTurnCancelled):
            workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech", cancelled=cancelled)
        assistant.respond.assert_not_called()
        media.end.assert_called_once()

    def test_cancellation_during_streamed_playback_exits_player_with_abort(self):
        cancelled = Event()
        media, assistant, player = Mock(), Mock(), Mock()
        player.__enter__ = Mock(return_value=player)
        player.__exit__ = Mock(return_value=False)
        player.write.side_effect = lambda _: cancelled.set()
        def respond(text, **kwargs):
            kwargs["on_audio"](object())
            return AssistantResponse("Hello")
        assistant.respond.side_effect = respond
        with patch("ui.workflow.AudioStreamPlayer", return_value=player):
            with self.assertRaises(VoiceTurnCancelled):
                ConversationWorkflow(assistant, media_session=media).execute(
                    text="Hello", mode="speech", cancelled=cancelled)
        self.assertIs(player.__exit__.call_args.args[0], VoiceTurnCancelled)
        media.end.assert_called_once()

    def test_media_stays_paused_through_transcription_and_playback(self):
        control = Mock(last_error="")
        control.pause.return_value = {"bundle": "test.player"}
        media = SpeechMediaSession(control)
        assistant = Mock()
        assistant.respond.return_value = AssistantResponse("Hello.", object())
        def transcribe(*args, **kwargs):
            self.assertTrue(media.active)
            control.resume.assert_not_called()
            return "Hello."
        def play(audio):
            self.assertTrue(media.active)
            control.resume.assert_not_called()
        workflow = ConversationWorkflow(assistant, transcriber=transcribe, player=play, media_session=media)
        workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech")
        control.resume.assert_called_once_with(control.pause.return_value)
        self.assertFalse(media.active)
        workflow.execute(text="Silent reply", mode="text")
        control.pause.assert_called_once()

    def test_media_resumes_after_streamed_player_drains(self):
        media = Mock()
        drained = []
        player = Mock()
        player.__enter__ = Mock(return_value=player)
        player.__exit__ = Mock(side_effect=lambda *_: drained.append(True))
        assistant = Mock()
        assistant.respond.return_value = AssistantResponse("Streamed reply.")
        media.end.side_effect = lambda: self.assertEqual(drained, [True])
        with patch("ui.workflow.AudioStreamPlayer", return_value=player):
            ConversationWorkflow(assistant, media_session=media).execute(text="Hello.", mode="speech")
        media.begin.assert_called_once()
        media.end.assert_called_once()

    def test_silence_and_transcription_model_or_playback_failure_release_media(self):
        for failure in ("silence", "transcription", "model", "playback"):
            with self.subTest(failure=failure):
                media = Mock()
                transcriber = Mock(return_value="" if failure == "silence" else "Question")
                assistant = Mock()
                assistant.respond.return_value = AssistantResponse("Reply", object())
                player = Mock()
                target = {"transcription": transcriber, "model": assistant.respond, "playback": player}.get(failure)
                if target is not None:
                    target.side_effect = RuntimeError("Failed turn")
                workflow = ConversationWorkflow(assistant, transcriber=transcriber, player=player, media_session=media)
                if failure == "silence":
                    self.assertIsNone(workflow.execute(audio=RecordedAudio([], 16000), mode="speech"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "Failed turn"):
                        workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech")
                media.begin.assert_called_once()
                media.end.assert_called_once()

    def test_invalid_request_does_not_pause_media(self):
        media = Mock()
        workflow = ConversationWorkflow(Mock(), media_session=media)
        for arguments in ({"mode": "speech"}, {"text": "Hello", "mode": "invalid"}):
            with self.assertRaises(ValueError):
                workflow.execute(**arguments)
        media.begin.assert_not_called()
        media.end.assert_not_called()

    def test_typed_and_spoken_input_share_history_and_text_mode_stays_silent(self):
        backend = Mock()
        backend.respond.side_effect = ["Hello, Alex.", "Your name is Alex."]
        speech = Mock()
        transcriber = Mock(return_value="My name is Alex.")
        player = Mock()
        events = []
        with Assistant(backend=backend, speech_backend=speech) as assistant:
            workflow = ConversationWorkflow(assistant, transcriber=transcriber, player=player)
            workflow.execute(audio=RecordedAudio([0.25], 16000), mode="text", emit=events.append)
            reply = workflow.execute(text="What is my name?", mode="text", emit=events.append)
            self.assertEqual(reply.text, "Your name is Alex.")
            self.assertEqual(assistant.history[0]["content"], "My name is Alex.")
            self.assertEqual(backend.respond.call_args.args[0][1]["content"], "My name is Alex.")
        transcriber.assert_called_once()
        speech.synthesize.assert_not_called()
        player.assert_not_called()
        self.assertIn(("user", "My name is Alex."), events)
        self.assertIn(("user", "What is my name?"), events)

    def test_speech_output_is_optional_and_occurs_after_text_is_available(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        speech = Mock()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(24000)
            recording.writeframes(b"\x00\x00" * 4)
        audio = SpeechAudio(buffer.getvalue(), 24000)
        speech.synthesize.return_value = audio
        events = []
        player = Mock(side_effect=lambda value: self.assertIn(("assistant_delta", "Hello."), events))
        with Assistant(backend=backend, speech_backend=speech) as assistant:
            workflow = ConversationWorkflow(assistant, player=player)
            workflow.execute(text="Hello.", mode="speech", language="English", emit=events.append)
            workflow.execute(text="Stay silent now.", mode="text", emit=events.append)
        player.assert_called_once_with(audio)
        speech.synthesize.assert_called_once_with("Hello.", language="English")

    def test_silence_does_not_create_a_conversation_turn(self):
        backend = Mock()
        events = []
        assistant = Assistant(backend=backend)
        workflow = ConversationWorkflow(assistant, transcriber=Mock(return_value=""))
        self.assertIsNone(workflow.execute(audio=RecordedAudio([], 16000), emit=events.append))
        self.assertEqual(assistant.history, [])
        backend.respond.assert_not_called()
        self.assertIn("No speech detected", events[-1][1])

    def test_invalid_input_combination_is_rejected_before_transcription(self):
        transcriber = Mock()
        workflow = ConversationWorkflow(Assistant(backend=Mock()), transcriber=transcriber)
        for arguments in ({}, {"text": "Hello.", "audio": RecordedAudio([], 16000)}):
            with self.assertRaises(ValueError):
                workflow.execute(**arguments)
        transcriber.assert_not_called()

    def test_wake_phrase_is_removed_only_from_wake_capture_prefix(self):
        backend = Mock()
        backend.respond.return_value = "Hello."
        transcriber = Mock(return_value="Hey Airis, what time is it?")
        assistant = Assistant(backend=backend)
        workflow = ConversationWorkflow(assistant, transcriber=transcriber)
        workflow.execute(audio=RecordedAudio([0.1], 16000), wake_word=True)
        self.assertEqual(assistant.history[0]["content"], "what time is it?")
        transcriber.return_value = "Tell me about Airis."
        workflow.execute(audio=RecordedAudio([0.1], 16000), wake_word=True)
        self.assertEqual(assistant.history[2]["content"], "Tell me about Airis.")
        transcriber.return_value = "Hey Airis, keep this greeting."
        workflow.execute(audio=RecordedAudio([0.1], 16000))
        self.assertEqual(assistant.history[4]["content"], "Hey Airis, keep this greeting.")

    def test_wake_phrase_without_question_never_calls_llm(self):
        backend = Mock()
        assistant = Assistant(backend=backend)
        workflow = ConversationWorkflow(assistant, transcriber=Mock(return_value="Hey Airis!"))
        self.assertIsNone(workflow.execute(audio=RecordedAudio([0.1], 16000), wake_word=True))
        backend.respond.assert_not_called()
        self.assertEqual(assistant.history, [])


if __name__ == "__main__":
    unittest.main()
