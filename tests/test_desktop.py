"""Headless desktop flow tests with simulated microphone and model backends."""

import importlib.util
import io
import wave
import os
import time
import unittest
from unittest.mock import Mock, patch
from threading import Event

from assistant import Assistant, AssistantResponse
from input.audio import RecordedAudio
from output.audio import SpeechAudio


class MemorySettings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class FakeWakeListener:
    """Simulate events only; never load a model or access a microphone."""

    def __init__(self):
        self.is_running = False
        self.start_count = 0
        self.command_count = 0
        self.command_waits = []
        self.emit = None

    def start(self, emit):
        self.start_count += 1
        self.is_running = True
        self.emit = emit
        emit(("state", "Listening"))

    def stop(self):
        self.is_running = False

    def start_command(self, emit, *, wait_seconds=10.0):
        self.command_count += 1
        self.command_waits.append(wait_seconds)
        self.is_running = True
        self.emit = emit
        emit(("state", "Awaiting question"))

    def wait_stopped(self):
        if self.is_running:
            raise RuntimeError("The fake stream must close before inference.")

    def deliver(self, kind, value):
        if kind in ("audio", "error", "activated", "candidate", "timeout"):
            self.is_running = False
        self.emit((kind, value))


class FakeShortcuts:
    def __init__(self):
        self.emit = None

    def start(self, emit):
        self.emit = emit

    def stop(self):
        self.emit = None

    def deliver(self, action):
        self.emit(action)


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "Install ui/requirements.txt in .venv.")
class DesktopTests(unittest.TestCase):
    def test_gmail_connection_runs_in_worker_and_updates_settings_status(self):
        self.window.assistant.connect_email = Mock()
        self.window.assistant.disconnect_email = Mock()
        self.window.run_email_connection("connect", "/tmp/test-client.json")
        self.window.poll_events()
        self.window.assistant.connect_email.assert_called_once_with("/tmp/test-client.json")
        self.assertIn("Gmail connected", self.window.email_connection_status.text())
        self.window.run_email_connection("disconnect", None)
        self.window.poll_events()
        self.window.assistant.disconnect_email.assert_called_once_with()
        self.assertIn("disconnected", self.window.email_connection_status.text())

    def test_email_setting_persists_and_disables_mailbox_reads(self):
        self.assertTrue(self.window.email_enabled.isChecked())
        self.window.email_enabled.setChecked(False)
        self.assertFalse(self.settings.value("email_enabled"))
        self.assertFalse(self.window.assistant._email_enabled)
        self.window.set_busy(True)
        self.assertFalse(self.window.email_enabled.isEnabled())
        self.window.set_busy(False)
        self.assertTrue(self.window.email_enabled.isEnabled())

    def test_web_setting_persists_and_source_links_require_an_explicit_click(self):
        from actions.web import WebSource
        from PySide6.QtCore import QUrl
        self.assertTrue(self.window.web_enabled.isChecked())
        self.window.web_enabled.setChecked(False)
        self.assertFalse(self.settings.value("web_enabled"))
        sources = (WebSource('<script>Unsafe title</script>', 'https://example.com/?a=1&b=2',
                             publisher="Publisher", published_at="2026-10-11T00:00:00+00:00"),)
        with patch("ui.desktop.QDesktopServices.openUrl") as open_url:
            self.window.events.put(("sources", sources))
            self.window.poll_events()
            open_url.assert_not_called()
            self.assertIn("<script>Unsafe title</script>", self.window.conversation.toPlainText())
            self.assertIn("Publisher · 2026-10-11", self.window.conversation.toPlainText())
            self.assertEqual(self.window.conversation.toPlainText(), self.window.voice_conversation.toPlainText())
            self.window.conversation.anchorClicked.emit(QUrl(sources[0].url))
            open_url.assert_called_once()
            self.window.conversation.anchorClicked.emit(QUrl("file:///private/data"))
            open_url.assert_called_once()
        self.window.append_message("You", "What is the current stock price?")
        self.window.insert_conversation("Airis: A factual answer.")
        from PySide6.QtGui import QTextCursor
        for view in (self.window.conversation, self.window.voice_conversation):
            text = view.toPlainText()
            for phrase in ("You: What is", "Airis: A factual"):
                cursor = QTextCursor(view.document())
                cursor.setPosition(text.index(phrase) + 1)
                self.assertFalse(cursor.charFormat().isAnchor())
            self.assertFalse(view.textCursor().charFormat().isAnchor())
        self.window.set_busy(True)
        self.assertFalse(self.window.web_enabled.isEnabled())
        self.window.set_busy(False)
        self.assertTrue(self.window.web_enabled.isEnabled())

    @classmethod
    def setUpClass(cls):
        if os.environ.get("AIRIS_RUN_QT_TESTS") != "1":
            raise unittest.SkipTest("Set AIRIS_RUN_QT_TESTS=1 where native Qt plugins can be accessed.")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from ui.qt_runtime import prepare_qt_plugins
        prepare_qt_plugins()
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui.desktop import AirisWindow

        self.backend = Mock()
        self.backend.respond.return_value = "Hello, Alex."
        self.recorder = Mock()
        self.recorder.is_recording = False
        self.recorder.limit_reached = False
        self.transcriber = Mock(return_value="My name is Alex.")
        self.player = Mock()
        self.speech = Mock()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(24000)
            recording.writeframes(b"\x00\x00" * 3)
        self.speech.synthesize.return_value = SpeechAudio(buffer.getvalue(), 24000)
        self.settings = MemorySettings()
        self.settings.setValue("background_input", "auto")
        self.wake = FakeWakeListener()
        self.shortcuts = FakeShortcuts()
        from actions.media_control import SpeechMediaSession
        self.media_control = Mock()
        self.media_control.pause.return_value = {"bundle": "test.player", "title": "Test item", "item": "1"}
        self.media_session = SpeechMediaSession(self.media_control)
        self.window = AirisWindow(
            assistant=Assistant(backend=self.backend, speech_backend=self.speech), recorder=self.recorder,
            transcriber=self.transcriber, player=self.player, settings=self.settings, wake_listener=self.wake,
            media_session=self.media_session,
            shortcut_listener=self.shortcuts,
        )
        self.window.timer.stop()

    def wait_for_reply(self):
        self.window._future.result(timeout=3)
        self.window.poll_events()
        self.assertFalse(self.window.busy)

    def setup_manual_shortcuts(self):
        self.window.background_input.setCurrentIndex(self.window.background_input.findData("hold"))
        self.recorder.start.side_effect = lambda: setattr(self.recorder, "is_recording", True)
        self.recorder.cancel.side_effect = lambda: setattr(self.recorder, "is_recording", False)
        self.recorder.stop.side_effect = lambda: (setattr(self.recorder, "is_recording", False)
                                                or RecordedAudio([0.1], 16000))
        self.window.start_shortcuts()

    def shortcut(self, action):
        self.shortcuts.deliver(action)
        self.window.poll_events()
        if action == "toggle_session" and self.window._background_starting:
            self.wait_for_reply()

    def tearDown(self):
        if self.window.busy:
            self.wait_for_reply()
        self.window.close()
        from PySide6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.application.processEvents()

    def test_typed_and_transcribed_messages_share_the_same_session(self):
        self.window.navigate("voice")
        self.window.submit(audio=RecordedAudio([0.25], 16000))
        self.wait_for_reply()
        self.window.navigate("text")
        self.window.message.setPlainText("What is my name?")
        self.window.send_text()
        self.wait_for_reply()
        history = self.window.assistant.history
        self.assertEqual(history[0]["content"], "My name is Alex.")
        self.assertEqual(history[2]["content"], "What is my name?")
        self.assertIn("You: My name is Alex.", self.window.conversation.toPlainText())
        self.player.assert_called_once()
        self.assertEqual(self.window.voice_conversation.toPlainText(), self.window.conversation.toPlainText())
        self.window.new_conversation()
        self.assertEqual(self.window.assistant.history, [])

    def test_speech_mode_plays_the_response_and_text_mode_stays_silent(self):
        audio = object()
        self.window.assistant = Mock()
        self.window.workflow.assistant = self.window.assistant
        self.window.assistant.respond.return_value = AssistantResponse("Hello.", audio)
        self.window.navigate("voice")
        self.window.submit(audio=RecordedAudio([0.25], 16000))
        self.wait_for_reply()
        self.player.assert_called_once_with(audio)
        self.assertEqual(self.window.assistant.respond.call_args.kwargs["mode"], "speech")
        self.assertEqual(self.window.assistant.respond.call_args.kwargs["language"], "Auto")

    def test_streamed_speech_starts_while_the_desktop_request_is_still_running(self):
        from output.audio import SpeechAudio

        played = Event()
        released = Event()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(24000)
            recording.writeframes(b"\x00\x00" * 24000)
        chunk = SpeechAudio(buffer.getvalue(), 24000)

        class LLM:
            def stream(self, messages, *, max_new_tokens):
                yield "Hello. "
                if not released.wait(3):
                    raise TimeoutError("Test did not release the LLM.")
                yield "Welcome back."

        class TTS:
            def synthesize(self, text, *, language):
                return chunk

            def stream(self, text, *, language):
                yield chunk

        self.window.assistant = Assistant(backend=LLM(), speech_backend=TTS())
        self.window.workflow.assistant = self.window.assistant
        self.player.side_effect = lambda audio: played.set()
        self.window.navigate("voice")
        self.window.submit(audio=RecordedAudio([0.25], 16000))
        try:
            self.assertTrue(played.wait(2))
            self.window.poll_events()
            self.assertTrue(self.window.busy)
            self.assertIn("Airis: Hello.", self.window.conversation.toPlainText())
        finally:
            released.set()
        self.wait_for_reply()
        self.assertEqual(self.player.call_count, 2)
        self.assertIn("Airis: Hello. Welcome back.", self.window.conversation.toPlainText())

    def test_empty_transcript_is_not_sent_to_the_assistant(self):
        self.transcriber.return_value = ""
        self.window.submit(audio=RecordedAudio([], 16000))
        self.wait_for_reply()
        self.backend.respond.assert_not_called()
        self.assertIn("No speech detected", self.window.status.text())

    def test_recording_starts_on_press_and_transcribes_on_release(self):
        self.window.navigate("voice")
        def start():
            self.recorder.is_recording = True
        def stop():
            self.recorder.is_recording = False
            return RecordedAudio([0.25], 16000)
        self.recorder.start.side_effect = start
        self.recorder.stop.side_effect = stop
        self.window.talk.pressed.emit()
        self.assertTrue(self.recorder.is_recording)
        self.window.talk.released.emit()
        self.wait_for_reply()
        self.recorder.start.assert_called_once()
        self.recorder.stop.assert_called_once()
        self.transcriber.assert_called_once()

    def test_live_transcript_appears_during_recording_and_is_reused_on_release(self):
        import numpy as np
        from input.live_transcription import LiveTranscriptionSession

        audio = RecordedAudio(np.full(16000, 0.25, np.float32), 16000)
        self.recorder.frame_count = 16000
        self.recorder.snapshot.return_value = audio
        self.recorder.start.side_effect = lambda: setattr(self.recorder, "is_recording", True)
        def stop():
            self.recorder.is_recording = False
            return RecordedAudio(np.concatenate([audio.samples, np.zeros(1600, np.float32)]), 16000)
        self.recorder.stop.side_effect = stop
        self.window._live_factory = lambda **kwargs: LiveTranscriptionSession(
            transcriber=self.transcriber, prepare=lambda: None, **kwargs,
        )
        self.window.navigate("voice")
        self.window.start_recording()
        self.window._live_future.result(timeout=2)
        self.window._live_next_update = 0.0
        self.window.poll_events()
        self.window._live_future.result(timeout=2)
        self.window.poll_events()
        self.assertTrue(self.recorder.is_recording)
        self.assertIn("My name is Alex.", self.window.transcription_preview.text())
        self.assertFalse(self.window.model.isEnabled())
        self.window.stop_recording()
        self.wait_for_reply()
        self.transcriber.assert_called_once()
        self.assertEqual(self.window.assistant.history[0]["content"], "My name is Alex.")
        self.assertEqual(self.window.transcription_preview.text(), "")

    def test_reply_limit_reaches_the_model_and_is_saved(self):
        self.assertEqual(self.window.reply_limit.value(), 2048)
        self.window.reply_limit.setValue(4096)
        self.window.message.setPlainText("Give me a detailed explanation.")
        self.window.send_text()
        self.wait_for_reply()
        self.assertEqual(self.backend.respond.call_args.kwargs["max_new_tokens"], 4096)
        self.assertEqual(self.settings.values["reply_limit"], 4096)

    def test_voice_selection_is_lazy_in_text_mode_and_survives_reopening(self):
        from ui.desktop import AirisWindow

        self.assertEqual(self.window.voice.currentData(), "mid")
        with patch("assistant.SubprocessSpeechBackend") as worker:
            self.window.voice.setCurrentText("High")
            self.window.submit(text="Hello.")
            self.wait_for_reply()
            worker.assert_not_called()
        self.assertEqual(self.window.assistant.voice, "high")
        self.assertEqual(self.settings.values["voice"], "high")
        self.window.close()
        self.window = AirisWindow(
            assistant=Assistant(backend=self.backend), recorder=self.recorder,
            transcriber=self.transcriber, player=self.player, settings=self.settings,
        )
        self.window.timer.stop()
        self.assertEqual(self.window.voice.currentText(), "High")
        self.assertEqual(self.window.assistant.voice, "high")

    def test_voice_change_warms_selected_profile_and_preserves_history(self):
        self.window.assistant = Assistant(backend=self.backend)
        self.window.workflow.assistant = self.window.assistant
        self.window.submit(text="Hello.")
        self.wait_for_reply()
        history = self.window.assistant.history
        released = Event()
        with patch("assistant.SubprocessSpeechBackend") as worker:
            worker.return_value.prepare.side_effect = lambda: released.wait(3)
            self.window.navigate("voice")
            self.window.navigate("settings")
            self.window.voice.setCurrentText("Low")
            try:
                self.assertTrue(self.window.busy)
                self.assertFalse(self.window.voice.isEnabled())
                self.assertFalse(self.window.send.isEnabled())
            finally:
                released.set()
            self.wait_for_reply()
            worker.assert_called_once_with(voice="low")
            worker.return_value.prepare.assert_called_once()
            self.window.voice.setCurrentText("High")
            self.wait_for_reply()
            worker.return_value.set_voice.assert_called_once_with("high")
            self.assertEqual(worker.call_count, 1)
        self.assertEqual(self.window.assistant.history, history)
        self.assertTrue(self.window.voice.isEnabled())
        self.assertIn("selected voice is prepared", self.window.status.text())

    def test_voice_selector_is_disabled_while_recording(self):
        self.window.navigate("voice")
        self.recorder.start.side_effect = lambda: setattr(self.recorder, "is_recording", True)
        def stop():
            self.recorder.is_recording = False
            return RecordedAudio([0.25], 16000)
        self.recorder.stop.side_effect = stop
        self.window.start_recording()
        self.assertFalse(self.window.voice.isEnabled())
        self.assertFalse(self.window.navigation.isEnabled())
        self.window.stop_recording()
        self.wait_for_reply()
        self.assertTrue(self.window.voice.isEnabled())

    def test_text_area_stays_silent_after_using_voice_chat(self):
        self.window.navigate("voice")
        self.window.submit(audio=RecordedAudio([0.25], 16000))
        self.wait_for_reply()
        self.player.reset_mock()
        self.speech.reset_mock()
        self.window.navigate("text")
        self.window.message.setPlainText("Reply in text, please.")
        self.window.send_text()
        self.wait_for_reply()
        self.player.assert_not_called()
        self.speech.synthesize.assert_not_called()
        self.recorder.start.assert_not_called()
        self.assertEqual(len(self.window.assistant.history), 4)

    def test_recording_cannot_start_from_text_or_settings(self):
        self.window.start_recording()
        self.window.navigate("settings")
        self.window.start_recording()
        self.recorder.start.assert_not_called()

    def test_selected_area_is_restored_and_text_chat_is_still_silent(self):
        from ui.desktop import AirisWindow

        self.window.navigate("voice")
        self.window.navigate("settings")
        self.window.close()
        self.window = AirisWindow(
            assistant=Assistant(backend=self.backend, speech_backend=self.speech), recorder=self.recorder,
            transcriber=self.transcriber, player=self.player, settings=self.settings,
        )
        self.window.timer.stop()
        self.assertEqual(self.window.current_page, "settings")
        self.window.navigate("text")
        self.window.message.setPlainText("Hello.")
        self.window.send_text()
        self.wait_for_reply()
        self.player.assert_not_called()
        self.speech.synthesize.assert_not_called()

    def test_message_editor_sends_with_enter_and_keeps_shift_enter_as_newline(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self.window.message.setPlainText("First line")
        self.window.message.moveCursor(self.window.message.textCursor().MoveOperation.End)
        QTest.keyClick(self.window.message, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.window.message.insertPlainText("Second line")
        self.backend.respond.assert_not_called()
        QTest.keyClick(self.window.message, Qt.Key.Key_Return)
        self.wait_for_reply()
        self.assertEqual(self.window.assistant.history[0]["content"], "First line\nSecond line")
        self.player.assert_not_called()

    def test_invalid_saved_voice_falls_back_to_mid(self):
        from ui.desktop import AirisWindow

        self.window.close()
        self.settings.values["voice"] = "missing"
        self.window = AirisWindow(
            assistant=Assistant(backend=self.backend), recorder=self.recorder,
            transcriber=self.transcriber, settings=self.settings,
        )
        self.window.timer.stop()
        self.assertEqual(self.window.voice.currentData(), "mid")
        self.assertEqual(self.window.assistant.voice, "mid")

    def test_partial_reply_appears_before_generation_finishes(self):
        released = Event()
        first_chunk = Event()

        class StreamingBackend:
            last_finish_reason = "stop"

            def stream(self, messages, *, max_new_tokens):
                yield "First "
                first_chunk.set()
                if not released.wait(timeout=3):
                    raise TimeoutError("Test did not release generation.")
                yield "second."

        self.window.assistant = Assistant(backend=StreamingBackend())
        self.window.workflow.assistant = self.window.assistant
        self.window.submit(text="Hello.")
        try:
            self.assertTrue(first_chunk.wait(timeout=2))
            self.window.poll_events()
            self.assertTrue(self.window.busy)
            self.assertIn("Airis: First", self.window.conversation.toPlainText())
        finally:
            released.set()
        self.wait_for_reply()
        document = self.window.conversation.toPlainText()
        self.assertEqual(document.count("Airis:"), 1)
        self.assertIn("Airis: First second.", document)

    def test_startup_preparation_runs_in_background_before_inputs_are_enabled(self):
        from ui.desktop import AirisWindow

        released = Event()
        warm_up = Mock(side_effect=lambda: released.wait(timeout=3))
        self.window.close()
        self.window = AirisWindow(
            assistant=Assistant(backend=self.backend), recorder=self.recorder,
            settings=self.settings, auto_prepare=False, prepare_callback=warm_up,
        )
        self.window.timer.stop()
        self.window.start_preparation()
        self.assertTrue(self.window.busy)
        self.assertFalse(self.window.send.isEnabled())
        released.set()
        self.wait_for_reply()
        warm_up.assert_called_once()
        self.assertTrue(self.window.send.isEnabled())
        self.assertIn("prepared", self.window.status.text())

    def test_token_limit_is_reported_in_the_status(self):
        self.backend.last_finish_reason = "length"
        self.window.submit(text="Explain this.")
        self.wait_for_reply()
        self.assertIn("response limit", self.window.status.text())

    def test_prepared_model_is_reused_and_switching_loads_only_the_new_model(self):
        from ui.desktop import LazyResponseBackend
        from processing.mlx_model import FAST_MODEL_ID, MODEL_ID

        first, second = Mock(), Mock()
        backend = LazyResponseBackend()
        with patch("ui.desktop.load_mlx_backend", side_effect=[first, second]) as loader:
            backend.prepare()
            backend.respond([{"role": "user", "content": "Hello."}], max_new_tokens=2048)
            backend.prepare(FAST_MODEL_ID)
            backend.prepare(FAST_MODEL_ID)
        self.assertEqual(loader.call_args_list[0].kwargs["model_id"], MODEL_ID)
        self.assertEqual(loader.call_args_list[1].kwargs["model_id"], FAST_MODEL_ID)
        first.warm_up.assert_called_once()
        second.warm_up.assert_called_once()

    def test_wake_listening_is_opt_in_and_test_counters_do_not_start_capture(self):
        self.window.poll_events()
        self.assertFalse(self.window.wake_enabled.isChecked())
        self.assertEqual(self.wake.start_count, 0)
        self.window.wake_test.setChecked(True)
        self.window.wake_missed.click()
        self.window.wake_false.click()
        self.assertIn("Marked missed: 1", self.window.wake_diagnostics.text())
        self.assertIn("Marked false: 1", self.window.wake_diagnostics.text())
        self.assertEqual(self.wake.start_count, 0)

    def test_wake_activation_routes_to_voice_and_resumes_after_spoken_reply(self):
        self.window.show()
        self.window.wake_enabled.setChecked(True)
        self.window.poll_events()
        self.assertIn("Listening", self.window.wake_status.text())
        self.wake.deliver("activated", None)
        with patch.object(self.window, "isActiveWindow", return_value=True):
            self.window.poll_events()
        self.wait_for_reply()
        self.assertEqual(self.window.current_page, "voice")
        self.assertFalse(self.window.navigation.isEnabled())
        self.assertEqual(self.window._wake_activations, 1)
        self.assertEqual(self.window.assistant.history, [])
        self.backend.respond.assert_not_called()
        self.transcriber.assert_not_called()
        self.assertEqual(self.wake.command_count, 1)
        self.transcriber.return_value = "tell me something."
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertEqual(self.window.assistant.history[0]["content"], "tell me something.")
        self.assertEqual(self.player.call_count, 2)
        self.assertTrue(self.window.navigation.isEnabled())
        self.assertTrue(self.wake.is_running)
        self.assertEqual(self.wake.start_count, 2)
        self.recorder.start.assert_not_called()

    def test_rejected_candidate_does_not_interrupt_media_or_start_conversation(self):
        self.window.wake_verifier = Mock()
        def verify(_):
            self.assertFalse(self.wake.is_running)
            return False
        self.window.wake_verifier.verify.side_effect = verify
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("candidate", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertEqual(self.window._wake_rejected, 1)
        self.assertEqual(self.wake.start_count, 2)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.media_control.pause.assert_not_called()
        self.player.assert_not_called()
        self.backend.respond.assert_not_called()
        self.assertFalse(self.window.assistant.history)

    def test_verified_candidate_enters_existing_acknowledgement_and_question_flow(self):
        self.window.show()
        self.window.wake_verifier = Mock()
        self.window.wake_verifier.verify.return_value = True
        self.window.wake_enabled.setChecked(True)
        audio = RecordedAudio([0.1], 16000)
        self.wake.deliver("candidate", audio)
        with patch.object(self.window, "isActiveWindow", return_value=True):
            self.window.poll_events()
        verification = self.window._future
        verification.result(timeout=3)
        with patch.object(self.window, "isActiveWindow", return_value=True):
            self.window.poll_events()
            if self.window.busy:
                self.wait_for_reply()
        self.window.wake_verifier.verify.assert_called_once_with(audio)
        self.assertEqual(self.window._wake_activations, 1)
        self.assertEqual(self.wake.command_count, 1)
        self.assertTrue(self.window._wake_command_active)
        self.media_control.pause.assert_called_once()
        self.media_control.resume.assert_not_called()
        self.player.assert_called_once()
        self.backend.respond.assert_not_called()
        self.assertFalse(self.window.assistant.history)
        self.wake.deliver("timeout", None)
        self.window.poll_events()
        self.media_control.resume.assert_called_once()

    def test_disabling_wake_during_verification_cannot_activate_late(self):
        self.window.wake_verifier = Mock()
        started, released = Event(), Event()
        def verify(_):
            started.set()
            if not released.wait(3):
                raise TimeoutError("Verification test did not release inference.")
            return True
        self.window.wake_verifier.verify.side_effect = verify
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("candidate", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        try:
            self.assertTrue(started.wait(2))
            self.window.wake_enabled.setChecked(False)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertFalse(self.wake.is_running)
        self.assertEqual(self.wake.command_count, 0)
        self.player.assert_not_called()
        self.media_control.pause.assert_not_called()

    def test_verification_failure_cannot_fall_back_to_an_unverified_activation(self):
        self.window.wake_verifier = Mock()
        self.window.wake_verifier.verify.side_effect = RuntimeError("Speech check unavailable")
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("candidate", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertTrue(self.window._wake_fault)
        self.assertIn("Speech check unavailable", self.window.status.text())
        self.assertEqual(self.wake.command_count, 0)
        self.media_control.pause.assert_not_called()

    def test_standby_microphone_hint_remains_visible_without_diagnostics(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("microphone", {"name": "MacBook microphone", "threshold": 0.85, "standby": True})
        self.window.poll_events()
        self.assertIn("MacBook microphone", self.window.wake_microphone_hint.text())
        self.wake.deliver("microphone", {"name": "EPOS", "threshold": 0.85, "standby": False})
        self.window.poll_events()
        self.assertIn("MacBook microphone", self.window.wake_microphone_hint.text())

    def test_disabling_wake_cancels_recording_and_ignores_queued_audio(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("state", "Recording")
        self.window.poll_events()
        old_emit = self.wake.emit
        self.window.wake_enabled.setChecked(False)
        old_emit(("audio", RecordedAudio([0.1], 16000)))
        self.window.poll_events()
        self.assertFalse(self.wake.is_running)
        self.assertTrue(self.window.navigation.isEnabled())
        self.assertEqual(self.window._wake_state, "Off")
        self.backend.respond.assert_not_called()
        self.assertFalse(self.settings.values["wake_enabled"])

    def test_typed_reply_stays_silent_with_wake_enabled(self):
        self.window.wake_enabled.setChecked(True)
        self.window.poll_events()
        self.window.message.setPlainText("Reply silently.")
        self.window.send_text()
        self.wait_for_reply()
        self.player.assert_not_called()
        self.speech.synthesize.assert_not_called()
        self.assertEqual(self.wake.start_count, 2)
        self.media_control.pause.assert_not_called()
        self.media_control.resume.assert_not_called()

    def test_push_to_talk_waits_for_listener_and_can_cancel_before_it_stops(self):
        self.window.navigate("voice")
        self.window.wake_enabled.setChecked(True)
        self.wake.stop = Mock()
        self.window.start_recording()
        self.assertTrue(self.window._pending_recording)
        self.recorder.start.assert_not_called()
        self.window.stop_recording()
        self.assertFalse(self.window._pending_recording)
        self.wake.is_running = False
        self.window.poll_events()
        self.recorder.start.assert_not_called()

    def test_wake_error_restores_controls_and_requires_explicit_retry(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("state", "Recording")
        self.window.poll_events()
        self.wake.deliver("error", "Microphone disconnected")
        self.window.poll_events()
        self.assertIn("Microphone disconnected", self.window.wake_status.text())
        self.assertTrue(self.window.navigation.isEnabled())
        self.assertEqual(self.wake.start_count, 1)
        self.window.poll_events()
        self.assertEqual(self.wake.start_count, 1)
        self.window.wake_enabled.setChecked(False)
        self.window.wake_enabled.setChecked(True)
        self.assertEqual(self.wake.start_count, 2)

    def test_cancel_button_discards_wake_recording(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("state", "Recording")
        self.window.poll_events()
        self.window.talk.pressed.emit()
        self.window.talk.released.emit()
        self.assertIn("cancelled", self.window.status.text())
        self.window.poll_events()
        self.backend.respond.assert_not_called()
        self.assertTrue(self.window.navigation.isEnabled())

    def test_closing_app_stops_wake_listener(self):
        self.window.wake_enabled.setChecked(True)
        self.assertTrue(self.wake.is_running)
        self.window.close()
        self.assertFalse(self.wake.is_running)
        self.assertTrue(self.window._closing)

    def test_wake_listener_stays_paused_until_audio_playback_finishes(self):
        self.window.wake_enabled.setChecked(True)
        released, playing = Event(), Event()
        def play(_):
            playing.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release playback.")
        self.player.side_effect = play
        self.window._wake_command_active = True
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.window.poll_events()
            self.assertTrue(self.window.busy)
            self.assertFalse(self.wake.is_running)
            self.assertEqual(self.wake.start_count, 1)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertEqual(self.wake.start_count, 2)

    def test_push_to_talk_opens_only_after_wake_microphone_closes(self):
        self.window.navigate("voice")
        self.window.wake_enabled.setChecked(True)
        self.wake.stop = Mock()
        self.window.start_recording()
        self.recorder.start.assert_not_called()
        self.wake.is_running = False
        self.recorder.start.side_effect = lambda: setattr(self.recorder, "is_recording", True)
        self.window.poll_events()
        self.recorder.start.assert_called_once()
        self.assertFalse(self.window._pending_recording)
        self.assertEqual(self.wake.start_count, 1)
        self.recorder.stop.side_effect = lambda: (setattr(self.recorder, "is_recording", False)
                                                or RecordedAudio([0.1], 16000))
        self.window.stop_recording()
        self.wait_for_reply()

    def test_wake_recording_continues_when_window_loses_focus(self):
        from PySide6.QtCore import QEvent
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("state", "Recording")
        self.window.poll_events()
        self.window.changeEvent(QEvent(QEvent.Type.ActivationChange))
        self.assertTrue(self.wake.is_running)
        self.assertEqual(self.window._wake_state, "Recording")

    def test_close_waits_for_wake_stream_shutdown(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.stop = Mock()
        self.assertFalse(self.window.close())
        self.assertIn("Stopping wake listening", self.window.status.text())
        self.wake.is_running = False
        self.window.poll_events()
        self.assertTrue(self.window._closing)

    def test_diagnostics_show_microphone_signal_and_retain_short_score_peaks(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("microphone", {"name": "EPOS headset", "threshold": 0.993588})
        self.wake.deliver("diagnostics", {"score": 0.98, "peak_score": 0.98, "confirmed_peak": 0.97,
                                           "rms": 0.02, "frames": 3})
        self.window.poll_events()
        self.assertIn("EPOS headset", self.window.wake_diagnostics.text())
        self.assertIn("Highest peak: 0.980000", self.window.wake_diagnostics.text())
        self.assertIn("two frames): 0.970000", self.window.wake_diagnostics.text())
        self.assertGreater(self.window.wake_level.value(), 0)
        self.wake.deliver("diagnostics", {"score": 0.1, "peak_score": 0.98, "confirmed_peak": 0.97,
                                           "rms": 0.0, "frames": 6})
        self.window.poll_events()
        self.assertIn("Highest peak: 0.980000", self.window.wake_diagnostics.text())
        self.assertIn("No input signal", self.window.wake_diagnostics.text())

    def test_custom_threshold_restarts_listener_and_can_restore_calibrated_default(self):
        self.window.wake_enabled.setChecked(True)
        self.assertIsNone(self.wake.threshold)
        self.window.wake_custom.setChecked(True)
        self.window.wake_threshold.setValue(0.85)
        self.assertEqual(self.wake.threshold, 0.85)
        self.assertTrue(self.settings.values["wake_custom_threshold"])
        self.assertEqual(self.settings.values["wake_threshold"], 0.85)
        self.assertTrue(self.window.wake_threshold.isEnabled())
        self.window.wake_restore.click()
        self.assertIsNone(self.wake.threshold)
        self.assertFalse(self.settings.values["wake_custom_threshold"])
        self.assertFalse(self.window.wake_threshold.isEnabled())

    def test_phrase_check_shows_recognition_and_does_not_send_or_speak(self):
        self.window.wake_enabled.setChecked(True)
        report = {"transcript": "Hey Airis", "peak_score": 0.95, "confirmed_peak": 0.94,
                  "threshold": 0.993588, "activated": False, "rms": 0.02, "audio_peak": 0.1}
        with patch("ui.desktop.check_wake_phrase", return_value=report) as check:
            self.window.start_wake_check()
            self.wait_for_reply()
        check.assert_called_once()
        self.assertIn("Whisper heard: Hey Airis", self.window.wake_check_result.text())
        self.assertIn("Activated: No", self.window.wake_check_result.text())
        self.assertEqual(self.window.assistant.history, [])
        self.backend.respond.assert_not_called()
        self.player.assert_not_called()
        self.assertTrue(self.wake.is_running)

    def test_threshold_and_phrase_check_are_disabled_during_wake_recording(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("state", "Recording")
        self.window.poll_events()
        self.assertFalse(self.window.wake_custom.isEnabled())
        self.assertFalse(self.window.wake_threshold.isEnabled())
        self.assertFalse(self.window.wake_check.isEnabled())

    def test_evaluated_preset_requires_explicit_application(self):
        self.window._wake_profile = {"threshold": 0.85, "positive_recordings": 6,
                                     "negative_audio_seconds": 21.95}
        self.window.wake_test.setChecked(True)
        self.window.wake_enabled.setChecked(True)
        self.assertFalse(self.window.wake_custom.isChecked())
        self.assertIsNone(self.wake.threshold)
        self.window.wake_profile_button.click()
        self.assertTrue(self.window.wake_custom.isChecked())
        self.assertEqual(self.wake.threshold, 0.85)
        self.assertEqual(self.settings.values["wake_threshold"], 0.85)
        self.assertIn("threshold applied", self.window.status.text())

    def test_wake_acknowledgement_finishes_before_question_capture_and_silent_timeout(self):
        self.window.show()
        self.window.wake_enabled.setChecked(True)
        playing, released = Event(), Event()
        def play(_):
            playing.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release the acknowledgement.")
        self.player.side_effect = play
        self.wake.deliver("activated", None)
        with patch.object(self.window, "isActiveWindow", return_value=True):
            self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.window.poll_events()
            self.assertTrue(self.window.busy)
            self.assertEqual(self.wake.command_count, 0)
            self.assertFalse(self.wake.is_running)
            self.assertEqual(self.window.assistant.history, [])
            self.transcriber.assert_not_called()
            self.backend.respond.assert_not_called()
            self.media_control.pause.assert_called_once()
            self.media_control.resume.assert_not_called()
        finally:
            released.set()
        self.wait_for_reply()
        self.assertEqual(self.wake.command_count, 1)
        self.assertTrue(self.media_session.active)
        self.wake.deliver("timeout", None)
        self.window.poll_events()
        self.assertFalse(self.window._wake_command_active)
        self.assertTrue(self.wake.is_running)
        self.assertEqual(self.wake.start_count, 2)
        self.transcriber.assert_not_called()
        self.backend.respond.assert_not_called()
        self.assertEqual(self.window.assistant.history, [])
        self.media_control.resume.assert_called_once()
        self.assertFalse(self.media_session.active)

    def test_cancelled_wake_capture_closes_before_media_resumes(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.wake.stop = Mock()
        self.window.cancel_background_listening()
        self.window.poll_events()
        self.media_control.resume.assert_not_called()
        self.assertTrue(self.window._media_release_pending)
        self.wake.is_running = False
        self.window.poll_events()
        self.media_control.resume.assert_called_once()
        self.assertFalse(self.media_session.active)

    def test_push_to_talk_pauses_before_microphone_and_releases_on_focus_loss(self):
        from PySide6.QtCore import QEvent
        self.window.navigate("voice")
        def capture():
            self.media_control.pause.assert_called_once()
            self.recorder.is_recording = True
        self.recorder.start.side_effect = capture
        self.recorder.cancel.side_effect = lambda: setattr(self.recorder, "is_recording", False)
        self.window.start_recording()
        self.assertTrue(self.media_session.active)
        with patch.object(self.window, "isActiveWindow", return_value=False):
            self.window.changeEvent(QEvent(QEvent.Type.ActivationChange))
        self.window.poll_events()
        self.media_control.resume.assert_called_once()
        self.assertFalse(self.media_session.active)

    def test_microphone_start_failure_restores_media(self):
        self.window.navigate("voice")
        self.recorder.start.side_effect = RuntimeError("Unavailable microphone")
        self.window.start_recording()
        self.media_control.pause.assert_called_once()
        self.media_control.resume.assert_called_once()
        self.assertFalse(self.media_session.active)

    def test_disabling_wake_during_acknowledgement_prevents_question_capture(self):
        self.window.wake_enabled.setChecked(True)
        playing, released = Event(), Event()
        self.player.side_effect = lambda _: (playing.set(), released.wait(3))
        self.wake.deliver("activated", None)
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.window.wake_enabled.setChecked(False)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertEqual(self.wake.command_count, 0)
        self.assertFalse(self.wake.is_running)
        self.assertFalse(self.window._wake_command_active)
        self.backend.respond.assert_not_called()

    def test_wake_audio_without_fresh_question_phase_cannot_enter_history(self):
        self.window.wake_enabled.setChecked(True)
        self.transcriber.return_value = "Hey Errors. Now it's working."
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.transcriber.assert_not_called()
        self.backend.respond.assert_not_called()
        self.assertEqual(self.window.assistant.history, [])

    def test_hidden_window_keeps_session_overlay_and_history_across_multiple_turns(self):
        self.window.hide()
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.assertFalse(self.window.isVisible())
        self.assertTrue(self.window.voice_overlay.isVisible())
        self.wait_for_reply()
        self.assertFalse(self.window.isVisible())
        self.assertTrue(self.window.voice_overlay.isVisible())
        self.assertEqual(self.window.voice_overlay.state, "Awaiting question")
        for question in ("My name is Alex.", "What is my name?", "Tell me more."):
            self.transcriber.return_value = question
            self.wake.deliver("audio", RecordedAudio([0.1], 16000))
            self.window.poll_events()
            self.wait_for_reply()
            self.assertFalse(self.window.isVisible())
            self.assertTrue(self.window.voice_overlay.isVisible())
            self.assertTrue(self.window.voice_overlay.animation.isActive())
            self.assertEqual(self.window.voice_overlay.state, "Awaiting question")
            self.assertTrue(self.window._background_session_active)
            self.assertTrue(self.media_session.active)
            self.media_control.resume.assert_not_called()
        self.assertEqual(len(self.window.assistant.history), 6)
        self.assertEqual(self.wake.start_count, 1)
        self.assertEqual(self.wake.command_count, 4)
        self.assertEqual(self.wake.command_waits, [None] * 4)
        self.assertEqual(self.window._wake_activations, 1)
        self.assertEqual(self.player.call_count, 4)  # One cue, three spoken replies.
        self.media_control.pause.assert_called_once()
        self.window.voice_overlay.cancel_requested.emit()
        self.window.poll_events()
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertFalse(self.window.voice_overlay.animation.isActive())
        self.assertFalse(self.window._background_session_active)
        self.media_control.resume.assert_called_once()
        self.assertEqual(self.wake.start_count, 2)

    def test_overlay_does_not_take_focus_and_hides_when_main_window_is_active(self):
        from PySide6.QtCore import Qt
        self.window._background_voice_turn = True
        self.window.set_wake_state("Processing")
        self.assertTrue(self.window.voice_overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertTrue(self.window.voice_overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)
        self.window.show()
        with patch.object(self.window, "isActiveWindow", return_value=True):
            self.window.update_voice_overlay()
            self.assertFalse(self.window.voice_overlay.isVisible())
        with patch.object(self.window, "isActiveWindow", return_value=False):
            self.window.update_voice_overlay()
            self.assertTrue(self.window.voice_overlay.isVisible())

    def test_clicking_background_indicator_cancels_waiting_for_question(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(self.window.voice_overlay, Qt.MouseButton.LeftButton)
        self.window.poll_events()
        self.assertFalse(self.window._wake_command_active)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertTrue(self.wake.is_running)
        self.backend.respond.assert_not_called()

    def test_background_session_can_be_started_again_after_explicit_end(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        first_cancelled = self.window._voice_cancelled
        self.window.cancel_background_listening()
        self.window.poll_events()
        self.assertTrue(first_cancelled.is_set())
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.assertIsNot(self.window._voice_cancelled, first_cancelled)
        self.assertFalse(self.window._voice_cancelled.is_set())
        self.assertTrue(self.window._background_session_active)
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertEqual(len(self.window.assistant.history), 2)
        self.assertTrue(self.window._background_session_active)

    def test_background_cancel_during_acknowledgement_prevents_late_question_capture(self):
        self.window.wake_enabled.setChecked(True)
        playing, released = Event(), Event()
        self.player.side_effect = lambda _: (playing.set(), released.wait(3))
        self.wake.deliver("activated", None)
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.assertTrue(self.window.voice_overlay.can_cancel)
            self.window.voice_overlay.cancel_requested.emit()
            self.assertFalse(self.window.voice_overlay.isVisible())
        finally:
            released.set()
        self.wait_for_reply()
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window._wake_command_active)
        self.assertEqual(self.wake.command_count, 0)
        self.media_control.resume.assert_called_once()
        self.assertTrue(self.wake.is_running)

    def test_visible_but_inactive_window_uses_continuous_background_session(self):
        self.window.show()
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        with patch.object(self.window, "isActiveWindow", return_value=False):
            self.window.poll_events()
            self.wait_for_reply()
            self.assertTrue(self.window._background_session_active)
            self.assertTrue(self.window.voice_overlay.isVisible())
            self.wake.deliver("audio", RecordedAudio([0.1], 16000))
            self.window.poll_events()
            self.wait_for_reply()
            self.assertTrue(self.window._background_session_active)
            self.assertTrue(self.window.voice_overlay.isVisible())
        self.assertEqual(self.wake.command_count, 2)

    def test_background_session_stays_active_on_silence_and_ignores_cancelled_audio(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.wake.deliver("timeout", None)
        self.window.poll_events()
        self.assertTrue(self.window._background_session_active)
        self.assertTrue(self.window.voice_overlay.isVisible())
        self.assertEqual(self.wake.command_count, 2)
        self.media_control.resume.assert_not_called()
        stale_emit = self.wake.emit
        self.window.voice_overlay.cancel_requested.emit()
        stale_emit(("audio", RecordedAudio([0.1], 16000)))
        self.window.poll_events()
        self.transcriber.assert_not_called()
        self.assertFalse(self.window._background_session_active)
        self.media_control.resume.assert_called_once()

    def test_background_session_waits_for_playback_before_next_capture_and_can_end_while_speaking(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        playing, released = Event(), Event()
        def play(_):
            playing.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release voice playback.")
        self.player.side_effect = play
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.window.poll_events()
            self.assertTrue(self.window.busy)
            self.assertFalse(self.wake.is_running)
            self.assertEqual(self.wake.command_count, 1)
            self.assertTrue(self.window.voice_overlay.can_cancel)
            self.window.voice_overlay.cancel_requested.emit()
            self.assertFalse(self.window.voice_overlay.isVisible())
            self.assertFalse(self.window._background_session_active)
            self.media_control.resume.assert_not_called()
        finally:
            released.set()
        self.wait_for_reply()
        self.assertEqual(self.wake.command_count, 1)
        self.assertEqual(self.wake.start_count, 2)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertEqual(self.window.assistant.history, [])
        self.assertNotIn("Request failed", self.window.status.text())
        self.media_control.resume.assert_called_once()

    def test_background_cancel_during_transcription_prevents_model_and_late_ui_updates(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        started, released = Event(), Event()
        def transcribe(*args, **kwargs):
            started.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release transcription.")
            return "Open Spotify."
        self.transcriber.side_effect = transcribe
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        try:
            self.assertTrue(started.wait(2))
            self.window.voice_overlay.cancel_requested.emit()
            self.window.events.put(("status", "Speaking..."))
        finally:
            released.set()
        self.wait_for_reply()
        self.backend.respond.assert_not_called()
        self.assertEqual(self.window.assistant.history, [])
        self.assertNotIn("Open Spotify", self.window.voice_conversation.toPlainText())
        self.assertFalse(self.window._background_session_active)
        self.assertEqual(self.wake.command_count, 1)
        self.media_control.resume.assert_called_once()

    def test_background_silence_from_transcription_rearms_without_a_new_wake_word(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.transcriber.return_value = ""
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertTrue(self.window._background_session_active)
        self.assertTrue(self.window.voice_overlay.isVisible())
        self.assertEqual(self.wake.command_count, 2)
        self.assertEqual(self.wake.start_count, 1)
        self.backend.respond.assert_not_called()
        self.media_control.resume.assert_not_called()

    def test_background_error_ends_session_and_restores_media(self):
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.backend.respond.side_effect = RuntimeError("Model unavailable")
        self.wake.deliver("audio", RecordedAudio([0.1], 16000))
        self.window.poll_events()
        self.wait_for_reply()
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertEqual(self.wake.command_count, 1)
        self.assertIn("Model unavailable", self.window.status.text())
        self.media_control.resume.assert_called_once()

    def test_keyboard_session_starts_without_wake_word_or_microphone_or_window_activation(self):
        self.setup_manual_shortcuts()
        self.assertFalse(self.window.wake_enabled.isChecked())
        with patch.object(self.window, "show_main_window") as show:
            self.shortcut("toggle_session")
            self.assertTrue(self.window._background_session_active)
            self.assertTrue(self.window._session_manual)
            self.assertTrue(self.window.voice_overlay.isVisible())
            self.assertEqual(self.window.voice_overlay.state, "Ready to talk")
            self.assertFalse(self.window.isVisible())
            show.assert_not_called()
        self.recorder.start.assert_not_called()
        self.backend.respond.assert_not_called()
        self.player.assert_called_once()
        self.speech.synthesize.assert_called_once_with("I'm ready.", language="English")
        self.media_control.pause.assert_called_once()
        self.assertEqual(self.window.assistant.history, [])
        self.assertEqual(self.wake.command_count, 0)
        self.shortcut("toggle_session")
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertFalse(self.window._background_session_active)

    def test_keyboard_start_reuses_readiness_cue_without_creating_conversation_history(self):
        self.setup_manual_shortcuts()
        for _ in range(2):
            self.shortcut("toggle_session")
            self.shortcut("toggle_session")
        self.speech.synthesize.assert_called_once_with("I'm ready.", language="English")
        self.assertEqual(self.player.call_count, 2)
        self.assertEqual(self.window.assistant.history, [])
        self.backend.respond.assert_not_called()
        self.recorder.start.assert_not_called()

    def test_keyboard_start_cue_finishes_before_a_held_talk_shortcut_records(self):
        self.setup_manual_shortcuts()
        playing, released = Event(), Event()
        def play(_):
            playing.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release the readiness cue.")
        self.player.side_effect = play
        self.shortcuts.deliver("toggle_session")
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.shortcut("talk_pressed")
            self.assertTrue(self.window.busy)
            self.recorder.start.assert_not_called()
            self.assertEqual(self.wake.command_count, 0)
            self.assertTrue(self.media_session.active)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertTrue(self.recorder.is_recording)
        self.assertTrue(self.window._background_session_active)
        self.window.cancel_background_listening()

    def test_cancelling_keyboard_start_during_synthesis_prevents_late_audio_or_capture(self):
        self.setup_manual_shortcuts()
        generating, released = Event(), Event()
        audio = self.speech.synthesize.return_value
        def synthesize(*args, **kwargs):
            generating.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release cue synthesis.")
            return audio
        self.speech.synthesize.side_effect = synthesize
        self.shortcuts.deliver("toggle_session")
        self.window.poll_events()
        try:
            self.assertTrue(generating.wait(2))
            self.shortcut("talk_pressed")
            self.shortcut("toggle_session")
        finally:
            released.set()
        self.wait_for_reply()
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.player.assert_not_called()
        self.recorder.start.assert_not_called()
        self.media_control.pause.assert_not_called()

    def test_cancelling_keyboard_start_during_playback_restores_media_without_capture(self):
        self.setup_manual_shortcuts()
        playing, released = Event(), Event()
        self.player.side_effect = lambda _: (playing.set(), released.wait(3))
        self.shortcuts.deliver("toggle_session")
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.shortcut("toggle_session")
            self.assertFalse(self.window.voice_overlay.isVisible())
            self.media_control.resume.assert_not_called()
        finally:
            released.set()
        self.wait_for_reply()
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window._background_starting)
        self.recorder.start.assert_not_called()
        self.media_control.resume.assert_called_once()

    def test_keyboard_start_cue_failure_ends_session_and_restores_media(self):
        self.setup_manual_shortcuts()
        self.player.side_effect = RuntimeError("Playback unavailable")
        self.shortcut("toggle_session")
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.assertIn("Playback unavailable", self.window.status.text())
        self.recorder.start.assert_not_called()
        self.media_control.resume.assert_called_once()

    def test_keyboard_hold_sends_on_release_and_keeps_microphone_closed_between_turns(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        for question in ("My name is Alex.", "What is my name?"):
            self.transcriber.return_value = question
            transcriptions = self.transcriber.call_count
            self.shortcut("talk_pressed")
            self.assertTrue(self.recorder.is_recording)
            self.assertTrue(self.window._manual_recording)
            self.assertEqual(self.transcriber.call_count, transcriptions)
            self.shortcut("talk_released")
            self.wait_for_reply()
            self.assertFalse(self.recorder.is_recording)
            self.assertTrue(self.window._background_session_active)
            self.assertEqual(self.window.voice_overlay.state, "Ready to talk")
            self.media_control.resume.assert_not_called()
        self.assertEqual(self.transcriber.call_count, 2)
        self.assertEqual(self.player.call_count, 3)  # One readiness cue, two replies.
        self.assertEqual(len(self.window.assistant.history), 4)
        self.assertEqual(self.wake.command_count, 0)
        self.assertEqual(self.wake.start_count, 0)
        self.media_control.pause.assert_called_once()
        self.shortcut("toggle_session")
        self.media_control.resume.assert_called_once()

    def test_manual_wake_activation_waits_for_shortcut_instead_of_automatic_capture(self):
        self.setup_manual_shortcuts()
        self.window.wake_enabled.setChecked(True)
        self.wake.deliver("activated", None)
        self.window.poll_events()
        self.wait_for_reply()
        self.assertTrue(self.window._background_session_active)
        self.assertEqual(self.window.voice_overlay.state, "Ready to talk")
        self.assertEqual(self.wake.command_count, 0)
        self.assertFalse(self.wake.is_running)
        self.recorder.start.assert_not_called()
        self.shortcut("talk_pressed")
        self.shortcut("talk_released")
        self.wait_for_reply()
        self.assertEqual(self.wake.command_count, 0)
        self.assertFalse(self.wake.is_running)
        self.assertEqual(self.player.call_count, 2)  # One cue, one reply.

    def test_hold_waits_for_standby_stream_to_close_and_release_can_cancel_pending_capture(self):
        self.setup_manual_shortcuts()
        self.window.wake_enabled.setChecked(True)
        self.wake.stop = Mock()
        closed = Event()
        self.wake.wait_stopped = Mock(side_effect=lambda: closed.wait(3))
        self.shortcuts.deliver("toggle_session")
        self.window.poll_events()
        try:
            self.shortcut("talk_pressed")
            self.recorder.start.assert_not_called()
            self.player.assert_not_called()
            self.shortcut("talk_released")
            self.wake.is_running = False
        finally:
            closed.set()
        self.wait_for_reply()
        self.recorder.start.assert_not_called()
        self.assertTrue(self.window._background_session_active)
        self.backend.respond.assert_not_called()

    def test_hold_during_wake_acknowledgement_waits_for_cue_without_cancelling_session(self):
        self.setup_manual_shortcuts()
        self.window.wake_enabled.setChecked(True)
        playing, released = Event(), Event()
        self.player.side_effect = lambda _: (playing.set(), released.wait(3))
        self.wake.deliver("activated", None)
        self.window.poll_events()
        try:
            self.assertTrue(playing.wait(2))
            self.shortcut("talk_pressed")
            self.recorder.start.assert_not_called()
        finally:
            released.set()
        self.wait_for_reply()
        self.assertTrue(self.recorder.is_recording)
        self.assertTrue(self.window._background_session_active)
        self.assertEqual(self.wake.command_count, 0)
        self.window.cancel_background_listening()

    def test_keyboard_hold_recording_survives_focus_loss_and_hiding_main_window(self):
        from PySide6.QtCore import QEvent
        self.setup_manual_shortcuts()
        self.window._background_service = True
        self.shortcut("toggle_session")
        self.shortcut("talk_pressed")
        with patch.object(self.window, "isActiveWindow", return_value=False):
            self.window.changeEvent(QEvent(QEvent.Type.ActivationChange))
        self.assertTrue(self.recorder.is_recording)
        self.window.close()
        self.assertTrue(self.recorder.is_recording)
        self.shortcut("talk_released")
        self.wait_for_reply()
        self.assertTrue(self.window._background_session_active)
        self.assertFalse(self.window._closing)
        self.window._background_service = False

    def test_ending_keyboard_session_during_hold_discards_audio(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        self.shortcut("talk_pressed")
        self.shortcut("toggle_session")
        self.shortcut("talk_released")
        self.recorder.cancel.assert_called_once()
        self.assertFalse(self.recorder.is_recording)
        self.recorder.stop.assert_not_called()
        self.transcriber.assert_not_called()
        self.backend.respond.assert_not_called()
        self.media_control.resume.assert_called_once()

    def test_hold_interrupts_a_spoken_reply_before_starting_the_next_capture(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        self.shortcut("talk_pressed")
        playing, released = Event(), Event()
        def play(_):
            playing.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release playback.")
        self.player.side_effect = play
        self.shortcut("talk_released")
        try:
            self.assertTrue(playing.wait(2))
            self.shortcut("talk_pressed")
            self.assertTrue(self.window._request_cancelled.is_set())
            self.assertTrue(self.window.busy)
            self.assertFalse(self.recorder.is_recording)
            self.assertTrue(self.window._background_session_active)
            self.assertEqual(self.recorder.start.call_count, 1)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertTrue(self.recorder.is_recording)
        self.assertEqual(self.recorder.start.call_count, 2)
        self.assertFalse(self.window._voice_cancelled.is_set())
        self.assertEqual(self.window.assistant.history, [])
        self.media_control.resume.assert_not_called()
        self.player.side_effect = None
        self.shortcut("talk_released")
        self.wait_for_reply()
        self.assertEqual(len(self.window.assistant.history), 2)
        self.assertEqual(self.window.voice_overlay.state, "Ready to talk")

    def test_keyboard_recording_limit_sends_once_and_late_release_does_nothing(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        self.shortcut("talk_pressed")
        self.recorder.limit_reached = True
        self.window.poll_events()
        self.recorder.limit_reached = False
        self.wait_for_reply()
        self.shortcut("talk_released")
        self.recorder.stop.assert_called_once()
        self.transcriber.assert_called_once()
        self.assertEqual(self.window.voice_overlay.state, "Ready to talk")

    def test_shortcut_settings_persist_and_registration_conflicts_are_visible(self):
        self.window.background_input.setCurrentIndex(self.window.background_input.findData("hold"))
        self.assertEqual(self.settings.value("background_input"), "hold")
        self.window.shortcuts_enabled.setChecked(False)
        self.assertFalse(self.settings.value("shortcuts_enabled"))
        self.shortcuts.start = Mock(side_effect=RuntimeError("Shortcut conflict"))
        self.window.shortcuts_enabled.setChecked(True)
        self.assertTrue(self.settings.value("shortcuts_enabled"))
        self.assertIn("Shortcut conflict", self.window.shortcuts_status.text())

    def test_disabling_shortcuts_discards_manual_capture_and_ignores_queued_keys(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        self.shortcut("talk_pressed")
        self.window.shortcuts_enabled.setChecked(False)
        self.window.shortcut_events.put("talk_released")
        self.window.shortcut_events.put("toggle_session")
        self.window.poll_events()
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.recorder.is_recording)
        self.backend.respond.assert_not_called()

    def test_keyboard_microphone_failure_ends_session_and_restores_media(self):
        self.setup_manual_shortcuts()
        self.shortcut("toggle_session")
        self.recorder.start.side_effect = RuntimeError("Unavailable microphone")
        self.shortcut("talk_pressed")
        self.assertFalse(self.window._background_session_active)
        self.assertFalse(self.window.voice_overlay.isVisible())
        self.media_control.resume.assert_called_once()

    def test_closing_background_window_keeps_listener_and_explicit_quit_releases_it(self):
        self.window._background_service = True
        self.window.wake_enabled.setChecked(True)
        self.window.show()
        self.assertFalse(self.window.close())
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.window._closing)
        self.assertFalse(self.window._shutdown_complete)
        self.assertTrue(self.wake.is_running)
        self.window.show_main_window()
        self.assertTrue(self.window.isVisible())
        self.window.request_quit()
        self.assertTrue(self.window._shutdown_complete)
        self.assertFalse(self.wake.is_running)

    def test_quit_waits_for_reply_without_reopening_hidden_window(self):
        self.window._background_service = True
        released, entered = Event(), Event()
        def reply(*_args, **_kwargs):
            entered.set()
            if not released.wait(3):
                raise TimeoutError("Test did not release the reply.")
            return "Hello."
        self.backend.respond.side_effect = reply
        self.window.submit(text="Hello")
        try:
            self.assertTrue(entered.wait(2))
            self.window.request_quit()
            self.assertTrue(self.window._closing)
            self.assertFalse(self.window._shutdown_complete)
        finally:
            released.set()
        self.wait_for_reply()
        self.assertTrue(self.window._shutdown_complete)

    def test_single_instance_launch_requests_existing_window(self):
        import uuid
        from ui.single_instance import SingleInstance
        name = "airis-test-" + uuid.uuid4().hex
        first, second = SingleInstance(name=name), SingleInstance(name=name)
        try:
            self.assertTrue(first.acquire())
            first.open_requested.connect(self.window.show_main_window)
            self.assertFalse(second.acquire())
            first.poll_requests()
            self.assertTrue(self.window.isVisible())
        finally:
            first.close()
            second.close()

    def test_background_menu_controls_explicit_open_wake_and_quit(self):
        from ui.background import BackgroundControls
        self.window._background_service = True
        with patch("ui.background.QSystemTrayIcon.isSystemTrayAvailable", return_value=False):
            controls = BackgroundControls(self.window, self.application)
        try:
            controls.wake_action.setChecked(True)
            self.assertTrue(self.window.wake_enabled.isChecked())
            controls.open_action.trigger()
            self.assertTrue(self.window.isVisible())
            with patch.object(self.application, "quit") as quit_app:
                controls.quit_action.trigger()
                quit_app.assert_called_once()
                self.assertTrue(controls.finished)
            self.assertTrue(self.window._shutdown_complete)
        finally:
            self.application.removeEventFilter(controls)
            controls.tray.hide()
            controls.deleteLater()


if __name__ == "__main__":
    unittest.main()
