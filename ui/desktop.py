"""Desktop chat with push-to-talk input and optional spoken replies."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import gc
import math
import sys
from time import perf_counter
from threading import Event
from html import escape
from urllib.parse import urlsplit

from PySide6.QtCore import QEvent, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QDesktopServices, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QLabel, QPlainTextEdit,
    QDoubleSpinBox, QFileDialog, QMainWindow, QProgressBar, QPushButton, QSpinBox, QTextBrowser,
)

from assistant import Assistant
from input.microphone import MicrophoneRecorder
from actions.media_control import SpeechMediaSession
from input.transcription import prepare_transcription, transcribe_audio
from input.live_transcription import LiveTranscriptionSession
from input.wakeword_listener import WakeWordListener
from input.wakeword_check import check_wake_phrase
from input.wakeword_profile import load_wake_profile
from input.wakeword_verification import WakePhraseVerifier, VERIFIED_CANDIDATE_THRESHOLD
from output.playback import play_audio
from output.voices import DEFAULT_VOICE, VOICE_PROFILES
from processing.mlx_model import FAST_MODEL_ID, MODEL_ID, load_mlx_backend
from .workflow import ConversationWorkflow, VoiceTurnCancelled
from .wake_workflow import WakeAcknowledgement, ACKNOWLEDGEMENT, READY_ACKNOWLEDGEMENT
from .qt_runtime import prepare_qt_plugins
from .layout import PAGE_IDS, build_layout
from .voice_overlay import VoiceOverlay
from .keyboard_shortcuts import GlobalVoiceShortcuts, TALK_SHORTCUT


class LazyResponseBackend:
    """Prepare and reuse the selected quantized model on the inference thread."""

    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self._backend = None
        self._warmed = False

    def prepare(self, model_id=None):
        if model_id is not None and model_id != self.model_id:
            self._backend = None
            self._warmed = False
            self.model_id = model_id
            gc.collect()
            mlx = sys.modules.get("mlx.core")
            if mlx is not None:
                mlx.clear_cache()
        if self._backend is None:
            self._backend = load_mlx_backend(model_id=self.model_id)
        if not self._warmed:
            self._backend.warm_up()
            self._warmed = True

    @property
    def last_finish_reason(self):
        return getattr(self._backend, "last_finish_reason", None)

    def respond(self, messages, *, max_new_tokens):
        self.prepare()
        return self._backend.respond(messages, max_new_tokens=max_new_tokens)

    def stream(self, messages, *, max_new_tokens):
        self.prepare()
        yield from self._backend.stream(messages, max_new_tokens=max_new_tokens)

    def reset_cache(self):
        if self._backend is not None:
            self._backend.reset_cache()


class TalkButton(QPushButton):
    def mouseReleaseEvent(self, event):
        # Qt normally cancels a click released outside the button. Recording
        # must also stop when the user releases there.
        was_down = self.isDown()
        outside = not self.rect().contains(event.position().toPoint())
        super().mouseReleaseEvent(event)
        if was_down and outside:
            self.released.emit()


class MessageEditor(QPlainTextEdit):
    """Send with Enter; keep Shift + Enter available for multiline messages."""

    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AirisWindow(QMainWindow):
    shutdown_finished = Signal()

    def __init__(self, *, assistant=None, recorder=None, transcriber=transcribe_audio,
                 player=play_audio, settings=None, auto_prepare=None, prepare_callback=None,
                 live_transcriber_factory=None, wake_listener=None, background_service=False, media_session=None,
                 wake_verifier=None, shortcut_listener=None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.settings = settings if settings is not None else QSettings("Airis", "Desktop")
        self._owns_assistant = assistant is None
        selected_model = self.settings.value("model_id", MODEL_ID)
        if selected_model not in (MODEL_ID, FAST_MODEL_ID):
            selected_model = MODEL_ID
        self._response_backend = LazyResponseBackend(selected_model) if self._owns_assistant else None
        self.assistant = assistant if assistant is not None else Assistant(backend=self._response_backend)
        self._prepare_callback = prepare_callback
        self._auto_prepare = self._owns_assistant if auto_prepare is None else auto_prepare
        self.recorder = recorder if recorder is not None else MicrophoneRecorder()
        self.transcriber = transcriber
        self._live_factory = live_transcriber_factory
        if self._live_factory is None and transcriber is transcribe_audio and callable(getattr(type(self.recorder), "snapshot", None)):
            self._live_factory = LiveTranscriptionSession
        self._live_session = None
        self._live_future = None
        self._live_frames = 0
        self._live_next_update = 0.0
        self._live_failed = False
        self.player = player
        self.media_session = media_session if media_session is not None else SpeechMediaSession()
        self._media_release_pending = False
        self.workflow = ConversationWorkflow(self.assistant, transcriber=transcriber, player=player,
                                              media_session=self.media_session)
        self.wake_acknowledgement = WakeAcknowledgement(self.assistant, player)
        self.background_acknowledgement = WakeAcknowledgement(self.assistant, player, text=READY_ACKNOWLEDGEMENT)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airis-inference")
        self.events = Queue()
        self.wake_events = Queue()
        self.wake_listener = wake_listener if wake_listener is not None else WakeWordListener()
        self.wake_verifier = (WakePhraseVerifier(transcriber=self.transcriber)
                              if wake_verifier is None and isinstance(self.wake_listener, WakeWordListener)
                              else wake_verifier)
        if self.wake_verifier is not None:
            self.wake_listener.candidate_threshold = VERIFIED_CANDIDATE_THRESHOLD
        self._wake_rejected = 0
        self._wake_profile = load_wake_profile(self.wake_listener.model_path) if hasattr(self.wake_listener, "model_path") else None
        self._wake_generation = 0
        self._wake_fault = False
        self._wake_state = "Off"
        self._wake_command_active = False
        self._wake_activations = self._wake_missed = self._wake_false = 0
        self._wake_score = 0.0
        self._wake_peak_score = 0.0
        self._wake_confirmed_peak = 0.0
        self._wake_rms = 0.0
        self._wake_frames = 0
        self._wake_microphone = "Waiting for audio"
        self._wake_active_threshold = None
        self._wake_check_cancelled = Event()
        self._pending_recording = False
        self.busy = False
        self._closing = False
        self._background_service = background_service
        self._quitting = False
        self._shutdown_complete = False
        self._background_voice_turn = False
        self._background_session_active = False
        self._background_starting = False
        self._voice_cancelled = Event()
        self._request_cancelled = None
        self._session_manual = False
        self._talk_shortcut_held = False
        self._manual_recording = False
        self.shortcuts = shortcut_listener if shortcut_listener is not None else GlobalVoiceShortcuts()
        self.shortcut_events = Queue()
        self.voice_overlay = VoiceOverlay()
        self.voice_overlay.cancel_requested.connect(self.cancel_background_listening)
        self.voice_overlay.open_requested.connect(self.show_main_window)
        self.voice_overlay.quit_requested.connect(self.request_quit)
        self._reply_position = None
        self._conversation_mode = "speech" if self.settings.value("mode", "Text") == "Speech" else "text"
        selected_page = self.settings.value("page", "voice" if self._conversation_mode == "speech" else "text")
        if selected_page not in PAGE_IDS:
            selected_page = "text"
        if selected_page in ("text", "voice"):
            self._conversation_mode = "speech" if selected_page == "voice" else "text"
        self.setWindowTitle("Airis")
        self.resize(1000, 780)
        self.setMinimumSize(820, 660)

        self.conversation = QTextBrowser()
        self.conversation.setOpenExternalLinks(False)
        self.conversation.setOpenLinks(False)
        self.conversation.anchorClicked.connect(self.open_source_link)
        self.conversation.setPlaceholderText("Your conversation starts here.")
        self.conversation.setAccessibleName("Conversation")
        self.voice_conversation = QTextBrowser()
        self.voice_conversation.setOpenExternalLinks(False)
        self.voice_conversation.setOpenLinks(False)
        self.voice_conversation.anchorClicked.connect(self.open_source_link)
        self.voice_conversation.setAccessibleName("Voice conversation transcript")
        self.message = MessageEditor()
        self.message.setPlaceholderText("Write a message, or try 'open Spotify'...")
        self.message.setAccessibleName("Message")
        self.message.setFixedHeight(100)
        self.send = QPushButton("Send")
        self.talk = TalkButton("Hold\nto talk")
        self.talk.setToolTip("Ask a question, or say 'open Outlook and make it full screen'.")
        self.voice = QComboBox()
        for profile in VOICE_PROFILES:
            self.voice.addItem(profile.title(), profile)
        selected_voice = self.settings.value("voice", DEFAULT_VOICE)
        if selected_voice not in VOICE_PROFILES:
            selected_voice = DEFAULT_VOICE
        self.voice.setCurrentIndex(self.voice.findData(selected_voice))
        self.voice.setToolTip("Reference voice for spoken replies. Text replies stay silent.")
        self.assistant.set_voice(selected_voice)
        self.language = QComboBox()
        self.language.addItems(["Auto", "English", "German"])
        self.language.setCurrentText(self.settings.value("language", "Auto"))
        self.model = QComboBox()
        self.model.addItem("Qwen3.5 9B · 4-bit", MODEL_ID)
        self.model.addItem("Qwen3.5 4B · 4-bit", FAST_MODEL_ID)
        self.model.setCurrentIndex(self.model.findData(selected_model))
        self.model.setEnabled(self._owns_assistant)
        self.model.setAccessibleName("Response model")
        self.voice.setAccessibleName("Voice profile")
        self.language.setAccessibleName("Speech language")
        self.reply_limit = QSpinBox()
        self.reply_limit.setRange(256, 8192)
        self.reply_limit.setSingleStep(256)
        self.reply_limit.setSuffix(" tokens")
        self.reply_limit.setValue(int(self.settings.value("reply_limit", 2048)))
        self.reply_limit.setAccessibleName("Maximum reply tokens")
        self.reply_limit.setToolTip("Maximum reply length. Higher values allow longer answers; complete replies still stop early.")
        self.prepare_button = QPushButton("Prepare / Retry")
        self.web_enabled = QCheckBox("Enable web search and page reading")
        self.web_enabled.setChecked(str(self.settings.value("web_enabled", True)).lower() in ("true", "1"))
        self.assistant.set_web_enabled(self.web_enabled.isChecked())
        self.web_enabled.toggled.connect(self.web_access_changed)
        self.email_enabled = QCheckBox("Enable read-only email access")
        self.email_enabled.setChecked(str(self.settings.value("email_enabled", True)).lower() in ("true", "1"))
        self.assistant.set_email_enabled(self.email_enabled.isChecked())
        self.email_enabled.toggled.connect(self.email_access_changed)
        self.email_connect = QPushButton("Connect Gmail")
        self.email_disconnect = QPushButton("Disconnect Gmail")
        self.email_connect.setEnabled(self.assistant.email_connection_supported is True)
        self.email_disconnect.setEnabled(self.assistant.email_connection_supported is True)
        self.email_connect.clicked.connect(self.connect_gmail)
        self.email_disconnect.clicked.connect(self.disconnect_gmail)
        self.email_connection_status = QLabel("Gmail sign-in uses read-only access and macOS Keychain. Connect once, then ask to read emails.")
        self.email_connection_status.setWordWrap(True)
        self.email_connection_status.setTextFormat(Qt.TextFormat.PlainText)
        self.prepare_button.setEnabled(self._owns_assistant or prepare_callback is not None)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.clear = QPushButton("New conversation")
        self.status = QLabel("Ready. Choose Text chat or Voice chat to begin.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.media_notice = QLabel("")
        self.media_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.media_notice.setWordWrap(True)
        self.media_notice.setVisible(False)
        self.transcription_preview = QLabel("")
        self.transcription_preview.setTextFormat(Qt.TextFormat.PlainText)
        self.transcription_preview.setWordWrap(True)
        self.wake_enabled = QCheckBox("Enable wake word (experimental)")
        self.wake_enabled.setChecked(str(self.settings.value("wake_enabled", False)).lower() in ("true", "1"))
        self.wake_enabled.setAccessibleName("Enable wake word")
        self.wake_microphone_hint = QLabel(
            "Wake listening uses a built-in or wired microphone to preserve Bluetooth audio quality. "
            "With Bluetooth headphones, say the wake phrase toward the MacBook."
        )
        self.wake_microphone_hint.setTextFormat(Qt.TextFormat.PlainText)
        self.wake_microphone_hint.setWordWrap(True)
        self.wake_test = QCheckBox("Show wake-word test controls")
        self.wake_custom = QCheckBox("Use a custom detection threshold")
        self.wake_custom.setChecked(str(self.settings.value("wake_custom_threshold", False)).lower() in ("true", "1"))
        self.wake_threshold = QDoubleSpinBox()
        self.wake_threshold.setRange(0.01, 1.0)
        self.wake_threshold.setDecimals(6)
        self.wake_threshold.setSingleStep(0.01)
        self.wake_threshold.setValue(float(self.settings.value("wake_threshold", 0.9)))
        self.wake_threshold.setAccessibleName("Wake detection threshold")
        self.wake_threshold.setToolTip(
            "With speech verification, this controls candidate sensitivity and is capped at 0.85. "
            "A candidate still needs spoken wake-phrase confirmation before activation."
        )
        self.wake_listener.threshold = self.wake_threshold.value() if self.wake_custom.isChecked() else None
        self.wake_restore = QPushButton("Use calibrated threshold")
        self.wake_reset = QPushButton("Reset peak reading")
        self.wake_check = QPushButton("Check wake phrase (5 seconds)")
        self.wake_profile_button = QPushButton("Use evaluated threshold")
        self.wake_profile_hint = QLabel("")
        self.wake_profile_hint.setWordWrap(True)
        if self._wake_profile is not None:
            self.wake_profile_button.setText(f"Use evaluated headset threshold ({self._wake_profile['threshold']:.2f})")
            self.wake_profile_hint.setText(
                f"Evaluated with {self._wake_profile['positive_recordings']} wake recordings and "
                f"{self._wake_profile['negative_audio_seconds']:.0f} seconds of ordinary speech. "
                "These recordings used the headset, so this preset does not calibrate the standby Mac microphone."
            )
        self.wake_check_result = QLabel("")
        self.wake_check_result.setTextFormat(Qt.TextFormat.PlainText)
        self.wake_check_result.setWordWrap(True)
        self.wake_level = QProgressBar()
        self.wake_level.setRange(0, 70)
        self.wake_level.setValue(0)
        self.wake_level.setTextVisible(False)
        self.wake_level.setFixedHeight(7)
        self.wake_level.setAccessibleName("Microphone input level")
        self.wake_status = QLabel("Wake word: Off")
        self.wake_status.setTextFormat(Qt.TextFormat.PlainText)
        self.wake_status.setWordWrap(True)
        self.wake_diagnostics = QLabel("")
        self.wake_diagnostics.setTextFormat(Qt.TextFormat.PlainText)
        self.wake_diagnostics.setWordWrap(True)
        self.wake_missed = QPushButton("Missed wake word")
        self.wake_false = QPushButton("False activation")
        self.background_input = QComboBox()
        self.background_input.addItem("Hold shortcut to talk", "hold")
        self.background_input.addItem("Automatic listening", "auto")
        self.background_input.setCurrentIndex(1 if self.settings.value("background_input", "hold") == "auto" else 0)
        self.shortcuts_enabled = QCheckBox("Enable global voice shortcuts")
        self.shortcuts_enabled.setChecked(str(self.settings.value("shortcuts_enabled", True)).lower() in ("true", "1"))
        self.shortcuts_status = QLabel("Shortcuts will be registered when Airis starts.")
        self.shortcuts_status.setTextFormat(Qt.TextFormat.PlainText)
        self.shortcuts_status.setWordWrap(True)
        self.shortcuts_retry = QPushButton("Retry shortcuts")
        build_layout(self, selected_page)
        self.update_voice_summary()
        self.update_wake_diagnostics()

        self.send.clicked.connect(self.send_text)
        self.message.submitted.connect(self.send_text)
        self.talk.pressed.connect(self.start_recording)
        self.talk.released.connect(self.stop_recording)
        self.clear.clicked.connect(self.new_conversation)
        self.prepare_button.clicked.connect(lambda: self.start_preparation())
        self.model.currentIndexChanged.connect(self.model_changed)
        self.navigation.currentRowChanged.connect(self.page_changed)
        self.voice.currentIndexChanged.connect(self.voice_changed)
        self.language.currentTextChanged.connect(self.language_changed)
        self.reply_limit.valueChanged.connect(lambda value: self.settings.setValue("reply_limit", value))
        self.wake_enabled.toggled.connect(self.wake_changed)
        self.wake_test.toggled.connect(self.update_wake_diagnostics)
        self.wake_custom.toggled.connect(self.wake_threshold_changed)
        self.wake_threshold.valueChanged.connect(self.wake_threshold_changed)
        self.wake_restore.clicked.connect(lambda: self.wake_custom.setChecked(False))
        self.wake_reset.clicked.connect(self.reset_wake_readings)
        self.wake_check.clicked.connect(self.start_wake_check)
        self.wake_profile_button.clicked.connect(self.apply_wake_profile)
        self.wake_missed.clicked.connect(lambda: self.mark_wake_result("missed"))
        self.wake_false.clicked.connect(lambda: self.mark_wake_result("false"))
        self.background_input.currentIndexChanged.connect(
            lambda: self.settings.setValue("background_input", self.background_input.currentData()))
        self.shortcuts_enabled.toggled.connect(self.shortcuts_changed)
        self.shortcuts_retry.clicked.connect(self.start_shortcuts)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_events)
        self.timer.start(50)
        if self._auto_prepare:
            self.set_busy(True)
            self.status.setText("Preparing the response model...")
            QTimer.singleShot(0, lambda: self.start_preparation(initial=True))

    def wake_changed(self, enabled):
        self.settings.setValue("wake_enabled", enabled)
        self._wake_fault = False
        self.clear_wake_readings()
        recording = (self._wake_command_active and not self.busy) or self._wake_state == "Recording"
        self._wake_command_active = False
        if not enabled:
            self.finish_background_voice_turn()
        self.pause_wake()
        self.set_wake_state("Paused" if enabled else "Off")
        if recording:
            self.talk.setText("Hold\nto talk")
            self.set_busy(False)
            self.status.setText("Wake recording cancelled.")
        if enabled and self._owns_assistant and not self.busy and not self.recorder.is_recording:
            self.start_preparation()
        else:
            self.sync_wake_listener()

    def clear_wake_readings(self):
        self._wake_score = self._wake_peak_score = self._wake_confirmed_peak = self._wake_rms = 0.0
        self._wake_frames = 0
        self.update_wake_diagnostics()

    def reset_wake_readings(self):
        if self.busy or self.recorder.is_recording or self._wake_command_active or self._wake_state == "Recording":
            return
        self.clear_wake_readings()
        self.pause_wake()
        self.sync_wake_listener()

    def start_wake_check(self):
        if self.busy or self.recorder.is_recording or self._pending_recording or self._wake_command_active or self._wake_state == "Recording":
            return
        self._wake_check_cancelled.clear()
        self.wake_check_result.setText("Preparing microphone check...")
        self.set_busy(True)
        language = None if self.language.currentText() == "Auto" else self.language.currentText()
        self._future = self.executor.submit(self.after_wake_stops, self.run_wake_check,
                                            self.wake_listener.threshold, self.recorder.device, language)

    def run_wake_check(self, threshold, device, language):
        try:
            result = check_wake_phrase(threshold=threshold, device=device, language=language,
                                       cancelled=self._wake_check_cancelled,
                                       transcriber=self.transcriber, emit=self.events.put,
                                       standby=isinstance(self.wake_listener, WakeWordListener),
                                       candidate_threshold=getattr(self.wake_listener, "candidate_threshold", None),
                                       verifier=self.wake_verifier)
            if result is not None:
                self.events.put(("wake_check_result", result))
        except Exception as error:
            self.events.put(("error", f"Wake phrase check failed: {error}"))
        finally:
            self.events.put(("done", None))

    def wake_threshold_changed(self, *_):
        custom = self.wake_custom.isChecked()
        self.settings.setValue("wake_custom_threshold", custom)
        self.settings.setValue("wake_threshold", self.wake_threshold.value())
        self.wake_listener.threshold = self.wake_threshold.value() if custom else None
        self.update_wake_controls()
        self._wake_fault = False
        self.reset_wake_readings()

    def apply_wake_profile(self):
        if self._wake_profile is None or self.busy or self.recorder.is_recording or self._wake_command_active or self._wake_state == "Recording":
            return
        self.wake_threshold.blockSignals(True)
        self.wake_custom.blockSignals(True)
        self.wake_threshold.setValue(self._wake_profile["threshold"])
        self.wake_custom.setChecked(True)
        self.wake_threshold.blockSignals(False)
        self.wake_custom.blockSignals(False)
        self.wake_threshold_changed()
        self.status.setText("Evaluated headset threshold applied. Test both wake phrases and ordinary conversation.")

    def update_wake_controls(self):
        idle = not (self.busy or self.recorder.is_recording or self._pending_recording
                    or self._wake_command_active or self._background_session_active
                    or self._wake_state == "Recording" or self._closing)
        self.wake_custom.setEnabled(idle)
        self.wake_threshold.setEnabled(idle and self.wake_custom.isChecked())
        self.wake_restore.setEnabled(idle and self.wake_custom.isChecked())
        self.wake_reset.setEnabled(idle and self.wake_enabled.isChecked())
        self.wake_check.setEnabled(idle)
        self.wake_profile_button.setEnabled(idle and self._wake_profile is not None)
        self.background_input.setEnabled(idle)

    def start_shortcuts(self):
        if not self.shortcuts_enabled.isChecked() or self._closing:
            return
        try:
            self.shortcuts.start(self.shortcut_events.put)
            self.shortcuts_status.setText("Global voice shortcuts are active.")
        except (OSError, RuntimeError) as error:
            self.shortcuts_status.setText(f"Shortcuts unavailable: {error}")

    def shortcuts_changed(self, enabled):
        self.settings.setValue("shortcuts_enabled", enabled)
        if enabled:
            self.start_shortcuts()
        else:
            if self._session_manual:
                self.cancel_background_listening()
            self.shortcuts.stop()
            self.shortcuts_status.setText("Global voice shortcuts are off.")

    def poll_shortcuts(self):
        while True:
            try:
                action = self.shortcut_events.get_nowait()
            except Empty:
                return
            if self._closing or not self.shortcuts_enabled.isChecked():
                continue
            if action == "toggle_session":
                if self._background_voice_turn:
                    self.cancel_background_listening()
                else:
                    self.start_background_session()
            elif action == "talk_pressed":
                self.begin_background_hold()
            elif action == "talk_released":
                self.release_background_hold()

    def start_background_session(self):
        """Arm manual voice input without opening a microphone or the main window."""
        if self._closing or self.busy or self.recorder.is_recording or self._pending_recording:
            self.status.setText("Airis is busy. Try the session shortcut when this request finishes.")
            return
        self.pause_wake()
        self._background_voice_turn = self._background_session_active = self._session_manual = True
        self._voice_cancelled = Event()
        self._wake_command_active = False
        self._wake_fault = False
        self._media_release_pending = False
        self._background_starting = True
        self.set_busy(True)
        self.set_wake_state("Acknowledging")
        self.status.setText("Preparing the voice session...")
        self._future = self.executor.submit(self.after_wake_stops, self.run_background_start,
                                            self._wake_generation, self._voice_cancelled,
                                            self.model.currentData(), self.wake_enabled.isChecked())

    def run_background_start(self, generation, cancelled, model_id, include_wake):
        """Prepare and play one cached readiness cue before manual capture can start."""
        error = None
        try:
            if cancelled.is_set() or generation != self._wake_generation or self._closing:
                return
            if self._owns_assistant:
                self._prepare_runtime(model_id, True, include_wake)
            audio = self.background_acknowledgement.prepare()
            if cancelled.is_set() or generation != self._wake_generation or self._closing:
                return
            self.media_session.begin()
            if not cancelled.is_set() and generation == self._wake_generation and not self._closing:
                self.events.put(("status", "Speaking readiness cue..."))
                self.player(audio)
        except Exception as exception:
            error = str(exception)
        finally:
            self.events.put(("background_ready", (generation, error)))

    def complete_background_start(self, generation, error):
        self._background_starting = False
        self.set_busy(False)
        if generation != self._wake_generation or self._closing or not self._background_session_active:
            self.finish_background_voice_turn()
            if not self._closing:
                self.status.setText("Voice session ended.")
            return
        if error is not None:
            self.finish_background_voice_turn()
            self.status.setText(f"Voice session startup failed: {error}. Try starting the session again.")
            return
        self.ready_for_background_hold()

    def ready_for_background_hold(self):
        self.set_wake_state("Ready to talk")
        self.talk.setText("Hold\nto talk")
        self.status.setText(f"Voice session ready. Hold {TALK_SHORTCUT} to speak; release to send.")
        for widget in (self.send, self.navigation, self.voice, self.language, self.clear,
                       self.message, self.model, self.reply_limit, self.prepare_button, self.web_enabled, self.email_enabled, self.email_connect, self.email_disconnect):
            widget.setEnabled(False)
        if self._talk_shortcut_held and not self.wake_listener.is_running:
            self._begin_recording(background=True)

    def begin_background_hold(self):
        if not self._background_session_active or self._closing or self._talk_shortcut_held:
            return
        self._talk_shortcut_held = True
        self._session_manual = True
        if self.busy and (self._wake_state == "Acknowledging" or self._background_starting):
            self.status.setText("Wait for the acknowledgement. Keep the talk shortcut held to record next.")
            return
        self._wake_command_active = False
        self.pause_wake()
        if self.busy:
            if self._request_cancelled is not None:
                self._request_cancelled.set()
            self.status.setText("Stopping the current reply. Keep the talk shortcut held to record when it stops.")
        else:
            self.sync_wake_listener()

    def release_background_hold(self):
        self._talk_shortcut_held = False
        if self._manual_recording:
            self.stop_recording()

    def pause_wake(self):
        """Invalidate queued wake events before cancelling capture."""
        self._wake_generation += 1
        self.wake_listener.stop()

    def set_wake_state(self, state):
        self._wake_state = state
        label = "Voice session" if self._background_session_active else "Wake word"
        self.wake_status.setText(f"● {label}: {state}")
        color = "#7bdac7" if state in ("Listening", "Awaiting question", "Ready to talk") else "#e8bd74" if state in ("Acknowledging", "Recording", "Speaking") else "#9ba9bc"
        self.wake_status.setStyleSheet(f"color: {color};")
        self.update_voice_overlay()

    def update_voice_overlay(self):
        active = self._background_voice_turn and not self._closing
        foreground = self.isVisible() and not self.isMinimized() and self.isActiveWindow()
        if active and not foreground:
            self.voice_overlay.set_state(self._wake_state,
                                         can_cancel=True)
            self.voice_overlay.present()
        else:
            self.voice_overlay.dismiss()

    def finish_background_voice_turn(self):
        """End the wake interaction and any continuous background session."""
        self._background_voice_turn = False
        self._background_session_active = False
        self._background_starting = False
        self._voice_cancelled.set()
        self._session_manual = False
        self._talk_shortcut_held = False
        if self._manual_recording:
            self._manual_recording = False
            self.recorder.cancel()
            if self._live_session is not None:
                self._live_session.cancel()
                self._live_session = None
            self.transcription_preview.clear()
        self.voice_overlay.dismiss()
        self.release_media_when_idle()

    def release_media_when_idle(self):
        """Wait for cancelled capture to close before restoring external audio."""
        if self.media_session.active:
            self._media_release_pending = True

    def finish_media_release(self):
        if (self._media_release_pending and not self.busy and not self.recorder.is_recording
                and not self._pending_recording and not self._wake_command_active and not self.wake_listener.is_running):
            self.media_session.end()
            self._media_release_pending = False

    def cancel_background_listening(self):
        """Cancel capture or a running reply without rearming the voice session."""
        if not self._background_voice_turn:
            return
        self._wake_command_active = False
        self.pause_wake()
        self.finish_background_voice_turn()
        self.talk.setText("Hold\nto talk")
        if not self.busy:
            self.set_busy(False)
            self.set_wake_state("Paused" if self.wake_enabled.isChecked() else "Off")
        self.status.setText("Voice session ended.")

    def show_main_window(self):
        if self._closing:
            return
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.voice_overlay.dismiss()

    def sync_wake_listener(self):
        self.finish_media_release()
        if self._media_release_pending:
            return
        if self._closing:
            self.set_wake_state("Off")
            return
        if self.busy or self.recorder.is_recording or self._pending_recording or self._wake_command_active:
            return
        if self._background_session_active and self._session_manual:
            self.ready_for_background_hold()
            return
        if self._wake_fault:
            return
        if not self.wake_enabled.isChecked():
            self.set_wake_state("Off")
            return
        if not self.wake_listener.is_running:
            generation = self._wake_generation
            if self._background_session_active:
                self._wake_command_active = True
                self.begin_wake_question(generation, None)
                return
            self.set_wake_state("Loading")
            try:
                self.wake_listener.start(lambda event: self.wake_events.put((generation, *event)))
            except Exception as error:
                self._wake_fault = True
                self.set_wake_state(f"Unavailable: {error}. Toggle Enable wake word to retry.")

    def update_wake_diagnostics(self, *_):
        visible = self.wake_test.isChecked()
        for widget in (self.wake_diagnostics, self.wake_missed, self.wake_false, self.wake_level,
                       self.wake_custom, self.wake_threshold, self.wake_restore, self.wake_reset,
                       self.wake_check, self.wake_check_result):
            widget.setVisible(visible)
        self.wake_profile_button.setVisible(visible and self._wake_profile is not None)
        self.wake_profile_hint.setVisible(visible and self._wake_profile is not None)
        level_db = 20 * math.log10(max(self._wake_rms, 1e-8))
        self.wake_level.setValue(max(0, min(70, round(level_db + 70))))
        threshold = f"{self._wake_active_threshold:.6f}" if self._wake_active_threshold is not None else "Waiting"
        signal = "No input signal detected" if self._wake_frames and self._wake_rms < 1e-7 else ""
        threshold_label = "Candidate threshold" if self.wake_verifier is not None else "Activation threshold"
        self.wake_diagnostics.setText(
            f"Microphone: {self._wake_microphone}\n"
            f"Audio received: {self._wake_frames * 0.08:.2f} s · Level: {level_db:.1f} dBFS {signal}\n"
            f"Recent peak: {self._wake_score:.6f} · Highest peak: {self._wake_peak_score:.6f}\n"
            f"Confirmed peak (two frames): {self._wake_confirmed_peak:.6f}\n"
            f"{threshold_label}: {threshold} · Activations: {self._wake_activations}\n"
            f"Rejected candidates: {self._wake_rejected}\n"
            f"Marked missed: {self._wake_missed} · Marked false: {self._wake_false}\n"
            "Lower thresholds detect more phrases and cause more false activations.\n"
            "Session counters only. No audio is saved."
        )
        self.update_wake_controls()

    def mark_wake_result(self, result):
        if result == "missed":
            self._wake_missed += 1
        else:
            self._wake_false += 1
        self.update_wake_diagnostics()

    def poll_wake_events(self):
        while True:
            try:
                generation, kind, value = self.wake_events.get_nowait()
            except Empty:
                break
            if generation != self._wake_generation or self._closing or not self.wake_enabled.isChecked():
                continue
            if kind == "candidate" and self.wake_verifier is not None:
                if not self.busy and not self.recorder.is_recording and not self._wake_command_active:
                    self.start_wake_verification(value)
            elif kind == "activated" and self.wake_verifier is None:
                if not self.busy and not self.recorder.is_recording and not self._wake_command_active:
                    self.start_wake_acknowledgement()
            elif kind == "state":
                self.set_wake_state(value)
                if value == "Recording":
                    self.update_wake_diagnostics()
                    self.navigate("voice")
                    self.status.setText("Listening to your question. A pause sends it automatically.")
                    self.talk.setText("Cancel\nrecording")
                    for widget in (self.send, self.navigation, self.voice, self.language, self.clear,
                                   self.message, self.model, self.reply_limit, self.prepare_button, self.web_enabled, self.email_enabled, self.email_connect, self.email_disconnect):
                        widget.setEnabled(False)
                elif value == "Awaiting question":
                    self.status.setText("I'm listening. Speak your question; a pause sends it automatically.")
            elif kind == "score":
                self._wake_score = value
                self.update_wake_diagnostics()
            elif kind == "diagnostics":
                self._wake_score = value["score"]
                self._wake_peak_score = max(self._wake_peak_score, value["peak_score"])
                self._wake_confirmed_peak = max(self._wake_confirmed_peak, value["confirmed_peak"])
                self._wake_rms = value["rms"]
                self._wake_frames = value["frames"]
                self.update_wake_diagnostics()
            elif kind == "microphone":
                self._wake_microphone = value["name"]
                self._wake_active_threshold = value["threshold"]
                if value.get("standby", True):
                    self.wake_microphone_hint.setText(
                        f"Wake microphone: {value['name']}. Speak toward this microphone. "
                        "Question recording follows your system input."
                    )
                self.update_wake_diagnostics()
            elif kind == "error":
                self._wake_command_active = False
                self.finish_background_voice_turn()
                self._wake_fault = True
                self.set_wake_state(f"Unavailable: {value}. Toggle Enable wake word to retry.")
                self.talk.setText("Hold\nto talk")
                if not self.busy:
                    self.set_busy(False)
            elif kind == "timeout":
                self._wake_command_active = False
                if self._background_session_active:
                    self.set_wake_state("Awaiting question")
                    continue
                self.finish_background_voice_turn()
                self.talk.setText("Hold\nto talk")
                self.set_busy(False)
                self.set_wake_state("Paused")
                self.status.setText("No question detected. Wake listening resumed.")
            elif kind == "audio" and self._wake_command_active and not self.busy and not self.recorder.is_recording:
                self._wake_command_active = False
                self.talk.setText("Hold\nto talk")
                self.submit(audio=value, released_at=perf_counter())
        self.sync_wake_listener()

    def start_wake_verification(self, audio):
        """Close standby capture and serialize confirmation with other MLX work."""
        self.set_busy(True)
        self.set_wake_state("Checking wake phrase")
        self.status.setText("Checking wake phrase...")
        generation = self._wake_generation
        self._future = self.executor.submit(self.after_wake_stops, self.run_wake_verification, generation, audio)

    def run_wake_verification(self, generation, audio):
        accepted, error = False, None
        try:
            if generation == self._wake_generation and not self._closing:
                accepted = self.wake_verifier.verify(audio)
        except Exception as exception:
            error = str(exception)
        finally:
            self.events.put(("wake_verified", (generation, accepted, error)))

    def complete_wake_verification(self, generation, accepted, error):
        self.set_busy(False)
        if generation != self._wake_generation or self._closing or not self.wake_enabled.isChecked():
            return
        if error is not None:
            self._wake_fault = True
            self.set_wake_state("Verification unavailable")
            self.status.setText(f"Wake verification failed: {error}. Toggle Enable wake word to retry.")
        elif accepted:
            self.start_wake_acknowledgement()
        else:
            self._wake_rejected += 1
            self.set_wake_state("Paused")
            self.status.setText("Wake phrase not confirmed. Listening resumed.")
            self.update_wake_diagnostics()

    def start_wake_acknowledgement(self):
        self._background_voice_turn = True
        self._background_session_active = not (
            self.isVisible() and not self.isMinimized() and self.isActiveWindow())
        self._session_manual = self._background_session_active and self.background_input.currentData() == "hold"
        self._voice_cancelled = Event()
        self._media_release_pending = False
        self._wake_activations += 1
        self._wake_command_active = True
        self.navigation.setCurrentRow(PAGE_IDS.index("voice"))
        self.set_busy(True)
        self.set_wake_state("Acknowledging")
        self.status.setText(ACKNOWLEDGEMENT)
        self.update_wake_diagnostics()
        generation = self._wake_generation
        self._future = self.executor.submit(self.after_wake_stops, self.run_wake_acknowledgement, generation)

    def run_wake_acknowledgement(self, generation):
        error = None
        try:
            if generation == self._wake_generation and not self._closing:
                self.media_session.begin()
                if generation == self._wake_generation and not self._closing:
                    self.wake_acknowledgement.play()
                else:
                    self.media_session.end()
        except Exception as exception:
            self.media_session.end()
            error = str(exception)
        finally:
            self.events.put(("wake_ack_ready", (generation, error)))

    def begin_wake_question(self, generation, error):
        self.set_busy(False)
        if generation != self._wake_generation or self._closing or not self.wake_enabled.isChecked():
            self._wake_command_active = False
            self.finish_background_voice_turn()
            return
        if error is not None:
            self._wake_command_active = False
            self.finish_background_voice_turn()
            self._wake_fault = True
            self.set_wake_state(f"Unavailable: {error}. Toggle Enable wake word to retry.")
            self.status.setText(f"Wake acknowledgement failed: {error}")
            return
        if self._session_manual:
            self._wake_command_active = False
            self.ready_for_background_hold()
            return
        self.set_wake_state("Awaiting question")
        self.talk.setText("Cancel\nrecording")
        self.status.setText("I'm listening. Speak your question; a pause sends it automatically.")
        for widget in (self.send, self.navigation, self.voice, self.language, self.clear,
                       self.message, self.model, self.reply_limit, self.prepare_button, self.web_enabled, self.email_enabled, self.email_connect, self.email_disconnect):
            widget.setEnabled(False)
        try:
            self.wake_listener.start_command(lambda event: self.wake_events.put((generation, *event)),
                                              wait_seconds=None if self._background_session_active else 10.0)
        except Exception as exception:
            self._wake_command_active = False
            self.finish_background_voice_turn()
            self._wake_fault = True
            self.set_busy(False)
            self.talk.setText("Hold\nto talk")
            self.set_wake_state(f"Unavailable: {exception}. Toggle Enable wake word to retry.")

    def after_wake_stops(self, callback, *args):
        try:
            self.wake_listener.wait_stopped()
        except Exception as error:
            self._wake_command_active = False
            self.events.put(("error", str(error)))
            self.events.put(("done", None))
            return
        callback(*args)

    def start_preparation(self, *, initial=False):
        if self.recorder.is_recording or self._pending_recording or self._wake_command_active or self._wake_state == "Recording":
            return
        if self.busy and not initial:
            return
        if self._closing:
            self.set_busy(False)
            self.close()
            return
        self.set_busy(True)
        self.status.setText("Loading and warming up the response model...")
        self._future = self.executor.submit(
            self.after_wake_stops, self.prepare_runtime, self.model.currentData(),
            self._conversation_mode == "speech" or self.wake_enabled.isChecked() or self._background_session_active,
            self.wake_enabled.isChecked(),
        )

    def prepare_runtime(self, model_id, include_voice, include_wake=False):
        try:
            self._prepare_runtime(model_id, include_voice, include_wake)
            self.events.put(("status", "Ready. Models are prepared for this session."))
        except Exception as error:
            self.events.put(("error", f"Model preparation failed: {error}. Use Prepare / Retry."))
        finally:
            self.events.put(("done", None))

    def _prepare_runtime(self, model_id, include_voice, include_wake=False):
        """Prepare on the inference worker; the caller owns completion events."""
        if self._response_backend is not None:
            self._response_backend.prepare(model_id)
        elif self._prepare_callback is not None:
            self._prepare_callback()
        if include_voice:
            if self.transcriber is transcribe_audio:
                self.events.put(("status", "Preparing speech recognition..."))
                prepare_transcription()
            self.events.put(("status", "Preparing the voice..."))
            self.assistant.prepare_speech()
            self.background_acknowledgement.prepare()
            if include_wake:
                if self.wake_verifier is not None:
                    self.wake_verifier.prepare(download=True)
                self.events.put(("status", "Preparing the wake acknowledgement..."))
                self.wake_acknowledgement.prepare()

    def model_changed(self):
        self.settings.setValue("model_id", self.model.currentData())
        if self._owns_assistant:
            self.start_preparation()

    @property
    def current_page(self):
        return PAGE_IDS[self.pages.currentIndex()]

    def navigate(self, page):
        """Switch areas only when no recording or model request is in progress."""
        if page not in PAGE_IDS:
            raise ValueError("Page must be 'text', 'voice', or 'settings'.")
        if not self.busy and not self.recorder.is_recording and not self._pending_recording and not self._wake_command_active and not self._closing:
            self.navigation.setCurrentRow(PAGE_IDS.index(page))

    def page_changed(self, index):
        if not 0 <= index < len(PAGE_IDS):
            return
        page = PAGE_IDS[index]
        self.pages.setCurrentIndex(index)
        self.settings.setValue("page", page)
        if page in ("text", "voice"):
            self._conversation_mode = "speech" if page == "voice" else "text"
            self.settings.setValue("mode", "Speech" if page == "voice" else "Text")
            if page == "voice" and self._owns_assistant:
                self.start_preparation()
            elif page == "text":
                self.message.setFocus()

    def language_changed(self, value):
        self.settings.setValue("language", value)
        self.update_voice_summary()

    def update_voice_summary(self):
        self.voice_summary.setText(f"{self.voice.currentText()} voice · {self.language.currentText()}")

    def voice_changed(self):
        previous_voice = self.assistant.voice
        try:
            self.assistant.set_voice(self.voice.currentData())
        except Exception as error:
            self.voice.blockSignals(True)
            self.voice.setCurrentIndex(self.voice.findData(previous_voice))
            self.voice.blockSignals(False)
            self.status.setText(f"Voice selection failed: {error}")
            return
        self.settings.setValue("voice", self.voice.currentData())
        self.update_voice_summary()
        if self._conversation_mode == "speech" or self.wake_enabled.isChecked() or self._background_session_active:
            self.set_busy(True)
            self.status.setText(f"Preparing the {self.voice.currentText().lower()} voice...")
            self._future = self.executor.submit(self.after_wake_stops, self.prepare_voice, self.wake_enabled.isChecked())

    def prepare_voice(self, include_wake=False):
        """Warm only the changed voice; keep response and recognition models ready."""
        try:
            self.assistant.prepare_speech()
            self.background_acknowledgement.prepare()
            if include_wake:
                if self.wake_verifier is not None:
                    self.wake_verifier.prepare(download=True)
                self.wake_acknowledgement.prepare()
            self.events.put(("status", "Ready. The selected voice is prepared."))
        except Exception as error:
            self.events.put(("error", f"Voice preparation failed: {error}. Use Prepare / Retry."))
        finally:
            self.events.put(("done", None))

    def web_access_changed(self, enabled):
        self.assistant.set_web_enabled(enabled)
        self.settings.setValue("web_enabled", enabled)

    def email_access_changed(self, enabled):
        self.assistant.set_email_enabled(enabled)
        self.settings.setValue("email_enabled", enabled)

    def connect_gmail(self):
        if self.busy or self.recorder.is_recording or self._background_session_active:
            return
        client_file, _ = QFileDialog.getOpenFileName(self, "Choose your Google OAuth Desktop client JSON", "", "JSON files (*.json)")
        if client_file:
            self.start_email_connection("connect", client_file)

    def disconnect_gmail(self):
        if not self.busy and not self.recorder.is_recording and not self._background_session_active:
            self.start_email_connection("disconnect")

    def start_email_connection(self, action, client_file=None):
        self.set_busy(True)
        self.email_connection_status.setText("Complete Google sign-in in your browser. Waiting up to three minutes..." if action == "connect" else "Removing Gmail credentials...")
        self._future = self.executor.submit(self.run_email_connection, action, client_file)

    def run_email_connection(self, action, client_file):
        try:
            if action == "connect":
                self.assistant.connect_email(client_file)
                message = "Gmail connected with read-only access. Ask to read your latest unread emails."
            else:
                self.assistant.disconnect_email()
                message = "Gmail disconnected. Saved credentials removed from Keychain."
            self.events.put(("email_connection", message))
        except Exception as error:
            self.events.put(("email_connection", str(error)))
        finally:
            self.events.put(("done", None))

    def open_source_link(self, url):
        # Only a user's explicit source-link click can open the browser.
        if urlsplit(url.toString()).scheme in ("http", "https"):
            QDesktopServices.openUrl(url)

    def append_sources(self, sources):
        rows = ["<b>Sources</b>"]
        for i, source in enumerate(sources, 1):
            url = source.url
            if urlsplit(url).scheme not in ("http", "https"):
                continue
            details = " · ".join(value for value in (source.publisher, source.published_at[:10]) if value)
            row = f'[{i}] <a href="{escape(url, quote=True)}">{escape(source.title)}</a>'
            if details:
                row += " — " + escape(details)
            rows.append(row)
        for view in (self.conversation, self.voice_conversation):
            cursor = view.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertHtml("<br>".join(rows))
            cursor.insertText("\n\n", QTextCharFormat())
            view.setTextCursor(cursor)
            view.ensureCursorVisible()

    def append_message(self, speaker, text):
        # Insert plain text: model output must not become executable markup.
        self.insert_conversation(f"{speaker}: {text}\n\n")

    def insert_conversation(self, text, *, replace_from=None):
        """Update both transcripts incrementally using independently owned documents."""
        for view in (self.conversation, self.voice_conversation):
            cursor = view.textCursor()
            if replace_from is None:
                cursor.movePosition(cursor.MoveOperation.End)
            else:
                cursor.setPosition(replace_from)
                cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
            cursor.insertText(text, QTextCharFormat())
            view.setTextCursor(cursor)
            view.ensureCursorVisible()
        return cursor.position()

    def set_busy(self, value):
        self.busy = value
        if value:
            self.pause_wake()
            if self.wake_enabled.isChecked() or self._background_session_active:
                self.set_wake_state("Processing")
        for widget in (self.send, self.talk, self.navigation, self.voice, self.language, self.clear, self.message):
            widget.setEnabled(not value)
        self.reply_limit.setEnabled(not value)
        self.web_enabled.setEnabled(not value)
        self.email_enabled.setEnabled(not value)
        self.email_connect.setEnabled(not value and self.assistant.email_connection_supported is True)
        self.email_disconnect.setEnabled(not value and self.assistant.email_connection_supported is True)
        self.model.setEnabled(not value and self._owns_assistant)
        self.prepare_button.setEnabled(not value and (self._owns_assistant or self._prepare_callback is not None))
        self.progress.setVisible(value)
        self.update_wake_controls()

    def send_text(self):
        text = self.message.toPlainText().strip()
        if self._closing or self.busy or self._pending_recording or self._wake_command_active or self._wake_state == "Recording" or self.recorder.is_recording or self.current_page != "text" or not text:
            return
        self.message.clear()
        self.submit(text=text)

    def start_recording(self):
        if self._background_session_active and self._session_manual:
            self.begin_background_hold()
            return
        if (self._wake_command_active or self._wake_state == "Recording") and not self.busy:
            self._wake_command_active = False
            self.finish_background_voice_turn()
            self.pause_wake()
            self.talk.setText("Hold\nto talk")
            self.set_busy(False)
            self.set_wake_state("Paused")
            self.status.setText("Wake recording cancelled.")
            return
        if self._closing or self.busy or self.recorder.is_recording or self.current_page != "voice":
            return
        self.pause_wake()
        if self.wake_listener.is_running:
            self._pending_recording = True
            self.status.setText("Switching microphone to push-to-talk...")
            for widget in (self.send, self.navigation, self.voice, self.language, self.clear,
                           self.message, self.model, self.reply_limit, self.prepare_button, self.web_enabled, self.email_enabled, self.email_connect, self.email_disconnect):
                widget.setEnabled(False)
            return
        self._begin_recording()

    def _begin_recording(self, *, background=False):
        try:
            self._media_release_pending = False
            self.media_session.begin()
            self.recorder.start()
        except Exception as error:
            self.media_session.end()
            if background:
                self.finish_background_voice_turn()
            self.status.setText(f"Microphone unavailable: {error}")
            self.set_busy(False)
            return
        self._manual_recording = background
        self.set_wake_state("Recording" if background else "Paused" if self.wake_enabled.isChecked() else "Off")
        self.send.setEnabled(False)
        self.clear.setEnabled(False)
        for widget in (self.model, self.navigation, self.voice, self.language, self.reply_limit, self.prepare_button, self.message, self.web_enabled, self.email_enabled, self.email_connect, self.email_disconnect):
            widget.setEnabled(False)
        if self._live_factory is not None:
            language = None if self.language.currentText() == "Auto" else self.language.currentText()
            self._live_session = self._live_factory(language=language)
            self._live_future = self.executor.submit(self._live_session.prepare)
            self._live_frames = 0
            self._live_next_update = perf_counter() + 1.0
            self._live_failed = False
        self.transcription_preview.clear()
        self.talk.setText("Release\nto send")
        self.status.setText("Listening... Release to send. Maximum recording: 30 seconds.")

    def stop_recording(self):
        self._talk_shortcut_held = False
        self._manual_recording = False
        if self._pending_recording:
            self._pending_recording = False
            self.set_busy(False)
            return
        if not self.recorder.is_recording:
            return
        released_at = perf_counter()
        session, self._live_session = self._live_session, None
        self.talk.setDown(False)
        self.talk.setText("Hold\nto talk")
        try:
            audio = self.recorder.stop()
        except Exception as error:
            if session is not None:
                session.cancel()
            self.set_busy(False)
            if self._background_session_active:
                self.finish_background_voice_turn()
            self.release_media_when_idle()
            self.status.setText(f"Recording failed: {error}")
            return
        self.transcription_preview.clear()
        self.submit(audio=audio, transcription_session=session, released_at=released_at)

    def submit(self, *, text=None, audio=None, transcription_session=None, released_at=None, wake_word=False):
        # Input determines output, regardless of persisted settings or navigation.
        mode = "speech" if audio is not None else "text"
        language = None if self.language.currentText() == "Auto" else self.language.currentText()
        self.set_busy(True)
        self.status.setText("Transcribing..." if audio is not None else "Thinking...")
        self._request_cancelled = self._voice_cancelled if self._background_voice_turn else None
        self._future = self.executor.submit(self.after_wake_stops, self.run_request, text, audio, mode, language,
                                            self.reply_limit.value(), transcription_session, released_at, wake_word,
                                            self._request_cancelled,
                                            self._background_session_active)

    def run_request(self, text, audio, mode, language, max_new_tokens,
                    transcription_session=None, released_at=None, wake_word=False,
                    cancelled=None, keep_media_paused=False):
        try:
            self.workflow.execute(
                text=text, audio=audio, mode=mode, language=language,
                max_new_tokens=max_new_tokens, emit=self.events.put,
                transcription_session=transcription_session, released_at=released_at,
                wake_word=wake_word,
                cancelled=cancelled, keep_media_paused=keep_media_paused,
            )
        except VoiceTurnCancelled:
            self.events.put(("status", "Voice session ended."))
        except Exception as error:
            self.events.put(("error", str(error)))
        finally:
            self.events.put(("done", None))

    def poll_events(self):
        self.poll_shortcuts()
        self.media_notice.setText(self.media_session.last_error)
        self.media_notice.setVisible(bool(self.media_session.last_error))
        if self._pending_recording and not self.wake_listener.is_running:
            self._pending_recording = False
            self.start_recording()
        self.poll_wake_events()
        if self._closing and not self.busy and (self._live_future is None or self._live_future.done()):
            self.close()
            return
        if self.recorder.limit_reached:
            self.stop_recording()
        if self.recorder.is_recording and self._live_session is not None and not self._live_failed:
            if self._live_future is None or self._live_future.done():
                if self._live_future is not None:
                    try:
                        text = self._live_future.result()
                        if text:
                            self.transcription_preview.setText(f"Hearing: {text}")
                    except Exception:
                        # Final transcription reports persistent failures; a
                        # provisional failure must not discard the recording.
                        self._live_failed = True
                if not self._live_failed and perf_counter() >= self._live_next_update:
                    frames = self.recorder.frame_count
                    if frames > self._live_frames:
                        self._live_frames = frames
                        self._live_future = self.executor.submit(self._live_session.update, self.recorder.snapshot())
                        self._live_next_update = perf_counter() + 1.2
        while True:
            try:
                kind, value = self.events.get_nowait()
            except Empty:
                break
            if (self._request_cancelled is not None and self._request_cancelled.is_set()
                    and kind in ("user", "assistant_start", "assistant_delta", "assistant", "sources", "status", "error")):
                continue
            if kind == "user":
                self.append_message("You", value)
            elif kind == "assistant_start":
                self._reply_position = self.insert_conversation("Airis: ")
            elif kind == "assistant_delta":
                self.insert_conversation(value)
                self.status.setText("Responding...")
            elif kind == "assistant":
                if self._reply_position is None:
                    self.append_message("Airis", value)
                else:
                    self.insert_conversation(value + "\n\n", replace_from=self._reply_position)
                    self._reply_position = None
            elif kind == "error":
                self.finish_background_voice_turn()
                self._request_cancelled = None
                if self._reply_position is not None:
                    self.insert_conversation("\n\n")
                    self._reply_position = None
                self.status.setText(f"Request failed: {value}")
            elif kind == "sources":
                self.append_sources(value)
            elif kind == "email_connection":
                self.email_connection_status.setText(value)
            elif kind == "wake_check_result":
                verification_note = (f"Classifier candidate: {'Yes' if value.get('classifier_activated') else 'No'} · "
                                     f"Spoken phrase verified: {'Yes' if value['activated'] else 'No'}\n"
                                     if value.get("verification_enabled") else "")
                microphone_note = f"Wake microphone: {value['microphone']}\n" if value.get("microphone") else ""
                self.wake_check_result.setText(
                    microphone_note + verification_note +
                    f"Whisper heard: {value['transcript'] or '(no speech recognized)'}\n"
                    f"Peak: {value['peak_score']:.6f} · Confirmed peak: {value['confirmed_peak']:.6f}\n"
                    f"Threshold: {value['threshold']:.6f} · Activated: {'Yes' if value['activated'] else 'No'}\n"
                    f"Microphone RMS: {value['rms']:.6f} · Audio peak: {value['audio_peak']:.6f}\n"
                    "Audio discarded. No question sent and no training performed."
                )
                self.status.setText("Wake phrase check complete. Results are shown in Settings.")
            elif kind == "wake_ack_ready":
                self.begin_wake_question(*value)
            elif kind == "background_ready":
                self.complete_background_start(*value)
            elif kind == "wake_verified":
                self.complete_wake_verification(*value)
            elif kind == "status":
                self.status.setText(value)
                if self.busy and (self.wake_enabled.isChecked() or self._background_session_active):
                    self.set_wake_state("Speaking" if "speaking" in value.lower() else "Processing")
            elif kind == "done":
                cancelled = self._request_cancelled is not None and self._request_cancelled.is_set()
                self._request_cancelled = None
                if self._background_session_active and self._session_manual and self._voice_cancelled.is_set():
                    self._voice_cancelled = Event()
                if not self._background_session_active:
                    self.finish_background_voice_turn()
                if self._wake_command_active and self._wake_state == "Acknowledging":
                    self._wake_command_active = False
                self.set_busy(False)
                if not self._background_session_active:
                    self.release_media_when_idle()
                if self._reply_position is not None:
                    self.insert_conversation("\n\n")
                    self._reply_position = None
                if cancelled:
                    self.status.setText("Reply interrupted." if self._background_session_active else "Voice session ended.")
                if self.current_page == "text":
                    self.message.setFocus()
                elif self.current_page == "voice":
                    self.talk.setFocus()
                if self._closing:
                    self.close()
        self.sync_wake_listener()
        self.update_wake_controls()
        self.update_voice_overlay()

    def new_conversation(self):
        if not self.busy and not self.recorder.is_recording and not self._pending_recording and not self._wake_command_active and self._wake_state != "Recording":
            self.assistant.reset_history()
            self.conversation.clear()
            self.voice_conversation.clear()
            self.status.setText("New conversation. Ready.")

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            if getattr(self, "_pending_recording", False):
                self._pending_recording = False
                self.set_busy(False)
            if hasattr(self, "recorder") and self.recorder.is_recording and not self._manual_recording:
                self.recorder.cancel()
                self.release_media_when_idle()
                if self._live_session is not None:
                    self._live_session.cancel()
                    self._live_session = None
                self.transcription_preview.clear()
                self.talk.setDown(False)
                self.talk.setText("Hold\nto talk")
                self.set_busy(False)
                self.status.setText("Recording cancelled when the window lost focus.")
        super().changeEvent(event)
        if hasattr(self, "voice_overlay"):
            self.update_voice_overlay()

    def request_quit(self):
        if self._shutdown_complete:
            return
        self._quitting = True
        self.close()

    def closeEvent(self, event):
        if self._shutdown_complete:
            event.accept()
            return
        if self._background_service and not self._quitting:
            if (self.recorder.is_recording or self._pending_recording) and not self._manual_recording:
                self.recorder.cancel()
                self.release_media_when_idle()
                self._pending_recording = False
                if self._live_session is not None:
                    self._live_session.cancel()
                    self._live_session = None
                self.transcription_preview.clear()
                self.talk.setDown(False)
                self.talk.setText("Hold\nto talk")
                self.set_busy(False)
                self.status.setText("Push-to-talk cancelled. Airis remains available in the background.")
            self.hide()
            self.update_voice_overlay()
            event.ignore()
            return
        self._closing = True
        self.shortcuts.stop()
        self.finish_background_voice_turn()
        self._wake_command_active = False
        self._wake_check_cancelled.set()
        self._pending_recording = False
        self.pause_wake()
        self.recorder.cancel()
        if self._live_session is not None:
            self._live_session.cancel()
            self._live_session = None
        if self.wake_listener.is_running:
            self.status.setText("Stopping wake listening before closing...")
            event.ignore()
            return
        if not self.busy and self._live_future is not None and not self._live_future.done():
            self._closing = True
            self.status.setText("Finishing speech recognition before closing...")
            event.ignore()
            return
        if self.busy:
            self._closing = True
            self.status.setText("Finishing the current request before closing...")
            event.ignore()
            return
        self.timer.stop()
        self.media_session.end()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.assistant.close()
        self.voice_overlay.close()
        self.voice_overlay.deleteLater()
        self._shutdown_complete = True
        self.shutdown_finished.emit()
        event.accept()


def main():
    prepare_qt_plugins()
    application = QApplication(sys.argv)
    application.setApplicationName("Airis")
    application.setApplicationDisplayName("Airis")
    application.setQuitOnLastWindowClosed(False)
    from .background import BackgroundControls
    from .single_instance import SingleInstance
    instance = SingleInstance(application)
    if not instance.acquire():
        return 0
    window = AirisWindow(background_service=True)
    instance.open_requested.connect(window.show_main_window)
    window.shutdown_finished.connect(instance.close)
    controls = BackgroundControls(window, application)
    window.show_main_window()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
