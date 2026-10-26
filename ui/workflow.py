"""Interface-independent routing of typed or transcribed messages."""

from assistant import Assistant
from input.transcription import transcribe_audio
from output.playback import play_audio
from output.playback import AudioStreamPlayer
from contextlib import nullcontext
from time import perf_counter
import re
from output.telemetry import logger
from actions.spotify import SPOTIFY_BUNDLE


class VoiceTurnCancelled(Exception):
    """Cooperatively stop a voice turn at an inference or playback boundary."""


class ConversationWorkflow:
    """Use one session for all input methods, with replaceable adapters."""

    def __init__(self, assistant: Assistant, *, transcriber=transcribe_audio, player=play_audio, media_session=None):
        self.assistant = assistant
        self.transcriber = transcriber
        self.player = player
        self.media_session = media_session

    def execute(self, *, text=None, audio=None, mode="text", language=None, max_new_tokens=2048,
                emit=lambda event: None, transcription_session=None, released_at=None, wake_word=False,
                cancelled=None, keep_media_paused=False):
        """Run one turn; a session owner may retain media focus and cancel it.

        With keep_media_paused=True, the owner must call media_session.end()
        after capture and processing stop, including on cancellation or failure.
        """
        if (text is None) == (audio is None):
            raise ValueError("Provide either text or recorded audio.")
        if mode not in ("text", "speech"):
            raise ValueError("mode must be 'text' or 'speech'.")
        if cancelled is not None and cancelled.is_set():
            raise VoiceTurnCancelled()
        interrupt = self.media_session if mode == "speech" else None
        try:
            if interrupt is not None:
                interrupt.begin()
            return self._execute(text=text, audio=audio, mode=mode, language=language, max_new_tokens=max_new_tokens,
                                 emit=emit, transcription_session=transcription_session, released_at=released_at,
                                 wake_word=wake_word, cancelled=cancelled)
        finally:
            if interrupt is not None and not keep_media_paused:
                interrupt.end()

    def _execute(self, *, text=None, audio=None, mode="text", language=None, max_new_tokens=2048,
                 emit=lambda event: None, transcription_session=None, released_at=None, wake_word=False,
                 cancelled=None):
        def check_cancelled():
            if cancelled is not None and cancelled.is_set():
                raise VoiceTurnCancelled()

        original_emit = emit
        def emit(event):
            check_cancelled()
            original_emit(event)

        check_cancelled()
        if (text is None) == (audio is None):
            raise ValueError("Provide either text or recorded audio.")
        if mode not in ("text", "speech"):
            raise ValueError("mode must be 'text' or 'speech'.")
        started = released_at if released_at is not None else perf_counter()
        transcription_seconds = 0.0
        first_text_at = None
        first_audio_at = None
        if audio is not None:
            emit(("status", "Transcribing..."))
            transcription_started = perf_counter()
            text = transcription_session.finish(audio) if transcription_session is not None else self.transcriber(audio, language=language)
            check_cancelled()
            transcription_seconds = perf_counter() - transcription_started
            logger.info("Transcription after release: elapsed_s=%.2f preview_reused=%s",
                        transcription_seconds, bool(transcription_session and transcription_session.final_reused))
            if wake_word:
                # Remove only an explicit leading wake phrase, never a name
                # occurring later in the actual question.
                text = re.sub(r"^\s*(?:hey[\s,.!?:;—-]+)?airis\b[\s,.!?:;—-]*", "", text,
                              count=1, flags=re.IGNORECASE).strip()
            if not text:
                emit(("status", "No speech detected. Please try again."))
                return None
        emit(("user", text))
        emit(("status", "Thinking..."))
        emit(("assistant_start", None))
        # Native playback holds one device open; injected players receive each
        # chunk so the workflow remains independent of a particular audio API.
        playback = AudioStreamPlayer() if mode == "speech" and self.player is play_audio else None
        def receive_text(chunk):
            check_cancelled()
            nonlocal first_text_at
            if first_text_at is None:
                first_text_at = perf_counter()
            emit(("assistant_delta", chunk))

        def record_first_audio():
            nonlocal first_audio_at
            if first_audio_at is None:
                first_audio_at = perf_counter()
                logger.info("Voice turn: release_to_first_audio_s=%.2f transcription_s=%.2f first_text_s=%s",
                            first_audio_at - started, transcription_seconds,
                            f"{first_text_at - started:.2f}" if first_text_at is not None else "unavailable")

        def receive_audio(chunk):
            check_cancelled()
            record_first_audio()
            result = playback.write(chunk) if playback is not None else self.player(chunk)
            check_cancelled()
            return result

        def receive_music_result(result):
            # The action has already happened. Update ownership even if the
            # user cancelled during the native call, so Pause stays paused.
            if self.media_session is not None:
                self.media_session.user_playback(SPOTIFY_BUNDLE, playing=result.playing)
            check_cancelled()

        playback_context = playback if playback is not None else nullcontext()
        with playback_context:
            response = self.assistant.respond(
                text, mode=mode, language=language or "Auto", max_new_tokens=max_new_tokens,
                on_text=receive_text,
                on_stage=lambda stage: emit(("status", stage)),
                on_audio=receive_audio if mode == "speech" else None,
                on_music_result=receive_music_result,
            )
            check_cancelled()
        emit(("assistant", response.text))
        if response.sources:
            emit(("sources", response.sources))
        if mode == "speech" and response.audio is not None:
            emit(("status", "Speaking..."))
            record_first_audio()
            self.player(response.audio)
        emit(("status", "Ready."))
        if response.truncated:
            emit(("status", "Reply reached the response limit. Increase Reply length or ask Airis to continue."))
        return response
