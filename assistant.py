"""Orchestrate an Airis conversation with optional speech output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from output.audio import SpeechAudio
from output.speech_backend import SpeechBackend, SubprocessSpeechBackend
from output.streaming import StreamingSpeechPipeline
from output.voices import DEFAULT_VOICE, validate_voice
from actions import AppControl
from actions.web import WebAccess, WebSource
from actions.spotify import SpotifyControl
from actions.email_reading import EmailReader, EmailRequest
from actions.gmail_reading import GmailMailReader, GmailConnectionError
from processing.email_requests import email_request
from processing.agent import process_agent_text
from processing.text_processing import ResponseBackend
from processing.prompts import build_system_prompt


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    audio: SpeechAudio | None = None
    truncated: bool = False
    sources: tuple[WebSource, ...] = ()


class Assistant:
    """A session with independent history and replaceable response/speech engines.

    Explicit app requests use a bounded, replaceable local tool service.
    Set app_control=False to disable computer actions for an embedded client.
    Email reading is an independent read-only provider; Gmail sign-in is explicit.
    Construction loads no models. Text mode never starts TTS. A caller-supplied
    speech backend belongs to the caller; close() releases only our own worker.
    """

    def __init__(
        self,
        *,
        backend: ResponseBackend | None = None,
        speech_backend: SpeechBackend | None = None,
        system_prompt: str | None = None,
        speech_buffer_seconds: float = 1.0,
        speech_priority: bool = True,
        voice: str = DEFAULT_VOICE,
        app_control: AppControl | Literal[False] | None = None,
        web_access: WebAccess | Literal[False] | None = None,
        spotify_control: SpotifyControl | Literal[False] | None = None,
        email_reader: EmailReader | Literal[False] | None = None,
    ):
        self._backend = backend
        self._speech_backend = speech_backend
        self._owns_speech_backend = False
        self._system_prompt = system_prompt
        self._app_control = (None if app_control is False else
                             app_control if app_control is not None else AppControl())
        self._web_access = (None if web_access is False else
                            web_access if web_access is not None else WebAccess())
        self._web_enabled = web_access is not False
        self._spotify_control = (None if spotify_control is False or (app_control is False and spotify_control is None)
                                 else spotify_control if spotify_control is not None else SpotifyControl())
        self._email_reader = (None if email_reader is False or (app_control is False and email_reader is None)
                              else email_reader if email_reader is not None else GmailMailReader())
        self._email_enabled = self._email_reader is not None
        self._email_history_start = None
        self._history: list[dict[str, str]] = []
        self._lock = Lock()
        self._closed = False
        if not 0 <= speech_buffer_seconds <= 2.0:
            raise ValueError("speech_buffer_seconds must be between 0 and 2.")
        self._speech_buffer_seconds = speech_buffer_seconds
        self._speech_priority = speech_priority
        self._voice = DEFAULT_VOICE
        self.set_voice(voice)

    @property
    def voice(self) -> str:
        with self._lock:
            return self._voice

    def set_web_enabled(self, enabled: bool) -> None:
        """Enable or disable web retrieval without loading another model."""
        if type(enabled) is not bool:
            raise ValueError("Web access must be enabled or disabled with a boolean.")
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            self._web_enabled = enabled and self._web_access is not None

    def set_voice(self, voice: str) -> None:
        """Select a session's voice lazily, preserving history and text mode."""
        validate_voice(voice)
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            if self._speech_backend is not None and voice != self._voice:
                setter = getattr(self._speech_backend, "set_voice", None)
                if not callable(setter):
                    raise ValueError("This speech backend does not support voice selection.")
                setter(voice)
            self._voice = voice

    def set_email_enabled(self, enabled: bool) -> None:
        """Gate mailbox reads for all conversation modes without opening Outlook."""
        if type(enabled) is not bool:
            raise ValueError("Email reading must be enabled or disabled with a boolean.")
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            self._email_enabled = enabled and self._email_reader is not None

    @property
    def email_connection_supported(self) -> bool:
        return callable(getattr(type(self._email_reader), "connect", None))

    def connect_email(self, client_file) -> None:
        """Explicit UI sign-in; never initiated by a model or an email message."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            if not self.email_connection_supported:
                raise GmailConnectionError("This mailbox reader doesn't support Google sign-in.")
            self._email_reader.connect(client_file)

    def disconnect_email(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            if not callable(getattr(type(self._email_reader), "disconnect", None)):
                raise GmailConnectionError("This mailbox reader doesn't support disconnecting.")
            self._email_reader.disconnect()

    @property
    def history(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(message) for message in self._history]

    def respond(
        self,
        text: str,
        *,
        mode: Literal["text", "speech"] = "text",
        language: str | None = None,
        max_new_tokens: int = 128,
        on_text: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_audio: Callable[[SpeechAudio], None] | None = None,
        on_music_result: Callable | None = None,
    ) -> AssistantResponse:
        """Return text and optionally synthesize speech for the same reply.

        With on_audio in speech mode, generate sentences and deliver audio chunks
        concurrently with text generation. The callback consumes them in order;
        the returned audio is None to avoid retaining the complete recording.
        The opening sentence's text is buffered before synthesis; playback
        starts after a short audio reserve is ready. Text generation yields to
        speech while that reserve is low. Device underruns increase buffering
        for later turns, up to two seconds.
        Without on_audio, speech mode returns a complete WAV as before.

        Modes can change between turns. History is committed after a complete
        response; a failed request leaves it unchanged. Audio is never played
        automatically, and a microphone adapter can later feed this same method.
        """
        if mode not in ("text", "speech"):
            raise ValueError("mode must be 'text' or 'speech'.")
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            system_prompt = build_system_prompt(self._system_prompt, mode=mode)
            sources = []
            pipeline = None
            if mode == "speech" and on_audio is not None:
                if self._speech_backend is None:
                    self._speech_backend = SubprocessSpeechBackend(voice=self._voice)
                    self._owns_speech_backend = True
                pipeline = StreamingSpeechPipeline(
                    self._speech_backend, on_audio, language=language, on_stage=on_stage,
                    initial_buffer_seconds=self._speech_buffer_seconds,
                    prioritize_speech=self._speech_priority,
                )

            def receive_text(chunk):
                if on_text is not None:
                    on_text(chunk)
                if pipeline is not None:
                    pipeline.feed(chunk)
                    pipeline.wait_for_text_generation()

            try:
                response, truncated = process_agent_text(
                    text, backend=self._backend, history=self._history,
                    system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                    on_text=receive_text if pipeline is not None else on_text,
                    on_stage=on_stage, app_control=self._app_control,
                    web_access=self._web_access, web_enabled=self._web_enabled,
                    mode=mode, on_sources=sources.extend,
                    spotify_control=self._spotify_control, on_music_result=on_music_result,
                    email_reader=self._email_reader, email_enabled=self._email_enabled,
                    web_history=self._history[:self._email_history_start] if self._email_history_start is not None else None,
                )
                if pipeline is not None:
                    pipeline.finish()
                    self._speech_buffer_seconds = pipeline.buffer_seconds
            except BaseException:
                if pipeline is not None:
                    pipeline.abort()
                raise
            audio = None
            if mode == "speech" and pipeline is None:
                if on_stage is not None:
                    on_stage("Generating speech...")
                if self._speech_backend is None:
                    self._speech_backend = SubprocessSpeechBackend(voice=self._voice)
                    self._owns_speech_backend = True
                audio = self._speech_backend.synthesize(response, language=language)
            history_reply = response
            if sources:
                history_reply += "\n\nSources retrieved:\n" + "\n".join(
                    f"[{i}] {source.title} ({'page read' if source.page_read else 'search snippet only'}): "
                    f"{source.url if len(source.url) <= 300 else urlsplit(source.url).hostname + ' (link shown in transcript)'}"
                    for i, source in enumerate(sources, 1))
            if isinstance(email_request(text), EmailRequest) and self._email_history_start is None:
                # Subsequent replies may also contain private email details.
                # Exclude this part of the chat from external retrieval planning.
                self._email_history_start = len(self._history)
            self._history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": history_reply},
            ])
            return AssistantResponse(text=response, audio=audio, truncated=truncated, sources=tuple(sources))

    def prepare_speech(self) -> None:
        """Prepare the selected voice without generating or playing an answer."""
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            if self._speech_backend is None:
                self._speech_backend = SubprocessSpeechBackend(voice=self._voice)
                self._owns_speech_backend = True
            prepare = getattr(self._speech_backend, "prepare", None)
            if callable(prepare):
                prepare()

    def synthesize_speech(self, text: str, *, language: str | None = None) -> SpeechAudio:
        """Render fixed interface speech without a model reply or history entry."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Interface speech requires non-empty text.")
        with self._lock:
            if self._closed:
                raise RuntimeError("The assistant session is closed.")
            if self._speech_backend is None:
                self._speech_backend = SubprocessSpeechBackend(voice=self._voice)
                self._owns_speech_backend = True
            return self._speech_backend.synthesize(text, language=language)

    def reset_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._email_history_start = None
            if callable(getattr(type(self._backend), "reset_cache", None)):
                self._backend.reset_cache()

    def close(self) -> None:
        with self._lock:
            if self._owns_speech_backend:
                self._speech_backend.close()
            self._closed = True

    def __enter__(self) -> Assistant:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="The message to send to Airis.")
    parser.add_argument("--mode", choices=("text", "speech"), default="text")
    parser.add_argument("--model", help="An explicit MLX model ID or checkpoint directory.")
    parser.add_argument("--language", help="Language for speech synthesis.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, help="WAV path; requires --mode speech.")
    parser.add_argument("--play", action="store_true", help="Play audio from memory; requires --mode speech.")
    args = parser.parse_args()
    if args.output is not None and args.mode != "speech":
        parser.error("--output requires --mode speech.")
    if args.play and args.mode != "speech":
        parser.error("--play requires --mode speech.")

    backend = None
    if args.model is not None:
        from processing.mlx_model import load_mlx_backend

        print("Loading the response model...", flush=True)
        backend = load_mlx_backend(model_id=args.model)
    with Assistant(backend=backend) as assistant:
        print("Generating the response...", flush=True)
        response = assistant.respond(
            args.text, mode=args.mode, language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
        print(response.text, flush=True)
        if response.audio is not None:
            if args.output is not None:
                print(f"Audio saved to {response.audio.save(args.output)}", flush=True)
            if args.play:
                from output.playback import play_audio

                print("Playing the response...", flush=True)
                play_audio(response.audio)
            elif args.output is None:
                print("Audio generated in memory. Use --play to listen or --output to save it.", flush=True)


if __name__ == "__main__":
    main()
