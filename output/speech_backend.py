"""Lazy persistent speech worker using the shared project environment."""

from __future__ import annotations

import atexit
import base64
import binascii
from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
import platform
from threading import Lock
from typing import Protocol

from .audio import SpeechAudio
from .voices import DEFAULT_VOICE, validate_voice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


class SpeechBackend(Protocol):
    """Replaceable speech engine; playback belongs to the integrating app."""

    def synthesize(self, text: str, *, language: str | None = None) -> SpeechAudio:
        ...


class SubprocessSpeechBackend:
    """Start TTS on the first speech request and reuse its model and voice.

    Apple Silicon defaults to streaming MLX in the shared .venv. The optional
    PyTorch adapter remains available for custom runtimes with engine="qwen". A custom
    command must implement the same JSON-lines protocol. No shell is used.
    """

    def __init__(self, *, command: Sequence[str] | None = None, engine: str | None = None,
                 voice: str = DEFAULT_VOICE):
        self._voice = validate_voice(voice)
        if engine is None:
            use_mlx = command is None and platform.system() == "Darwin" and platform.machine() == "arm64"
            engine = "mlx" if use_mlx else "qwen"
        if engine not in ("mlx", "qwen"):
            raise ValueError("Speech engine must be 'mlx' or 'qwen'.")
        self._engine = engine
        self._command = list(command) if command is not None else None
        self._process: subprocess.Popen | None = None
        self._lock = Lock()
        self._closed = False

    def set_voice(self, voice: str) -> None:
        """Select the next request's reference without loading or restarting TTS."""
        validate_voice(voice)
        with self._lock:
            if self._closed:
                raise RuntimeError("The speech backend is closed.")
            self._voice = voice

    def _start(self) -> subprocess.Popen:
        if self._closed:
            raise RuntimeError("The speech backend is closed.")
        if self._process is None or self._process.poll() is not None:
            command = self._command
            if command is None:
                python = RUNTIME_PYTHON
                module = "output.mlx_speech_worker" if self._engine == "mlx" else "output.speech_worker"
                if not python.is_file():
                    raise RuntimeError(f"The TTS Python environment was not found at {python}.")
                command = [str(python), "-u", "-m", module]
            self._process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Model logs remain visible; only JSON uses stdout.
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            atexit.unregister(self.close)
            atexit.register(self.close)
        return self._process

    def synthesize(self, text: str, *, language: str | None = None) -> SpeechAudio:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech text must be a nonempty string.")
        with self._lock:
            process = self._start()
            try:
                process.stdin.write(json.dumps({
                    "action": "synthesize", "text": text, "language": language, "voice": self._voice,
                }) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError("The TTS worker connection failed.") from error
            if not line:
                raise RuntimeError("The TTS worker exited before returning audio. Check its terminal logs.")
            try:
                response = json.loads(line)
                if not response["ok"]:
                    raise RuntimeError(f"Speech generation failed: {response.get('error', 'Unknown error.')}")
                wav_bytes = base64.b64decode(response["wav_base64"], validate=True)
                sample_rate = response["sample_rate"]
                if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
                    raise ValueError("Invalid sample rate.")
                if not wav_bytes.startswith(b"RIFF") or wav_bytes[8:12] != b"WAVE":
                    raise ValueError("Invalid WAV data.")
            except (ValueError, KeyError, TypeError, binascii.Error) as error:
                raise RuntimeError("The TTS worker returned an invalid audio response.") from error
            return SpeechAudio(wav_bytes=wav_bytes, sample_rate=sample_rate)

    def stream(self, text: str, *, language: str | None = None):
        """Yield incremental audio; interrupted streams discard their worker protocol."""
        if self._engine != "mlx":
            yield self.synthesize(text, language=language)
            return
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech text must be a nonempty string.")
        with self._lock:
            process = self._start()
            complete = False
            try:
                process.stdin.write(json.dumps({
                    "action": "stream", "text": text, "language": language, "voice": self._voice,
                }) + "\n")
                process.stdin.flush()
                while True:
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError("The TTS worker exited while streaming. Check logs/desktop.log.")
                    response = json.loads(line)
                    if not response.get("ok"):
                        complete = True
                        raise RuntimeError(f"Speech generation failed: {response.get('error', 'Unknown error.')}")
                    if response.get("event") == "done":
                        complete = True
                        break
                    if response.get("event") != "audio":
                        raise ValueError("Invalid speech stream event.")
                    wav_bytes = base64.b64decode(response["wav_base64"], validate=True)
                    rate = response["sample_rate"]
                    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
                        raise ValueError("Invalid sample rate.")
                    if not wav_bytes.startswith(b"RIFF") or wav_bytes[8:12] != b"WAVE":
                        raise ValueError("Invalid WAV data.")
                    yield SpeechAudio(wav_bytes, rate)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError("The TTS worker returned an invalid speech stream or disconnected.") from error
            finally:
                if not complete:
                    # Unread chunks cannot be mistaken for the next request's reply.
                    process.kill()
                    process.wait()
                    process.stdin.close()
                    process.stdout.close()
                    self._process = None

    def prepare(self) -> None:
        """Prepare the TTS model and selected voice without playing or saving audio."""
        with self._lock:
            process = self._start()
            try:
                process.stdin.write(json.dumps({"action": "prepare", "voice": self._voice}) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                response = json.loads(line)
            except (OSError, ValueError) as error:
                raise RuntimeError("The TTS worker could not prepare the voice.") from error
            if not isinstance(response, dict) or not response.get("ok"):
                raise RuntimeError(f"Voice preparation failed: {response.get('error', 'Invalid worker response.') if isinstance(response, dict) else 'Invalid worker response.'}")

    def close(self) -> None:
        """Release the worker and its model memory; safe to call repeatedly."""
        with self._lock:
            self._closed = True
            process = self._process
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
                    self._process = None
            atexit.unregister(self.close)
