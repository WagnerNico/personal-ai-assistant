"""Optional local audio playback, independent of speech generation."""

import io
import wave

from audio_runtime import audio_session
from .audio import SpeechAudio


class AudioStreamPlayer:
    """Follow the system output for each reply and keep it open across chunks."""

    def __init__(self, *, device=None):
        self.device = device
        self.stream = None
        self.audio_format = None
        self._session = None

    def __enter__(self):
        return self

    def write(self, audio: SpeechAudio):
        """Write PCM samples and return PortAudio's underrun signal."""
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError(
                "Local playback requires sounddevice. Install "
                "output/requirements-playback.txt in the calling Python environment."
            ) from error

        with wave.open(io.BytesIO(audio.wav_bytes), "rb") as recording:
            audio_format = (recording.getframerate(), recording.getnchannels(), recording.getsampwidth())
            if self.stream is None:
                dtype = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}.get(audio_format[2])
                if dtype is None:
                    raise ValueError("Unsupported PCM sample width.")
                session = audio_session(sd, follow_system_default=self.device is None)
                session.__enter__()
                self._session = session
                try:
                    self.stream = sd.RawOutputStream(
                        samplerate=audio_format[0], channels=audio_format[1], dtype=dtype, device=self.device,
                    )
                    self.stream.start()
                except BaseException:
                    try:
                        if self.stream is not None:
                            self.stream.close()
                    finally:
                        self.stream = None
                        self._session.__exit__(None, None, None)
                        self._session = None
                    raise
                self.audio_format = audio_format
            elif self.audio_format != audio_format:
                raise ValueError("Streaming audio format changed during playback.")
            return self.stream.write(recording.readframes(recording.getnframes()))

    def __exit__(self, error_type, error, traceback):
        try:
            if self.stream is not None:
                try:
                    if error_type is None:
                        self.stream.stop()
                    else:
                        self.stream.abort()
                finally:
                    self.stream.close()
        finally:
            self.stream = None
            if self._session is not None:
                self._session.__exit__(error_type, error, traceback)
                self._session = None


def play_audio(audio: SpeechAudio, *, device: int | str | None = None) -> None:
    """Play WAV bytes from memory using the current system output by default."""
    with AudioStreamPlayer(device=device) as player:
        player.write(audio)
