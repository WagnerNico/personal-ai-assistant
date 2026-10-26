"""Portable audio returned by speech backends."""

from dataclasses import dataclass
from pathlib import Path
import io
import wave


@dataclass(frozen=True)
class SpeechAudio:
    """WAV bytes suitable for playback, saving, or a future HTTP response."""

    wav_bytes: bytes
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Return the PCM duration without decoding or copying the samples."""
        with wave.open(io.BytesIO(self.wav_bytes), "rb") as recording:
            return recording.getnframes() / recording.getframerate()

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.wav_bytes)
        return destination


def combine_audio(chunks) -> SpeechAudio:
    """Combine compatible PCM WAV chunks for callers requesting complete audio."""
    buffer = io.BytesIO()
    sample_rate = None
    audio_format = None
    with wave.open(buffer, "wb") as output:
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk.wav_bytes), "rb") as recording:
                current = (recording.getnchannels(), recording.getsampwidth(), recording.getframerate())
                if audio_format is None:
                    audio_format = current
                    sample_rate = current[2]
                    output.setnchannels(current[0])
                    output.setsampwidth(current[1])
                    output.setframerate(sample_rate)
                elif audio_format != current:
                    raise ValueError("Speech chunks must share the same PCM format.")
                output.writeframesraw(recording.readframes(recording.getnframes()))
        if audio_format is None:
            # wave requires parameters even when the generator produced no audio.
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            raise RuntimeError("The speech backend returned no audio.")
    return SpeechAudio(buffer.getvalue(), sample_rate)
