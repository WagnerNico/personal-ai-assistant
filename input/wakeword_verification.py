"""Confirm candidate wake phrases locally before starting a conversation."""

import re
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .audio import RecordedAudio
from .speech_activity import SpeechActivity, MODEL_PATH, install_speech_activity_model
from .transcription import transcribe_audio
from output.telemetry import logger


VERIFIED_CANDIDATE_THRESHOLD = 0.85


def is_wake_phrase(text):
    """Accept the name and observed Whisper spellings only as a whole utterance."""
    punctuation = r"[\s,.!?:;—\-]*"
    # These spellings were observed in supplied wake recordings, not accepted
    # as general synonyms. The classifier and speech gate must also agree.
    name = r"(?:airis|aris|ares|eris|errors|arrows|aerith)"
    return bool(re.fullmatch(rf"{punctuation}(?:hey[\s,.!?:;—\-]+)?{name}{punctuation}",
                             text.casefold()))


@dataclass(frozen=True)
class WakeVerification:
    accepted: bool
    speech: bool
    transcript: str = ""


class WakePhraseVerifier:
    """Reject non-speech first; reuse the prepared Whisper only for candidates."""

    def __init__(self, *, transcriber=transcribe_audio, speech_activity=None):
        self.transcriber = transcriber
        self.activity = speech_activity

    def prepare(self, *, download=False):
        if self.activity is None:
            if download and not MODEL_PATH.is_file():
                try:
                    install_speech_activity_model()
                except (OSError, ValueError) as error:
                    raise RuntimeError("Could not download the speech-check model. Check your connection and use Prepare / Retry.") from error
            self.activity = SpeechActivity()

    def verify(self, audio):
        return self.inspect(audio).accepted

    def inspect(self, audio):
        if audio.sample_rate != 16000:
            raise ValueError("Wake verification requires 16 kHz audio.")
        self.prepare()
        samples = np.asarray(audio.samples, np.float32)
        started = perf_counter()
        if not self.activity.has_speech(samples):
            logger.info("Wake verification: accepted=False speech=False elapsed_s=%.3f", perf_counter() - started)
            return WakeVerification(False, False)
        # Normalize only a speech-confirmed candidate for recognition. Neither
        # the wake classifier nor its calibration receives this adjusted audio.
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if 0 < peak < 0.1:
            samples = samples * min(12.0, 0.1 / peak)
        options = {"language": "English"}
        if self.transcriber is transcribe_audio:
            options["max_new_tokens"] = 24
        text = self.transcriber(RecordedAudio(samples, 16000), **options)
        accepted = is_wake_phrase(text)
        logger.info("Wake verification: accepted=%s speech=True elapsed_s=%.3f", accepted, perf_counter() - started)
        return WakeVerification(accepted, True, text)
