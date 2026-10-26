"""Provisional transcription during capture, with final verification on release."""

from threading import Event, Lock
from time import perf_counter

from .transcription import prepare_transcription, transcribe_audio
from output.telemetry import logger


class LiveTranscriptionSession:
    """Process complete recording snapshots serially on the app's inference worker.

    Preview text can change as more audio arrives. Reuse a preview only when
    the final recording matches its samples and adds at most a short silent tail.
    Otherwise transcribe the complete final recording to preserve context.
    """

    def __init__(self, *, language=None, transcriber=transcribe_audio, prepare=prepare_transcription):
        self.language = language
        self.transcriber = transcriber
        self.prepare_callback = prepare
        self.cancelled = Event()
        self.lock = Lock()
        self.last_audio = None
        self.last_text = ""
        self.preview_count = 0
        self.final_reused = False

    def prepare(self):
        if not self.cancelled.is_set():
            self.prepare_callback()

    def update(self, audio):
        if self.cancelled.is_set():
            return ""
        started = perf_counter()
        text = self.transcriber(audio, language=self.language)
        with self.lock:
            if self.cancelled.is_set():
                return ""
            self.last_audio = audio
            self.last_text = text
            self.preview_count += 1
        logger.info("Transcription preview: recording_s=%.2f elapsed_s=%.2f",
                    len(audio.samples) / audio.sample_rate, perf_counter() - started)
        return text

    def finish(self, audio):
        import numpy as np

        self.final_reused = False
        if self.cancelled.is_set():
            raise RuntimeError("The recording transcription was cancelled.")
        with self.lock:
            previous, text = self.last_audio, self.last_text
        if previous is not None and text and previous.sample_rate == audio.sample_rate:
            count = len(previous.samples)
            tail = np.asarray(audio.samples[count:], dtype=np.float32)
            matches = count <= len(audio.samples) and np.array_equal(audio.samples[:count], previous.samples)
            short_silence = tail.size <= audio.sample_rate * 0.5 and (
                not tail.size or float(np.max(np.abs(tail))) < 0.001
            )
            if matches and short_silence:
                self.final_reused = True
                logger.info("Transcription final: preview_reused=true previews=%d", self.preview_count)
                return text
        started = perf_counter()
        text = self.transcriber(audio, language=self.language)
        logger.info("Transcription final: preview_reused=false previews=%d elapsed_s=%.2f",
                    self.preview_count, perf_counter() - started)
        return text

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            self.last_audio = None
            self.last_text = ""
