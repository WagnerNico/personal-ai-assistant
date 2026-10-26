"""Bounded sentence-to-audio pipeline, independent of models and playback devices."""

from queue import Empty, Full, Queue
import re
from threading import Condition, Event, Lock, Thread
from time import perf_counter

from .telemetry import logger


class SentenceBuffer:
    """Keep unfinished sentences until a boundary or a practical length limit."""

    def __init__(self, max_chars=240, *, wait_for_first_sentence=False):
        self.pending = ""
        self.max_chars = max_chars
        self.first_pending = wait_for_first_sentence

    def feed(self, text):
        self.pending += text
        chunks = []
        while self.pending:
            boundary = None
            for match in re.finditer(r'[.!?]+[\"\u201d\u2019)]*(?=\s)|\n+', self.pending):
                if self.first_pending and match.group().startswith("\n"):
                    continue
                prefix = self.pending[:match.end()]
                # Avoid splitting common abbreviations, initials, and decimals.
                if match.group().startswith(".") and re.search(
                    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|e\.g|i\.e|etc)|\b[A-Z])\.$", prefix
                ):
                    continue
                boundary = match.end()
                break
            if not self.first_pending and (boundary is None or boundary > self.max_chars) and len(self.pending) > self.max_chars:
                boundary = self.pending.rfind(" ", 0, self.max_chars + 1)
                if boundary <= 0:
                    boundary = self.max_chars
            if boundary is None:
                break
            chunk = self.pending[:boundary].strip()
            self.pending = self.pending[boundary:].lstrip()
            if chunk:
                chunks.append(chunk)
                self.first_pending = False
        return chunks

    def finish(self):
        tail, self.pending = self.pending.strip(), ""
        return [tail] if tail else []


class StreamingSpeechPipeline:
    """Generate and consume audio concurrently, with backpressure and error propagation."""

    def __init__(self, backend, on_audio, *, language=None, on_stage=None,
                 initial_buffer_seconds=1.0, prioritize_speech=True):
        if not 0 <= initial_buffer_seconds <= 2.0:
            raise ValueError("initial_buffer_seconds must be between 0 and 2.")
        self.backend = backend
        self.on_audio = on_audio
        self.language = language
        self.on_stage = on_stage
        self.buffer_seconds = initial_buffer_seconds
        self.prioritize_speech = prioritize_speech
        self.condition = Condition()
        self.pending_sentences = 0
        self.queued_seconds = 0.0
        self.started_at = perf_counter()
        self.playback_started_at = None
        self.generation_seconds = 0.0
        self.audio_seconds = 0.0
        self.chunk_count = 0
        self.max_chunk_seconds = 0.0
        self.underruns = 0
        self.text_wait_seconds = 0.0
        self.sentences = SentenceBuffer(wait_for_first_sentence=True)
        self.text_queue = Queue(maxsize=8)
        self.audio_queue = Queue(maxsize=16)
        self.stopped = Event()
        self.error = None
        self.error_lock = Lock()
        self.threads = [
            Thread(target=self._generate, name="Airis speech generation"),
            Thread(target=self._play, name="Airis speech playback"),
        ]
        for thread in self.threads:
            thread.start()

    def _fail(self, error):
        with self.error_lock:
            if self.error is None:
                self.error = error
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()

    def _check(self):
        if self.error is not None:
            raise self.error
        if self.stopped.is_set():
            raise RuntimeError("Speech streaming was cancelled.")

    def _put(self, queue, value):
        while not self.stopped.is_set():
            try:
                queue.put(value, timeout=0.05)
                return
            except Full:
                pass
        self._check()

    def _get(self, queue):
        while not self.stopped.is_set():
            try:
                return queue.get(timeout=0.05)
            except Empty:
                pass
        return None

    def _generate(self):
        try:
            while not self.stopped.is_set():
                text = self._get(self.text_queue)
                if text is None:
                    break
                if self.on_stage:
                    self.on_stage("Generating speech...")
                setup_started = perf_counter()
                # Each request already contains buffered text. Stream audio for
                # the opening sentence too, without waiting for its full waveform.
                if callable(getattr(type(self.backend), "stream", None)):
                    chunks = self.backend.stream(text, language=self.language)
                else:
                    chunks = iter([self.backend.synthesize(text, language=self.language)])
                try:
                    produced = False
                    sentence_started = perf_counter()
                    iterator = iter(chunks)
                    sentence_generation = perf_counter() - setup_started
                    sentence_duration = 0.0
                    sentence_chunks = 0
                    first_chunk_seconds = 0.0
                    while True:
                        chunk_started = perf_counter()
                        audio = next(iterator, None)
                        elapsed = perf_counter() - chunk_started
                        sentence_generation += elapsed
                        if audio is None:
                            break
                        duration = audio.duration_seconds
                        self.chunk_count += 1
                        self.audio_seconds += duration
                        sentence_duration += duration
                        sentence_chunks += 1
                        if sentence_chunks == 1:
                            first_chunk_seconds = elapsed
                        if sentence_chunks > 1:
                            self.max_chunk_seconds = max(self.max_chunk_seconds, elapsed)
                        with self.condition:
                            self.queued_seconds += duration
                            self.condition.notify_all()
                        self._put(self.audio_queue, audio)
                        produced = True
                    if not produced:
                        raise RuntimeError("The speech backend returned no audio.")
                    self.generation_seconds += sentence_generation
                    logger.info(
                        "Sentence: chunks=%d audio_s=%.2f generation_s=%.2f first_chunk_s=%.2f elapsed_s=%.2f",
                        sentence_chunks, sentence_duration, sentence_generation, first_chunk_seconds,
                        perf_counter() - sentence_started,
                    )
                finally:
                    close = getattr(chunks, "close", None)
                    if close:
                        close()
                    with self.condition:
                        self.pending_sentences -= 1
                        self.condition.notify_all()
            if not self.stopped.is_set():
                self._put(self.audio_queue, None)
        except Exception as error:
            self._fail(error)

    def _play(self):
        try:
            buffering = True
            pending = []
            duration = 0.0
            while not self.stopped.is_set():
                audio = self._get(self.audio_queue)
                if audio is None:
                    # A short reply must play even when it never reaches the
                    # startup threshold. The end marker flushes the remainder.
                    for chunk in pending:
                        self._deliver(chunk)
                    break
                pending.append(audio)
                duration += audio.duration_seconds
                if buffering and duration < self.buffer_seconds:
                    continue
                buffering = False
                for index, chunk in enumerate(pending):
                    if self._deliver(chunk):
                        # Preserve any buffered remainder when recovering from
                        # an underrun, instead of starting every tiny chunk alone.
                        pending = pending[index + 1:]
                        duration = sum(item.duration_seconds for item in pending)
                        buffering = True
                        break
                else:
                    pending = []
                    duration = 0.0
        except Exception as error:
            self._fail(error)

    def _deliver(self, audio):
        self._check()
        if self.playback_started_at is None:
            self.playback_started_at = perf_counter()
            logger.info("Playback started: startup_s=%.2f buffer_target_s=%.2f",
                        self.playback_started_at - self.started_at, self.buffer_seconds)
        if self.on_stage:
            self.on_stage("Speaking...")
        underflowed = self.on_audio(audio) is True
        with self.condition:
            self.queued_seconds = max(0.0, self.queued_seconds - audio.duration_seconds)
            self.condition.notify_all()
        if underflowed:
            self.underruns += 1
            self.buffer_seconds = min(2.0, self.buffer_seconds + 0.5)
            logger.info("Playback underrun: count=%d new_buffer_target_s=%.2f", self.underruns, self.buffer_seconds)
        return underflowed

    def wait_for_text_generation(self):
        """Give speech GPU time while its queued audio reserve is low.

        Called from the text callback between LLM text chunks.
        Playback needs no model computation, so text generation can resume once
        the speech queue has a reserve or the current synthesis has finished.
        """
        if not self.prioritize_speech:
            return
        started = perf_counter()
        with self.condition:
            while (self.pending_sentences and self.queued_seconds < self.buffer_seconds + 0.5
                   and not self.stopped.is_set()):
                self.condition.wait(timeout=0.05)
        self.text_wait_seconds += perf_counter() - started
        self._check()

    def feed(self, text):
        self._check()
        for sentence in self.sentences.feed(text):
            with self.condition:
                self.pending_sentences += 1
            self._put(self.text_queue, sentence)

    def finish(self):
        for sentence in self.sentences.finish():
            with self.condition:
                self.pending_sentences += 1
            self._put(self.text_queue, sentence)
        self._put(self.text_queue, None)
        for thread in self.threads:
            thread.join()
        self._check()
        logger.info("Reply: audio_s=%.2f generation_s=%.2f max_chunk_s=%.2f underruns=%d text_wait_s=%.2f",
                    self.audio_seconds, self.generation_seconds, self.max_chunk_seconds,
                    self.underruns, self.text_wait_seconds)

    def abort(self):
        """Discard queued work and wait for any active backend/playback call."""
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()
