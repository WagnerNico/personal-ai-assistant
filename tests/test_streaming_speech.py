"""Check early playback, ordered chunks, bounded failure cleanup, and worker framing."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import sys
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from assistant import Assistant
from output.audio import SpeechAudio, combine_audio
from output.speech_backend import SubprocessSpeechBackend
from output.streaming import SentenceBuffer, StreamingSpeechPipeline
from output.playback import AudioStreamPlayer
from ui.workflow import ConversationWorkflow


def audio_chunk(frames=b"\x01\x00" * 4):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(frames)
    return SpeechAudio(buffer.getvalue(), 24000)


class StreamingSpeechTests(unittest.TestCase):
    def test_sentence_boundaries_survive_token_splits_abbreviations_and_decimals(self):
        buffer = SentenceBuffer()
        pieces = []
        for text in ("Dr.", " Smith has 3.", "14 apples. Ne", "xt sentence!\nLast fragment"):
            pieces.extend(buffer.feed(text))
        pieces.extend(buffer.finish())
        self.assertEqual(pieces, ["Dr. Smith has 3.14 apples.", "Next sentence!", "Last fragment"])
        long = SentenceBuffer(max_chars=20)
        result = long.feed("a long sentence " * 10) + long.finish()
        self.assertTrue(all(len(piece) <= 20 for piece in result))
        self.assertEqual(" ".join(result), ("a long sentence " * 10).strip())

    def test_complete_first_text_sentence_starts_audio_before_llm_and_tts_finish(self):
        first_played = Event()
        partial_text_ready = Event()
        release_text = Event()
        release_audio = Event()
        later_text_generated = Event()
        requested = []
        played = []
        events = []

        class LLM:
            def stream(self, messages, *, max_new_tokens):
                yield "First "
                partial_text_ready.set()
                if not release_text.wait(2):
                    raise TimeoutError("Test did not complete the text sentence.")
                yield "sentence. "
                if not first_played.wait(2):
                    raise TimeoutError("Audio did not start before later text.")
                yield "Second sentence. "
                later_text_generated.set()

        first, second = audio_chunk(b"\x01\x00" * 15360), audio_chunk(b"\x02\x00" * 15360)

        class TTS:
            def synthesize(self, text, *, language):
                raise AssertionError("Streaming replies must not wait for complete audio.")

            def stream(self, text, *, language):
                requested.append(("stream", text, language))
                yield first
                yield second
                if not release_audio.wait(2):
                    raise TimeoutError("TTS was not released.")

        def play(chunk):
            played.append(chunk)
            first_played.set()

        assistant = Assistant(backend=LLM(), speech_backend=TTS())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                ConversationWorkflow(assistant, player=play).execute,
                text="Hello.", mode="speech", language="English", emit=events.append,
            )
            try:
                self.assertTrue(partial_text_ready.wait(2))
                self.assertEqual(requested, [])
                self.assertFalse(first_played.is_set())
                self.assertEqual(played, [])
                release_text.set()
                self.assertTrue(first_played.wait(2))
                self.assertFalse(later_text_generated.is_set())
                self.assertFalse(future.done())
            finally:
                release_text.set()
                release_audio.set()
            reply = future.result(timeout=3)
        self.assertTrue(later_text_generated.is_set())
        self.assertEqual(requested, [
            ("stream", "First sentence.", "English"), ("stream", "Second sentence.", "English"),
        ])
        self.assertEqual(played, [first, second, first, second])
        self.assertEqual(reply.text, "First sentence. Second sentence.")
        self.assertIsNone(reply.audio)
        self.assertEqual(assistant.history[-1]["content"], reply.text)
        self.assertIn(("assistant", reply.text), events)

    def test_opening_sentence_waits_across_line_breaks_and_length_limit(self):
        buffer = SentenceBuffer(max_chars=20, wait_for_first_sentence=True)
        self.assertEqual(buffer.feed("This opening sentence\n"), [])
        self.assertEqual(buffer.feed("needs its final words"), [])
        self.assertEqual(buffer.feed(". Next sentence. "), [
            "This opening sentence\nneeds its final words.", "Next sentence.",
        ])
        self.assertEqual(buffer.finish(), [])
        unfinished = SentenceBuffer(wait_for_first_sentence=True)
        self.assertEqual(unfinished.feed("A brief reply without punctuation"), [])
        self.assertEqual(unfinished.finish(), ["A brief reply without punctuation"])

    def test_playback_failure_unblocks_bounded_queues_and_does_not_commit_history(self):
        class LLM:
            def stream(self, messages, *, max_new_tokens):
                for _ in range(100):
                    yield "A sentence. "

        class TTS:
            def stream(self, text, *, language):
                for _ in range(100):
                    yield audio_chunk(b"\x01\x00" * 7680)

        assistant = Assistant(backend=LLM(), speech_backend=TTS())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                assistant.respond, "Hello.", mode="speech",
                on_audio=Mock(side_effect=RuntimeError("Playback unavailable.")),
            )
            with self.assertRaisesRegex(RuntimeError, "Playback unavailable"):
                future.result(timeout=3)
        self.assertEqual(assistant.history, [])

    def test_startup_buffers_one_second_and_short_replies_flush(self):
        first_ready, released, last_released, played = Event(), Event(), Event(), Event()
        chunk = audio_chunk(b"\x01\x00" * 7680)

        class TTS:
            def stream(self, text, *, language):
                yield chunk
                first_ready.set()
                if not released.wait(2):
                    raise TimeoutError("Test did not release buffered audio.")
                for _ in range(3):
                    yield chunk
                if not last_released.wait(2):
                    raise TimeoutError("Test did not finish speech generation.")

        output = []
        def consume(audio):
            output.append(audio)
            played.set()

        pipeline = StreamingSpeechPipeline(TTS(), consume)
        pipeline.feed("A complete sentence. ")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.finish)
            try:
                self.assertTrue(first_ready.wait(2))
                self.assertFalse(played.is_set())
                released.set()
                self.assertTrue(played.wait(2))
                self.assertFalse(future.done())
            finally:
                released.set()
                last_released.set()
            future.result(timeout=3)
        self.assertEqual(output, [chunk] * 4)

        backend = Mock()
        backend.synthesize.return_value = chunk
        consumer = Mock()
        short = StreamingSpeechPipeline(backend, consumer)
        short.feed("Short reply. ")
        short.finish()
        consumer.assert_called_once_with(chunk)

    def test_underrun_increases_buffer_and_collects_audio_before_resuming(self):
        first_played, partial_ready, released = Event(), Event(), Event()
        first = audio_chunk(b"\x01\x00" * 26400)
        following = audio_chunk(b"\x01\x00" * 19200)
        output = []

        class TTS:
            def stream(self, text, *, language):
                yield first
                if not first_played.wait(2):
                    raise TimeoutError("Playback did not start.")
                yield following
                partial_ready.set()
                if not released.wait(2):
                    raise TimeoutError("Test did not release recovery audio.")
                yield following

        def consume(audio):
            output.append(audio)
            if len(output) == 1:
                first_played.set()
                return True
            return False

        pipeline = StreamingSpeechPipeline(TTS(), consume)
        pipeline.feed("A complete sentence. ")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pipeline.finish)
            try:
                self.assertTrue(partial_ready.wait(2))
                self.assertEqual(output, [first])
            finally:
                released.set()
            future.result(timeout=3)
        self.assertEqual(output, [first, following, following])
        self.assertEqual(pipeline.underruns, 1)
        self.assertEqual(pipeline.buffer_seconds, 1.5)

    def test_player_returns_device_underrun_feedback(self):
        device_module = Mock()
        device_module.RawOutputStream.return_value.write.return_value = True
        with patch.dict(sys.modules, {"sounddevice": device_module}):
            with AudioStreamPlayer() as player:
                self.assertIs(player.write(audio_chunk()), True)

    def test_synthesis_failure_propagates_and_next_turn_can_recover(self):
        llm = Mock()
        llm.respond.return_value = "Hello."
        speech = Mock()
        speech.synthesize.side_effect = RuntimeError("Voice failed.")
        assistant = Assistant(backend=llm, speech_backend=speech)
        with self.assertRaisesRegex(RuntimeError, "Voice failed"):
            assistant.respond("Hello.", mode="speech", on_audio=Mock())
        self.assertEqual(assistant.history, [])
        speech.synthesize.side_effect = None
        speech.synthesize.return_value = audio_chunk()
        player = Mock()
        assistant.respond("Try again.", mode="speech", on_audio=player)
        player.assert_called_once()

    def test_mlx_adapter_uses_the_same_reference_and_encodes_incremental_samples(self):
        import numpy as np
        from output.mlx_speech import stream_speech

        reference = object()
        model = Mock()
        model.generate.return_value = iter([
            SimpleNamespace(audio=np.array([0.25, -0.25]), sample_rate=24000),
            SimpleNamespace(audio=np.array([0.5]), sample_rate=24000),
        ])
        with patch("output.mlx_speech._get_voice", return_value=(model, reference, "Original transcript.")):
            chunks = list(stream_speech("Hello.", language="German"))
        arguments = model.generate.call_args.kwargs
        self.assertTrue(arguments["stream"])
        self.assertIs(arguments["ref_audio"], reference)
        self.assertEqual(arguments["ref_text"], "Original transcript.")
        self.assertEqual(arguments["lang_code"], "German")
        self.assertEqual(arguments["streaming_interval"], 0.32)
        with wave.open(io.BytesIO(combine_audio(chunks).wav_bytes), "rb") as recording:
            self.assertEqual(recording.getnframes(), 3)
            self.assertEqual(recording.readframes(3), b"\x00\x20\x00\xe0\x00\x40")

    def test_worker_returns_first_chunk_before_done_and_cleans_interrupted_protocol(self):
        wav = base64.b64encode(audio_chunk().wav_bytes).decode()
        script = (
            "import json, sys\n"
            "for line in sys.stdin:\n"
            " request = json.loads(line)\n"
            " if request['action'] == 'prepare':\n"
            "  print(json.dumps({'ok': True}), flush=True)\n"
            " elif request['action'] == 'synthesize':\n"
            f"  print(json.dumps({{'ok': True, 'sample_rate': 24000, 'wav_base64': {wav!r}}}), flush=True)\n"
            " else:\n"
            f"  chunk = {{'ok': True, 'event': 'audio', 'sample_rate': 24000, 'wav_base64': {wav!r}}}\n"
            "  print(json.dumps(chunk), flush=True)\n"
            "  print(json.dumps(chunk), flush=True)\n"
            "  print(json.dumps({'ok': True, 'event': 'done'}), flush=True)\n"
        )
        backend = SubprocessSpeechBackend(command=[sys.executable, "-u", "-c", script], engine="mlx")
        try:
            backend.prepare()
            first_process = backend._process
            chunks = backend.stream("Hello.")
            self.assertEqual(next(chunks).sample_rate, 24000)
            chunks.close()
            self.assertIsNotNone(first_process.poll())
            self.assertIsNone(backend._process)
            recording = backend.synthesize("Another turn.")
            with wave.open(io.BytesIO(recording.wav_bytes), "rb") as audio:
                self.assertEqual(audio.getnframes(), 4)
            self.assertIsNot(backend._process, first_process)
            second_process = backend._process
            self.assertEqual(len(list(backend.stream("Next."))), 2)
            self.assertIs(backend._process, second_process)
        finally:
            backend.close()

    def test_streaming_player_keeps_one_device_open_and_closes_it_on_failure(self):
        with patch.dict(sys.modules, {"sounddevice": Mock()}) as modules:
            device = modules["sounddevice"].RawOutputStream.return_value
            with AudioStreamPlayer() as player:
                player.write(audio_chunk())
                player.write(audio_chunk())
            modules["sounddevice"].RawOutputStream.assert_called_once()
            device.start.assert_called_once()
            self.assertEqual(device.write.call_count, 2)
            device.stop.assert_called_once()
            device.close.assert_called_once()
            device.reset_mock()
            device.write.side_effect = RuntimeError("Device disconnected.")
            with self.assertRaisesRegex(RuntimeError, "Device disconnected"):
                with AudioStreamPlayer() as player:
                    player.write(audio_chunk())
            device.abort.assert_called_once()
            device.close.assert_called_once()

    def test_actual_mlx_worker_frames_chunks_and_done_without_model_stdout(self):
        from output.mlx_speech_worker import main

        stdout, stderr = io.StringIO(), io.StringIO()
        requests = io.StringIO(
            '{"action":"prepare"}\n'
            '{"action":"synthesize","text":"A complete opening sentence."}\n'
            '{"action":"stream","text":"Hello.","language":"English"}\n'
        )

        def generate(*args, **kwargs):
            print("Model log.")
            yield audio_chunk()
            yield audio_chunk()

        with patch("sys.stdin", requests), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            with patch("output.mlx_speech.prepare"), patch("output.mlx_speech.stream_speech", side_effect=generate), patch(
                "output.mlx_speech.synthesize_speech", return_value=audio_chunk(),
            ):
                main()
        import json
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(responses[0], {"ok": True})
        self.assertIn("wav_base64", responses[1])
        self.assertNotIn("event", responses[1])
        self.assertEqual([response["event"] for response in responses[2:]], ["audio", "audio", "done"])
        self.assertIn("Model log.", stderr.getvalue())

    def test_mlx_complete_sentence_uses_non_streaming_waveform_decoding(self):
        import numpy as np
        from output.mlx_speech import synthesize_speech

        model = Mock()
        model.generate.return_value = iter([
            SimpleNamespace(audio=np.array([0.25, -0.25]), sample_rate=24000),
        ])
        with patch("output.mlx_speech._get_voice", return_value=(model, object(), "Original transcript.")):
            audio = synthesize_speech("A complete opening sentence.", language="English")
        self.assertFalse(model.generate.call_args.kwargs["stream"])
        with wave.open(io.BytesIO(audio.wav_bytes), "rb") as recording:
            self.assertEqual(recording.getnframes(), 2)

    def test_voice_preparation_warms_the_actual_streaming_decoder(self):
        from output.mlx_speech import prepare
        model = Mock()
        model.generate.return_value = iter([object(), object()])
        with patch("output.mlx_speech._get_voice", return_value=(model, object(), "Original transcript.")):
            prepare()
        arguments = model.generate.call_args.kwargs
        self.assertTrue(arguments["stream"])
        self.assertEqual(arguments["streaming_interval"], 0.32)
        self.assertGreaterEqual(arguments["max_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
