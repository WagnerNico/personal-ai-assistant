"""Compare speech scheduling and model sizes silently on the target Mac."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
from time import perf_counter, sleep


class SimulatedPlayback:
    """Consume audio at its natural rate and count missed delivery deadlines."""

    def __init__(self):
        self.started = perf_counter()
        self.first_audio_seconds = None
        self.deadline = None
        self.gaps = 0
        self.gap_seconds = 0.0
        self.audio_seconds = 0.0

    def write(self, audio):
        now = perf_counter()
        if self.first_audio_seconds is None:
            self.first_audio_seconds = now - self.started
        underflowed = self.deadline is not None and now > self.deadline + 0.08
        if underflowed:
            self.gaps += 1
            self.gap_seconds += now - self.deadline
        duration = audio.duration_seconds
        self.audio_seconds += duration
        self.deadline = now + duration
        sleep(duration)
        return underflowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["9b", "4b"], default=["9b", "4b"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    processes = subprocess.check_output(["ps", "-A", "-o", "command="], text=True)
    if any(" -m ui.desktop" in line or " -m output.mlx_speech_worker" in line
           for line in processes.splitlines()):
        raise RuntimeError("Quit Airis before benchmarking to avoid loading duplicate models.")

    from assistant import Assistant
    from output.speech_backend import SubprocessSpeechBackend
    from processing.mlx_model import MODEL_ID, FAST_MODEL_ID, load_mlx_backend
    import mlx.core as mx

    report = []
    speech = SubprocessSpeechBackend(engine="mlx")
    prompt = "Explain why the sky looks blue in three short, complete sentences."
    try:
        speech.prepare()
        for size in args.models:
            model_id = MODEL_ID if size == "9b" else FAST_MODEL_ID
            print(f"Preparing {size.upper()}...", flush=True)
            backend = load_mlx_backend(model_id=model_id)
            backend.temperature = 0.0
            backend.warm_up()
            policies = [("concurrent", 0.0, False), ("buffered_priority", 1.0, True)] if size == "9b" else [("buffered_priority", 1.0, True)]
            for policy, buffer_seconds, priority in policies:
                player = SimulatedPlayback()
                with Assistant(backend=backend, speech_backend=speech,
                               speech_buffer_seconds=buffer_seconds, speech_priority=priority) as assistant:
                    reply = assistant.respond(prompt, mode="speech", language="English",
                                              max_new_tokens=args.max_new_tokens, on_audio=player.write)
                result = {
                    "model": model_id, "policy": policy,
                    "first_audio_s": round(player.first_audio_seconds, 2),
                    "elapsed_s": round(perf_counter() - player.started, 2),
                    "audio_s": round(player.audio_seconds, 2),
                    "delivery_gaps": player.gaps, "gap_s": round(player.gap_seconds, 2),
                    "reply_characters": len(reply.text), "truncated": reply.truncated,
                }
                report.append(result)
                print(json.dumps(result), flush=True)
            del backend, assistant
            gc.collect()
            mx.clear_cache()
    finally:
        speech.close()
    destination = Path(__file__).resolve().parents[1] / "logs" / "speech_benchmark.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {destination}", flush=True)


if __name__ == "__main__":
    main()
