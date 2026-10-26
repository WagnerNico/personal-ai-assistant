"""Explicit, bounded microphone evaluation without saving audio or training on it."""

from math import gcd
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from scipy.signal import resample_poly

from input.microphone import MicrophoneRecorder
from input.wakeword import WakeWordDetector


def check_microphone(model_path, *, seconds=15.0, device=None):
    """Capture one short session, print activations, and discard all audio on exit."""
    if not 0 < seconds <= 30:
        raise ValueError("Microphone evaluation must last between zero and thirty seconds.")
    detector = WakeWordDetector(Path(model_path))
    recorder = MicrophoneRecorder(device=device, max_duration=seconds)
    consumed, activations, highest_score = 0, [], 0.0
    recorder.start()
    started = perf_counter()
    print(f"Microphone check active for {seconds:g} seconds. Say Hey Airis or Airis, with pauses between attempts.", flush=True)
    try:
        while perf_counter() - started < seconds and not recorder.limit_reached:
            captured = recorder.snapshot()
            if not len(captured.samples):
                sleep(0.08)
                continue
            if captured.sample_rate != 16000:
                factor = gcd(captured.sample_rate, 16000)
                samples = resample_poly(captured.samples, 16000 // factor, captured.sample_rate // factor)
                # Keep the resampler's unfinished tail out of streaming inference.
                safe_end = max(0, len(samples) - 640)
            else:
                samples, safe_end = captured.samples, len(captured.samples)
            for frame in detector.feed(np.asarray(samples[consumed:safe_end], np.float32)):
                highest_score = max(highest_score, frame.score)
                if frame.activated:
                    activations.append(frame.time_seconds)
                    print(f"Wake activation at {frame.time_seconds:.2f} seconds (score {frame.score:.5f})", flush=True)
            consumed = safe_end
            sleep(0.08)
        captured = recorder.stop()
        if captured.sample_rate != 16000:
            factor = gcd(captured.sample_rate, 16000)
            samples = resample_poly(captured.samples, 16000 // factor, captured.sample_rate // factor)
        else:
            samples = captured.samples
        for frame in detector.feed(np.asarray(samples[consumed:], np.float32)):
            highest_score = max(highest_score, frame.score)
            if frame.activated:
                activations.append(frame.time_seconds)
                print(f"Wake activation at {frame.time_seconds:.2f} seconds (score {frame.score:.5f})", flush=True)
    finally:
        if recorder.is_recording:
            recorder.cancel()
    return {"duration_seconds": len(captured.samples) / captured.sample_rate,
            "activation_times_seconds": activations, "highest_score": highest_score,
            "audio_saved": False, "used_for_training": False,
            "limits": "A short microphone check is not an accuracy or false-activation-rate benchmark."}
