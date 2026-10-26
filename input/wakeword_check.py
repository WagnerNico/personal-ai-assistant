"""Bounded microphone diagnosis without model replies or saved recordings."""

from math import gcd
from threading import Event
from time import monotonic

import numpy as np
from scipy.signal import resample_poly

from .microphone import MicrophoneRecorder
from .wakeword import WakeWordDetector
from .wakeword_listener import DEFAULT_WAKE_MODEL
from .audio import RecordedAudio
from .audio_devices import standby_input_device
from audio_runtime import audio_session


def check_wake_phrase(*, model_path=DEFAULT_WAKE_MODEL, threshold=None, device=None,
                      seconds=5.0, cancelled=None, recorder_factory=MicrophoneRecorder,
                      detector_factory=WakeWordDetector, transcriber=None, language=None,
                      emit=lambda event: None, standby=False, candidate_threshold=None, verifier=None):
    """Capture a user-requested phrase, score it, and optionally transcribe it.

    Recognition is diagnostic only: neither successful nor failed tests change
    model weights, thresholds, conversation history, or training datasets.
    """
    if not 0 < seconds <= 10:
        raise ValueError("Wake phrase checks must last between zero and ten seconds.")
    cancelled = cancelled if cancelled is not None else Event()
    detector = detector_factory(model_path, threshold=threshold)
    if candidate_threshold is not None:
        detector.set_threshold(min(candidate_threshold, detector.trigger.threshold))
    microphone = ""
    if standby:
        import sounddevice as sd
        with audio_session(sd, follow_system_default=device is None):
            device = standby_input_device(sd, device)
            microphone = sd.query_devices(device, "input")["name"]
            recorder = recorder_factory(device=device, max_duration=seconds)
            if not cancelled.is_set():
                recorder.start()
    else:
        recorder = recorder_factory(device=device, max_duration=seconds)
    if cancelled.is_set():
        if recorder.is_recording:
            recorder.cancel()
        return None
    if not standby:
        recorder.start()
    try:
        started = monotonic()
        emit(("status", 'Speak "Hey Airis" or "Airis" now. Checking the microphone for five seconds...'))
        while not recorder.limit_reached and monotonic() - started < seconds:
            if cancelled.wait(0.05):
                return None
        audio = recorder.stop()
    finally:
        if recorder.is_recording:
            recorder.cancel()
    if cancelled.is_set():
        return None
    samples = np.asarray(audio.samples, np.float32)
    if not samples.size:
        raise RuntimeError("The microphone delivered no audio. Check microphone permission and your input device.")
    if audio.sample_rate != 16000:
        factor = gcd(audio.sample_rate, 16000)
        samples = resample_poly(samples, 16000 // factor, audio.sample_rate // factor).astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("The microphone produced invalid samples.")
    detector.reset()
    # Match the live frontend's quiet starting history and confirmation rule.
    frames = detector.feed(np.concatenate([samples, np.zeros(7680, np.float32)]))
    scores = [frame.score for frame in frames]
    peak = max(scores, default=0.0)
    confirmed = max((min(a, b) for a, b in zip(scores, scores[1:])), default=0.0)
    rms = float(np.sqrt(np.mean(samples * samples)))
    result = {"duration_seconds": len(audio.samples) / audio.sample_rate,
              "rms": rms, "audio_peak": float(np.max(np.abs(samples))),
              "peak_score": peak, "confirmed_peak": confirmed,
              "threshold": detector.trigger.threshold,
              "activated": any(frame.activated for frame in frames),
              "classifier_activated": any(frame.activated for frame in frames),
              "verification_enabled": verifier is not None, "microphone": microphone,
              "transcript": "", "audio_saved": False, "used_for_training": False}
    inspected = False
    if verifier is not None:
        result["activated"] = False
        for index, frame in enumerate(frames):
            if not frame.activated or cancelled.is_set():
                continue
            end = min(len(samples), (index + 1) * 1280 + 3840)
            candidate = RecordedAudio(samples[max(0, end - 38400):end], 16000)
            verification = verifier.inspect(candidate)
            inspected = True
            result["transcript"] = verification.transcript
            if verification.accepted:
                result["activated"] = True
                break
    if not inspected and transcriber is not None and rms >= 1e-7 and not cancelled.is_set():
        emit(("status", "Checking what Whisper heard..."))
        result["transcript"] = transcriber(audio, language=language)
    return None if cancelled.is_set() else result
