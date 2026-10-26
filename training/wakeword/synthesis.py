"""Generate a resumable corpus using Qwen-designed synthetic speakers."""

import json
from pathlib import Path
from time import perf_counter

from .data import write_plan


def generate_corpus(recipe, root, *, synthesizer=None):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    root = Path(root)
    plan = write_plan(recipe, root)
    audio_dir = root / "audio"
    audio_dir.mkdir(exist_ok=True)
    started = perf_counter()
    generated, reused, audio_seconds = 0, 0, 0.0
    for index, example in enumerate(plan):
        path = audio_dir / f"{example.id}.wav"
        if path.exists():
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or not info.frames:
                raise ValueError(f"Invalid cached synthetic clip: {path}")
            reused += 1
            continue
        if synthesizer is None:
            from .qwen_synthesis import QwenSynthesizer
            synthesizer = QwenSynthesizer(recipe, root)
            synthesizer.configure_batches(plan)
        samples, rate = synthesizer(example)
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
            raise ValueError(f"The synthesizer produced invalid audio for {example.id}.")
        if not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError("The synthesizer must return a positive integer sample rate.")
        divisor = gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // divisor, int(rate) // divisor)
        temporary = path.with_suffix(".partial.wav")
        sf.write(temporary, samples, 16000, subtype="PCM_16")
        temporary.replace(path)
        generated += 1
        audio_seconds += len(samples) / 16000
        if generated == 1 or generated % 25 == 0:
            print(f"Synthetic clips: {index + 1}/{len(plan)}", flush=True)
    report = {"generated": generated, "reused": reused, "audio_seconds": audio_seconds,
              "elapsed_seconds": perf_counter() - started, "model": recipe["tts_model"]}
    source_count = getattr(synthesizer, "generated_sources", None)
    if source_count is not None:
        report["generated_source_utterances"] = source_count
        report["generated_batches"] = synthesizer.generated_batches
        report["synthesis_batch_size"] = recipe.get("synthesis_batch_size", 1)
    (root / "synthesis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
