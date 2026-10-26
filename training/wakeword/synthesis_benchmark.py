"""Measure Qwen cloning batches without loading a second copy of the TTS model."""

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

from .data import examples
from .qwen_synthesis import QwenSynthesizer


def benchmark_synthesis(recipe, root):
    recipe = dict(recipe)
    voice = recipe["voices"]["train"][0]
    recipe["voices"] = {"train": [voice], "validation": [], "test": []}
    recipe["speeds"] = [1.0]
    recipe["positive_takes"] = 4
    recipe["negative_takes"] = 1
    recipe["synthesis_batch_size"] = 1
    root = Path(root) / "synthesis_benchmark"
    report_path = root / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    synthesizer = QwenSynthesizer(recipe, root)
    prototype = next(example for example in examples(recipe) if example.label)
    # Separate keys ensure this benchmark never becomes training data.
    warmup = replace(prototype, take=100000)
    synthesizer(warmup)
    singles = [replace(prototype, take=100001 + index) for index in range(4)]
    started = perf_counter()
    sequential = [synthesizer(example) for example in singles]
    sequential_seconds = perf_counter() - started
    batched_examples = [replace(prototype, take=200001 + index) for index in range(4)]
    recipe["synthesis_batch_size"] = 4
    synthesizer.configure_batches(batched_examples)
    started = perf_counter()
    batched = [synthesizer(example) for example in batched_examples]
    batch_seconds = perf_counter() - started
    for samples, rate in sequential + batched:
        if rate != 24000 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
            raise ValueError("Invalid audio from the synthesis benchmark.")
    report = {"utterances_per_method": 4, "sequential_seconds": sequential_seconds,
              "batch_seconds": batch_seconds, "speedup": sequential_seconds / batch_seconds,
              "sequential_audio_seconds": [len(samples) / rate for samples, rate in sequential],
              "batch_audio_seconds": [len(samples) / rate for samples, rate in batched],
              "peak_gpu_memory_gb": mx.get_peak_memory() / 1e9,
              "model": recipe["tts_model"], "training_audio_used": False}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
