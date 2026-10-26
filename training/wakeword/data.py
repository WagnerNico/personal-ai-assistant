"""Deterministic synthetic examples with voice-separated dataset partitions."""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path


RECIPE_PATH = Path(__file__).with_name("recipe.json")


def load_recipe(path=RECIPE_PATH):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    voices = [voice for split in recipe["voices"].values() for voice in split]
    if len(voices) != len(set(voices)):
        raise ValueError("Training, validation, and test voices must be disjoint.")
    if any(voice not in recipe["voice_designs"] for voice in voices):
        raise ValueError("Each synthetic speaker must have a voice design instruction.")
    if recipe.get("pronunciation") != "qwen":
        raise ValueError("Use Qwen's natural pronunciation for the wake phrases.")
    if not recipe.get("phrases") or any(not isinstance(phrase, str) or not phrase.strip() for phrase in recipe["phrases"]):
        raise ValueError("Provide nonempty positive wake phrases.")
    positives = {phrase.strip().rstrip(".!?").casefold() for phrase in recipe["phrases"]}
    if any(phrase.strip().rstrip(".!?").casefold() in positives for phrase in recipe["negative_phrases"]):
        raise ValueError("A wake phrase cannot also be a negative example.")
    if recipe["sample_rate"] != 16000 or recipe["frame_samples"] != 1280:
        raise ValueError("The openWakeWord pipeline requires 16 kHz audio and 80 ms frames.")
    for key in ("takes", "positive_takes", "negative_takes", "synthesis_batch_size"):
        value = recipe.get(key, 1 if key == "synthesis_batch_size" else recipe["takes"])
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if not recipe["speeds"] or any(speed <= 0 for speed in recipe["speeds"]):
        raise ValueError("Synthesis speeds must be positive.")
    return recipe


@dataclass(frozen=True)
class Example:
    id: str
    split: str
    voice: str
    text: str
    label: int
    speed: float
    take: int = 0


def examples(recipe):
    if recipe.get("pronunciation") != "qwen":
        raise ValueError("Use Qwen's natural pronunciation for the wake phrases.")
    phrases = [(phrase.rstrip(".!?") + ".", 1) for phrase in recipe["phrases"]]
    phrases += [("Hey.", 0)]
    phrases += [(phrase, 0) for phrase in recipe["negative_phrases"]]
    for split, voices in recipe["voices"].items():
        for voice in voices:
            for text, label in phrases:
                takes = recipe.get("positive_takes" if label else "negative_takes", recipe["takes"])
                for take in range(takes):
                    for speed in recipe["speeds"]:
                        identity = json.dumps([
                            recipe["tts_model"], recipe["design_model"], recipe["reference_text"],
                            recipe["voice_designs"][voice], recipe["seed"], split, voice,
                            text, label, speed, take,
                        ])
                        identifier = hashlib.sha256(identity.encode()).hexdigest()[:24]
                        yield Example(identifier, split, voice, text, label, speed, take)


def write_plan(recipe, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = list(examples(recipe))
    (root / "plan.jsonl").write_text(
        "".join(json.dumps(asdict(example), ensure_ascii=False) + "\n" for example in plan),
        encoding="utf-8",
    )
    (root / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return plan


def augment(samples, *, seed, window_samples=32000, label=1):
    """Simulate gain, room echoes, placement, and noise without real recordings."""
    import numpy as np
    from scipy.signal import resample

    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all() or not len(samples):
        raise ValueError("Synthetic source audio must be finite, nonempty, and mono.")
    if label and len(samples) > window_samples:
        raise ValueError("The positive utterance exceeds the training window.")
    # Bound stretching so a positive remains complete inside the training window.
    maximum_stretch = min(1.1, window_samples / len(samples)) if label else 1.1
    samples = resample(samples, max(1, int(len(samples) * rng.uniform(0.9, maximum_stretch)))).astype(np.float32)
    if len(samples) > window_samples:
        start = int(rng.integers(0, len(samples) - window_samples + 1))
        samples = samples[start:start + window_samples]
    output = np.zeros(window_samples, dtype=np.float32)
    offset = int(rng.integers(0, window_samples - len(samples) + 1))
    output[offset:offset + len(samples)] = samples * rng.uniform(0.25, 1.0)
    if rng.random() < 0.7:
        delay = int(rng.uniform(0.02, 0.12) * 16000)
        output[delay:] += rng.uniform(0.05, 0.3) * output[:-delay].copy()
    noise = rng.standard_normal(window_samples).astype(np.float32)
    if rng.random() < 0.5:
        spectrum = np.fft.rfft(noise)
        spectrum /= np.sqrt(np.maximum(1, np.arange(len(spectrum))))
        noise = np.fft.irfft(spectrum, n=window_samples).astype(np.float32)
    rms = max(float(np.sqrt(np.mean(output ** 2))), 0.005)
    noise *= rms / (10 ** (rng.uniform(8, 30) / 20) * max(float(np.std(noise)), 1e-6))
    return np.clip(output + noise, -1, 1).astype(np.float32)
