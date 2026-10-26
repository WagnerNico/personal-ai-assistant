"""Design synthetic Qwen speakers once, then clone them with Airis's TTS model."""

import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter


def reference_identity(recipe, voice):
    identity = {
        "voice": voice, "model": recipe["design_model"], "seed": recipe["seed"],
        "text": recipe["reference_text"], "instruction": recipe["voice_designs"][voice],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24], identity


def _collect_audio(results):
    import numpy as np

    results = list(results)
    if not results:
        raise ValueError("Qwen generated no speech.")
    rate = int(results[0].sample_rate)
    if rate <= 0 or any(int(result.sample_rate) != rate for result in results):
        raise ValueError("Qwen returned inconsistent sample rates.")
    audio = np.concatenate([np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in results])
    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-4:
        raise ValueError("Qwen generated invalid or silent speech.")
    return audio, rate


def prepare_references(recipe, root, *, renderer=None):
    """Generate each speaker once; release VoiceDesign before loading Base."""
    import soundfile as sf

    root = Path(root)
    directory = root / "references"
    directory.mkdir(parents=True, exist_ok=True)
    references, missing = {}, []
    for voices in recipe["voices"].values():
        for voice in voices:
            key, identity = reference_identity(recipe, voice)
            path = directory / f"{key}.wav"
            references[voice] = path
            if path.exists():
                info = sf.info(path)
                if info.channels != 1 or not info.frames or info.samplerate <= 0:
                    raise ValueError(f"Invalid cached Qwen reference: {path}")
            else:
                missing.append((voice, path, key, identity))
    if not missing:
        return references
    model = None
    if renderer is None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model
        model = load_model(recipe["design_model"])
        def renderer(**kwargs):
            return model.generate(**kwargs)
    started = perf_counter()
    try:
        for index, (voice, path, key, identity) in enumerate(missing):
            if model is not None:
                mx.random.seed((recipe["seed"] + int(key[:8], 16)) % 2**32)
            samples, rate = _collect_audio(renderer(
                text=recipe["reference_text"], instruct=identity["instruction"],
                lang_code="English", stream=False, verbose=False, max_tokens=512,
            ))
            temporary = path.with_suffix(".partial.wav")
            sf.write(temporary, samples, rate, subtype="PCM_16")
            temporary.replace(path)
            path.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            print(f"Qwen reference voices: {index + 1}/{len(missing)} ({voice})", flush=True)
        (root / "reference_report.json").write_text(json.dumps({
            "generated": len(missing), "elapsed_seconds": perf_counter() - started,
            "model": recipe["design_model"],
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        if model is not None:
            del model
            gc.collect()
            mx.clear_cache()
    return references


class QwenSynthesizer:
    """Reuse speaker conditioning and source utterances across speed variants."""

    def __init__(self, recipe, root):
        self.recipe = recipe
        self.root = Path(root)
        self.references = prepare_references(recipe, root)
        self.sources = self.root / "sources"
        self.sources.mkdir(exist_ok=True)
        self.model = None
        self.reference_audio = {}
        self.generated_sources = 0
        self.generated_batches = 0
        self.batch_groups = {}

    def source_identity(self, example):
        reference_key, _ = reference_identity(self.recipe, example.voice)
        return hashlib.sha256(json.dumps([
            self.recipe["tts_model"], reference_key, example.text, example.take,
        ]).encode()).hexdigest()[:24]

    def configure_batches(self, plan):
        """Fix group membership before checking caches so resume preserves batch seeds."""
        size = self.recipe.get("synthesis_batch_size", 1)
        self.batch_groups = {}
        groups = {}
        for example in plan:
            if example.speed != self.recipe["speeds"][0]:
                continue
            # Keep similar lengths together to limit padding and early-finished work.
            bucket = (example.voice, example.label, len(example.text) > 80)
            groups.setdefault(bucket, []).append(example)
        for examples in groups.values():
            for start in range(0, len(examples), size):
                group = examples[start:start + size]
                for example in group:
                    self.batch_groups[self.source_identity(example)] = group

    def _load_base(self, voice):
        from mlx_audio.tts.utils import load_model
        from mlx_audio.utils import load_audio

        if self.model is None:
            self.model = load_model(self.recipe["tts_model"])
        if voice not in self.reference_audio:
            self.reference_audio[voice] = load_audio(
                str(self.references[voice]), sample_rate=self.model.sample_rate,
            )

    def _generate_batch(self, group):
        import mlx.core as mx
        import soundfile as sf

        voice = group[0].voice
        self._load_base(voice)
        keys = [self.source_identity(example) for example in group]
        batch_key = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
        mx.random.seed((self.recipe["seed"] + int(batch_key[:8], 16)) % 2**32)
        chunks = [[] for _ in group]
        for result in self.model.batch_generate(
            texts=[example.text for example in group], ref_audio=self.reference_audio[voice],
            ref_text=self.recipe["reference_text"], lang_code="English", stream=False,
            verbose=False, max_tokens=256, temperature=0.7,
        ):
            if not 0 <= result.sequence_idx < len(group):
                raise ValueError("Qwen returned an invalid batch sequence index.")
            chunks[result.sequence_idx].append(result)
        # Validate every output before saving any member of the batch.
        outputs = [_collect_audio(results) for results in chunks]
        for key, (samples, rate) in zip(keys, outputs):
            path = self.sources / f"{key}.wav"
            if path.exists():
                continue
            temporary = path.with_suffix(".partial.wav")
            sf.write(temporary, samples, rate, subtype="PCM_16")
            temporary.replace(path)
            self.generated_sources += 1
        self.generated_batches += 1

    def __call__(self, example):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample

        source_key = self.source_identity(example)
        path = self.sources / f"{source_key}.wav"
        if not path.exists() and self.recipe.get("synthesis_batch_size", 1) > 1:
            group = self.batch_groups.get(source_key)
            if group is None:
                raise ValueError("Configure synthesis batches with the complete plan first.")
            self._generate_batch(group)
        if path.exists():
            samples, rate = sf.read(path, dtype="float32")
        else:
            import mlx.core as mx
            self._load_base(example.voice)
            mx.random.seed((self.recipe["seed"] + int(source_key[:8], 16)) % 2**32)
            samples, rate = _collect_audio(self.model.generate(
                text=example.text, ref_audio=self.reference_audio[example.voice],
                ref_text=self.recipe["reference_text"], lang_code="English",
                stream=False, verbose=False, max_tokens=256, temperature=0.7,
            ))
            temporary = path.with_suffix(".partial.wav")
            sf.write(temporary, samples, rate, subtype="PCM_16")
            temporary.replace(path)
            self.generated_sources += 1
        if samples.ndim != 1 or not len(samples) or rate <= 0 or not np.isfinite(samples).all():
            raise ValueError(f"Invalid cached Qwen utterance: {path}")
        if example.speed != 1.0:
            samples = resample(samples, max(1, round(len(samples) / example.speed))).astype(np.float32)
        return samples, rate
