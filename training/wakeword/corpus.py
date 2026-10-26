"""Audit generated audio and prepare reproducible synthetic training variants."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .data import augment, examples, load_recipe


def audit_corpus(root, *, training_only=False):
    """Check every planned recording and record file hashes and partition counts."""
    import numpy as np
    import soundfile as sf
    from huggingface_hub import try_to_load_from_cache

    root = Path(root)
    recipe = load_recipe(root / "recipe.json")
    plan = list(examples(recipe))
    saved_plan = [json.loads(line) for line in (root / "plan.jsonl").read_text(encoding="utf-8").splitlines()]
    if saved_plan != [asdict(row) for row in plan]:
        raise ValueError("The saved corpus plan does not match its recipe.")
    if training_only:
        plan = [row for row in plan if row.split == "train"]
        saved_plan = [row for row in saved_plan if row["split"] == "train"]
    prefix = "training_" if training_only else ""
    rows, errors, oversized = [], [], []
    for example, saved_row in zip(plan, saved_plan):
        path = root / "audio" / f"{example.id}.wav"
        try:
            info = sf.info(path)
            samples, rate = sf.read(path, dtype="float32")
            if rate != recipe["sample_rate"] or info.channels != 1 or info.subtype != "PCM_16":
                raise ValueError("Expected mono 16 kHz PCM16 audio.")
            if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
                raise ValueError("Audio is empty, nonfinite, or silent.")
            if example.label and len(samples) > recipe["window_samples"]:
                oversized.append(example.id)
            rows.append({
                **saved_row, "path": str(path.relative_to(root)),
                "samples": len(samples), "duration_seconds": len(samples) / rate,
                "peak": float(np.max(np.abs(samples))),
                "rms": float(np.sqrt(np.mean(samples ** 2))),
                "clipped_sample_fraction": float(np.mean(np.abs(samples) >= 0.999)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        except (OSError, ValueError, RuntimeError) as error:
            errors.append({"id": example.id, "error": str(error)})
    snapshots = {}
    for key in ("tts_model", "design_model"):
        cached = try_to_load_from_cache(recipe[key], "config.json")
        if isinstance(cached, str):
            snapshots[recipe[key]] = Path(cached).parent.name
    report = {
        "planned_clips": len(plan), "valid_clips": len(rows), "errors": errors,
        "by_split": dict(Counter(row["split"] for row in rows)),
        "positives": sum(row["label"] for row in rows),
        "negatives": sum(not row["label"] for row in rows),
        "audio_seconds": sum(row["duration_seconds"] for row in rows),
        "maximum_positive_seconds": max((row["duration_seconds"] for row in rows if row["label"]), default=0),
        "positives_exceeding_window": oversized,
        "cached_checkpoint_revisions": snapshots,
        "audited_partitions": ["train"] if training_only else list(recipe["voices"]),
    }
    (root / f"{prefix}validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"Corpus audit found {len(errors)} invalid or missing clips; see {prefix}validation_report.json.")
    manifest = root / f"{prefix}corpus_manifest.jsonl"
    temporary = manifest.with_suffix(".partial.jsonl")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(manifest)
    return report


def prepare_variants(root):
    """Cache noise and echo variants without loading a speech model."""
    import numpy as np
    import soundfile as sf

    root = Path(root)
    audit = audit_corpus(root)
    recipe = load_recipe(root / "recipe.json")
    source_hashes = {row["id"]: row["sha256"] for row in (
        json.loads(line) for line in (root / "corpus_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    )}
    directory = root / "augmented"
    directory.mkdir(exist_ok=True)
    manifest = root / "augmentation_manifest.jsonl"
    temporary_manifest = manifest.with_suffix(".partial.jsonl")
    generated, reused, excluded = 0, 0, []
    splits, labels = Counter(), Counter()
    started = perf_counter()
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for example in examples(recipe):
            samples, _ = sf.read(root / "audio" / f"{example.id}.wav", dtype="float32")
            if example.id in audit["positives_exceeding_window"]:
                excluded.append(example.id)
                continue
            for index in range(recipe["augmentations_per_clip"]):
                identity = json.dumps([recipe["seed"], example.id, source_hashes[example.id], index,
                                       recipe["sample_rate"], recipe["window_samples"], "noise-echo-v1"])
                digest = hashlib.sha256(identity.encode()).hexdigest()
                seed = int(digest[:16], 16)
                path = directory / f"{digest[:24]}.wav"
                if path.exists():
                    info = sf.info(path)
                    if (info.samplerate != recipe["sample_rate"] or info.channels != 1
                            or info.frames != recipe["window_samples"] or info.subtype != "PCM_16"):
                        raise ValueError(f"Invalid cached augmented clip: {path}")
                    cached, _ = sf.read(path, dtype="float32")
                    if not np.isfinite(cached).all() or np.max(np.abs(cached)) < 1e-4:
                        raise ValueError(f"Invalid cached augmented clip: {path}")
                    reused += 1
                else:
                    waveform = augment(samples, seed=seed, window_samples=recipe["window_samples"], label=example.label)
                    temporary = path.with_suffix(".partial.wav")
                    sf.write(temporary, waveform, recipe["sample_rate"], subtype="PCM_16")
                    temporary.replace(path)
                    generated += 1
                row = {"id": path.stem, "parent_id": example.id, "split": example.split,
                       "voice": example.voice, "text": example.text, "label": example.label,
                       "source_sha256": source_hashes[example.id],
                       "seed": seed, "path": str(path.relative_to(root)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                handle.write(json.dumps(row) + "\n")
                splits[example.split] += 1
                labels[example.label] += 1
            completed = generated + reused
            if completed and completed % 1000 == 0:
                print(f"Augmented clips: {completed}", flush=True)
    temporary_manifest.replace(manifest)
    report = {"generated": generated, "reused": reused, "by_split": dict(splits),
              "positives": labels[1], "negatives": labels[0],
              "excluded_positive_sources": excluded,
              "audio_seconds": (generated + reused) * recipe["window_samples"] / recipe["sample_rate"],
              "elapsed_seconds": perf_counter() - started}
    (root / "augmentation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
