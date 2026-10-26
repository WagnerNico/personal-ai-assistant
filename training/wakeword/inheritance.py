"""Reuse matching training caches while keeping new evaluation speakers unseen."""

import json
import os
from pathlib import Path

from .data import examples, load_recipe, write_plan
from .qwen_synthesis import reference_identity
import hashlib


def inherit_training_cache(recipe, root, baseline):
    root, baseline = Path(root), Path(baseline)
    previous = load_recipe(baseline / "recipe.json")
    old_voices = {voice for voices in previous["voices"].values() for voice in voices}
    evaluation_voices = set(recipe["voices"]["validation"] + recipe["voices"]["test"])
    if evaluation_voices & old_voices:
        raise ValueError("The next iteration must use fresh validation and test speaker identities.")
    allowed = set(previous["voices"]["train"]) & set(recipe["voices"]["train"])
    plan = write_plan(recipe, root)
    reused = {"reference_files": 0, "source_files": 0, "audio_files": 0, "base_files": 0}

    def link(source, destination, group):
        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            reused[group] += 1

    for voice in allowed:
        key, _ = reference_identity(recipe, voice)
        for extension in (".wav", ".json"):
            link(baseline / "references" / (key + extension), root / "references" / (key + extension), "reference_files")
    for example in plan:
        if example.split != "train" or example.voice not in allowed:
            continue
        reference_key, _ = reference_identity(recipe, example.voice)
        source_key = hashlib.sha256(json.dumps([
            recipe["tts_model"], reference_key, example.text, example.take,
        ]).encode()).hexdigest()[:24]
        link(baseline / "sources" / (source_key + ".wav"), root / "sources" / (source_key + ".wav"), "source_files")
        link(baseline / "audio" / (example.id + ".wav"), root / "audio" / (example.id + ".wav"), "audio_files")
    for source in (baseline / "google_base").rglob("*"):
        if source.is_file():
            link(source, root / "google_base" / source.relative_to(baseline / "google_base"), "base_files")
    report = {"baseline": str(baseline), "reused": reused, "inherited_partitions": ["train"],
              "fresh_validation_and_test": True, "planned_clips": len(plan),
              "speakers": {split: len(voices) for split, voices in recipe["voices"].items()},
              "distinct_positive_sources": sum(len(voices) * len(recipe["phrases"]) * recipe.get("positive_takes", recipe["takes"])
                                               for voices in recipe["voices"].values())}
    # Keep the original inheritance record when an interrupted run is resumed.
    report_path = root / "iteration_provenance.json"
    if not report_path.exists():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
