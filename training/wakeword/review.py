"""Prepare unchanged training recordings for pronunciation and label review."""

import hashlib
import json
import os
from pathlib import Path
import re

from .data import load_recipe
from .evaluate import read_rows


def prepare_pronunciation_review(root):
    root = Path(root)
    recipe = load_recipe(root / "recipe.json")
    rows = read_rows(root / "corpus_manifest.jsonl")
    preferred = ["us_female_clear", "us_male_low", "de_male_english"]
    voices = [voice for voice in preferred if voice in recipe["voices"]["train"]]
    if not voices:
        voices = recipe["voices"]["train"][:3]
    texts = ["Hey Airis.", "Hey Eris.", "Hey Iris.", "Hey Harris.", "Airis.", "Aris."]
    directory = root / "review"
    directory.mkdir(exist_ok=True)
    records, lines = [], ["# Pronunciation review", "",
        "These unchanged normal-speed recordings come only from training speakers.",
        "Compare the intended wake phrases with similar names before changing any labels.",
        "The log-mel similarity report is a proxy; it does not establish phoneme equivalence.",
        "The confusers remain in the strict benchmark and have not been relabelled or excluded.", ""]
    for voice in voices:
        voice_name = re.sub(r"[^A-Za-z0-9_-]", "_", voice)
        lines += [f"## {voice}", ""]
        for text in texts:
            candidates = [row for row in rows if row["split"] == "train" and row["voice"] == voice
                          and row["text"] == text and row["speed"] == 1.0 and row["take"] == 0]
            if not candidates:
                continue
            row = candidates[0]
            source = root / row["path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("A pronunciation review waveform has changed since the audit.")
            filename = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_") + ".wav"
            target = directory / voice_name / filename
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                os.link(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("A cached pronunciation review clip does not match its source.")
            records.append({"voice": voice, "text": text, "label": row["label"], "partition": "train",
                            "source_id": row["id"], "sha256": row["sha256"], "path": str(target.relative_to(root))})
            lines.append(f"- [{text}]({voice_name}/{filename}) — {'positive' if row['label'] else 'negative'} label")
        lines.append("")
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return {"clips": len(records), "voices": voices, "partition": "train", "readme": str(directory / "README.md")}
