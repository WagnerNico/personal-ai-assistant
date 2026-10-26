"""Cache complete clean training utterances at their actual streaming endpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from input.google_wakeword_features import GoogleEmbedding
from .evaluate import read_rows


def cache_positive_features(root):
    root = Path(root)
    manifest = root / "corpus_manifest.jsonl"
    if not manifest.exists():
        manifest = root / "training_corpus_manifest.jsonl"
    rows = [row for row in read_rows(manifest) if row["split"] == "train" and row["label"] == 1]
    if not rows:
        raise ValueError("Clean positive features require positive training recordings.")
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()
                                 + (root / "google_base" / "conversion_report.json").read_bytes()
                                 + b"positive-stream-endpoints0-240ms-v1").hexdigest()
    directory = root / "features"
    directory.mkdir(exist_ok=True)
    path, metadata_path = directory / "positive_stream.npy", directory / "positive_stream.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] == fingerprint:
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.shape != (len(rows) * 4, 16, 96) or features.dtype != np.float32:
                raise ValueError("Invalid clean positive feature shape or dtype.")
            return features, metadata
    frontend = GoogleEmbedding(root / "google_base")
    temporary = path.with_suffix(".partial.npy")
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(rows) * 4, 16, 96))
    for index, row in enumerate(rows):
        audio_path = root / row["path"]
        if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"The clean training waveform has changed: {audio_path}")
        samples, rate = sf.read(audio_path, dtype="float32")
        if rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f"Invalid clean positive training waveform: {audio_path}")
        frontend.reset()
        samples = np.pad(samples, (0, 3840 + (-len(samples)) % 1280))
        for start in range(0, len(samples), 1280):
            context = frontend.push(samples[start:start + 1280])
            if start >= len(samples) - 5120:
                destination = index * 4 + (start - (len(samples) - 5120)) // 1280
                features[destination] = context
        if (index + 1) % 500 == 0:
            print(f"Clean positive streams: {index + 1}/{len(rows)} recordings", flush=True)
    features.flush()
    del features
    temporary.replace(path)
    metadata = {"fingerprint": fingerprint, "contexts": len(rows) * 4, "source_recordings": len(rows),
                "source_partition": "train", "held_out_audio_read": False,
                "records": [{"parent_id": row["id"], "voice": row["voice"], "text": row["text"]} for row in rows]}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return np.load(path, mmap_mode="r", allow_pickle=False), metadata
