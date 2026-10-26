"""Cache original Google embeddings with waveform and model fingerprints."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from input.google_wakeword_features import GoogleEmbedding


def cache_google_features(root):
    root = Path(root)
    manifest = root / "augmentation_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    owners = {}
    for row in rows:
        owners.setdefault(row["voice"], set()).add(row["split"])
    if any(len(values) != 1 for values in owners.values()):
        raise ValueError("Google feature extraction requires disjoint speaker partitions.")
    report = root / "google_base" / "conversion_report.json"
    fingerprint = hashlib.sha256(manifest.read_bytes() + report.read_bytes() + b"google-context16-v1").hexdigest()
    directory = root / "features"
    directory.mkdir(exist_ok=True)
    destination, metadata_path = directory / "google.npy", directory / "google_metadata.json"
    shape = (len(rows), 16, 96)
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] == fingerprint:
            features = np.load(destination, mmap_mode="r")
            if features.shape != shape or features.dtype != np.float32:
                raise ValueError("Invalid cached Google feature shape or dtype.")
            return rows, features
    frontend = GoogleEmbedding(root / "google_base", threads=4)
    temporary = directory / "google.partial.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    started = perf_counter()
    for start in range(0, len(rows), 8):
        audio = []
        for row in rows[start:start + 8]:
            path = root / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"The waveform has changed: {path}")
            samples, rate = sf.read(path, dtype="float32")
            if rate != 16000 or samples.shape != (32000,) or not np.isfinite(samples).all():
                raise ValueError(f"Invalid waveform for Google features: {path}")
            audio.append(samples)
        features[start:start + len(audio)] = frontend.clips(np.stack(audio))
        if start % 512 == 0:
            print(f"Google features: {start + len(audio)}/{len(rows)}", flush=True)
    features.flush()
    del features
    temporary.replace(destination)
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "shape": shape,
                                        "elapsed_seconds": perf_counter() - started,
                                        "base_conversion": json.loads(report.read_text())}, indent=2) + "\n")
    return rows, np.load(destination, mmap_mode="r")
