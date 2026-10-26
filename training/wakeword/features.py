"""Cache verified mel features once, with dataset and frontend fingerprints."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from input.wakeword_features import FEATURE_FRAMES, MEL_BANDS, MelFeatures, WINDOW_SAMPLES


def cache_features(root):
    root = Path(root)
    manifest = root / "augmentation_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Prepare an augmented corpus before feature extraction.")
    owners = {}
    for row in rows:
        owners.setdefault(row["voice"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in owners.values()):
        raise ValueError("Feature extraction requires disjoint speaker partitions.")
    fingerprint = hashlib.sha256(manifest.read_bytes() + b"airis-mel-v1-float32").hexdigest()
    directory = root / "features"
    directory.mkdir(exist_ok=True)
    metadata_path = directory / "metadata.json"
    destination = directory / "mel.npy"
    shape = (len(rows), FEATURE_FRAMES, MEL_BANDS)
    if metadata_path.exists() and destination.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] == fingerprint:
            cached = np.load(destination, mmap_mode="r")
            if cached.shape != shape or cached.dtype != np.float32:
                raise ValueError("Invalid cached feature shape or dtype.")
            return rows, cached
    temporary = directory / "mel.partial.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    extractor, started = MelFeatures(), perf_counter()
    for start in range(0, len(rows), 64):
        group = rows[start:start + 64]
        audio = []
        for row in group:
            path = root / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"The augmented waveform has changed: {path}")
            samples, rate = sf.read(path, dtype="float32")
            if rate != 16000 or samples.shape != (WINDOW_SAMPLES,) or not np.isfinite(samples).all():
                raise ValueError(f"Invalid waveform for feature extraction: {path}")
            audio.append(samples)
        features[start:start + len(group)] = extractor(np.stack(audio))
        if start % 2048 == 0:
            print(f"Cached features: {start + len(group)}/{len(rows)}", flush=True)
    features.flush()
    del features
    temporary.replace(destination)
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "frontend": "airis-mel-v1",
                                        "shape": shape, "elapsed_seconds": perf_counter() - started}, indent=2) + "\n")
    return rows, np.load(destination, mmap_mode="r")
