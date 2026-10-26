"""Cache exact streaming validation contexts for classifier checkpoint selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from input.google_wakeword_features import GoogleEmbedding
from .evaluate import read_rows, target_bounds


def cache_validation_features(root):
    """Read validation audio only; reuse the frozen base across classifier candidates."""
    root = Path(root)
    augmented = read_rows(root / "augmentation_manifest.jsonl")
    originals = {row["id"]: row for row in read_rows(root / "corpus_manifest.jsonl")}
    selected = [row for row in augmented if row["split"] == "validation"]
    negatives = [row for row in selected if row["label"] == 0]
    positives = [row for row in selected if row["label"] == 1]
    positives += [row for row in originals.values() if row["split"] == "validation" and row["label"] == 1]
    conversion = (root / "google_base" / "conversion_report.json").read_bytes()
    fingerprint = hashlib.sha256(json.dumps(negatives + positives, sort_keys=True).encode()
                                 + conversion + b"validation-stream-grace480ms-v1").hexdigest()
    directory = root / "features"
    directory.mkdir(exist_ok=True)
    path, metadata_path = directory / "validation_stream.npy", directory / "validation_stream.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] == fingerprint:
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.shape != tuple(metadata["shape"]) or features.dtype != np.float32:
                raise ValueError("Invalid cached streaming validation feature shape or dtype.")
            return features, metadata
    negative_frames = len(negatives) * 25
    counts = [(originals[row["id"]]["samples"] if "parent_id" not in row else 32000) for row in positives]
    counts = [(samples + 7680 + 1279) // 1280 for samples in counts]
    shape = (negative_frames + sum(counts), 16, 96)
    temporary = path.with_suffix(".partial.npy")
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    frontend = GoogleEmbedding(root / "google_base")
    offset, records = 0, []
    for index, row in enumerate(negatives + positives):
        audio_path = root / row["path"]
        if hashlib.sha256(audio_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"The validation waveform has changed: {audio_path}")
        samples, rate = sf.read(audio_path, dtype="float32")
        if rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f"Invalid streaming validation waveform: {audio_path}")
        start = offset
        if row["label"]:
            frontend.reset()
            samples = np.pad(samples, (0, 7680))
            samples = np.pad(samples, (0, (-len(samples)) % 1280))
        elif len(samples) != 32000:
            raise ValueError("Streaming validation negatives must be two-second variants.")
        for frame_start in range(0, len(samples), 1280):
            features[offset] = frontend.push(samples[frame_start:frame_start + 1280])
            offset += 1
        if row["label"]:
            bounds = target_bounds(row, originals)
            records.append({"id": row["id"], "voice": row["voice"], "text": row["text"],
                            "kind": "augmented" if "parent_id" in row else "original",
                            "target_start_seconds": bounds[0], "target_end_seconds": bounds[1],
                            "feature_start": start, "feature_end": offset})
        if (index + 1) % 500 == 0:
            print(f"Streaming validation features: {index + 1}/{len(negatives) + len(positives)} clips", flush=True)
    if offset != shape[0]:
        raise ValueError("Streaming validation feature counts do not match the manifest.")
    features.flush()
    del features
    temporary.replace(path)
    metadata = {"fingerprint": fingerprint, "shape": shape, "negative_frames": negative_frames,
                "positive_records": records, "partition": "validation", "test_audio_read": False}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return np.load(path, mmap_mode="r", allow_pickle=False), metadata


def score_validation_stream(network, features, metadata, device, maximum_false_activations,
                            maximum_negative_score_fraction=0.001):
    from scipy.special import expit
    from .evaluate import calibrate, positive_metrics
    from .train import predict

    scores = expit(predict(network, features, np.arange(len(features)), device))
    records = [{**row, "scores": scores[row["feature_start"]:row["feature_end"]].tolist()}
               for row in metadata["positive_records"]]
    selected, _ = calibrate(scores[:metadata["negative_frames"]], records, maximum_false_activations,
                           maximum_negative_score_fraction)
    metrics = positive_metrics(records, selected["threshold"])
    return {**selected, "minimum_phrase_recall": min(row["recall"] for row in metrics["by_phrase"].values()),
            "by_phrase": metrics["by_phrase"]}
