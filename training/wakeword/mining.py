"""Collect difficult and uniformly sampled negative contexts from training speech."""

from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from input.wakeword import WakeWordDetector
from .data import load_recipe
from .evaluate import read_rows


def training_negatives(rows, allowed_voices):
    """Filter before reading audio; held-out partitions never enter a mining pool."""
    selected = [row for row in rows if row["split"] == "train" and row["label"] == 0]
    if not selected or any(row["voice"] not in allowed_voices for row in selected):
        raise ValueError("Negative mining requires training speakers from the target recipe.")
    return selected


def mine_negatives(root, source_root, model_path, *, limit=4096, uniform_limit=1024):
    root, source_root, model_path = Path(root), Path(source_root), Path(model_path)
    if limit < 1 or uniform_limit < 1:
        raise ValueError("Negative mining limits must be positive.")
    recipe = load_recipe(root / "recipe.json")
    manifest = source_root / "corpus_manifest.jsonl"
    if not manifest.exists():
        manifest = source_root / "training_corpus_manifest.jsonl"
    rows = training_negatives(read_rows(manifest), recipe["voices"]["train"])
    model_metadata = json.loads(model_path.with_suffix(".json").read_text())
    if model_metadata["frontend"] != "google-speech-embedding-v1":
        raise ValueError("Negative embedding mining requires the Google speech base.")
    for filename, expected in model_metadata["embedding_hashes"].items():
        if hashlib.sha256((root / "google_base" / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("The mining model and target corpus must share the same speech base.")
    identity = {"rows": rows, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "embedding_hashes": model_metadata["embedding_hashes"], "limit": limit,
                "uniform_limit": uniform_limit, "seed": recipe["seed"], "version": "continuous-negative-v1"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root / "mining"
    directory.mkdir(exist_ok=True)
    metadata_path = directory / f"{fingerprint[:24]}.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    detector = WakeWordDetector(model_path)
    rng = np.random.default_rng(recipe["seed"])
    order = rng.permutation(len(rows))
    hard, uniform, seen = [], [], 0
    started = perf_counter()
    for position, row_index in enumerate(order):
        row = rows[int(row_index)]
        path = source_root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"The training waveform has changed: {path}")
        samples, rate = sf.read(path, dtype="float32")
        if rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f"Invalid training waveform for mining: {path}")
        # Preserve the real streaming frontend across clips, including boundaries.
        local_candidates = []
        for start in range(0, len(samples), 1280):
            frames = detector.feed(samples[start:start + 1280])
            if not frames or detector.samples_processed % 5120:
                continue
            score = frames[-1].score
            record = {"parent_id": row["id"], "voice": row["voice"], "split": "train", "label": 0,
                      "time_seconds": frames[-1].time_seconds, "score": score}
            features = detector.frontend.features.copy()
            local_candidates.append((score, seen, features, record))
            seen += 1
            if len(uniform) < uniform_limit:
                uniform.append((features, record))
            else:
                replace = int(rng.integers(seen))
                if replace < uniform_limit:
                    uniform[replace] = (features, record)
        # A single utterance cannot occupy the whole hard-negative pool.
        for candidate in sorted(local_candidates, key=lambda item: item[0], reverse=True)[:4]:
            if len(hard) < limit:
                heapq.heappush(hard, candidate)
            elif candidate[:2] > hard[0][:2]:
                heapq.heapreplace(hard, candidate)
        if (position + 1) % 250 == 0:
            print(f"Training negative mining: {position + 1}/{len(rows)} recordings", flush=True)
    if not hard or not uniform:
        raise ValueError("The training speech did not provide enough streaming contexts.")
    chosen = [(item[2], {**item[3], "sampling": "hard"}) for item in sorted(hard, key=lambda item: item[1])]
    chosen += [(features, {**record, "sampling": "uniform"}) for features, record in uniform]
    feature_path = directory / f"{fingerprint[:24]}.npy"
    temporary = feature_path.with_suffix(".partial.npy")
    np.save(temporary, np.stack([features for features, _ in chosen]))
    temporary.replace(feature_path)
    records_path = feature_path.with_suffix(".jsonl")
    records_path.write_text("".join(json.dumps(record) + "\n" for _, record in chosen))
    report = {"fingerprint": fingerprint, "source_root": str(source_root),
              "source_partition": "train", "source_recordings": len(rows),
              "source_manifest_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              "source_model_sha256": identity["model_sha256"], "embedding_hashes": identity["embedding_hashes"],
              "features": str(feature_path.relative_to(root)), "records": str(records_path.relative_to(root)),
              "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
              "contexts": len(chosen), "hard_contexts": len(hard), "uniform_contexts": len(uniform),
              "by_voice": dict(Counter(record["voice"] for _, record in chosen)),
              "stream_hours": detector.samples_processed / 16000 / 3600,
              "elapsed_seconds": perf_counter() - started,
              "held_out_audio_read": False}
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_mined_negatives(root, allowed_voices):
    """Validate provenance and hashes before training on an optional mining pool."""
    root = Path(root)
    pools, reports = [], []
    for path in sorted((root / "mining").glob("*.json")):
        report = json.loads(path.read_text())
        if report["source_partition"] != "train" or not set(report["by_voice"]).issubset(allowed_voices):
            raise ValueError("A negative mining pool contains a held-out speaker or partition.")
        for filename, expected in report["embedding_hashes"].items():
            if hashlib.sha256((root / "google_base" / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("The negative mining pool uses a different speech base.")
        features_path, records_path = root / report["features"], root / report["records"]
        for artifact, key in ((features_path, "features_sha256"), (records_path, "records_sha256")):
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != report[key]:
                raise ValueError("A cached negative mining artifact has changed.")
        records = read_rows(records_path)
        if any(row["split"] != "train" or row["label"] != 0 or row["voice"] not in allowed_voices for row in records):
            raise ValueError("Negative mining records must contain only training negatives.")
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(records), 16, 96) or not np.isfinite(features).all():
            raise ValueError("Invalid mined negative embedding shape or values.")
        pools.append(features)
        reports.append(report)
    return (np.concatenate(pools) if pools else np.empty((0, 16, 96), np.float32)), reports
