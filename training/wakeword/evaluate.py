"""Calibrate on validation streams, then measure held-out detection and CPU latency."""

from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from input.wakeword import Trigger, WakeWordDetector
from input.wakeword_features import FRAME_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def target_bounds(row, originals):
    """Locate the complete dry utterance using the augmentation's deterministic seed."""
    if "parent_id" not in row:
        return 0.0, row["samples"] / SAMPLE_RATE
    count = originals[row["parent_id"]]["samples"]
    rng = np.random.default_rng(row["seed"])
    stretched = max(1, int(count * rng.uniform(0.9, min(1.1, WINDOW_SAMPLES / count))))
    offset = int(rng.integers(0, WINDOW_SAMPLES - stretched + 1))
    return offset / SAMPLE_RATE, (offset + stretched) / SAMPLE_RATE


def collect_scores(root, detector, split, augmented, originals):
    root = Path(root)
    directory = root / "models" / "evaluation"
    directory.mkdir(exist_ok=True)
    selected = [row for row in augmented if row["split"] == split]
    positives = [row for row in selected if row["label"]]
    positives += [row for row in originals.values() if row["split"] == split and row["label"]]
    negatives = [row for row in selected if not row["label"]]
    path = directory / f"{split}_scores.npz"
    meta_path = directory / f"{split}_scores.json"
    model_metadata = json.loads(detector.model_path.with_suffix(".json").read_text())
    frontend_metadata = {key: model_metadata.get(key) for key in
                         ("frontend", "sample_rate", "window_samples", "frame_samples", "embedding_hashes")}
    fingerprint = hashlib.sha256(
        detector.model_path.read_bytes()
        + json.dumps(frontend_metadata, sort_keys=True).encode()
        + json.dumps(selected + positives, sort_keys=True).encode() + b"stream-v2-grace480ms"
    ).hexdigest()
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] == fingerprint:
            saved = np.load(path, allow_pickle=False)
            lookup = {row["id"]: row for row in positives}
            for row in meta["positive_records"]:
                row["target_start_seconds"] = target_bounds(lookup[row["id"]], originals)[0]
            return saved["negative"], meta["positive_records"]
    detector.reset()
    scores = array("f")
    started = perf_counter()
    for index, row in enumerate(negatives):
        samples, rate = sf.read(root / row["path"], dtype="float32")
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("A negative evaluation waveform has changed since the audit.")
        if rate != SAMPLE_RATE:
            raise ValueError("Streaming evaluation requires 16 kHz audio.")
        scores.extend(frame.score for frame in detector.feed(samples))
        if (index + 1) % 1000 == 0:
            print(f"{split.capitalize()} negative stream: {index + 1}/{len(negatives)} clips", flush=True)
    negative_scores = np.asarray(scores, np.float32)
    records = []
    for row in positives:
        detector.reset()
        samples, rate = sf.read(root / row["path"], dtype="float32")
        if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("A positive evaluation waveform has changed since the audit.")
        if rate != SAMPLE_RATE:
            raise ValueError("Positive evaluation requires 16 kHz audio.")
        samples = np.pad(samples, (0, 7680))
        samples = np.pad(samples, (0, (-len(samples)) % FRAME_SAMPLES))
        frames = detector.feed(samples)
        records.append({"id": row["id"], "voice": row["voice"], "text": row["text"],
                        "kind": "augmented" if "parent_id" in row else "original",
                        "target_start_seconds": target_bounds(row, originals)[0],
                        "target_end_seconds": target_bounds(row, originals)[1],
                        "scores": [frame.score for frame in frames]})
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, negative=negative_scores)
    temporary.replace(path)
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "positive_records": records,
                                    "scoring_seconds": perf_counter() - started}, indent=2) + "\n")
    return negative_scores, records


def negative_events(scores, threshold):
    scores = np.asarray(scores)
    above = scores >= threshold
    # Only the second frame of an above-threshold run can fire an armed trigger.
    candidates = np.flatnonzero(above[1:] & above[:-1] & ~np.r_[False, above[:-2]]) + 1
    quiet = np.flatnonzero(scores < threshold * 0.5)
    count, candidate_position = 0, 0
    while candidate_position < len(candidates):
        fired = int(candidates[candidate_position])
        count += 1
        # The default 1.5-second cooldown needs nineteen 80 ms frame intervals.
        quiet_position = np.searchsorted(quiet, fired + 19)
        if quiet_position == len(quiet):
            break
        candidate_position = int(np.searchsorted(candidates, quiet[quiet_position], side="right"))
    hours = len(scores) * 0.08 / 3600
    return {"episodes": count, "hours": hours, "episodes_per_hour": count / hours if hours else None,
            "above_threshold_fraction": float(np.mean(above)) if len(scores) else None}


def positive_metrics(records, threshold):
    total, detected = Counter(), Counter()
    kinds, detected_kinds = Counter(), Counter()
    delays, starts, failures, early_activations = [], [], [], 0
    for row in records:
        trigger = Trigger(threshold)
        activation = None
        for index, score in enumerate(row["scores"]):
            when = (index + 1) * 0.08
            if trigger.update(score, when):
                if when >= row["target_start_seconds"]:
                    activation = when
                    break
                early_activations += 1
        total[row["text"]] += 1
        kinds[row["kind"]] += 1
        if activation is not None:
            detected[row["text"]] += 1
            detected_kinds[row["kind"]] += 1
            delays.append(activation - row["target_end_seconds"])
            starts.append(activation)
        else:
            failures.append(row["id"])
    return {
        "examples": sum(total.values()), "detected": sum(detected.values()),
        "recall": sum(detected.values()) / sum(total.values()) if total else None,
        "by_phrase": {phrase: {"examples": count, "detected": detected[phrase], "recall": detected[phrase] / count}
                      for phrase, count in total.items()},
        "by_kind": {kind: {"examples": count, "detected": detected_kinds[kind], "recall": detected_kinds[kind] / count}
                    for kind, count in kinds.items()},
        "activation_from_clip_start_p50_seconds": float(np.median(starts)) if starts else None,
        "delay_from_dry_waveform_end_p50_seconds": float(np.median(delays)) if delays else None,
        "delay_from_dry_waveform_end_p95_seconds": float(np.percentile(delays, 95)) if delays else None,
        "missed_example_ids": failures,
        "activations_before_target": early_activations,
    }


def calibrate(negative, positives, maximum_false_activations, maximum_negative_score_fraction=0.001):
    from scipy.special import expit
    if len(negative) < 2 or not positives:
        raise ValueError("Threshold calibration requires positive examples and a negative audio stream.")
    if not 0 <= maximum_negative_score_fraction < 1:
        raise ValueError("The negative score fraction limit must be between zero and one.")
    consecutive_peak = float(np.minimum(negative[:-1], negative[1:]).max()) if len(negative) > 1 else 0
    paired = np.minimum(negative[:-1], negative[1:])
    peaks = paired[1:-1][(paired[1:-1] >= paired[:-2]) & (paired[1:-1] >= paired[2:])]
    upper_peaks = np.unique(peaks[peaks > 0.5])[-100:]
    candidates = sorted(set([float(x) for x in expit(np.linspace(-3, 16, 90))]
                            + [min(1.0, float(value) + 1e-6) for value in upper_peaks]
                            + [min(1.0, consecutive_peak + 1e-6), 1.0]))
    measurements = []
    for threshold in candidates:
        negatives = negative_events(negative, threshold)
        positives_score = positive_metrics(positives, threshold)
        measurements.append({"threshold": threshold, "false_activations_per_hour": negatives["episodes_per_hour"],
                             "negative_episodes": negatives["episodes"], "recall": positives_score["recall"],
                             "negative_above_threshold_fraction": negatives["above_threshold_fraction"],
                             "minimum_phrase_recall": min(row["recall"] for row in positives_score["by_phrase"].values())})
    eligible = [point for point in measurements if point["false_activations_per_hour"] <= maximum_false_activations
                and point["negative_above_threshold_fraction"] <= maximum_negative_score_fraction]
    if eligible:
        selected = max(eligible, key=lambda point: (point["minimum_phrase_recall"], point["recall"],
                                                  -point["false_activations_per_hour"], point["threshold"]))
    else:
        selected = min(measurements, key=lambda point: (point["negative_above_threshold_fraction"],
                                                       point["false_activations_per_hour"], -point["recall"]))
    return {**selected, "constraints_satisfied": bool(eligible)}, measurements


def benchmark(model_path):
    detector = WakeWordDetector(model_path)
    rng = np.random.default_rng(123)
    frames = (rng.standard_normal((1100, FRAME_SAMPLES)) * 0.02).astype(np.float32)
    durations = []
    for index, samples in enumerate(frames):
        started = perf_counter()
        detector.feed(samples)
        elapsed = (perf_counter() - started) * 1000
        if index >= 100:
            durations.append(elapsed)
    return {"audio_frame_ms": 80, "measured_frames": len(durations), "cpu_threads": 1,
            "p50_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
            "p99_ms": float(np.percentile(durations, 99)),
            "real_time_factor_p50": float(np.median(durations) / 80),
            "includes": "Mel update, ONNX inference, audio buffering, and trigger logic; excludes microphone I/O."}


def calibrate_detector(root, *, model_path=None):
    root = Path(root)
    augmented = read_rows(root / "augmentation_manifest.jsonl")
    originals = {row["id"]: row for row in read_rows(root / "corpus_manifest.jsonl")}
    path = Path(model_path) if model_path else root / "models" / "airis_wakeword.onnx"
    if model_path is None and not path.exists():
        path = root / "models" / "airis_dscnn.onnx"
    if not path.exists():
        raise ValueError("Train the selected detector backend before threshold calibration.")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    detector = WakeWordDetector(path)
    validation_negative, validation_positive = collect_scores(root, detector, "validation", augmented, originals)
    recipe = json.loads((root / "recipe.json").read_text())
    selected, calibration = calibrate(validation_negative, validation_positive, recipe["max_validation_false_activations_per_hour"],
                                      recipe.get("max_validation_negative_score_fraction", 0.001))
    threshold = selected["threshold"]
    metadata.update({"threshold": threshold, "threshold_calibrated": True, "ready_for_ui": False,
                     "calibration_constraints_satisfied": selected["constraints_satisfied"]})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Validation selected threshold {threshold:.6f}, recall {selected['recall']:.3f}", flush=True)
    report = {"threshold": threshold, "threshold_selected_on": "validation", "calibration": calibration,
              "calibration_constraints_satisfied": selected["constraints_satisfied"],
              "validation": {"positive": positive_metrics(validation_positive, threshold),
                             "negative_stream": negative_events(validation_negative, threshold)}}
    (root / "models" / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate_detector(root, *, model_path=None):
    root = Path(root)
    calibrated = calibrate_detector(root, model_path=model_path)
    augmented = read_rows(root / "augmentation_manifest.jsonl")
    originals = {row["id"]: row for row in read_rows(root / "corpus_manifest.jsonl")}
    path = Path(model_path) if model_path else root / "models" / "airis_wakeword.onnx"
    if model_path is None and not path.exists():
        path = root / "models" / "airis_dscnn.onnx"
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    detector = WakeWordDetector(path)
    recipe = json.loads((root / "recipe.json").read_text())
    threshold = calibrated["threshold"]
    test_negative, test_positive = collect_scores(root, detector, "test", augmented, originals)
    latency = benchmark(path)
    test_positives = positive_metrics(test_positive, threshold)
    test_negatives = negative_events(test_negative, threshold)
    passed = (calibrated["calibration_constraints_satisfied"]
              and all(row["recall"] >= 0.9 for row in test_positives["by_phrase"].values())
              and test_negatives["episodes_per_hour"] <= recipe["max_validation_false_activations_per_hour"]
              and test_negatives["above_threshold_fraction"] <= recipe.get("max_validation_negative_score_fraction", 0.001)
              and latency["p95_ms"] < 80)
    report = {"model": metadata["architecture"], "pretrained_weights": metadata["pretrained_weights"], "threshold": threshold,
              "threshold_selected_on": "validation", "calibration": calibrated["calibration"],
              "calibration_constraints_satisfied": calibrated["calibration_constraints_satisfied"],
              "validation": calibrated["validation"],
              "test": {"positive": test_positives, "negative_stream": test_negatives},
              "cpu_latency": latency, "synthetic_acceptance_passed": passed, "ready_for_ui": False,
              "limits": "Synthetic development benchmark only. Negative streams concatenate augmented negative clips. "
                        "Positive clips receive 480 ms silence plus frame alignment after the clip. "
                        "Delay is measured from the complete dry waveform end, which can include silence. "
                        "No live microphone accuracy, battery use, or concurrent LLM/TTS load has been measured."}
    metadata["synthetic_acceptance_passed"] = passed
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "models" / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if recipe.get("iteration", 1) >= 2 and metadata["frontend"] == "google-speech-embedding-v1":
        from .model_card import write_iteration_card
        write_iteration_card(root, report)
    return report
