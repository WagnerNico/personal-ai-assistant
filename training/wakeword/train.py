"""Train a compact detector on balanced batches and select on validation only."""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import soundfile as sf
from scipy.special import expit

from input.wakeword_features import FEATURE_FRAMES, MEL_BANDS, MelFeatures
from .data import load_recipe
from .features import cache_features
from .network import ProbabilityNetwork, WakeNetwork


def positive_bounds(row, originals):
    count = originals[row["parent_id"]]["samples"]
    rng = np.random.default_rng(row["seed"])
    length = max(1, int(count * rng.uniform(0.9, min(1.1, 32000 / count))))
    offset = int(rng.integers(0, 32000 - length + 1))
    return offset / 16000, (offset + length) / 16000


def clean_examples(root, originals, rng):
    """Include complete clean positives and clean negative crops, using training voices."""
    positives = [row for row in originals.values() if row["split"] == "train" and row["label"] and row["samples"] <= 32000]
    negatives = [row for row in originals.values() if row["split"] == "train" and not row["label"]]
    negatives = [negatives[i] for i in rng.choice(len(negatives), min(256, len(negatives)), replace=False)]
    groups = []
    extractor = MelFeatures()
    for group in (positives, negatives):
        audio = []
        for row in group:
            samples, rate = sf.read(root / row["path"], dtype="float32")
            if rate != 16000:
                raise ValueError("Clean examples must use 16 kHz audio.")
            if len(samples) > 32000:
                start = int(rng.integers(0, len(samples) - 32000 + 1))
                samples = samples[start:start + 32000]
            output = np.zeros(32000, np.float32)
            offset = int(rng.integers(0, 32000 - len(samples) + 1))
            output[offset:offset + len(samples)] = samples
            audio.append(output)
        groups.append(extractor(np.stack(audio)))
    return groups


def predict(network, features, indices, device):
    predictions = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            batch = torch.from_numpy(np.asarray(features[indices[start:start + 128]])).to(device)
            predictions.extend(network(batch).detach().cpu().numpy().tolist())
    return np.asarray(predictions, np.float32)


def balanced_loss(logits, labels):
    losses = np.logaddexp(0, logits) - labels * logits
    return float((losses[labels == 1].mean() + losses[labels == 0].mean()) / 2)


def train_detector(root, *, steps=None, device="auto"):
    root = Path(root)
    recipe = load_recipe(root / "recipe.json")
    rows, features = cache_features(root)
    steps = recipe["steps"] if steps is None else steps
    if steps < 1:
        raise ValueError("Training steps must be positive.")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in ("cpu", "mps"):
        raise ValueError("Choose the CPU or Apple MPS training device.")
    torch.set_num_threads(4)
    torch.manual_seed(recipe["seed"])
    rng = np.random.default_rng(recipe["seed"])
    positive = np.array([i for i, row in enumerate(rows) if row["split"] == "train" and row["label"]], dtype=np.int64)
    negative = np.array([i for i, row in enumerate(rows) if row["split"] == "train" and not row["label"]], dtype=np.int64)
    validation = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"], dtype=np.int64)
    labels = np.array([row["label"] for row in rows], np.float32)
    if not len(positive) or not len(negative) or not len(validation) or len(set(labels[validation])) != 2:
        raise ValueError("Training and validation require positive and negative examples.")
    network = WakeNetwork().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=recipe["learning_rate"], weight_decay=1e-3)
    directory = root / "models"
    directory.mkdir(exist_ok=True)
    started, best_loss, history = perf_counter(), float("inf"), []
    best_state, best_step = None, 0
    batch_size = recipe["batch_size"]
    background_rng = np.random.default_rng(recipe["seed"] + 101)
    background_audio = (background_rng.standard_normal((128, 32000))
                        * 10 ** background_rng.uniform(-5, -1.5, (128, 1))).astype(np.float32)
    background_audio[:16] = 0
    background = MelFeatures()(background_audio)
    originals = {row["id"]: row for row in (
        json.loads(line) for line in (root / "corpus_manifest.jsonl").read_text().splitlines()
    )}
    clean_positive, clean_negative = clean_examples(root, originals, background_rng)
    for step in range(1, steps + 1):
        network.train()
        indices = np.concatenate((rng.choice(positive, batch_size // 2), rng.choice(negative, batch_size - batch_size // 2)))
        rng.shuffle(indices)
        training_features = np.asarray(features[indices]).copy()
        for position, index in enumerate(indices):
            if labels[index] == 1:
                if rng.random() < 0.3:
                    training_features[position] = clean_positive[rng.integers(len(clean_positive))]
                else:
                    onset, end = positive_bounds(rows[index], originals)
                    lower, upper = int(np.ceil((end - 1.98) * 100)), int(np.floor(onset * 100))
                    shift = int(rng.integers(lower, upper + 1)) if lower <= upper else 0
                    fill = background[rng.integers(len(background))]
                    if shift > 0:
                        training_features[position, :-shift] = features[index, shift:]
                        training_features[position, -shift:] = fill[-shift:]
                    elif shift < 0:
                        count = -shift
                        training_features[position, count:] = features[index, :-count]
                        training_features[position, :count] = fill[:count]
            else:
                draw = rng.random()
                if draw < 0.15:
                    training_features[position] = background[rng.integers(len(background))]
                elif draw < 0.35:
                    training_features[position] = clean_negative[rng.integers(len(clean_negative))]
                elif draw < 0.65:
                    boundary = int(rng.integers(16, FEATURE_FRAMES - 16))
                    other = int(rng.choice(negative))
                    training_features[position, :boundary] = features[other, -boundary:]
                    training_features[position, boundary:] = features[index, :FEATURE_FRAMES - boundary]
                elif draw < 0.85:
                    shift = int(rng.integers(8, FEATURE_FRAMES - 8))
                    training_features[position, :shift] = background[rng.integers(len(background)), :shift]
                    training_features[position, shift:] = features[index, :FEATURE_FRAMES - shift]
            # Frequency variation reduces dependence on a training speaker's timbre.
            shift = int(rng.integers(-2, 3))
            if shift:
                training_features[position] = np.roll(training_features[position], shift, axis=-1)
                if shift > 0:
                    training_features[position, :, :shift] = np.float32((np.log(np.float32(1e-8)) + 8) / 8)
                else:
                    training_features[position, :, shift:] = np.float32((np.log(np.float32(1e-8)) + 8) / 8)
        batch = torch.from_numpy(training_features).to(device)
        targets = torch.from_numpy(labels[indices]).to(device)
        rate = recipe["learning_rate"] * (0.05 + 0.95 * (1 + math.cos(math.pi * step / steps)) / 2)
        optimizer.param_groups[0]["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(network(batch), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            logits = predict(network, features, validation, device)
            score = balanced_loss(logits, labels[validation])
            if not np.isfinite(score):
                raise ValueError("Detector training produced a nonfinite validation loss.")
            history.append({"step": step, "training_loss": float(loss.detach().cpu()), "validation_balanced_loss": score})
            if score < best_loss:
                best_loss, best_step = score, step
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
            print(f"Training step {step}/{steps}: validation balanced loss {score:.4f}", flush=True)
    network = network.cpu().eval()
    network.load_state_dict(best_state)
    checkpoint = directory / "airis_dscnn.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save({"state_dict": best_state, "architecture": network.architecture, "channels": 24}, temporary)
    temporary.replace(checkpoint)
    probability = ProbabilityNetwork(network).eval()
    path = directory / "airis_dscnn.onnx"
    temporary = path.with_suffix(".partial.onnx")
    torch.onnx.export(probability, torch.zeros(1, FEATURE_FRAMES, MEL_BANDS), str(temporary),
                      input_names=["mel"], output_names=["wake_probability"],
                      dynamic_axes={"mel": {0: "batch"}, "wake_probability": {0: "batch"}},
                      opset_version=17, dynamo=False, external_data=False)
    temporary.replace(path)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    sample = np.asarray(features[validation[:32]])
    expected = expit(predict(network, features, validation[:32], "cpu"))
    actual = session.run(None, {"mel": sample})[0]
    parity = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(parity) or parity > 1e-4:
        raise ValueError(f"Exported model prediction mismatch: {parity}")
    metadata = {"architecture": network.architecture, "frontend": "airis-mel-v1", "sample_rate": 16000,
                "window_samples": 32000, "frame_samples": 1280, "threshold": 0.99,
                "threshold_calibrated": False, "ready_for_ui": False,
                "phrases": recipe["phrases"], "parameter_count": sum(p.numel() for p in network.parameters()),
                "pretrained_weights": False, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {"steps": steps, "best_step": best_step, "device": device, "seed": recipe["seed"],
              "best_validation_balanced_loss": best_loss, "history": history,
              "training_seconds": perf_counter() - started, "model_bytes": path.stat().st_size,
              "parameter_count": metadata["parameter_count"], "onnx_max_prediction_difference": parity,
              "test_used_for_training_or_selection": False,
              "training_strategy": "Balanced batches, frequency shifts, complete shifted positive windows, clean recordings, composite/partial negative windows, and procedural background.",
              "training_revision": 3,
              "feature_fingerprint": json.loads((root / "features" / "metadata.json").read_text())["fingerprint"]}
    (directory / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
