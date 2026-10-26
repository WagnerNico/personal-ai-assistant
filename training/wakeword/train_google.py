"""Train a new Airis classifier on the original Google speech embedding base."""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from scipy.special import expit

from input.google_wakeword_features import GoogleEmbedding
from .data import load_recipe
from .google_features import cache_google_features
from .train import balanced_loss, predict, positive_bounds
from .mining import load_mined_negatives


class GoogleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 96, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class GoogleProbability(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, features):
        return torch.sigmoid(self.network(features))


def train_google_detector(root, *, steps=None, device="auto"):
    root = Path(root)
    recipe = load_recipe(root / "recipe.json")
    rows, features = cache_google_features(root)
    steps = recipe["steps"] if steps is None else steps
    if steps < 1:
        raise ValueError("Training steps must be positive.")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(recipe["seed"])
    rng = np.random.default_rng(recipe["seed"])
    indices = {split: np.array([i for i, row in enumerate(rows) if row["split"] == split], dtype=np.int64)
               for split in ("train", "validation", "test")}
    labels = np.array([row["label"] for row in rows], np.float32)
    positives = indices["train"][labels[indices["train"]] == 1]
    negatives = indices["train"][labels[indices["train"]] == 0]
    if not len(positives) or not len(negatives) or len(set(labels[indices["validation"]])) != 2:
        raise ValueError("Training and validation require both positive and negative recordings.")
    originals = {row["id"]: row for row in (
        json.loads(line) for line in (root / "corpus_manifest.jsonl").read_text().splitlines()
    )}
    zero = GoogleEmbedding(root / "google_base").zero_embedding
    mined, mining_reports = load_mined_negatives(root, recipe["voices"]["train"])
    clean_positives = np.empty((0, 16, 96), np.float32)
    if recipe.get("clean_positive_streams", False):
        from .positive_features import cache_positive_features
        clean_positives, _ = cache_positive_features(root)
    streaming_selection = recipe.get("streaming_checkpoint_selection", False)
    if streaming_selection:
        from .validation_features import cache_validation_features, score_validation_stream
        validation_features, validation_metadata = cache_validation_features(root)
    network = GoogleClassifier().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=recipe["learning_rate"], weight_decay=0.01)
    history, best_loss, best_state, best_step, best_rank = [], float("inf"), None, 0, None
    started = perf_counter()
    batch_size = recipe["batch_size"]
    for step in range(1, steps + 1):
        network.train()
        batch_indices = np.concatenate((rng.choice(positives, batch_size // 2),
                                        rng.choice(negatives, batch_size - batch_size // 2)))
        rng.shuffle(batch_indices)
        batch = np.asarray(features[batch_indices]).copy()
        for position, index in enumerate(batch_indices):
            if labels[index]:
                onset, end = positive_bounds(rows[index], originals)
                lower, upper = int(np.ceil((end - 1.95) / 0.08)), int(np.floor(onset / 0.08))
                shift = int(rng.integers(lower, upper + 1)) if lower <= upper else 0
                if shift > 0:
                    batch[position, :-shift] = features[index, shift:]
                    batch[position, -shift:] = zero
                elif shift < 0:
                    batch[position, -shift:] = features[index, :shift]
                    batch[position, :-shift] = zero
            elif rng.random() < 0.15:
                batch[position] = zero
            elif rng.random() < 0.3:
                boundary = int(rng.integers(2, 14))
                other = int(rng.choice(negatives))
                batch[position, :boundary] = features[other, -boundary:]
        if len(mined):
            positions = np.flatnonzero(labels[batch_indices] == 0)
            positions = rng.choice(positions, max(1, len(positions) // 3), replace=False)
            batch[positions] = mined[rng.integers(len(mined), size=len(positions))]
        if len(clean_positives):
            positions = np.flatnonzero(labels[batch_indices] == 1)
            positions = rng.choice(positions, max(1, len(positions) // 3), replace=False)
            batch[positions] = clean_positives[rng.integers(len(clean_positives), size=len(positions))]
        optimizer.param_groups[0]["lr"] = recipe["learning_rate"] * (0.05 + 0.95 * (1 + math.cos(math.pi * step / steps)) / 2)
        inputs = torch.from_numpy(batch).to(device)
        targets = torch.from_numpy(labels[batch_indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(network(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
        interval = 500 if streaming_selection else 250
        if step == 1 or step % interval == 0 or step == steps:
            logits = predict(network, features, indices["validation"], device)
            score = balanced_loss(logits, labels[indices["validation"]])
            if not np.isfinite(score):
                raise ValueError("Classifier training produced a nonfinite validation loss.")
            point = {"step": step, "training_loss": float(loss.detach().cpu()), "validation_balanced_loss": score}
            rank = (-score,)
            if streaming_selection:
                stream = score_validation_stream(network, validation_features, validation_metadata, device,
                                                 recipe["max_validation_false_activations_per_hour"],
                                                 recipe.get("max_validation_negative_score_fraction", 0.001))
                point["streaming_validation"] = stream
                rank = (stream["constraints_satisfied"], stream["minimum_phrase_recall"], stream["recall"],
                        -stream["false_activations_per_hour"], -score)
            history.append(point)
            if best_rank is None or rank > best_rank:
                best_loss, best_step = score, step
                best_rank = rank
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
            print(f"Google classifier step {step}/{steps}: validation balanced loss {score:.4f}", flush=True)
            if streaming_selection:
                print(f"Streaming validation: minimum phrase recall {stream['minimum_phrase_recall']:.3f}, "
                      f"false activations/hour {stream['false_activations_per_hour']:.3f}", flush=True)
    network = network.cpu().eval()
    network.load_state_dict(best_state)
    directory = root / "models"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "airis_wakeword.pt"
    torch.save({"architecture": "airis-google-dnn-v1", "state_dict": best_state}, checkpoint)
    path = directory / "airis_wakeword.onnx"
    temporary = path.with_suffix(".partial.onnx")
    torch.onnx.export(GoogleProbability(network), torch.zeros(1, 16, 96), str(temporary),
                      input_names=["embedding"], output_names=["wake_probability"],
                      dynamic_axes={"embedding": {0: "batch"}, "wake_probability": {0: "batch"}},
                      opset_version=17, dynamo=False, external_data=False)
    temporary.replace(path)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    subset = indices["validation"][:32]
    expected = expit(predict(network, features, subset, "cpu"))
    actual = session.run(None, {"embedding": np.asarray(features[subset])})[0]
    difference = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(difference) or difference > 1e-4:
        raise ValueError(f"Google classifier ONNX export parity failed: {difference}")
    conversion = json.loads((root / "google_base" / "conversion_report.json").read_text())
    metadata = {"architecture": "airis-google-dnn-v1", "frontend": "google-speech-embedding-v1", "sample_rate": 16000,
                "window_samples": 32000, "frame_samples": 1280, "threshold": 0.99, "threshold_calibrated": False,
                "ready_for_ui": False, "phrases": recipe["phrases"], "pretrained_weights": True,
                "classifier_pretrained": False, "embedding_directory": "../google_base",
                "embedding_hashes": {"google_speech_embedding.onnx": conversion["onnx_sha256"],
                                     "frontend_assets.npz": conversion["frontend_assets_sha256"]},
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {"steps": steps, "best_step": best_step, "device": device, "seed": recipe["seed"],
              "best_validation_balanced_loss": best_loss, "history": history,
              "selected_validation_balanced_loss": best_loss,
              "minimum_validation_balanced_loss": min(point["validation_balanced_loss"] for point in history),
              "training_seconds": perf_counter() - started, "classifier_bytes": path.stat().st_size,
              "classifier_parameters": sum(parameter.numel() for parameter in network.parameters()),
              "onnx_max_prediction_difference": difference, "base_conversion": conversion,
              "test_used_for_training_or_selection": False,
              "checkpoint_selection": "streaming_validation_minimum_phrase_recall" if streaming_selection else "validation_balanced_loss",
              "mined_negative_contexts": len(mined), "mining_pools": mining_reports,
              "clean_positive_contexts": len(clean_positives),
              "feature_fingerprint": json.loads((root / "features" / "google_metadata.json").read_text())["fingerprint"]}
    (directory / "google_training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
