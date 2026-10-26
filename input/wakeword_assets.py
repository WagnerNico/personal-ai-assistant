"""Install only verified inference artifacts, independently of training data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parent / "models" / "wakeword"


def install_wake_model(source, destination=DEFAULT_MODEL_DIRECTORY):
    """Copy a classifier and frozen frontend without changing acceptance flags.

    Quit Airis before replacing an installed model. Training corpora, feature
    caches, PyTorch checkpoints, and TensorFlow conversions are not copied.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    config = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if config.get("frontend") != "google-speech-embedding-v1":
        raise ValueError("Only the Google speech embedding wake frontend can be installed.")
    directory = source.parent / config["embedding_directory"]
    files = [(source, Path("airis_wakeword.onnx"), config["sha256"])]
    for filename, expected in config["embedding_hashes"].items():
        if Path(filename).name != filename:
            raise ValueError("Embedding filenames must not contain directory components.")
        files.append((directory / filename, Path("google_base") / filename, expected))
    if set(config["embedding_hashes"]) != {"google_speech_embedding.onnx", "frontend_assets.npz"}:
        raise ValueError("The wake model does not declare the required embedding artifacts.")
    for path, _, expected in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Wake artifact hash mismatch: {path.name}")
    config["embedding_directory"] = "google_base"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wake-install-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for path, relative, _ in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        metadata = staging / "airis_wakeword.json"
        metadata.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        destination.mkdir(parents=True, exist_ok=True)
        for _, relative, _ in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)
        os.replace(metadata, destination / metadata.name)
    return destination / "airis_wakeword.onnx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trained ONNX classifier with JSON sidecar.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    arguments = parser.parse_args()
    print(install_wake_model(arguments.source, arguments.destination))


if __name__ == "__main__":
    main()
