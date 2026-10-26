"""Check evaluation isolation, mining integrity, and explicit microphone cleanup."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.signal import resample_poly

from input.audio import RecordedAudio
from training.wakeword.mining import load_mined_negatives


class TrainingSafetyTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "Optional ONNX runtime is not installed.")
    def test_validation_cache_matches_runtime_and_never_reads_test_audio(self):
        import soundfile as sf
        from input.google_wakeword_features import GoogleEmbedding
        from training.wakeword.data import augment
        from training.wakeword.validation_features import cache_validation_features
        from training.wakeword.positive_features import cache_positive_features

        rng = np.random.default_rng(14)
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="mel")]
        session.run.side_effect = lambda _, inputs: [np.tile(inputs["mel"].mean(axis=1), (1, 3))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "google_base").mkdir()
            np.savez(root / "google_base" / "frontend_assets.npz",
                     window=np.hanning(401)[:-1].astype(np.float32),
                     filters=rng.uniform(0, 0.1, (257, 32)).astype(np.float32), epsilon=np.float32(0.001))
            (root / "google_base" / "conversion_report.json").write_text("{}")
            original = (np.sin(np.arange(16000) * 0.05) * 0.1).astype(np.float32)
            audio = {"negative": (rng.standard_normal(32000) * 0.03).astype(np.float32),
                     "augmented_positive": augment(original, seed=12), "raw_positive": original}
            rows = {}
            for identifier, samples in audio.items():
                path = root / f"{identifier}.wav"
                sf.write(path, samples, 16000, subtype="PCM_16")
                rows[identifier] = {"id": identifier, "path": path.name, "samples": len(samples),
                                    "split": "validation", "voice": "validation_voice",
                                    "label": int(identifier != "negative"), "text": "Hey Airis.",
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            augmented = [rows["negative"], {**rows["augmented_positive"], "parent_id": "raw_positive", "seed": 12},
                         {"id": "test_negative", "split": "test", "label": 0, "path": "must_not_be_read.wav"}]
            originals = [rows["raw_positive"], {"id": "test_positive", "split": "test", "label": 1,
                                               "path": "must_not_be_read.wav"},
                         {**rows["raw_positive"], "id": "train_positive", "voice": "training_voice", "split": "train"}]
            for filename, records in (("augmentation_manifest.jsonl", augmented), ("corpus_manifest.jsonl", originals)):
                (root / filename).write_text("".join(json.dumps(row) + "\n" for row in records))
            with patch("onnxruntime.InferenceSession", return_value=session):
                cached, metadata = cache_validation_features(root)
                frontend, expected = GoogleEmbedding(root / "google_base"), []
                for identifier in ("negative", "augmented_positive", "raw_positive"):
                    samples, _ = sf.read(root / rows[identifier]["path"], dtype="float32")
                    if identifier != "negative":
                        frontend.reset()
                        samples = np.pad(samples, (0, 7680))
                        samples = np.pad(samples, (0, (-len(samples)) % 1280))
                    for start in range(0, len(samples), 1280):
                        expected.append(frontend.push(samples[start:start + 1280]).copy())
                np.testing.assert_array_equal(cached, np.stack(expected))
                self.assertEqual(metadata["negative_frames"], 25)
                self.assertFalse(metadata["test_audio_read"])
                clean, clean_metadata = cache_positive_features(root)
                np.testing.assert_array_equal(clean, np.stack(expected[-7:-3]))
                self.assertEqual(clean_metadata["records"][0]["parent_id"], "train_positive")
                self.assertFalse(clean_metadata["held_out_audio_read"])

    def test_mining_loader_rejects_changed_or_held_out_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mining").mkdir()
            (root / "google_base").mkdir()
            base = root / "google_base" / "base.bin"
            base.write_bytes(b"test base")
            features = root / "mining" / "pool.npy"
            np.save(features, np.zeros((1, 16, 96), np.float32))
            records = root / "mining" / "pool.jsonl"
            record = {"voice": "train_voice", "split": "train", "label": 0}
            records.write_text(json.dumps(record) + "\n")
            report = {"source_partition": "train", "by_voice": {"train_voice": 1},
                      "embedding_hashes": {"base.bin": hashlib.sha256(base.read_bytes()).hexdigest()},
                      "features": "mining/pool.npy", "records": "mining/pool.jsonl",
                      "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
                      "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest()}
            metadata = root / "mining" / "pool.json"
            metadata.write_text(json.dumps(report))
            pooled, _ = load_mined_negatives(root, ["train_voice"])
            self.assertEqual(pooled.shape, (1, 16, 96))
            with self.assertRaisesRegex(ValueError, "held-out"):
                load_mined_negatives(root, ["test_voice"])
            records.write_text(json.dumps({**record, "split": "test"}) + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_mined_negatives(root, ["train_voice"])
            report["records_sha256"] = hashlib.sha256(records.read_bytes()).hexdigest()
            metadata.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "training negatives"):
                load_mined_negatives(root, ["train_voice"])

    def test_microphone_check_resamples_in_memory_and_releases_capture(self):
        from training.wakeword.microphone_check import check_microphone

        samples = (np.sin(np.arange(24000) * 0.03) * 0.1).astype(np.float32)
        recorder, detector = Mock(), Mock()
        recorder.is_recording = False
        recorder.limit_reached = False
        recorder.snapshot.side_effect = [RecordedAudio(samples[:12000], 48000), RecordedAudio(samples[:18000], 48000)]
        recorder.stop.return_value = RecordedAudio(samples, 48000)
        fed = []
        detector.feed.side_effect = lambda audio: (fed.append(audio.copy()) or [])
        with patch("training.wakeword.microphone_check.MicrophoneRecorder", return_value=recorder), \
             patch("training.wakeword.microphone_check.WakeWordDetector", return_value=detector), \
             patch("training.wakeword.microphone_check.perf_counter", side_effect=[0, 0.1, 0.2, 0.6]), \
             patch("training.wakeword.microphone_check.sleep"):
            report = check_microphone("test.onnx", seconds=0.5)
        np.testing.assert_allclose(np.concatenate(fed), resample_poly(samples, 1, 3), atol=1e-6)
        recorder.stop.assert_called_once()
        self.assertFalse(report["audio_saved"])
        self.assertFalse(report["used_for_training"])

    def test_microphone_check_cancels_on_processing_error(self):
        from training.wakeword.microphone_check import check_microphone

        recorder, detector = Mock(), Mock()
        recorder.is_recording, recorder.limit_reached = True, False
        recorder.snapshot.return_value = RecordedAudio(np.ones(1280, np.float32) * 0.1, 16000)
        detector.feed.side_effect = RuntimeError("Test inference failure")
        with patch("training.wakeword.microphone_check.MicrophoneRecorder", return_value=recorder), \
             patch("training.wakeword.microphone_check.WakeWordDetector", return_value=detector), \
             patch("training.wakeword.microphone_check.perf_counter", side_effect=[0, 0.1]):
            with self.assertRaisesRegex(RuntimeError, "inference failure"):
                check_microphone("test.onnx")
        recorder.cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
