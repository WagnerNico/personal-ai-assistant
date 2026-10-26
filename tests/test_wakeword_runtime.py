"""Check streaming feature equivalence, trigger behavior, and native ONNX parity."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from input.wakeword import Trigger, WakeWordDetector
from input.wakeword_features import MelFeatures, StreamingMel


class WakeRuntimeTests(unittest.TestCase):
    def test_incremental_features_match_full_audio_windows(self):
        rng = np.random.default_rng(123)
        frontend, extractor = StreamingMel(), MelFeatures()
        for _ in range(45):
            samples = (rng.standard_normal(1280) * 0.03).astype(np.float32)
            incremental = frontend.push(samples)
            np.testing.assert_allclose(incremental, extractor(frontend.audio), atol=1e-6, rtol=1e-6)
            self.assertEqual(incremental.shape, (198, 40))
        frontend.reset()
        np.testing.assert_array_equal(frontend.features, extractor(np.zeros(32000, np.float32)))

    def test_trigger_requires_confirmation_and_rearms_after_cooldown(self):
        trigger = Trigger(0.8)
        self.assertFalse(trigger.update(0.9, 0.08))
        self.assertTrue(trigger.update(0.9, 0.16))
        self.assertFalse(trigger.update(0.9, 1.8))
        self.assertFalse(trigger.update(0.1, 1.9))
        self.assertFalse(trigger.update(0.9, 1.98))
        self.assertTrue(trigger.update(0.9, 2.06))
        trigger.reset()
        self.assertFalse(trigger.update(0.9, 0.08))

    def test_calibration_respects_false_activation_limit_and_target_timing(self):
        from training.wakeword.evaluate import calibrate, positive_metrics

        negatives = np.full(4000, 0.1, np.float32)
        negatives[100:103] = 0.9
        positives = [{"id": "positive", "voice": "validation_voice", "text": "Hey Airis.",
                      "kind": "original", "target_start_seconds": 0.0, "target_end_seconds": 0.24,
                      "scores": [0.1, 0.96, 0.96]}]
        selected, _ = calibrate(negatives, positives, 0.5)
        self.assertGreater(selected["threshold"], 0.9)
        self.assertEqual(selected["recall"], 1)
        self.assertEqual(selected["negative_episodes"], 0)
        early = [{**positives[0], "target_start_seconds": 1.0}]
        metrics = positive_metrics(early, 0.8)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["activations_before_target"], 1)

    def test_vectorized_episode_count_matches_streaming_trigger(self):
        from training.wakeword.evaluate import negative_events

        rng = np.random.default_rng(71)
        for threshold in (0.2, 0.5, 0.8, 0.99, 1.0):
            for scores in (rng.uniform(0, 1, 8000), np.zeros(100), np.ones(100),
                           np.tile([0.9, 0.9, 0.1, 0.6, 0.6], 80)):
                trigger = Trigger(threshold)
                expected = sum(trigger.update(float(score), (index + 1) * 0.08)
                               for index, score in enumerate(scores))
                self.assertEqual(negative_events(scores, threshold)["episodes"], expected)

    def test_calibration_cannot_reward_a_trigger_that_never_rearms(self):
        from training.wakeword.evaluate import calibrate

        # One permanent activation across two hours would otherwise meet 0.5/hour.
        negatives = np.full(90000, 0.6, np.float32)
        positives = [{"id": "positive", "voice": "validation_voice", "text": "Airis.",
                      "kind": "original", "target_start_seconds": 0, "target_end_seconds": 0.24,
                      "scores": [0.1, 0.55, 0.55]}]
        selected, _ = calibrate(negatives, positives, 0.5)
        self.assertGreater(selected["threshold"], 0.6)
        self.assertEqual(selected["recall"], 0)
        self.assertEqual(selected["negative_above_threshold_fraction"], 0)
        self.assertTrue(selected["constraints_satisfied"])
        saturated, _ = calibrate(np.ones_like(negatives), positives, 0.5)
        self.assertFalse(saturated["constraints_satisfied"])

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "Optional ONNX runtime is not installed.")
    def test_google_incremental_contexts_match_batch_features(self):
        from input.google_wakeword_features import GoogleEmbedding

        rng = np.random.default_rng(12)
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="mel")]
        session.run.side_effect = lambda _, inputs: [np.tile(inputs["mel"].mean(axis=1), (1, 3))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "frontend_assets.npz", window=np.hanning(401)[:-1].astype(np.float32),
                     filters=rng.uniform(0, 0.1, (257, 32)).astype(np.float32), epsilon=np.float32(0.001))
            with patch("onnxruntime.InferenceSession", return_value=session):
                frontend = GoogleEmbedding(root)
                audio = (rng.standard_normal((1, 32000)) * 0.02).astype(np.float32)
                expected = frontend.clips(audio)[0]
                for start in range(0, 32000, 1280):
                    frontend.push(audio[0, start:start + 1280])
                np.testing.assert_allclose(frontend.features, expected, atol=1e-6)
                self.assertEqual(frontend.features.shape, (16, 96))
                frontend.reset()
                np.testing.assert_array_equal(frontend.features, np.tile(frontend.zero_embedding, (16, 1)))

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"), "Optional ONNX dependencies are not installed.")
    def test_exported_cpu_model_matches_torch_and_arbitrary_audio_chunks(self):
        import torch
        from training.wakeword.network import ProbabilityNetwork, WakeNetwork

        torch.set_num_threads(1)
        torch.manual_seed(42)
        network = ProbabilityNetwork(WakeNetwork()).eval()
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(40000) * 800).astype(np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            torch.onnx.export(network, torch.zeros(1, 198, 40), path,
                              input_names=["mel"], output_names=["score"], opset_version=17, dynamo=False)
            path.with_suffix(".json").write_text(json.dumps({"frontend": "airis-mel-v1", "sample_rate": 16000, "threshold": 0.99}))
            first, second = WakeWordDetector(path), WakeWordDetector(path)
            whole = first.feed(audio)
            chunks = []
            for start in range(0, len(audio), 137):
                chunks.extend(second.feed(audio[start:start + 137]))
            self.assertEqual(whole, chunks)
            self.assertEqual(len(whole), len(audio) // 1280)
            self.assertLess(len(second.pending), 1280)
            with torch.inference_mode():
                expected = float(network(torch.from_numpy(first.frontend.features[None])).item())
            self.assertAlmostEqual(whole[-1].score, expected, places=5)
            first.reset()
            self.assertEqual(first.samples_processed, 0)
            self.assertEqual(len(first.pending), 0)
            with self.assertRaisesRegex(ValueError, "mono"):
                first.feed(np.zeros((1280, 2), np.float32))
            with self.assertRaisesRegex(ValueError, "finite"):
                first.feed(np.full(1280, np.nan, np.float32))


if __name__ == "__main__":
    unittest.main()
