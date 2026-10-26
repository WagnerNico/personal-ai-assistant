"""Check synthetic corpus partitioning, augmentation, and cache safety."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np

from training.wakeword.data import augment, examples, load_recipe
from training.wakeword.synthesis import generate_corpus


class WakeWordDataTests(unittest.TestCase):
    def test_more_positive_takes_keep_existing_clip_identities(self):
        recipe = load_recipe()
        baseline = list(examples(recipe))
        expanded = copy.deepcopy(recipe)
        expanded["positive_takes"] = 24
        expanded["negative_takes"] = 1
        plan = list(examples(expanded))
        self.assertTrue({row.id for row in baseline if row.label or row.take == 0}.issubset({row.id for row in plan}))
        self.assertEqual(sum(row.label for row in plan), 12 * 2 * 24 * 3)
        self.assertTrue(all(row.take == 0 for row in plan if not row.label))

    def test_negative_mining_filters_held_out_rows_before_audio_loading(self):
        from training.wakeword.mining import training_negatives

        rows = [{"id": "train_negative", "voice": "train_voice", "split": "train", "label": 0},
                {"id": "train_positive", "voice": "train_voice", "split": "train", "label": 1},
                {"id": "validation", "voice": "validation_voice", "split": "validation", "label": 0},
                {"id": "test", "voice": "test_voice", "split": "test", "label": 0}]
        self.assertEqual(training_negatives(rows, ["train_voice"]), rows[:1])
        with self.assertRaisesRegex(ValueError, "training speakers"):
            training_negatives(rows, ["validation_voice"])

    def test_both_wake_phrases_use_qwen_text_and_neither_is_negative(self):
        plan = list(examples(load_recipe()))
        positive_text = {example.text for example in plan if example.label == 1}
        negative_text = {example.text for example in plan if example.label == 0}
        self.assertEqual(positive_text, {"Hey Airis.", "Airis."})
        self.assertTrue(positive_text.isdisjoint(negative_text))
        self.assertTrue(all("/" not in text and "[" not in text for text in positive_text))

    def test_source_voices_never_leak_across_dataset_partitions(self):
        recipe = load_recipe()
        plan = list(examples(recipe))
        owners = {}
        for example in plan:
            owners.setdefault(example.voice, set()).add(example.split)
        self.assertTrue(all(len(splits) == 1 for splits in owners.values()))
        self.assertTrue(any(example.label == 1 for example in plan))
        self.assertFalse(any(example.label == 0 and example.text == "Airis." for example in plan))
        invalid = copy.deepcopy(recipe)
        invalid["voices"]["test"].append(invalid["voices"]["train"][0])
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load_recipe(path)

    def test_augmentation_preserves_the_full_positive_and_is_reproducible(self):
        samples = np.sin(np.arange(8000, dtype=np.float32) * 0.1) * 0.2
        first = augment(samples, seed=42)
        np.testing.assert_array_equal(first, augment(samples, seed=42))
        self.assertEqual(first.shape, (32000,))
        self.assertTrue(np.isfinite(first).all())
        self.assertLessEqual(float(np.max(np.abs(first))), 1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            augment(np.ones(50000, np.float32), seed=42, label=1)
        # A complete positive near the window length must also remain usable.
        self.assertEqual(augment(np.ones(31990, np.float32) * 0.2, seed=0).shape, (32000,))

    def test_augmented_corpus_preserves_partitions_and_resumes_identically(self):
        from training.wakeword.corpus import prepare_variants

        recipe = load_recipe()
        recipe["voices"] = {split: voices[:1] for split, voices in recipe["voices"].items()}
        recipe["takes"] = 1
        recipe["speeds"] = [1.0]
        recipe["negative_phrases"] = []
        recipe["augmentations_per_clip"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_corpus(recipe, root, synthesizer=lambda example: (np.full(16000, 0.2, np.float32), 16000))
            first = prepare_variants(root)
            manifest = (root / "augmentation_manifest.jsonl").read_text()
            second = prepare_variants(root)
            self.assertEqual(first["generated"], 18)
            self.assertEqual(first["positives"], 12)
            self.assertEqual(first["negatives"], 6)
            self.assertEqual(first["by_split"], {"train": 6, "validation": 6, "test": 6})
            self.assertEqual(second["reused"], 18)
            self.assertEqual(second["generated"], 0)
            self.assertEqual((root / "augmentation_manifest.jsonl").read_text(), manifest)
            parents = {example.id: example for example in examples(recipe)}
            for row in map(json.loads, manifest.splitlines()):
                self.assertEqual(row["split"], parents[row["parent_id"]].split)
                self.assertEqual(row["voice"], parents[row["parent_id"]].voice)

    def test_corpus_audit_rejects_missing_audio_and_does_not_crop_long_positives(self):
        from training.wakeword.corpus import audit_corpus, prepare_variants

        recipe = load_recipe()
        recipe["voices"] = {"train": ["us_female_clear"], "validation": [], "test": []}
        recipe["takes"] = 1
        recipe["speeds"] = [1.0]
        recipe["negative_phrases"] = []
        recipe["augmentations_per_clip"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_corpus(recipe, root, synthesizer=lambda example: (np.full(40000, 0.2, np.float32), 16000))
            first = audit_corpus(root)
            self.assertEqual(first["valid_clips"], 3)
            self.assertEqual(len(first["positives_exceeding_window"]), 2)
            variants = prepare_variants(root)
            self.assertEqual(variants["generated"], 1)
            self.assertEqual(len(variants["excluded_positive_sources"]), 2)
            self.assertEqual(variants["positives"], 0)
            self.assertEqual(variants["negatives"], 1)
            path = root / "audio" / f"{next(examples(recipe)).id}.wav"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "invalid or missing"):
                audit_corpus(root)
            failure = json.loads((root / "validation_report.json").read_text())
            self.assertEqual(failure["valid_clips"], 2)
            self.assertEqual(len(failure["errors"]), 1)

    def test_generated_corpus_resumes_without_resynthesizing_valid_clips(self):
        recipe = load_recipe()
        recipe["voices"] = {"train": ["us_female_clear"], "validation": [], "test": []}
        recipe["takes"] = 1
        recipe["speeds"] = [1.0]
        recipe["negative_phrases"] = []
        calls = []
        def synthesize(example):
            calls.append(example.id)
            return np.full(2400, 0.2, np.float32), 24000
        with tempfile.TemporaryDirectory() as directory:
            first = generate_corpus(recipe, directory, synthesizer=synthesize)
            second = generate_corpus(recipe, directory, synthesizer=synthesize)
            self.assertEqual(first["generated"], 3)
            self.assertEqual(second["generated"], 0)
            self.assertEqual(second["reused"], 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(list((Path(directory) / "audio").glob("*.wav"))), 3)

    def test_training_audit_does_not_mark_an_unfinished_corpus_complete(self):
        from training.wakeword.corpus import audit_corpus

        recipe = load_recipe()
        recipe["voices"] = {split: voices[:1] for split, voices in recipe["voices"].items()}
        recipe["takes"], recipe["speeds"], recipe["negative_phrases"] = 1, [1.0], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_corpus(recipe, root, synthesizer=lambda example: (np.full(16000, 0.2, np.float32), 16000))
            for row in examples(recipe):
                if row.split != "train":
                    (root / "audio" / f"{row.id}.wav").unlink()
            training = audit_corpus(root, training_only=True)
            self.assertEqual(training["valid_clips"], 3)
            self.assertEqual(training["audited_partitions"], ["train"])
            self.assertTrue((root / "training_corpus_manifest.jsonl").exists())
            self.assertFalse((root / "corpus_manifest.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "invalid or missing"):
                audit_corpus(root)

    def test_qwen_designs_references_once_and_uses_airis_base_for_samples(self):
        from training.wakeword.qwen_synthesis import QwenSynthesizer
        from output.mlx_speech import MODEL_ID

        recipe = load_recipe()
        recipe["voices"] = {"train": ["us_female_clear"], "validation": [], "test": []}
        recipe["takes"] = 1
        recipe["negative_phrases"] = []
        design, base = Mock(sample_rate=24000), Mock(sample_rate=24000)
        result = SimpleNamespace(audio=np.full(2400, 0.2, np.float32), sample_rate=24000)
        design.generate.side_effect = lambda **kwargs: iter([result])
        base.generate.side_effect = lambda **kwargs: iter([result])
        loader = Mock(side_effect=[design, base])
        reference = object()
        audio_loader = Mock(return_value=reference)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "mlx.core": Mock(), "mlx_audio.tts.utils": Mock(load_model=loader),
            "mlx_audio.utils": Mock(load_audio=audio_loader),
        }):
            synthesizer = QwenSynthesizer(recipe, directory)
            positive = [example for example in examples(recipe) if example.text == "Hey Airis."]
            first, rate = synthesizer(positive[0])
            second, _ = synthesizer(positive[1])
            self.assertEqual(rate, 24000)
            self.assertNotEqual(len(first), len(second))
            self.assertEqual(synthesizer.generated_sources, 1)
            base.generate.assert_called_once()
            audio_loader.assert_called_once()
            self.assertEqual(loader.call_args_list[0].args[0], recipe["design_model"])
            self.assertEqual(loader.call_args_list[1].args[0], MODEL_ID)
            self.assertEqual(design.generate.call_args.kwargs["instruct"], recipe["voice_designs"]["us_female_clear"])
            self.assertEqual(base.generate.call_args.kwargs["text"], "Hey Airis.")
            self.assertIs(base.generate.call_args.kwargs["ref_audio"], reference)
            self.assertEqual(base.generate.call_args.kwargs["ref_text"], recipe["reference_text"])
            self.assertNotIn("instruct", base.generate.call_args.kwargs)
            QwenSynthesizer(recipe, directory)
            self.assertEqual(loader.call_count, 2)

    def test_batched_cloning_preserves_membership_and_reuses_source_audio(self):
        from training.wakeword.qwen_synthesis import QwenSynthesizer

        recipe = load_recipe()
        recipe["voices"] = {"train": ["us_female_clear"], "validation": [], "test": []}
        recipe["positive_takes"] = 4
        recipe["negative_takes"] = 1
        recipe["synthesis_batch_size"] = 4
        recipe["negative_phrases"] = []
        design, base = Mock(sample_rate=24000), Mock(sample_rate=24000)
        reference_result = SimpleNamespace(audio=np.full(2400, 0.2, np.float32), sample_rate=24000)
        design.generate.side_effect = lambda **kwargs: iter([reference_result])
        base.batch_generate.side_effect = lambda **kwargs: iter([
            SimpleNamespace(sequence_idx=index, audio=np.full(2400 + index * 100, 0.1, np.float32), sample_rate=24000)
            for index in reversed(range(len(kwargs["texts"])))
        ])
        loader = Mock(side_effect=[design, base])
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "mlx.core": Mock(), "mlx_audio.tts.utils": Mock(load_model=loader),
            "mlx_audio.utils": Mock(load_audio=Mock(return_value=object())),
        }):
            plan = list(examples(recipe))
            synthesizer = QwenSynthesizer(recipe, directory)
            synthesizer.configure_batches(plan)
            positives = [example for example in plan if example.text == "Hey Airis."]
            first, _ = synthesizer(positives[0])
            faster, _ = synthesizer(positives[2])
            another_take, _ = synthesizer(positives[3])
            self.assertLess(len(faster), len(first))
            self.assertGreater(len(another_take), len(first))
            self.assertEqual(synthesizer.generated_sources, 4)
            self.assertEqual(synthesizer.generated_batches, 1)
            base.batch_generate.assert_called_once()
            self.assertEqual(base.batch_generate.call_args.kwargs["texts"], ["Hey Airis."] * 4)
            base.generate.assert_not_called()
            first_group = [synthesizer.source_identity(item) for item in synthesizer.batch_groups[synthesizer.source_identity(positives[0])]]
            synthesizer.configure_batches(plan)
            self.assertEqual(first_group, [synthesizer.source_identity(item) for item in synthesizer.batch_groups[synthesizer.source_identity(positives[0])]])

    def test_cached_corpus_does_not_load_a_speech_model(self):
        recipe = load_recipe()
        recipe["voices"] = {"train": ["us_female_clear"], "validation": [], "test": []}
        recipe["takes"] = 1
        recipe["speeds"] = [1.0]
        recipe["negative_phrases"] = []
        with tempfile.TemporaryDirectory() as directory:
            generate_corpus(recipe, directory, synthesizer=lambda example: (np.full(2400, 0.2, np.float32), 24000))
            with patch("training.wakeword.qwen_synthesis.QwenSynthesizer") as loader:
                report = generate_corpus(recipe, directory)
                loader.assert_not_called()
                self.assertEqual(report["reused"], 3)


if __name__ == "__main__":
    unittest.main()
