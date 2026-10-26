"""Verify that personal threshold presets remain optional and model-specific."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from input.wakeword_profile import load_wake_profile


class WakeProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / 'head.onnx'
        self.model.write_bytes(b'example-model')
        self.profile = {'version': 1, 'model_sha256': hashlib.sha256(self.model.read_bytes()).hexdigest(),
                        'threshold': 0.85, 'minimum_positive_confirmed_peak': 0.885,
                        'maximum_negative_confirmed_peak': 0.811, 'negative_audio_seconds': 21.95,
                        'positive_recordings': 6}

    def write_profile(self):
        self.model.with_name('personal_calibration.json').write_text(json.dumps(self.profile))

    def test_valid_preset_loads_without_modifying_model(self):
        self.write_profile()
        self.assertEqual(load_wake_profile(self.model)['threshold'], 0.85)
        self.assertEqual(self.model.read_bytes(), b'example-model')

    def test_stale_model_or_missing_profile_is_ignored(self):
        self.assertIsNone(load_wake_profile(self.model))
        self.write_profile()
        self.model.write_bytes(b'replacement-model')
        self.assertIsNone(load_wake_profile(self.model))

    def test_invalid_or_nonseparating_presets_are_ignored(self):
        for changes in ({'threshold': float('nan')}, {'threshold': 0.8}, {'threshold': 0.9},
                        {'negative_audio_seconds': 0}, {'positive_recordings': False}):
            original = self.profile.copy()
            self.profile.update(changes)
            self.write_profile()
            self.assertIsNone(load_wake_profile(self.model))
            self.profile = original


if __name__ == '__main__':
    unittest.main()
