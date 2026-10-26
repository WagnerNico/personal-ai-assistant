"""Keep standby capture off Bluetooth inputs without changing system defaults."""

import unittest
from unittest.mock import Mock, patch

from input.audio_devices import standby_input_device


class StandbyInputTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch("input.audio_devices.platform.system", return_value="Darwin")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.devices = [
            {"name": "Headset", "max_input_channels": 1},
            {"name": "Laptop speakers", "max_input_channels": 0},
            {"name": "MacBook Air-Mikrofon", "max_input_channels": 1},
            {"name": "USB microphone", "max_input_channels": 1},
        ]
        self.sd = Mock()
        self.sd.query_devices.side_effect = lambda **kwargs: self.devices[0] if kwargs else self.devices
        self.transports = {"Headset": "blue", "MacBook Air-Mikrofon": "bltn", "USB microphone": "usb "}

    def test_bluetooth_default_uses_builtin_even_with_localized_name(self):
        self.assertEqual(standby_input_device(self.sd, transports=self.transports), 2)
        self.sd.RawInputStream.assert_not_called()

    def test_safe_default_follows_system_without_replacement(self):
        self.transports["Headset"] = "usb "
        self.assertIsNone(standby_input_device(self.sd, transports=self.transports))
        self.sd.query_devices.assert_called_once_with(kind="input")

    def test_explicit_embedding_selection_is_preserved(self):
        self.assertEqual(standby_input_device(self.sd, requested="Headset"), "Headset")
        self.sd.query_devices.assert_not_called()

    def test_wired_input_is_used_if_no_builtin_exists(self):
        self.devices.pop(2)
        self.assertEqual(standby_input_device(self.sd, transports=self.transports), 2)

    def test_unknown_headset_only_does_not_silently_open(self):
        self.devices[:] = self.devices[:1]
        with self.assertRaisesRegex(RuntimeError, "built-in or wired"):
            standby_input_device(self.sd, transports={})
        self.sd.RawInputStream.assert_not_called()

    def test_coreaudio_failure_can_still_select_named_builtin(self):
        with patch("input.audio_devices.macos_input_transports", side_effect=OSError("Unavailable")):
            self.assertEqual(standby_input_device(self.sd), 2)

    def test_other_platforms_preserve_default(self):
        with patch("input.audio_devices.platform.system", return_value="Linux"):
            self.assertIsNone(standby_input_device(self.sd))
        self.sd.query_devices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
