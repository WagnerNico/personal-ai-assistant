"""Regression checks for macOS Qt plugin discovery flags."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from ui.qt_runtime import _clear_hidden_flag


@unittest.skipUnless(sys.platform == "darwin", "macOS file flags are required.")
class QtRuntimeTests(unittest.TestCase):
    def test_hidden_flag_is_cleared_without_changing_contents_or_other_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.dylib"
            path.write_bytes(b"unchanged plugin contents")
            flags = path.stat().st_flags & ~stat.UF_HIDDEN
            os.chflags(path, flags | stat.UF_HIDDEN)
            _clear_hidden_flag(path)
            self.assertEqual(path.stat().st_flags, flags)
            self.assertEqual(path.read_bytes(), b"unchanged plugin contents")

    def test_discoverable_files_do_not_need_a_metadata_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugin.dylib"
            path.write_bytes(b"plugin")
            os.chflags(path, path.stat().st_flags & ~stat.UF_HIDDEN)
            with patch("ui.qt_runtime.os.chflags") as write_flags:
                _clear_hidden_flag(path)
                write_flags.assert_not_called()


if __name__ == "__main__":
    unittest.main()
