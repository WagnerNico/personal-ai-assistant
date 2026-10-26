"""Validate the local macOS app bundle without launching a visible UI."""

from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest

from ui.build_launcher import build_launcher


class LauncherTests(unittest.TestCase):
    def test_bundle_uses_the_project_environment_and_does_not_open_a_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Project with spaces"
            root.mkdir()
            bundle = build_launcher(project_root=root)
            with (bundle / "Contents" / "Info.plist").open("rb") as file:
                info = plistlib.load(file)
            self.assertEqual(info["CFBundleExecutable"], "Airis")
            self.assertEqual(info["LSArchitecturePriority"][0], "arm64")
            self.assertTrue(info["LSRequiresNativeExecution"])
            self.assertIn("NSMicrophoneUsageDescription", info)
            executable = bundle / "Contents" / "MacOS" / "Airis"
            self.assertTrue(executable.stat().st_mode & 0o111)
            content = executable.read_text()
            self.assertIn(".venv/bin/python", content)
            self.assertIn("-m ui.desktop", content)
            self.assertIn("logs/desktop.log", content)
            subprocess.run(["zsh", "-n", str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
