"""Verify interruption ownership and recovery without controlling real media."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import Mock, patch

from actions.media_control import MacOSMediaControl, SpeechMediaSession
from actions.macos_media_keys import MacOSMediaKeys


class MediaSessionTests(unittest.TestCase):
    def setUp(self):
        self.token = {"bundle": "test.player", "title": "Track", "item": "1"}
        self.control = Mock(last_error="")
        self.control.pause.return_value = self.token
        self.session = SpeechMediaSession(self.control)

    def test_cue_and_question_share_one_interruption(self):
        self.session.begin()
        self.session.begin()
        self.assertTrue(self.session.active)
        self.control.pause.assert_called_once()
        self.control.resume.assert_not_called()
        self.session.end()
        self.session.end()
        self.assertFalse(self.session.active)
        self.control.resume.assert_called_once_with(self.token)

    def test_previously_paused_or_missing_media_is_never_started(self):
        self.control.pause.return_value = None
        self.session.begin()
        self.session.end()
        self.control.resume.assert_not_called()

    def test_failed_pause_does_not_create_resume_ownership(self):
        self.control.pause.side_effect = subprocess.TimeoutExpired("media helper", 2)
        self.session.begin()
        self.assertIn("could not be paused", self.session.last_error)
        self.session.end()
        self.control.resume.assert_not_called()
        self.assertFalse(self.session.active)

    def test_resume_failure_releases_session_for_next_turn(self):
        self.control.resume.side_effect = RuntimeError("Unavailable player")
        self.session.begin()
        self.session.end()
        self.assertFalse(self.session.active)
        self.assertIn("could not be resumed", self.session.last_error)
        self.control.resume.side_effect = None
        self.session.begin()
        self.assertTrue(self.session.active)
        self.assertEqual(self.session.last_error, "")
        self.session.end()
        self.assertEqual(self.control.pause.call_count, 2)

    def test_disabled_session_has_no_external_effects(self):
        session = SpeechMediaSession(self.control, enabled=False)
        session.begin()
        session.end()
        self.assertFalse(session.active)
        self.control.pause.assert_not_called()

    def test_permission_notice_is_exposed_to_ui(self):
        self.control.last_error = "Accessibility permission is needed."
        self.control.pause.return_value = None
        self.session.begin()
        self.assertEqual(self.session.last_error, self.control.last_error)
        self.session.end()
        self.control.resume.assert_not_called()


class MediaHelperTests(unittest.TestCase):
    def setUp(self):
        self.token = {"bundle": "test.player", "title": "Track", "item": "1", "playing": True}
        self.keys = Mock()
        self.keys.press_play_pause.return_value = False

    @unittest.skipUnless(shutil.which("node"), "Node is needed only for the isolated JavaScript helper simulation.")
    def test_native_helper_state_transitions_without_native_media_access(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([shutil.which("node"), str(root / "tests/media_control_simulation.js"),
                                 str(root / "actions/media_control.js")],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_titles_are_data_arguments_and_never_evaluated_as_code(self):
        control = MacOSMediaControl(keys=self.keys)
        token = {"bundle": "test.player", "title": "$(open example); 'quoted'\ntrack", "item": "1"}
        results = [Mock(stdout=json.dumps({**token, "playing": False})), Mock(stdout='{"resumed":true}')]
        with patch("actions.media_control.subprocess.run", side_effect=results) as run:
            control.resume(token)
        args = run.call_args.args[0]
        self.assertEqual(args[:3], ["/usr/bin/osascript", "-l", "JavaScript"])
        self.assertEqual(args[-2], "resume")
        self.assertEqual(json.loads(args[-1]), token)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_unsuccessful_pause_never_returns_a_token(self):
        control = MacOSMediaControl(keys=self.keys)
        with patch.object(control, "_run", side_effect=[self.token, {"paused": False, "token": None,
                                                                   "needsAccessibility": True}]):
            self.assertIsNone(control.pause())
        self.assertIn("Accessibility", control.last_error)

    def test_resume_skipped_for_changed_item_is_not_reported_as_failure(self):
        control = MacOSMediaControl(keys=self.keys)
        with patch.object(control, "_run", return_value={**self.token, "title": "Another track"}):
            control.resume(self.token)
        self.assertEqual(control.last_error, "")
        self.keys.press_play_pause.assert_not_called()

    def test_native_key_success_is_verified_and_creates_resume_ownership(self):
        self.keys.press_play_pause.return_value = True
        control = MacOSMediaControl(keys=self.keys)
        with patch.object(control, "_run", side_effect=[self.token, {"changed": True},
                                                       {**self.token, "playing": False}, {"changed": True}]) as run:
            token = control.pause()
            self.assertEqual(token, self.token)
            control.resume(token)
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         ["status", "wait-paused", "status", "wait-resumed"])
        self.assertEqual(self.keys.press_play_pause.call_count, 2)

    def test_native_key_failure_never_toggles_twice(self):
        self.keys.press_play_pause.return_value = True
        control = MacOSMediaControl(keys=self.keys)
        with patch.object(control, "_run", side_effect=[self.token, {"changed": False}]) as run:
            self.assertIsNone(control.pause())
        self.assertEqual([call.args[0] for call in run.call_args_list], ["status", "wait-paused"])
        self.keys.press_play_pause.assert_called_once()
        self.assertIn("did not respond", control.last_error)

    def test_inactive_media_never_receives_keys_or_commands(self):
        for before in (None, {**self.token, "playing": False}):
            with self.subTest(before=before):
                control = MacOSMediaControl(keys=self.keys)
                with patch.object(control, "_run", return_value=before) as run:
                    self.assertIsNone(control.pause())
                run.assert_called_once_with("status")
        self.keys.press_play_pause.assert_not_called()

    def test_manually_resumed_media_is_not_toggled(self):
        control = MacOSMediaControl(keys=self.keys)
        with patch.object(control, "_run", return_value=self.token) as run:
            control.resume(self.token)
        run.assert_called_once_with("status")
        self.keys.press_play_pause.assert_not_called()


class NativeMediaKeyTests(unittest.TestCase):
    def setUp(self):
        self.keys = MacOSMediaKeys()
        self.keys._loaded = True
        self.keys.ax = Mock()
        self.keys.ax.AXIsProcessTrusted.return_value = True
        self.keys.objc = Mock()
        self.keys.pointer = Mock(return_value=10)
        self.keys.release = Mock()

    def test_no_permission_means_no_event_construction_or_posting(self):
        self.keys.ax.AXIsProcessTrusted.return_value = False
        with patch.object(self.keys, "_events") as events:
            self.assertFalse(self.keys.press_play_pause())
        events.assert_not_called()
        self.keys.ax.CGEventPost.assert_not_called()

    def test_invalid_event_cannot_send_a_partial_key_press(self):
        with patch.object(self.keys, "_events", side_effect=RuntimeError("Invalid event")):
            with self.assertRaisesRegex(RuntimeError, "Invalid event"):
                self.keys.press_play_pause()
        self.keys.ax.CGEventPost.assert_not_called()
        self.keys.release.assert_called_once()

    def test_posts_down_and_up_once_and_releases_native_pool(self):
        with patch.object(self.keys, "_events", return_value=[101, 102]), patch("actions.macos_media_keys.time.sleep"):
            self.assertTrue(self.keys.press_play_pause())
        self.assertEqual([call.args for call in self.keys.ax.CGEventPost.call_args_list], [(0, 101), (0, 102)])
        self.keys.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
