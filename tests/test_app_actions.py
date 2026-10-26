"""Verify app scope, structured plans, truthful receipts and chat integration."""

from contextlib import contextmanager
import ctypes
import io
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from actions.app_control import AppControl, AppRequest, InstalledApp, installed_apps
from actions.macos_windows import MacOSWindows
from assistant import Assistant
from input.audio import RecordedAudio
from output.audio import SpeechAudio
from processing.agent import process_agent_text
from ui.workflow import ConversationWorkflow


OUTLOOK = InstalledApp("Microsoft Outlook", Path("/Applications/Microsoft Outlook.app"),
                       frozenset(("outlook", "microsoft outlook")))
SPOTIFY = InstalledApp("Spotify", Path("/Applications/Spotify.app"), frozenset(("spotify",)))


def test_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(b"\x00\x00" * 24)
    return SpeechAudio(buffer.getvalue(), 24000)


class AppActionTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch("actions.app_control.sys.platform", "darwin")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.runner = Mock()
        self.windows = Mock()
        self.control = AppControl(catalog=lambda: [OUTLOOK, SPOTIFY], runner=self.runner,
                                  windows=self.windows)
        self.backend = Mock()
        self.backend.last_finish_reason = "length"

    def request(self, text, **kwargs):
        return process_agent_text(text, app_control=self.control, backend=self.backend, **kwargs)

    def test_common_commands_open_exact_installed_bundle_without_inference(self):
        for text, app in (("open Outlook", OUTLOOK), ("Open Spotify.", SPOTIFY),
                          ("Could you please launch the Outlook app?", OUTLOOK),
                          ("Airis, I would like to open Spotify, please", SPOTIFY),
                          ("Bitte öffne Spotify", SPOTIFY)):
            with self.subTest(text=text):
                reply, truncated = self.request(text)
                self.assertEqual(reply, f"Opened {app.name}.")
                self.assertFalse(truncated)
                self.runner.assert_called_with(["/usr/bin/open", "-a", str(app.path)],
                                               check=True, capture_output=True, timeout=10)
        self.windows.make_full_screen.assert_not_called()
        self.backend.respond.assert_not_called()

    def test_full_screen_variants_and_result_order(self):
        calls = []
        self.runner.side_effect = lambda *a, **k: calls.append("open")
        self.windows.make_full_screen.side_effect = lambda *a: calls.append("full_screen")
        self.windows.wait_until_open.side_effect = lambda *a: calls.append("window_ready")
        for text in ("open Outlook and make it full screen", "open Outlook in fullscreen",
                     "open Outlook and put it in full-screen, please", "öffne Outlook im Vollbild",
                     "Could you open Outlook, and make it full screen for me?"):
            calls.clear()
            chunks, stages = [], []
            result, _ = self.request(text, on_text=chunks.append, on_stage=stages.append)
            self.assertEqual(calls, ["open", "window_ready", "full_screen"])
            self.assertEqual(chunks, ["Opened Microsoft Outlook in full screen."])
            self.assertEqual(result, chunks[0])
            self.assertIn("Making Microsoft Outlook full screen...", stages)
            self.assertIn("Waiting for Microsoft Outlook to finish opening...", stages)
            self.assertLess(stages.index("Waiting for Microsoft Outlook to finish opening..."),
                            stages.index("Making Microsoft Outlook full screen..."))
        self.windows.make_full_screen.assert_called_with(OUTLOOK.path)
        self.windows.wait_until_open.assert_called_with(OUTLOOK.path)

    def test_switch_commands_activate_the_exact_bundle_without_model_generation(self):
        for text, app in (("Switch to Spotify.", SPOTIFY), ("Go to Outlook", OUTLOOK),
                          ("Could you please switch over to the Spotify app?", SPOTIFY),
                          ("Airis, go back to Outlook, please", OUTLOOK),
                          ("focus on Spotify", SPOTIFY), ("Bitte wechsle zu Outlook", OUTLOOK)):
            with self.subTest(text=text):
                stages, chunks = [], []
                reply, truncated = self.request(text, on_stage=stages.append, on_text=chunks.append)
                self.assertEqual(reply, f"Switched to {app.name}.")
                self.assertFalse(truncated)
                self.assertEqual(chunks, [reply])
                self.assertEqual(stages, [f"Switching to {app.name}..."])
                self.runner.assert_called_with(["/usr/bin/open", "-a", str(app.path)],
                                               check=True, capture_output=True, timeout=10)
                self.windows.wait_until_active.assert_called_with(app.path)
        self.windows.wait_until_open.assert_not_called()
        self.windows.make_full_screen.assert_not_called()
        self.backend.respond.assert_not_called()

    def test_switch_receipt_follows_activation_before_optional_full_screen(self):
        calls = []
        self.runner.side_effect = lambda *a, **k: calls.append("open_or_activate")
        self.windows.wait_until_active.side_effect = lambda *a: calls.append("active")
        self.windows.wait_until_open.side_effect = lambda *a: calls.append("window_ready")
        self.windows.make_full_screen.side_effect = lambda *a: calls.append("full_screen")
        result = self.control.execute(AppRequest("Spotify", True, "switch_app"))
        self.assertEqual(calls, ["open_or_activate", "active", "window_ready", "full_screen"])
        self.assertEqual(result.text, "Switched to Spotify in full screen.")
        self.assertTrue(result.switched)
        self.assertTrue(result.full_screen)

    def test_switch_failure_never_claims_activation_or_attempts_full_screen(self):
        self.windows.wait_until_active.side_effect = RuntimeError("Target did not activate")
        with self.assertLogs("actions.app_control", level="WARNING"):
            result = self.control.execute(AppRequest("Spotify", True, "switch_app"))
        self.assertIn("couldn't switch", result.text)
        self.assertTrue(result.opened)
        self.assertFalse(result.switched)
        self.windows.wait_until_open.assert_not_called()
        self.windows.make_full_screen.assert_not_called()

    def test_switch_launch_failure_does_not_wait_or_claim_activation(self):
        self.runner.side_effect = subprocess.CalledProcessError(1, "open")
        self.assertEqual(self.request("go to Spotify")[0], "macOS couldn't open Spotify.")
        self.windows.wait_until_active.assert_not_called()

    def test_switch_full_screen_permission_failure_keeps_activation_receipt(self):
        self.windows.wait_until_open.side_effect = PermissionError()
        result, _ = self.request("switch to Spotify in full screen")
        self.assertIn("Switched to Spotify.", result)
        self.assertIn("Accessibility", result)
        self.windows.make_full_screen.assert_not_called()

    def test_standalone_full_screen_targets_running_app_without_opening_or_switching(self):
        self.windows.running_pid.return_value = 42
        for text, app in (("Make Spotify full screen.", SPOTIFY),
                          ("Put Outlook in fullscreen", OUTLOOK),
                          ("Could you please make the Outlook app full-screen?", OUTLOOK),
                          ("Airis, set Spotify to full screen, please", SPOTIFY),
                          ("Switch Outlook to full screen for me", OUTLOOK),
                          ("Bitte mache Spotify im Vollbild", SPOTIFY)):
            with self.subTest(text=text):
                chunks, stages = [], []
                reply, truncated = self.request(text, on_text=chunks.append, on_stage=stages.append)
                self.assertEqual(reply, f"Made {app.name} full screen.")
                self.assertEqual(chunks, [reply])
                self.assertFalse(truncated)
                self.windows.running_pid.assert_called_with(app.path)
                self.windows.wait_until_open.assert_called_with(app.path)
                self.windows.make_full_screen.assert_called_with(app.path)
                self.assertEqual(stages, [f"Waiting for {app.name} to finish opening...",
                                          f"Making {app.name} full screen..."])
        self.runner.assert_not_called()
        self.windows.wait_until_active.assert_not_called()
        self.backend.respond.assert_not_called()

    def test_standalone_full_screen_does_not_launch_a_closed_app(self):
        self.windows.running_pid.return_value = None
        self.assertEqual(self.request("Make Spotify full screen")[0],
                         "Spotify isn't open. Please open it first.")
        self.runner.assert_not_called()
        self.windows.wait_until_open.assert_not_called()
        self.windows.make_full_screen.assert_not_called()

    def test_standalone_full_screen_reports_permission_readiness_and_control_failures(self):
        self.windows.running_pid.return_value = 42
        self.windows.wait_until_open.side_effect = PermissionError()
        result = self.control.execute(AppRequest("Spotify", True, "full_screen_app"))
        self.assertIn("Accessibility", result.text)
        self.assertNotIn("Opened", result.text)
        self.assertFalse(result.opened)
        self.assertFalse(result.full_screen)
        self.windows.make_full_screen.assert_not_called()
        self.windows.wait_until_open.side_effect = RuntimeError("No ready window")
        with self.assertLogs("actions.app_control", level="WARNING"):
            result = self.control.execute(AppRequest("Spotify", True, "full_screen_app"))
        self.assertIn("isn't ready", result.text)
        self.windows.make_full_screen.assert_not_called()
        self.windows.wait_until_open.side_effect = None
        self.windows.make_full_screen.side_effect = RuntimeError("Fullscreen refused")
        with self.assertLogs("actions.app_control", level="WARNING"):
            result = self.control.execute(AppRequest("Spotify", True, "full_screen_app"))
        self.assertIn("couldn't make", result.text)
        self.assertFalse(result.full_screen)
        self.runner.assert_not_called()

    def test_standalone_full_screen_success_does_not_claim_opening_or_switching(self):
        self.windows.running_pid.return_value = 42
        result = self.control.execute(AppRequest("Spotify", True, "full_screen_app"))
        self.assertTrue(result.full_screen)
        self.assertFalse(result.opened)
        self.assertFalse(result.switched)

    def test_unready_app_cannot_proceed_to_full_screen_or_claim_it(self):
        self.windows.wait_until_open.side_effect = RuntimeError("No ready window")
        with self.assertLogs("actions.app_control", level="WARNING"):
            text, _ = self.request("open Spotify in full screen")
        self.assertIn("isn't ready yet", text)
        self.windows.make_full_screen.assert_not_called()

    def test_discussion_quoted_and_negated_commands_never_execute(self):
        self.backend.respond.return_value = "Here is an explanation."
        for text in ("How do I open Outlook?", "Do not open Spotify.",
                     "Explain 'open Outlook'.", "What happens when I say open Spotify?",
                     '"open Spotify"', "If I say open Spotify, what will happen?",
                     "Don't switch to Spotify", "How can I go to Outlook?",
                     '"switch to Spotify"', "Explain how to switch to Outlook",
                     "Don't make Spotify full screen", "How can I make Outlook full screen?",
                     '"make Spotify full screen"', "If I say make Spotify full screen, what happens?"):
            self.assertEqual(self.request(text)[0], "Here is an explanation.")
        self.runner.assert_not_called()
        self.windows.make_full_screen.assert_not_called()

    def test_extra_actions_paths_urls_and_shell_text_are_not_executed(self):
        for text in ("open Spotify and play music", "open Outlook then send an email",
                     "open Spotify; rm -rf files", "open /tmp/Spotify.app",
                     "open https://example.com", "open $(whoami)", "open --args",
                     "open Outlook and Spotify", "open Outlook\nand send email",
                     "switch to Spotify and play music", "go to /tmp/Spotify.app",
                     "switch to $(whoami)", "go to https://example.com",
                     "switch to Outlook; send email", "go to Outlook and Spotify",
                     "make /tmp/Spotify.app full screen", "make $(whoami) full screen",
                     "make Outlook and Spotify full screen", "make Outlook; send email full screen"):
            self.request(text)
        self.runner.assert_not_called()
        self.windows.make_full_screen.assert_not_called()
        self.backend.respond.assert_not_called()

    def test_unknown_or_ambiguous_app_is_reported_without_launch(self):
        result, _ = self.request("open Missing App")
        self.assertIn("couldn't find", result)
        self.control._catalog = lambda: [SPOTIFY, InstalledApp("Other Spotify", Path("/other.app"), SPOTIFY.aliases)]
        self.assertIn("More than one", self.request("open Spotify")[0])
        self.runner.assert_not_called()

    def test_open_failure_does_not_attempt_full_screen_or_claim_success(self):
        self.runner.side_effect = subprocess.CalledProcessError(1, "open")
        result, _ = self.request("open Outlook in full screen")
        self.assertEqual(result, "macOS couldn't open Microsoft Outlook.")
        self.windows.make_full_screen.assert_not_called()

    def test_permission_and_unsupported_window_report_partial_success(self):
        self.windows.make_full_screen.side_effect = PermissionError()
        result, _ = self.request("open Outlook in full screen")
        self.assertIn("Opened Microsoft Outlook.", result)
        self.assertIn("Accessibility", result)
        self.windows.make_full_screen.side_effect = RuntimeError()
        self.assertIn("couldn't make", self.request("open Outlook in full screen")[0])

    def test_full_screen_failure_logs_the_actual_reason(self):
        self.windows.make_full_screen.side_effect = RuntimeError("The focused window was not ready.")
        with self.assertLogs("actions.app_control", level="WARNING") as logs:
            text, _ = self.request("open Outlook in full screen")
        self.assertIn("couldn't make", text)
        self.assertIn("The focused window was not ready.", logs.output[0])
        self.assertIn("Microsoft Outlook", logs.output[0])

    def test_model_planner_sees_only_current_named_apps_and_never_streams_json(self):
        self.backend.respond.return_value = '{"action":"open_app","app":"Spotify","full_screen":false}'
        chunks = []
        result, _ = self.request("open the music app Spotify on my screen", on_text=chunks.append,
                                 history=[{"role": "user", "content": "Outlook"}])
        self.assertEqual(result, "Opened Spotify.")
        self.assertEqual(chunks, [result])
        messages = self.backend.respond.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertIn('Installed app names: ["Spotify"]', messages[0]["content"])
        self.assertNotIn("Outlook", messages[0]["content"])

    def test_switch_planner_keeps_requested_action_and_current_target(self):
        self.backend.respond.return_value = '{"action":"switch_app","app":"Spotify","full_screen":false}'
        reply, _ = self.request("switch to the music app Spotify on my screen")
        self.assertEqual(reply, "Switched to Spotify.")
        prompt = self.backend.respond.call_args.args[0][0]["content"]
        self.assertIn("Requested action: switch_app", prompt)
        self.assertIn('Installed app names: ["Spotify"]', prompt)

    def test_standalone_full_screen_planner_preserves_target_action_and_full_screen(self):
        self.backend.respond.return_value = '{"action":"full_screen_app","app":"Spotify","full_screen":true}'
        self.assertEqual(self.request("make the music app Spotify on my screen full screen")[0],
                         "Made Spotify full screen.")
        prompt = self.backend.respond.call_args.args[0][0]["content"]
        self.assertIn("Requested action: full_screen_app", prompt)
        self.assertIn('Installed app names: ["Spotify"]', prompt)
        self.runner.assert_not_called()

    def test_standalone_full_screen_planner_cannot_launch_or_change_target(self):
        for plan in ('{"action":"open_app","app":"Spotify","full_screen":true}',
                     '{"action":"switch_app","app":"Spotify","full_screen":true}',
                     '{"action":"full_screen_app","app":"Spotify","full_screen":false}',
                     '{"action":"full_screen_app","app":"Outlook","full_screen":true}'):
            self.backend.respond.return_value = plan
            reply, _ = self.request("make the music app Spotify on my screen full screen")
            self.assertIn("couldn't identify", reply)
        self.runner.assert_not_called()
        self.windows.make_full_screen.assert_not_called()

    def test_planner_cannot_change_opening_into_switching_or_reverse(self):
        for text, action in (("switch to the music app Spotify on my screen", "open_app"),
                             ("open the music app Spotify on my screen", "switch_app")):
            self.backend.respond.return_value = (
                '{"action":"' + action + '","app":"Spotify","full_screen":false}')
            self.assertIn("couldn't identify", self.request(text)[0])
        self.runner.assert_not_called()

    def test_invalid_unrequested_or_extended_model_plans_never_execute(self):
        for plan in ('{"action":"delete_file","app":"Spotify","full_screen":false}',
                     '{"action":"open_app","app":"Outlook","full_screen":false}',
                     '{"action":"open_app","app":"Spotify","full_screen":true}',
                     '{"action":"open_app","app":"Spotify","full_screen":"false"}',
                     '{"action":"open_app","app":"Spotify","full_screen":false,"args":["secret"]}',
                     "not JSON", '{"action":"none"}'):
            self.backend.respond.return_value = plan
            self.assertIn("couldn't identify", self.request("open the music app Spotify on my screen")[0])
        self.runner.assert_not_called()

    def test_invalid_inputs_are_rejected_before_actions(self):
        for kwargs in ({"max_new_tokens": 0}, {"max_new_tokens": True}):
            with self.assertRaises(ValueError):
                self.request("open Spotify", **kwargs)
        self.runner.assert_not_called()
        with self.assertRaises(ValueError):
            AppRequest("Spotify", "yes")
        for action in ("close_app", "", None, True):
            with self.assertRaises(ValueError):
                AppRequest("Spotify", action=action)
        with self.assertRaises(ValueError):
            AppRequest("Spotify", action="full_screen_app")

    def test_action_receipt_uses_speech_pipeline_and_shared_conversation_history(self):
        speech, player = Mock(), Mock()
        speech.synthesize.return_value = test_audio()
        assistant = Assistant(backend=self.backend, speech_backend=speech, app_control=self.control,
                              speech_buffer_seconds=0)
        events = []
        workflow = ConversationWorkflow(assistant, transcriber=Mock(return_value="open Spotify"), player=player)
        workflow.execute(text="open Outlook", mode="text", emit=events.append)
        speech.synthesize.assert_not_called()
        response = workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech", emit=events.append)
        self.assertEqual(response.text, "Opened Spotify.")
        self.assertEqual(len(assistant.history), 4)
        self.assertEqual(assistant.history[-1]["content"], response.text)
        self.assertEqual(speech.synthesize.call_args.args[0], response.text)
        player.assert_called_once()
        self.backend.respond.assert_not_called()

    def test_switch_receipt_uses_shared_text_and_speech_workflow(self):
        speech, player = Mock(), Mock()
        speech.synthesize.return_value = test_audio()
        assistant = Assistant(backend=self.backend, speech_backend=speech, app_control=self.control,
                              speech_buffer_seconds=0)
        workflow = ConversationWorkflow(assistant, transcriber=Mock(return_value="switch to Spotify"),
                                        player=player)
        workflow.execute(text="go to Outlook", mode="text")
        speech.synthesize.assert_not_called()
        response = workflow.execute(audio=RecordedAudio([0.1], 16000), mode="speech")
        self.assertEqual(response.text, "Switched to Spotify.")
        self.assertEqual(assistant.history[-1]["content"], response.text)
        speech.synthesize.assert_called_once()
        self.assertEqual(speech.synthesize.call_args.args[0], response.text)
        player.assert_called_once()
        self.backend.respond.assert_not_called()

    def test_app_control_can_be_disabled_for_embedded_text_only_clients(self):
        self.backend.respond.return_value = "No local actions here."
        assistant = Assistant(backend=self.backend, app_control=False)
        self.assertEqual(assistant.respond("open Spotify").text, "No local actions here.")
        self.runner.assert_not_called()


class AppCatalogTests(unittest.TestCase):
    def test_bundle_metadata_aliases_nested_utilities_and_symlink_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Applications"
            app = root / "Utilities/Microsoft Outlook.app"
            (app / "Contents").mkdir(parents=True)
            with (app / "Contents/Info.plist").open("wb") as file:
                plistlib.dump({"CFBundlePackageType": "APPL", "CFBundleName": "Outlook"}, file)
            outside = Path(directory) / "Outside.app"
            outside.mkdir()
            (root / "linked.app").symlink_to(outside, target_is_directory=True)
            (root / "Broken.app").mkdir()
            apps = installed_apps([root])
            self.assertEqual(len(apps), 1)
            self.assertEqual(apps[0].path, app.resolve())
            self.assertEqual(apps[0].aliases, frozenset(("outlook", "microsoft outlook")))


class NativeWindowLogicTests(unittest.TestCase):
    def windows(self, full_screen, settable=True):
        windows = object.__new__(MacOSWindows)
        windows.cf = Mock()
        windows.cf.CFGetTypeID.return_value = 1
        windows.cf.CFBooleanGetTypeID.return_value = 1
        windows.cf.CFBooleanGetValue.return_value = full_screen
        windows.cf.CFEqual.side_effect = lambda a, b: a == b
        windows.finished_launching = Mock(return_value=True)
        windows._perform_full_screen_action = Mock(return_value=False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax = Mock()
        windows.true = 99
        @contextmanager
        def attribute(element, name):
            yield 10, None if name in ("AXRole", "AXSubrole", "AXMinimized", "AXWindows") else 20
        windows._attribute = attribute
        def is_settable(window, key, pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte))[0] = settable
            return 0
        windows.ax.AXUIElementIsAttributeSettable.side_effect = is_settable
        windows.ax.AXUIElementSetAttributeValue.return_value = 0
        windows.ax.AXUIElementCreateApplication.return_value = 11
        return windows

    def test_switch_wait_handles_an_existing_app_without_accessibility(self):
        windows = self.windows(False)
        windows.is_trusted.return_value = False
        windows.is_active = Mock(side_effect=[False, True])
        with patch("actions.macos_windows.time.sleep"):
            windows.wait_until_active(SPOTIFY.path)
        self.assertEqual(windows.is_active.call_count, 2)
        windows.is_active.assert_called_with(42)
        windows.is_trusted.assert_not_called()
        windows.ax.AXUIElementCreateApplication.assert_not_called()

    def test_switch_wait_handles_initial_launch_and_late_activation(self):
        windows = self.windows(False)
        windows.running_pid.side_effect = [None, 42, 42, 42]
        windows.finished_launching.side_effect = [False, True, True]
        windows.is_active = Mock(side_effect=[False, True])
        with patch("actions.macos_windows.time.sleep"):
            windows.wait_until_active(SPOTIFY.path)
        self.assertEqual(windows.running_pid.call_count, 4)
        self.assertEqual(windows.is_active.call_count, 2)

    def test_switch_wait_times_out_when_another_app_stays_active(self):
        windows = self.windows(False)
        windows.is_active = Mock(return_value=False)
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0, 21]), \
                patch("actions.macos_windows.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become active"):
                windows.wait_until_active(SPOTIFY.path)
        windows.is_active.assert_called_once_with(42)

    def test_wait_stage_requires_finished_launch_and_stable_window_before_returning(self):
        windows = self.windows(False)
        windows.finished_launching.side_effect = [False, True, True]
        windows._window_ready = Mock(return_value=True)
        @contextmanager
        def candidates(application):
            self.assertGreaterEqual(windows.finished_launching.call_count, 2)
            yield [20]
        windows._windows = candidates
        with patch("actions.macos_windows.time.sleep"):
            windows.wait_until_open(SPOTIFY.path)
        self.assertEqual(windows.finished_launching.call_count, 3)
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()
        windows.cf.CFRetain.assert_called_once_with(20)
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [20, 11])

    def test_wait_stage_restarts_stability_check_when_startup_window_is_replaced(self):
        windows = self.windows(False)
        windows._window_ready = Mock(return_value=True)
        snapshots = iter(([101], [102], [102]))
        @contextmanager
        def candidates(application):
            yield next(snapshots)
        windows._windows = candidates
        with patch("actions.macos_windows.time.sleep"):
            windows.wait_until_open(SPOTIFY.path)
        self.assertEqual(windows.finished_launching.call_count, 3)
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [101, 102, 11])
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()

    def test_wait_stage_times_out_without_trying_to_set_full_screen(self):
        windows = self.windows(False)
        windows.running_pid.return_value = None
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0.1, 21]), \
                patch("actions.macos_windows.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "finish opening.*20 seconds"):
                windows.wait_until_open(SPOTIFY.path)
        windows.ax.AXUIElementCreateApplication.assert_not_called()
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()

    def test_acknowledged_but_ignored_setter_uses_one_targeted_action_and_verifies(self):
        windows = self.windows(False)
        windows.cf.CFBooleanGetValue.side_effect = [False, False, True]
        windows._perform_full_screen_action.return_value = True
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0.1, 1.0, 1.1]), \
                patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(SPOTIFY.path)
        windows.ax.AXUIElementSetAttributeValue.assert_called_once_with(20, 10, 99)
        windows._perform_full_screen_action.assert_called_once_with(11, 20)

    def test_replaced_full_screen_window_is_verified_from_fresh_app_windows(self):
        windows = self.windows(False)
        snapshots = iter(([101], [102]))
        @contextmanager
        def candidates(application):
            yield next(snapshots)
        windows._windows = candidates
        windows.cf.CFBooleanGetValue.side_effect = [False, True]
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(SPOTIFY.path)
        windows.ax.AXUIElementSetAttributeValue.assert_called_once_with(101, 10, 99)
        windows._perform_full_screen_action.assert_not_called()
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [101, 11])

    def test_unrelated_background_full_screen_window_cannot_confirm_the_request(self):
        windows = self.windows(False)
        @contextmanager
        def candidates(application):
            yield [101, 102]
        windows._windows = candidates
        windows._full_screen_state = Mock(side_effect=lambda window: False if window == 101 else True)
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0.1, 0.2, 13]), \
                patch("actions.macos_windows.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not enter full screen"):
                windows.make_full_screen(SPOTIFY.path)
        windows._full_screen_state.assert_called_once_with(101)

    def test_window_array_is_used_if_focused_and_main_windows_are_missing(self):
        windows = self.windows(True)
        @contextmanager
        def attribute(element, name):
            yield 10, {(11, "AXWindows"): 900, (101, "AXFullScreen"): 20}.get((element, name))
        windows._attribute = attribute
        windows._array = Mock(side_effect=lambda value: [101] if value == 900 else [])
        windows.make_full_screen(SPOTIFY.path)
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()
        windows._perform_full_screen_action.assert_not_called()

    def test_menu_action_targets_only_enabled_enter_command(self):
        windows = self.windows(True)
        @contextmanager
        def attribute(element, name):
            yield 10, {(11, "AXMenuBar"): 100, (201, "AXTitle"): 701,
                       (202, "AXTitle"): 702, (202, "AXEnabled"): 20}.get((element, name))
        windows._attribute = attribute
        windows._text = Mock(side_effect=lambda value: {701: "Close", 702: "Enter Full Screen"}.get(value))
        windows._menu_nodes = Mock(return_value=iter([201, 202]))
        windows._press = Mock(return_value=True)
        self.assertTrue(windows._enter_full_screen_menu(11))
        windows._press.assert_called_once_with(202)

    def test_exit_full_screen_menu_never_triggers_another_button(self):
        windows = self.windows(True)
        windows._perform_full_screen_action = MacOSWindows._perform_full_screen_action.__get__(windows)
        @contextmanager
        def attribute(element, name):
            yield 10, 100 if name == "AXMenuBar" else 701
        windows._attribute = attribute
        windows._text = Mock(return_value="Exit Full Screen")
        windows._menu_nodes = Mock(return_value=iter([201]))
        windows._press = Mock()
        self.assertFalse(windows._perform_full_screen_action(11, 20))
        windows._press.assert_not_called()

    def test_dedicated_full_screen_button_is_used_without_zoom_or_global_keys(self):
        windows = self.windows(False)
        windows._perform_full_screen_action = MacOSWindows._perform_full_screen_action.__get__(windows)
        windows._enter_full_screen_menu = Mock(return_value=False)
        windows._full_screen_state = Mock(return_value=False)
        reads = []
        @contextmanager
        def attribute(element, name):
            reads.append((element, name))
            yield 10, 203 if name == "AXFullScreenButton" else None
        windows._attribute = attribute
        windows._press = Mock(return_value=True)
        self.assertTrue(windows._perform_full_screen_action(11, 20))
        self.assertEqual(reads, [(20, "AXFullScreenButton")])
        windows._press.assert_called_once_with(203)

    def test_already_full_screen_is_not_toggled_off(self):
        windows = self.windows(True)
        self.assertTrue(windows._ensure_full_screen(123))
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()

    def test_full_screen_sets_true_instead_of_sending_a_keyboard_shortcut(self):
        windows = self.windows(False)
        self.assertFalse(windows._ensure_full_screen(123))
        windows.ax.AXUIElementSetAttributeValue.assert_called_once_with(123, 10, 99)

    def test_read_only_or_refused_window_cannot_report_success(self):
        windows = self.windows(False, settable=False)
        with self.assertRaises(RuntimeError):
            windows._ensure_full_screen(123)
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()
        windows = self.windows(False)
        windows.ax.AXUIElementSetAttributeValue.return_value = -25205
        with self.assertRaises(RuntimeError):
            windows._ensure_full_screen(123)

    def test_full_screen_waits_for_observed_state_and_releases_application(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        windows.cf.CFBooleanGetValue.side_effect = [False, True]
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(SPOTIFY.path)
        windows.ax.AXUIElementSetAttributeValue.assert_called_once()
        windows.cf.CFRetain.assert_called_once_with(20)
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [20, 11])

    def test_unsupported_focused_dialog_falls_back_to_main_window(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        windows.cf.CFBooleanGetValue.side_effect = [False, True]
        @contextmanager
        def attribute(element, name):
            value = {(11, "AXFocusedWindow"): 101, (11, "AXMainWindow"): 102,
                     (101, "AXFullScreen"): None, (102, "AXFullScreen"): 20}.get((element, name))
            yield 10, value
        windows._attribute = attribute
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(OUTLOOK.path)
        windows.ax.AXUIElementSetAttributeValue.assert_called_once_with(102, 10, 99)
        windows.cf.CFRetain.assert_called_once_with(102)
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [102, 11])

    def test_temporary_rejection_is_retried_until_window_is_ready(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        windows.ax.AXUIElementSetAttributeValue.side_effect = [-25204, -25204, 0]
        windows.cf.CFBooleanGetValue.side_effect = [False] * 6 + [True]
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(OUTLOOK.path)
        self.assertEqual(windows.ax.AXUIElementSetAttributeValue.call_count, 3)
        windows.cf.CFRetain.assert_called_once_with(20)

    def test_confirmation_follows_requested_window_when_focus_changes(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        windows.cf.CFBooleanGetValue.side_effect = [False, False, True]
        reads = []
        @contextmanager
        def attribute(element, name):
            reads.append((element, name))
            if element == 11:
                if name == "AXFocusedWindow":
                    yield 10, 101 if reads.count((11, name)) == 1 else 999
                else:
                    yield 10, 101 if name == "AXMainWindow" else None
            else:
                yield 10, 20 if name == "AXFullScreen" else None
        windows._attribute = attribute
        windows._regular_window = Mock(side_effect=lambda window: window != 999)
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(OUTLOOK.path)
        self.assertEqual(reads.count((11, "AXFocusedWindow")), 3)
        self.assertEqual(reads.count((101, "AXFullScreen")), 3)
        windows.ax.AXUIElementSetAttributeValue.assert_called_once_with(101, 10, 99)

    def test_replaced_startup_window_is_reacquired(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        focus_reads = [0]
        @contextmanager
        def attribute(element, name):
            if element == 11:
                if name == "AXFocusedWindow":
                    focus_reads[0] += 1
                    yield 10, 101 if focus_reads[0] == 1 else 102
                else:
                    yield 10, None
            else:
                yield 10, 20 if element == 102 and name == "AXFullScreen" else None
        windows._attribute = attribute
        # The initial startup window accepts the request, then disappears.
        windows._ensure_full_screen = Mock(side_effect=[False, False])
        windows.cf.CFBooleanGetValue.side_effect = [False, True]
        with patch("actions.macos_windows.time.sleep"):
            windows.make_full_screen(OUTLOOK.path)
        self.assertEqual([call.args[0] for call in windows._ensure_full_screen.call_args_list], [101, 102])
        self.assertEqual([call.args[0] for call in windows.cf.CFRelease.call_args_list], [101, 102, 11])

    def test_persistent_rejection_times_out_with_reason_without_claiming_success(self):
        windows = self.windows(False, settable=False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0.1, 13]), \
                patch("actions.macos_windows.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "does not support setting full screen.*12 seconds"):
                windows.make_full_screen(OUTLOOK.path)
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()
        windows.cf.CFRetain.assert_not_called()
        windows.cf.CFRelease.assert_called_once_with(11)

    def test_missing_permission_does_not_attempt_window_control(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=False)
        windows._request_permission = Mock()
        windows.running_pid = Mock()
        with self.assertRaises(PermissionError):
            windows.make_full_screen(SPOTIFY.path)
        windows._request_permission.assert_called_once()
        windows.running_pid.assert_not_called()
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()

    def test_missing_window_times_out_and_releases_application(self):
        windows = self.windows(False)
        windows.is_trusted = Mock(return_value=True)
        windows.running_pid = Mock(return_value=42)
        windows.ax.AXUIElementCreateApplication.return_value = 11
        @contextmanager
        def absent(element, name):
            yield 10, None
        windows._attribute = absent
        with patch("actions.macos_windows.time.monotonic", side_effect=[0, 0.1, 13]), \
                patch("actions.macos_windows.time.sleep"):
            with self.assertRaises(RuntimeError):
                windows.make_full_screen(SPOTIFY.path)
        windows.cf.CFRelease.assert_called_once_with(11)
        windows.ax.AXUIElementSetAttributeValue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
