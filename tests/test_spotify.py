"""Verify explicit music routing, safe song selection and playback ownership."""

import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from actions.app_control import InstalledApp
from actions.spotify import SpotifyControl, SpotifyRequest, SpotifyResult, SPOTIFY_BUNDLE, track_uri
from actions.web import WebAccess, WebSource
from actions.media_control import SpeechMediaSession
from assistant import Assistant
from processing.agent import process_agent_text
from ui.workflow import ConversationWorkflow


URI = "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
URL = "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh"
APP = InstalledApp("Spotify", Path("/Applications/Spotify.app"), frozenset(("spotify",)))


class SpotifyControlTests(unittest.TestCase):
    def setUp(self):
        self.platform = patch("actions.spotify.sys.platform", "darwin")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.runner = Mock(return_value=Mock(stdout=json.dumps({"ok": True, "playing": True,
                                                               "title": "Hello", "artist": "Adele"})))
        self.control = SpotifyControl(runner=self.runner)
        self.control._app = Mock(return_value=APP)

    def test_resume_opens_only_installed_spotify_and_confirms_actual_track(self):
        result = self.control.execute(SpotifyRequest("play"))
        self.assertEqual(result.text, "Playing Hello by Adele on Spotify.")
        self.assertTrue(result.applied)
        self.assertTrue(result.playing)
        self.assertEqual(self.runner.call_args_list[0].args[0], ["/usr/bin/open", "-g", "-a", str(APP.path)])
        self.assertEqual(self.runner.call_args.args[0][-2:], ["play", ""])
        self.assertNotIn("shell", self.runner.call_args.kwargs)

    def test_pause_does_not_launch_spotify_and_verifies_pause(self):
        self.runner.return_value.stdout = '{"ok":true,"playing":false}'
        result = self.control.execute(SpotifyRequest("pause"))
        self.assertEqual(result.text, "Spotify is paused.")
        self.assertTrue(result.applied)
        self.assertFalse(result.playing)
        self.runner.assert_called_once()
        self.assertEqual(self.runner.call_args.args[0][-2:], ["pause", ""])

    def test_named_track_uses_indexed_spotify_match_without_reading_pages(self):
        search, reader = Mock(), Mock()
        search.search.return_value = [WebSource("Hello - song and lyrics by Adele", URL)]
        web = WebAccess(search=search, reader=reader)
        result = self.control.execute(SpotifyRequest("play", query="Hello by Adele"), web_access=web)
        self.assertEqual(result.uri, URI)
        self.assertTrue(result.applied)
        self.assertIn("site:open.spotify.com/track", search.search.call_args.args[0])
        reader.read.assert_not_called()
        self.assertEqual(self.runner.call_args.args[0][-1], URI)

    def test_unrelated_results_other_hosts_and_other_uri_types_never_play(self):
        search = Mock()
        search.search.return_value = [WebSource("Hello by Adele", "https://evil.example/track/" + URI.split(":")[-1]),
                                      WebSource("A different song", URL),
                                      WebSource("Hello by Adele", URL.replace("/track/", "/album/"))]
        result = self.control.execute(SpotifyRequest("play", query="Hello by Adele"),
                                      web_access=WebAccess(search=search, reader=Mock()))
        self.assertFalse(result.applied)
        self.assertIn("couldn't find", result.text)
        self.runner.assert_not_called()

    def test_disabled_web_blocks_named_search_but_allows_direct_track(self):
        web = Mock()
        result = self.control.execute(SpotifyRequest("play", query="Hello"), web_access=web, web_enabled=False)
        self.assertFalse(result.applied)
        self.runner.assert_not_called()
        web.search_provider.search.assert_not_called()
        self.assertTrue(self.control.execute(SpotifyRequest("play", uri=URI), web_enabled=False).applied)

    def test_automation_denial_and_unconfirmed_state_never_claim_success(self):
        for value in ({"ok": False, "error": "automation"}, {"ok": False, "error": "not_confirmed"},
                      {"ok": True, "playing": False}, {"ok": True, "playing": "true"}, {"arbitrary": True}):
            self.runner.return_value.stdout = json.dumps(value)
            result = self.control.execute(SpotifyRequest("play"))
            self.assertFalse(result.applied)
        self.runner.side_effect = subprocess.TimeoutExpired("Spotify", 15)
        self.assertFalse(self.control.execute(SpotifyRequest("pause")).applied)

    def test_catalog_requires_the_actual_spotify_bundle_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Spotify.app"
            (path / "Contents").mkdir(parents=True)
            app = InstalledApp("Spotify", path, APP.aliases)
            control = SpotifyControl(catalog=lambda: [app], runner=self.runner)
            for bundle_id, expected in (("fake.spotify", None), (SPOTIFY_BUNDLE, app)):
                with (path / "Contents/Info.plist").open("wb") as file:
                    plistlib.dump({"CFBundleIdentifier": bundle_id}, file)
                self.assertEqual(control._app(), expected)

    def test_track_links_are_strict_and_no_other_uri_or_host_is_accepted(self):
        self.assertEqual(track_uri(URL + "?si=share"), URI)
        self.assertEqual(track_uri(URL.replace("/track/", "/intl-de/track/")), URI)
        for value in (URL.replace("open.spotify.com", "open.spotify.com.evil.example"),
                      URL.replace("https://", "file://"), "spotify:search:Hello", "spotify:track:$(whoami)",
                      URL.replace("open.spotify.com", "user@open.spotify.com"), URI + ";open Spotify"):
            self.assertIsNone(track_uri(value))
        for arguments in (("close",), ("pause", "Hello"), ("play", "Hello", URI), ("play", "", "spotify:search:Hello")):
            with self.assertRaises(ValueError):
                SpotifyRequest(*arguments)


class SpotifyRoutingTests(unittest.TestCase):
    def setUp(self):
        self.control = Mock()
        self.control.execute.return_value = SpotifyResult("Spotify is paused.", applied=True, playing=False)
        self.backend = Mock()
        self.backend.respond.return_value = "A normal chat response."
        self.app = Mock()
        self.app.apps.return_value = [APP]

    def route(self, text, **kwargs):
        return process_agent_text(text, spotify_control=self.control, app_control=self.app,
                                  backend=self.backend, **kwargs)

    def test_music_commands_are_typed_and_do_not_generate_model_text(self):
        for text, action in (("Play Spotify.", "play"), ("Please resume the music", "play"),
                             ("Spiele Musik auf Spotify", "play"), ("Spiel die Musik ab", "play"),
                             ("Pause Spotify", "pause"), ("Pausiere die Musik auf Spotify", "pause"),
                             ("Kannst du Musik auf Spotify abspielen?", "play"),
                             ("Kannst du bitte die Musik pausieren?", "pause"),
                             ("Pausier mal Spotify", "pause")):
            with self.subTest(text=text):
                self.route(text)
                self.assertEqual(self.control.execute.call_args.args[0], SpotifyRequest(action))
        self.backend.respond.assert_not_called()
        self.app.execute.assert_not_called()

    def test_named_music_and_direct_links_are_routed_before_web_reading(self):
        for text, expected in (("Play Hello by Adele on Spotify", SpotifyRequest("play", query="Hello by Adele")),
                               ("Spiele Hello von Adele auf Spotify", SpotifyRequest("play", query="Hello von Adele")),
                               ("Kannst du Hello von Adele auf Spotify abspielen?", SpotifyRequest("play", query="Hello von Adele")),
                               ("Play " + URL, SpotifyRequest("play", uri=URI)),
                               ("Play " + URI, SpotifyRequest("play", uri=URI))):
            self.route(text, web_access=Mock())
            self.assertEqual(self.control.execute.call_args.args[0], expected)

    def test_discussion_negation_quotes_and_nonmusic_requests_do_not_control_spotify(self):
        for text in ("How do I pause Spotify?", "Don't play Spotify", '"Play Spotify"',
                     "Explain Spotify playback", "Play chess", "Continue writing my poem"):
            self.assertEqual(self.route(text)[0], "A normal chat response.")
        self.control.execute.assert_not_called()

    def test_shell_actions_and_multi_action_requests_do_not_execute(self):
        for text in ("Play Hello on Spotify; delete files", "Play https://evil.example on Spotify",
                     "Play Hello and delete files on Spotify", "Play spotify:search:Hello"):
            self.route(text)
        self.control.execute.assert_not_called()

    def test_successful_pause_updates_media_ownership_before_receipt(self):
        events = []
        reply, _ = self.route("Pause Spotify", on_music_result=lambda _: events.append("ownership"),
                               on_text=lambda _: events.append("receipt"))
        self.assertEqual(reply, "Spotify is paused.")
        self.assertEqual(events, ["ownership", "receipt"])
        self.control.execute.return_value = SpotifyResult("Permission needed.")
        owner = Mock()
        self.route("Pause Spotify", on_music_result=owner)
        owner.assert_not_called()

    def test_pause_in_voice_session_is_not_undone_when_media_session_ends(self):
        media_control = Mock(last_error="")
        media_control.pause.return_value = {"bundle": SPOTIFY_BUNDLE, "item": "old"}
        session = SpeechMediaSession(media_control)
        assistant = Assistant(backend=self.backend, spotify_control=self.control, web_access=False)
        workflow = ConversationWorkflow(assistant, media_session=session)
        # A typed confirmation is enough to exercise the action lifecycle while
        # avoiding real speech generation in this ownership test.
        session.begin()
        workflow.execute(text="Pause Spotify")
        session.end()
        media_control.resume.assert_not_called()
        self.assertEqual(assistant.history[-1]["content"], "Spotify is paused.")

    def test_music_control_can_be_disabled_for_an_embedded_client(self):
        assistant = Assistant(backend=self.backend, app_control=False, web_access=False)
        self.assertEqual(assistant.respond("Play Spotify").text, "A normal chat response.")


class SpotifyHelperTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is needed for the isolated Spotify helper simulation.")
    def test_fixed_script_verifies_play_pause_and_denial_without_native_effects(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([shutil.which("node"), str(root / "tests/spotify_simulation.js"),
                                 str(root / "actions/spotify.js")], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
