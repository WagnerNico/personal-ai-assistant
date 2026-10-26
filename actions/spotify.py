"""Bounded Spotify desktop playback through its installed scripting interface."""

from dataclasses import dataclass
import json
from pathlib import Path
import plistlib
import re
import subprocess
import sys
from urllib.parse import urlsplit

from .app_control import installed_apps, normalize_name
from .web import WebError


SPOTIFY_BUNDLE = "com.spotify.client"
TRACK_URI = re.compile(r"spotify:track:[A-Za-z0-9]{22}")


def track_uri(value):
    """Accept a track URI or a Spotify track link, never other URL handlers."""
    if not isinstance(value, str):
        return None
    if TRACK_URI.fullmatch(value):
        return value
    try:
        link = urlsplit(value)
        if (link.scheme != "https" or link.hostname != "open.spotify.com" or link.username is not None
                or link.password is not None or link.port is not None):
            return None
        match = re.fullmatch(r"/(?:intl-[a-z]{2}/)?track/([A-Za-z0-9]{22})/?", link.path)
        return "spotify:track:" + match[1] if match else None
    except ValueError:
        return None


@dataclass(frozen=True)
class SpotifyRequest:
    action: str
    query: str = ""
    uri: str = ""

    def __post_init__(self):
        if self.action not in ("play", "pause"):
            raise ValueError("Spotify supports play and pause only.")
        if not isinstance(self.query, str) or len(self.query) > 180 or any(ord(c) < 32 for c in self.query):
            raise ValueError("Use a music query of at most 180 characters without control characters.")
        if not isinstance(self.uri, str) or (self.uri and not TRACK_URI.fullmatch(self.uri)):
            raise ValueError("Only a Spotify track URI can be played.")
        if self.query and self.uri or self.action == "pause" and (self.query or self.uri):
            raise ValueError("A Spotify pause cannot select music; play accepts one query or URI.")


@dataclass(frozen=True)
class SpotifyResult:
    text: str
    applied: bool = False
    playing: bool | None = None
    uri: str = ""


class SpotifyControl:
    """Replaceable local music tool; no OAuth credentials or generated scripts."""

    def __init__(self, *, catalog=installed_apps, runner=subprocess.run):
        self._catalog, self._runner = catalog, runner

    def _app(self):
        for app in self._catalog():
            if "spotify" not in app.aliases:
                continue
            try:
                with (app.path / "Contents/Info.plist").open("rb") as file:
                    if plistlib.load(file).get("CFBundleIdentifier") == SPOTIFY_BUNDLE:
                        return app
            except (OSError, ValueError, plistlib.InvalidFileException):
                continue
        return None

    def _run(self, action, uri=""):
        result = self._runner(["/usr/bin/osascript", "-l", "JavaScript",
                               str(Path(__file__).with_suffix(".js")), action, uri],
                              check=True, capture_output=True, text=True, timeout=15)
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise RuntimeError("Spotify returned an invalid playback result.")
        return value

    def execute(self, request, *, web_access=None, web_enabled=True, on_stage=None):
        if not isinstance(request, SpotifyRequest):
            raise TypeError("SpotifyControl requires a SpotifyRequest.")
        if sys.platform != "darwin":
            return SpotifyResult("Spotify desktop control is currently available only on macOS.")
        app = self._app()
        if app is None:
            return SpotifyResult("I couldn't find the Spotify desktop app. Install Spotify and sign in first.")
        uri = request.uri
        if request.query:
            if not web_enabled or web_access is None:
                return SpotifyResult("Enable web search in Settings to find songs by name, or provide a Spotify track link.")
            if on_stage:
                on_stage("Finding the requested song on Spotify...")
            try:
                # The indexed Spotify title and verified track URL are enough;
                # avoid loading several dynamic pages before starting music.
                query = re.sub(r"\bsite:\S+", "", request.query, flags=re.I)
                sources = web_access.search_provider.search(query + " song site:open.spotify.com/track")
                words = set(normalize_name(request.query).split()) - {
                    "by", "von", "song", "track", "music", "musik", "the", "a", "please", "bitte"}
                for source in list(sources)[:4]:
                    candidate = track_uri(source.url)
                    if candidate and words and words <= set(normalize_name(source.title).split()):
                        uri = candidate
                        break
                if not uri:
                    return SpotifyResult("I couldn't find a matching Spotify song. Add the artist name or provide its Spotify track link.")
            except (WebError, OSError, RuntimeError):
                return SpotifyResult("I couldn't search for that song. Please try again or provide its Spotify track link.")
        if on_stage:
            on_stage("Pausing Spotify..." if request.action == "pause" else "Starting Spotify playback...")
        try:
            if request.action == "play":
                # Launch an existing installed bundle in the background; never
                # pass user text, URLs or launch arguments to the OS opener.
                self._runner(["/usr/bin/open", "-g", "-a", str(app.path)],
                             check=True, capture_output=True, timeout=10)
            value = self._run(request.action, uri)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError):
            return SpotifyResult("I couldn't control Spotify. Check that Spotify is signed in and allow Airis to control Spotify in System Settings > Privacy & Security > Automation.")
        if not value["ok"]:
            if value.get("error") == "automation":
                return SpotifyResult("Allow Airis or its launcher to control Spotify in System Settings > Privacy & Security > Automation, then try again.")
            return SpotifyResult("Spotify didn't confirm the requested playback. Open Spotify, sign in and select a song, then try again.")
        desired = request.action == "play"
        if type(value.get("playing")) is not bool or value["playing"] != desired:
            return SpotifyResult("Spotify didn't confirm the requested playback state. Please check the Spotify app.")
        if not desired:
            return SpotifyResult("Spotify is paused.", applied=True, playing=False)
        title = value.get("title", "")
        artist = value.get("artist", "")
        description = " by ".join(part[:120] for part in (title, artist) if isinstance(part, str) and part.strip())
        return SpotifyResult(f"Playing {description} on Spotify." if description else "Spotify is playing.",
                             applied=True, playing=True, uri=uri)
