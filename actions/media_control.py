"""Temporarily pause the current media item without launching or closing apps."""

import json
from pathlib import Path
import platform
import subprocess
from threading import RLock

from output.telemetry import logger
from .macos_media_keys import MacOSMediaKeys


class MacOSMediaControl:
    """Use a fixed helper; never interpolate media titles into executable code."""

    def __init__(self, *, keys=None):
        self.last_error = ""
        self.keys = keys if keys is not None else MacOSMediaKeys()

    def _run(self, action, token=None):
        arguments = ["/usr/bin/osascript", "-l", "JavaScript", str(Path(__file__).with_suffix(".js")), action]
        if token is not None:
            arguments.append(json.dumps(token, ensure_ascii=True))
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=2, check=True)
        return json.loads(result.stdout)

    def pause(self):
        self.last_error = ""
        before = self._run("status")
        if not before or not before.get("playing"):
            return None
        if self.keys.press_play_pause():
            result = self._run("wait-paused", before)
            if result.get("changed"):
                return before
            self.last_error = "The media player did not respond to pause. You can pause it manually for this conversation."
            return None
        result = self._run("pause")
        if result.get("needsAccessibility"):
            self.last_error = "To pause media, allow Airis in System Settings > Privacy & Security > Accessibility."
        elif "token" in result and not result.get("paused"):
            self.last_error = "The media player did not respond to pause. You can pause it manually for this conversation."
        return result.get("token") if result.get("paused") else None

    def resume(self, token):
        before = self._run("status")
        if not self._same_item(before, token) or before.get("playing"):
            return
        if self.keys.press_play_pause():
            result = self._run("wait-resumed", token)
            if not result.get("changed"):
                self.last_error = "The paused media could not be resumed. Please resume it in its player."
            return
        result = self._run("resume", token)
        if not result.get("resumed") and not result.get("skipped"):
            self.last_error = "The paused media could not be resumed. Please resume it in its player."

    @staticmethod
    def _same_item(a, b):
        return bool(a and b and all(a.get(key) == b.get(key) for key in ("bundle", "title", "item")))


class SpeechMediaSession:
    """Hold playback paused across capture, processing and the complete reply.

    Start/end are idempotent so a wake cue and its subsequent question share one
    interruption. Only an item actually paused by Airis can be resumed.
    """

    def __init__(self, control=None, *, enabled=True):
        self.control = control if control is not None else (MacOSMediaControl() if platform.system() == "Darwin" else None)
        self.enabled = enabled
        self._active = False
        self._token = None
        self.last_error = ""
        self._lock = RLock()

    @property
    def active(self):
        with self._lock:
            return self._active

    def begin(self):
        with self._lock:
            if self._active or not self.enabled or self.control is None:
                return
            self._active = True
            self.last_error = ""
            try:
                self._token = self.control.pause()
                error = getattr(self.control, "last_error", "")
                self.last_error = error if isinstance(error, str) else ""
            except Exception:
                self.last_error = "Background media could not be paused. You can pause it manually for this conversation."
            if self.last_error:
                logger.warning(self.last_error)

    def end(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
            token, self._token = self._token, None
            if token is not None:
                try:
                    self.control.resume(token)
                    error = getattr(self.control, "last_error", "")
                    if isinstance(error, str) and error:
                        self.last_error = error
                except Exception:
                    self.last_error = "The paused media could not be resumed. Please resume it in its player."
                    logger.warning(self.last_error)

    def user_playback(self, bundle, *, playing):
        """Respect an explicit player command instead of restoring stale playback.

        Starting music supersedes the old interruption. An explicit pause only
        drops resume ownership for that player, leaving other media unchanged.
        """
        with self._lock:
            if playing or (self._token and self._token.get("bundle") == bundle):
                self._token = None
                self._active = False
