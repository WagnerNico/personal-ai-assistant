"""Resolve installed applications for opening, switching and full screen."""

from dataclasses import dataclass
import logging
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import unicodedata


logger = logging.getLogger(__name__)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


@dataclass(frozen=True)
class InstalledApp:
    name: str
    path: Path
    aliases: frozenset[str]


@dataclass(frozen=True)
class AppRequest:
    app: str
    full_screen: bool = False
    action: str = "open_app"

    def __post_init__(self):
        if not isinstance(self.app, str) or not self.app.strip():
            raise ValueError("An app request requires a nonempty app name.")
        if type(self.full_screen) is not bool:
            raise ValueError("full_screen must be a boolean.")
        if self.action not in ("open_app", "switch_app", "full_screen_app"):
            raise ValueError("An app action must be open_app, switch_app or full_screen_app.")
        if self.action == "full_screen_app" and not self.full_screen:
            raise ValueError("A full_screen_app request requires full_screen=True.")


@dataclass(frozen=True)
class AppResult:
    text: str
    opened: bool = False
    full_screen: bool = False
    switched: bool = False


def installed_apps(roots=None) -> list[InstalledApp]:
    """Read bundles in standard app folders, without traversing app contents."""
    if roots is None:
        roots = (Path("/Applications"), Path.home() / "Applications",
                 Path("/System/Applications"), Path("/System/Library/CoreServices"))
    apps = []
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        candidates = list(root.glob("*.app")) + list(root.glob("*/*.app"))
        for candidate in candidates:
            path = candidate.resolve()
            if path in seen or not path.is_relative_to(root):
                continue
            seen.add(path)
            try:
                with (path / "Contents/Info.plist").open("rb") as file:
                    info = plistlib.load(file)
            except (OSError, ValueError, plistlib.InvalidFileException):
                continue
            if not isinstance(info, dict) or info.get("CFBundlePackageType") != "APPL":
                continue
            names = {path.stem}
            names.update(info[key] for key in ("CFBundleName", "CFBundleDisplayName")
                         if isinstance(info.get(key), str) and info[key].strip())
            aliases = {normalize_name(name) for name in names}
            for name in tuple(aliases):
                if name.startswith(("microsoft ", "google ")):
                    aliases.add(name.split(" ", 1)[1])
            apps.append(InstalledApp(path.stem, path, frozenset(aliases)))
    return sorted(apps, key=lambda app: str(app.path))


class AppControl:
    """Execute a typed request against installed bundles, never a shell command.

    The catalog and native window adapter are replaceable for other platforms
    or an agent's tool registry. Paths, URLs, documents and launch arguments are
    not part of this tool's interface.
    """

    def __init__(self, *, catalog=installed_apps, runner=subprocess.run, windows=None):
        self._catalog = catalog
        self._runner = runner
        self._windows = windows

    def apps(self):
        return self._catalog()

    def execute(self, request: AppRequest, *, on_stage=None) -> AppResult:
        if not isinstance(request, AppRequest):
            raise TypeError("AppControl requires an AppRequest.")
        if sys.platform != "darwin":
            return AppResult("App control is currently available only on macOS.")
        name = normalize_name(request.app)
        matches = [app for app in self.apps() if name in app.aliases]
        if not matches:
            return AppResult(f"I couldn't find an installed app named {request.app}.")
        if len(matches) != 1:
            return AppResult("More than one installed app matches that name. Please use its full app name.")
        app = matches[0]
        if request.action == "full_screen_app":
            try:
                if self._window_adapter().running_pid(app.path) is None:
                    return AppResult(f"{app.name} isn't open. Please open it first.")
            except (OSError, RuntimeError) as error:
                logger.warning("App lookup failed for %s: %s", app.name, error)
                return AppResult(f"I couldn't find a running window for {app.name}.")
            return self._make_full_screen(app, on_stage=on_stage)
        switching = request.action == "switch_app"
        if on_stage:
            on_stage(f"Switching to {app.name}..." if switching else f"Opening {app.name}...")
        try:
            # Launch Services activates an existing app or launches it first.
            # Never use -n (new instance) or -g (background launch).
            self._runner(["/usr/bin/open", "-a", str(app.path)],
                         check=True, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return AppResult(f"macOS couldn't open {app.name}.")
        if switching:
            try:
                self._window_adapter().wait_until_active(app.path)
            except (OSError, RuntimeError) as error:
                logger.warning("App activation failed for %s: %s", app.name, error)
                return AppResult(f"Opened {app.name}, but couldn't switch to it. Please try again.", opened=True)
        receipt = f"Switched to {app.name}" if switching else f"Opened {app.name}"
        if not request.full_screen:
            return AppResult(f"{receipt}.", opened=True, switched=switching)
        return self._make_full_screen(app, on_stage=on_stage, receipt=receipt,
                                      opened=True, switched=switching)

    def _make_full_screen(self, app, *, on_stage=None, receipt=None, opened=False, switched=False):
        """Use the same readiness and confirmation steps with or without launch."""
        stage = "opening"
        try:
            self._window_adapter()
            if on_stage:
                on_stage(f"Waiting for {app.name} to finish opening...")
            self._windows.wait_until_open(app.path)
            stage = "full_screen"
            if on_stage:
                on_stage(f"Making {app.name} full screen...")
            self._windows.make_full_screen(app.path)
        except PermissionError:
            prefix = f"{receipt}." if receipt else f"I couldn't make {app.name} full screen."
            return AppResult(
                f"{prefix} To enable full screen, allow Airis or its Python launcher "
                "in System Settings > Privacy & Security > Accessibility, then try again.",
                opened=opened, switched=switched)
        except (OSError, RuntimeError) as error:
            logger.warning("App action failed for %s during %s: %s", app.name, stage, error)
            if stage == "opening":
                text = (f"{receipt}, but its window isn't ready yet. Please try again." if receipt else
                        f"{app.name}'s window isn't ready yet. Please try again.")
            else:
                text = (f"{receipt}, but couldn't make its window full screen." if receipt else
                        f"I couldn't make {app.name}'s window full screen.")
            return AppResult(text, opened=opened, switched=switched)
        text = f"{receipt} in full screen." if receipt else f"Made {app.name} full screen."
        return AppResult(text, opened=opened, full_screen=True, switched=switched)

    def _window_adapter(self):
        if self._windows is None:
            from .macos_windows import MacOSWindows
            self._windows = MacOSWindows()
        return self._windows
