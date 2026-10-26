"""Build a local macOS launcher using the shared Airis environment."""

from pathlib import Path
import plistlib
import shlex


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_launcher(destination: str | Path | None = None, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Create a Finder/Dock launchable .app; this is not a standalone distribution."""
    bundle = Path(destination) if destination is not None else project_root / "Airis.app"
    contents = bundle / "Contents"
    executable = contents / "MacOS" / "Airis"
    executable.parent.mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(exist_ok=True)
    info = {
        "CFBundleName": "Airis",
        "CFBundleDisplayName": "Airis",
        "CFBundleIdentifier": "com.airis.desktop",
        "CFBundleExecutable": "Airis",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSArchitecturePriority": ["arm64", "x86_64"],
        "LSRequiresNativeExecution": True,
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "Airis uses your microphone for Hold to talk and optional wake-word listening while enabled.",
        "NSAppleEventsUsageDescription": "Airis reads Outlook emails without changing their unread status and controls Spotify playback when you request it.",
    }
    with (contents / "Info.plist").open("wb") as file:
        plistlib.dump(info, file)
    executable.write_text(
        "#!/bin/zsh\n"
        f"AIRIS_PROJECT_DIR={shlex.quote(str(project_root.resolve()))}\n"
        'cd -- "$AIRIS_PROJECT_DIR" || exit 1\n'
        'mkdir -p "$AIRIS_PROJECT_DIR/logs" || exit 1\n'
        '"$AIRIS_PROJECT_DIR/.venv/bin/python" -u -m ui.desktop '
        '>> "$AIRIS_PROJECT_DIR/logs/desktop.log" 2>&1\n'
        'AIRIS_EXIT_CODE=$?\n'
        'if [[ "$AIRIS_EXIT_CODE" -ne 0 ]]; then\n'
        "  /usr/bin/osascript -e 'display alert \"Airis could not start\" "
        "message \"See logs/desktop.log in the Airis project for details.\" as critical'\n"
        'fi\n'
        'exit "$AIRIS_EXIT_CODE"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bundle


if __name__ == "__main__":
    print(build_launcher())
