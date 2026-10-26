"""Prepare bundled Qt plugins for native desktop startup."""

from pathlib import Path
import os
import stat
import sys


def _clear_hidden_flag(path: Path) -> None:
    flags = path.stat().st_flags
    if flags & stat.UF_HIDDEN:
        try:
            os.chflags(path, flags & ~stat.UF_HIDDEN)
        except OSError as error:
            raise RuntimeError(f"Could not make the Qt plugin discoverable: {path}") from error


def prepare_qt_plugins() -> Path:
    """Ensure Qt can discover its bundled plugins on macOS.

    Qt skips files marked UF_HIDDEN. Some installations preserve that macOS
    flag on plugin libraries; clear only this flag, preserving other flags.
    """
    if sys.platform == "darwin":
        # Qt's Cocoa raise() otherwise activates the process even for a
        # non-focusing tool window. Explicit Open uses activateWindow().
        os.environ["QT_MAC_SET_RAISE_PROCESS"] = "0"
    import PySide6
    from PySide6.QtCore import QCoreApplication, QLibraryInfo

    plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    if not plugins.is_dir():
        plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    if not plugins.is_dir():
        raise RuntimeError("Qt plugins were not found. Reinstall ui/requirements.txt.")
    if sys.platform == "darwin":
        for path in plugins.rglob("*.dylib"):
            _clear_hidden_flag(path)
    QCoreApplication.addLibraryPath(str(plugins))
    return plugins
