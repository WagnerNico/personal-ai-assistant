"""Menu-bar controls and graceful application shutdown for background Airis."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


def tray_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(Qt.GlobalColor.black, 2.5))
    painter.drawEllipse(3, 3, 26, 26)
    painter.drawLine(10, 22, 16, 10)
    painter.drawLine(16, 10, 22, 22)
    painter.drawLine(13, 18, 19, 18)
    painter.end()
    icon = QIcon(pixmap)
    icon.setIsMask(True)
    return icon


class BackgroundControls(QObject):
    """Keep an explicit Open/Quit path while the main window is hidden."""

    def __init__(self, window, application):
        super().__init__(application)
        self.window, self.application = window, application
        self.finished = False
        self.menu = QMenu(window)
        self.open_action = self.menu.addAction("Open Airis", window.show_main_window)
        self.voice_action = self.menu.addAction("Start / end voice session", self.toggle_voice_session)
        self.wake_action = self.menu.addAction("Enable wake word")
        self.wake_action.setCheckable(True)
        self.wake_action.setChecked(window.wake_enabled.isChecked())
        self.wake_action.toggled.connect(window.wake_enabled.setChecked)
        window.wake_enabled.toggled.connect(self.wake_action.setChecked)
        self.menu.addSeparator()
        self.quit_action = QAction("Quit Airis", self)
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(window.request_quit)
        self.menu.addAction(self.quit_action)
        window.start_shortcuts()
        native_menu = window.menuBar().addMenu("Airis")
        native_menu.addAction(self.open_action)
        native_menu.addAction(self.quit_action)
        self.tray = QSystemTrayIcon(tray_icon(), self)
        self.tray.setToolTip("Airis")
        self.tray.setContextMenu(self.menu)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()
        else:
            window._background_service = False
        window.shutdown_finished.connect(self.shutdown_finished)
        application.installEventFilter(self)

    def toggle_voice_session(self):
        if self.window._background_voice_turn:
            self.window.cancel_background_listening()
        else:
            self.window.start_background_session()

    def eventFilter(self, watched, event):
        if watched is self.application and event.type() == QEvent.Type.Quit and not self.finished:
            self.window.request_quit()
            return True
        return super().eventFilter(watched, event)

    def shutdown_finished(self):
        self.finished = True
        self.tray.hide()
        self.application.removeEventFilter(self)
        self.application.quit()
