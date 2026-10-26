"""A small, non-activating voice indicator drawn with native Qt primitives."""

import math
import sys
from .keyboard_shortcuts import TALK_SHORTCUT

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QMenu, QWidget


STATE_LABELS = {
    "Acknowledging": "Airis",
    "Awaiting question": "Listening",
    "Ready to talk": "Hold to talk",
    "Recording": "Listening",
    "Processing": "Thinking",
    "Speaking": "Speaking",
}


class VoiceOverlay(QWidget):
    """Show above other windows without activating Airis or taking keyboard focus."""

    cancel_requested = Signal()
    open_requested = Signal()
    quit_requested = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("Airis voice activity")
        self.setAccessibleName("Airis voice activity")
        self.setFixedSize(144, 170)
        self.state = "Acknowledging"
        self.can_cancel = False
        self.phase = 0.0
        self._platform_ready = False
        self.animation = QTimer(self)
        self.animation.setInterval(40)
        self.animation.timeout.connect(self.animate)

    def set_state(self, state, *, can_cancel=False):
        self.state, self.can_cancel = state, can_cancel
        label = STATE_LABELS.get(state, "Airis")
        self.setAccessibleDescription(label)
        hint = f" Hold {TALK_SHORTCUT} to speak." if state == "Ready to talk" else ""
        self.setToolTip(f"{label}.{hint} Click to end the voice session." if can_cancel else f"{label}. Right-click for Airis controls.")
        self.update()

    def present(self):
        if self.isVisible():
            return
        if not self._platform_ready:
            if sys.platform == "darwin" and QGuiApplication.platformName() == "cocoa":
                from .macos_overlay import configure_panel
                self._platform_ready = configure_panel(self.winId())
            else:
                self._platform_ready = True
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(available.right() - self.width() - 28, available.bottom() - self.height() - 28)
        self.show()
        if self.windowHandle() is not None:
            # Screen geometry changes must keep the indicator on a visible display.
            screen = self.windowHandle().screen()
            if screen is not None:
                available = screen.availableGeometry()
                self.move(available.right() - self.width() - 28, available.bottom() - self.height() - 28)
        self.animation.start()

    def dismiss(self):
        self.animation.stop()
        self.hide()

    def animate(self):
        self.phase += 0.12
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(72, 70)
        pulse = 0.5 + 0.5 * math.sin(self.phase)
        radius = 41 + pulse * 3
        glow = QRadialGradient(center, 68)
        glow.setColorAt(0, QColor(75, 221, 204, 145))
        glow.setColorAt(0.68, QColor(62, 169, 190, 55))
        glow.setColorAt(1, QColor(32, 80, 105, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, 68, 68)
        orb = QRadialGradient(QPointF(57, 53), 85)
        orb.setColorAt(0, QColor("#b7f8e7"))
        orb.setColorAt(0.4, QColor("#58cdbd"))
        orb.setColorAt(0.78, QColor("#548bd9" if self.state == "Processing" else "#326ea1"))
        orb.setColorAt(1, QColor("#252e62"))
        painter.setBrush(orb)
        painter.setPen(QPen(QColor(190, 245, 245, 170), 1.2))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(QColor("#f4fff9"), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for index in range(5):
            height = 5 + 14 * (0.5 + 0.5 * math.sin(self.phase * 1.7 + index * 0.85))
            x = center.x() + (index - 2) * 10
            painter.drawLine(QPointF(x, center.y() - height), QPointF(x, center.y() + height))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(16, 23, 34, 235))
        painter.drawRoundedRect(QRectF(12, 132, 120, 29), 14, 14)
        painter.setPen(QColor("#eaf7f4"))
        painter.setFont(QFont("Helvetica Neue", 12))
        painter.drawText(QRectF(12, 132, 120, 29), Qt.AlignmentFlag.AlignCenter,
                         STATE_LABELS.get(self.state, "Airis"))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.can_cancel:
            self.cancel_requested.emit()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("Open Airis", self.open_requested.emit)
        cancel = menu.addAction("End voice session", self.cancel_requested.emit)
        cancel.setEnabled(self.can_cancel)
        menu.addSeparator()
        menu.addAction("Quit Airis", self.quit_requested.emit)
        menu.exec(event.globalPos())
