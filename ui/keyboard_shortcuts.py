"""Register two macOS voice shortcuts without monitoring arbitrary keystrokes."""

import ctypes as C
import sys
from threading import current_thread, main_thread


SESSION_SHORTCUT = "Control + A"
TALK_SHORTCUT = "Control + Space"
MODIFIERS = 1 << 12
SIGNATURE = int.from_bytes(b"AIRS", "big")
KEYBOARD_CLASS = int.from_bytes(b"keyb", "big")
PRESSED, RELEASED = 5, 6
NOT_HANDLED = -9874


class HotKeyID(C.Structure):
    _fields_ = [("signature", C.c_uint32), ("id", C.c_uint32)]


class EventType(C.Structure):
    _fields_ = [("event_class", C.c_uint32), ("kind", C.c_uint32)]


HANDLER = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p)


class GlobalVoiceShortcuts:
    """Use public Carbon hot-key registration on the application's main loop.

    Only registered chords reach this adapter. It needs no keyboard-monitoring
    permission and keeps native references and its callback alive until stop.
    """

    def __init__(self):
        self._library = None
        self._handler = None
        self._callback = None
        self._keys = []
        self._pressed = set()
        self._emit = None

    @staticmethod
    def _main_thread():
        if current_thread() is not main_thread():
            raise RuntimeError("Global shortcuts must be configured on the main thread.")

    def _load(self):
        if self._library is not None:
            return
        library = C.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        signatures = {
            "GetApplicationEventTarget": (C.c_void_p, []),
            "InstallEventHandler": (C.c_int32, [C.c_void_p, HANDLER, C.c_uint32,
                                               C.POINTER(EventType), C.c_void_p, C.POINTER(C.c_void_p)]),
            "RemoveEventHandler": (C.c_int32, [C.c_void_p]),
            "RegisterEventHotKey": (C.c_int32, [C.c_uint32, C.c_uint32, HotKeyID,
                                               C.c_void_p, C.c_uint32, C.POINTER(C.c_void_p)]),
            "UnregisterEventHotKey": (C.c_int32, [C.c_void_p]),
            "GetEventKind": (C.c_uint32, [C.c_void_p]),
            "GetEventParameter": (C.c_int32, [C.c_void_p, C.c_uint32, C.c_uint32,
                                              C.POINTER(C.c_uint32), C.c_ulong,
                                              C.POINTER(C.c_ulong), C.c_void_p]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(library, name)
            function.restype, function.argtypes = result, arguments
        self._library = library

    def start(self, emit):
        self._main_thread()
        if self._handler is not None:
            return
        if sys.platform != "darwin":
            raise RuntimeError("Global voice shortcuts are currently available only on macOS.")
        self._load()
        self._emit = emit
        self._callback = HANDLER(self._handle)
        target = self._library.GetApplicationEventTarget()
        types = (EventType * 2)(EventType(KEYBOARD_CLASS, PRESSED), EventType(KEYBOARD_CLASS, RELEASED))
        handler = C.c_void_p()
        try:
            status = self._library.InstallEventHandler(target, self._callback, 2, types, None, C.byref(handler))
            if status:
                raise RuntimeError(f"macOS could not install voice shortcuts (error {status}).")
            self._handler = handler.value
            for key_code, key_id, label in ((0, 1, SESSION_SHORTCUT), (49, 2, TALK_SHORTCUT)):
                reference = C.c_void_p()
                status = self._library.RegisterEventHotKey(
                    key_code, MODIFIERS, HotKeyID(SIGNATURE, key_id), target, 1, C.byref(reference))
                if status:
                    guidance = (" Check System Settings > Keyboard > Keyboard Shortcuts > Input Sources."
                                if key_id == 2 else " Check for a shortcut conflict.")
                    raise RuntimeError(f"macOS could not register {label} (error {status}).{guidance}")
                self._keys.append(reference.value)
        except BaseException:
            self.stop()
            raise

    def _handle(self, _call, event, _data):
        try:
            key = HotKeyID()
            status = self._library.GetEventParameter(
                event, int.from_bytes(b"----", "big"), int.from_bytes(b"hkid", "big"),
                None, C.sizeof(key), None, C.byref(key))
            if status or key.signature != SIGNATURE or key.id not in (1, 2):
                return NOT_HANDLED
            self.dispatch(key.id, self._library.GetEventKind(event))
            return 0
        except Exception:
            # Never let a Python exception cross the native callback boundary.
            return NOT_HANDLED

    def dispatch(self, key_id, kind):
        if self._emit is None or key_id not in (1, 2):
            return
        if kind == PRESSED and key_id not in self._pressed:
            self._pressed.add(key_id)
            self._emit("toggle_session" if key_id == 1 else "talk_pressed")
        elif kind == RELEASED and key_id in self._pressed:
            self._pressed.remove(key_id)
            if key_id == 2:
                self._emit("talk_released")

    def stop(self):
        self._main_thread()
        if self._library is not None:
            for reference in self._keys:
                self._library.UnregisterEventHotKey(reference)
            if self._handler is not None:
                self._library.RemoveEventHandler(self._handler)
        self._keys.clear()
        self._handler = self._callback = self._emit = None
        self._pressed.clear()
