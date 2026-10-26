"""Post a play/pause media key through native AppKit with Accessibility access."""

import ctypes as C
import time


class _Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


class MacOSMediaKeys:
    """Load lazily; construct both key events before sending either one."""

    def __init__(self):
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self.objc = C.CDLL("/usr/lib/libobjc.A.dylib")
        self.appkit = C.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        self.ax = C.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        self.objc.objc_getClass.argtypes = [C.c_char_p]
        self.objc.objc_getClass.restype = C.c_void_p
        self.objc.sel_registerName.argtypes = [C.c_char_p]
        self.objc.sel_registerName.restype = C.c_void_p
        self.pointer = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.release = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.make_event = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong, _Point, C.c_ulong,
                                     C.c_double, C.c_long, C.c_void_p, C.c_short, C.c_long, C.c_long)(
                                         ("objc_msgSend", self.objc))
        self.ax.AXIsProcessTrusted.argtypes = []
        self.ax.AXIsProcessTrusted.restype = C.c_bool
        self.ax.CGEventGetType.argtypes = [C.c_void_p]
        self.ax.CGEventGetType.restype = C.c_uint32
        self.ax.CGEventPost.argtypes = [C.c_uint32, C.c_void_p]
        self.ax.CGEventPost.restype = None
        self._loaded = True

    def _events(self):
        selector = self.objc.sel_registerName(
            b"otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:")
        events = []
        for flag in (0xA00, 0xB00):
            event = self.make_event(self.objc.objc_getClass(b"NSEvent"), selector,
                                    14, _Point(0, 0), flag, 0, 0, None, 8, (16 << 16) | flag, -1)
            native = self.pointer(event, self.objc.sel_registerName(b"CGEvent")) if event else None
            if not native or self.ax.CGEventGetType(native) != 14:
                raise RuntimeError("macOS could not construct a media-key event.")
            events.append(native)
        return events

    def press_play_pause(self):
        self._load()
        if not self.ax.AXIsProcessTrusted():
            return False
        pool = self.pointer(self.objc.objc_getClass(b"NSAutoreleasePool"), self.objc.sel_registerName(b"new"))
        try:
            for event in self._events():
                self.ax.CGEventPost(0, event)
                time.sleep(0.01)
        finally:
            self.release(pool, self.objc.sel_registerName(b"drain"))
        return True
