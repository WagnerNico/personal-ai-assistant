"""Verify app activation and set full-screen state through public macOS APIs."""

from contextlib import contextmanager, ExitStack
import ctypes as C
from pathlib import Path
from threading import current_thread, main_thread
import time


class MacOSWindows:
    """Target only the requested bundle's window; never toggle the frontmost app."""

    def __init__(self):
        self.ax = C.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        self.cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.objc = C.CDLL("/usr/lib/libobjc.A.dylib")
        self.appkit = C.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        self.objc.objc_getClass.argtypes = [C.c_char_p]
        self.objc.objc_getClass.restype = C.c_void_p
        self.objc.sel_registerName.argtypes = [C.c_char_p]
        self.objc.sel_registerName.restype = C.c_void_p
        self.pointer = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.number = C.CFUNCTYPE(C.c_long, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.item = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_ulong)(("objc_msgSend", self.objc))
        self.string = C.CFUNCTYPE(C.c_char_p, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.void = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.boolean = C.CFUNCTYPE(C.c_ubyte, C.c_void_p, C.c_void_p)(("objc_msgSend", self.objc))
        self.application_for_pid = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p, C.c_int)(("objc_msgSend", self.objc))
        self._bind(self.cf, "CFRelease", None, C.c_void_p)
        self._bind(self.cf, "CFRetain", C.c_void_p, C.c_void_p)
        self._bind(self.cf, "CFEqual", C.c_ubyte, C.c_void_p, C.c_void_p)
        self._bind(self.cf, "CFArrayGetTypeID", C.c_ulong)
        self._bind(self.cf, "CFArrayGetCount", C.c_long, C.c_void_p)
        self._bind(self.cf, "CFArrayGetValueAtIndex", C.c_void_p, C.c_void_p, C.c_long)
        self._bind(self.cf, "CFStringGetTypeID", C.c_ulong)
        self._bind(self.cf, "CFStringCreateWithCString", C.c_void_p, C.c_void_p, C.c_char_p, C.c_uint32)
        self._bind(self.cf, "CFBooleanGetValue", C.c_ubyte, C.c_void_p)
        self._bind(self.cf, "CFBooleanGetTypeID", C.c_ulong)
        self._bind(self.cf, "CFGetTypeID", C.c_ulong, C.c_void_p)
        self._bind(self.cf, "CFRunLoopRunInMode", C.c_int, C.c_void_p, C.c_double, C.c_ubyte)
        self._bind(self.cf, "CFDictionaryCreate", C.c_void_p, C.c_void_p,
                   C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.c_long, C.c_void_p, C.c_void_p)
        self._bind(self.ax, "AXIsProcessTrustedWithOptions", C.c_ubyte, C.c_void_p)
        self._bind(self.ax, "AXUIElementCreateApplication", C.c_void_p, C.c_int)
        self._bind(self.ax, "AXUIElementCopyAttributeValue", C.c_int, C.c_void_p,
                   C.c_void_p, C.POINTER(C.c_void_p))
        self._bind(self.ax, "AXUIElementIsAttributeSettable", C.c_int, C.c_void_p,
                   C.c_void_p, C.POINTER(C.c_ubyte))
        self._bind(self.ax, "AXUIElementSetAttributeValue", C.c_int, C.c_void_p, C.c_void_p, C.c_void_p)
        self._bind(self.ax, "AXUIElementSetMessagingTimeout", C.c_int, C.c_void_p, C.c_float)
        self._bind(self.ax, "AXUIElementPerformAction", C.c_int, C.c_void_p, C.c_void_p)
        self.true = C.c_void_p.in_dll(self.cf, "kCFBooleanTrue").value
        self._permission_prompted = False

    @staticmethod
    def _bind(library, name, result, *arguments):
        function = getattr(library, name)
        function.restype = result
        function.argtypes = list(arguments)

    def _sel(self, name):
        return self.objc.sel_registerName(name.encode())

    def is_trusted(self):
        return bool(self.ax.AXIsProcessTrustedWithOptions(None))

    def _request_permission(self):
        if self._permission_prompted:
            return
        self._permission_prompted = True
        key = C.c_void_p.in_dll(self.ax, "kAXTrustedCheckOptionPrompt").value
        options = self.cf.CFDictionaryCreate(None, (C.c_void_p * 1)(key),
                                            (C.c_void_p * 1)(self.true), 1, None, None)
        try:
            self.ax.AXIsProcessTrustedWithOptions(options)
        finally:
            self.cf.CFRelease(options)

    def running_pid(self, path: Path):
        """Look up an exact bundle URL, including duplicate bundle identifiers."""
        pool = self.pointer(self.objc.objc_getClass(b"NSAutoreleasePool"), self._sel("new"))
        try:
            workspace = self.pointer(self.objc.objc_getClass(b"NSWorkspace"), self._sel("sharedWorkspace"))
            # NSWorkspace's app list refreshes on the main run loop. Qt runs it
            # in the product; synchronous CLI callers need to service it here.
            if current_thread() is main_thread():
                mode = C.c_void_p.in_dll(self.cf, "kCFRunLoopDefaultMode").value
                self.cf.CFRunLoopRunInMode(mode, 0.01, True)
            apps = self.pointer(workspace, self._sel("runningApplications"))
            for index in range(self.number(apps, self._sel("count"))):
                app = self.item(apps, self._sel("objectAtIndex:"), index)
                url = self.pointer(app, self._sel("bundleURL"))
                value = self.pointer(url, self._sel("path")) if url else None
                raw = self.string(value, self._sel("UTF8String")) if value else None
                if raw and Path(raw.decode()).resolve() == path.resolve():
                    return self.number(app, self._sel("processIdentifier"))
            return None
        finally:
            self.void(pool, self._sel("drain"))

    def finished_launching(self, pid):
        """Read the requested process's launch state without activating another app."""
        pool = self.pointer(self.objc.objc_getClass(b"NSAutoreleasePool"), self._sel("new"))
        try:
            app = self.application_for_pid(self.objc.objc_getClass(b"NSRunningApplication"),
                                           self._sel("runningApplicationWithProcessIdentifier:"), pid)
            return bool(app and self.boolean(app, self._sel("isFinishedLaunching"))
                        and not self.boolean(app, self._sel("isTerminated")))
        finally:
            self.void(pool, self._sel("drain"))

    def is_active(self, pid):
        """Check the target process's foreground state without Accessibility access."""
        pool = self.pointer(self.objc.objc_getClass(b"NSAutoreleasePool"), self._sel("new"))
        try:
            app = self.application_for_pid(self.objc.objc_getClass(b"NSRunningApplication"),
                                           self._sel("runningApplicationWithProcessIdentifier:"), pid)
            return bool(app and self.boolean(app, self._sel("isActive"))
                        and not self.boolean(app, self._sel("isTerminated")))
        finally:
            self.void(pool, self._sel("drain"))

    def wait_until_active(self, path: Path, *, timeout=20.0):
        """Confirm activation after open -a, including an app's initial launch."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pid = self.running_pid(path)
            if pid and self.finished_launching(pid) and self.is_active(pid):
                return
            time.sleep(0.1)
        raise RuntimeError(f"The requested app did not become active within {timeout:g} seconds.")

    @contextmanager
    def _attribute(self, element, name):
        key = self.cf.CFStringCreateWithCString(None, name.encode(), 0x08000100)
        value = C.c_void_p()
        try:
            error = self.ax.AXUIElementCopyAttributeValue(element, key, C.byref(value))
            yield key, value.value if error == 0 else None
        finally:
            if value.value:
                self.cf.CFRelease(value.value)
            self.cf.CFRelease(key)

    def _text(self, value):
        if value is None or self.cf.CFGetTypeID(value) != self.cf.CFStringGetTypeID():
            return None
        raw = self.string(value, self._sel("UTF8String"))
        return raw.decode("utf-8") if raw else None

    def _array(self, value):
        if value is None or self.cf.CFGetTypeID(value) != self.cf.CFArrayGetTypeID():
            return []
        return [self.cf.CFArrayGetValueAtIndex(value, index)
                for index in range(min(128, self.cf.CFArrayGetCount(value)))]

    @contextmanager
    def _windows(self, application):
        """Keep candidate references alive while checking fresh app-owned windows."""
        with ExitStack() as stack:
            windows = []
            for name in ("AXFocusedWindow", "AXMainWindow", "AXWindows"):
                _, value = stack.enter_context(self._attribute(application, name))
                candidates = self._array(value) if name == "AXWindows" else ([value] if value else [])
                for window in candidates:
                    if not any(self.cf.CFEqual(window, previous) for previous in windows):
                        windows.append(window)
            yield windows

    def _full_screen_state(self, window):
        with self._attribute(window, "AXFullScreen") as (_, value):
            if value is not None and self.cf.CFGetTypeID(value) == self.cf.CFBooleanGetTypeID():
                return bool(self.cf.CFBooleanGetValue(value))
        return None

    def _regular_window(self, window):
        with self._attribute(window, "AXRole") as (_, value):
            role = self._text(value)
            if role is not None and role != "AXWindow":
                return False
        with self._attribute(window, "AXSubrole") as (_, value):
            if self._text(value) in ("AXDialog", "AXSystemDialog", "AXSheet"):
                return False
        return True

    def _window_ready(self, window):
        if not self._regular_window(window):
            return False
        with self._attribute(window, "AXMinimized") as (_, value):
            if value is not None and self.cf.CFGetTypeID(value) == self.cf.CFBooleanGetTypeID():
                if self.cf.CFBooleanGetValue(value):
                    return False
        return self._full_screen_state(window) is not None

    def _require_access(self):
        if not self.is_trusted():
            self._request_permission()
            raise PermissionError("Full screen requires macOS Accessibility permission.")

    def wait_until_open(self, path: Path, *, timeout=20.0):
        """Wait for completed launch and the same reachable window on two polls."""
        self._require_access()
        deadline = time.monotonic() + timeout
        application, application_pid, previous_window = None, None, None
        try:
            while time.monotonic() < deadline:
                pid = self.running_pid(path)
                if pid != application_pid:
                    if previous_window:
                        self.cf.CFRelease(previous_window)
                        previous_window = None
                    if application:
                        self.cf.CFRelease(application)
                        application = None
                    application_pid = pid
                if pid and self.finished_launching(pid):
                    if application is None:
                        application = self.ax.AXUIElementCreateApplication(pid)
                        self.ax.AXUIElementSetMessagingTimeout(application, 0.25)
                    with self._windows(application) as windows:
                        ready = next((window for window in windows if self._window_ready(window)), None)
                        if ready and previous_window and self.cf.CFEqual(ready, previous_window):
                            return
                        if previous_window:
                            self.cf.CFRelease(previous_window)
                            previous_window = None
                        if ready:
                            self.cf.CFRetain(ready)
                            previous_window = ready
                elif previous_window:
                    self.cf.CFRelease(previous_window)
                    previous_window = None
                time.sleep(0.1)
            raise RuntimeError(f"The app did not finish opening a usable window within {timeout:g} seconds.")
        finally:
            if previous_window:
                self.cf.CFRelease(previous_window)
            if application:
                self.cf.CFRelease(application)

    def _menu_nodes(self, element, *, deadline, depth=0, budget=None):
        budget = [96] if budget is None else budget
        if depth >= 4 or budget[0] <= 0 or time.monotonic() >= deadline:
            return
        with self._attribute(element, "AXChildren") as (_, children):
            for child in self._array(children):
                if budget[0] <= 0 or time.monotonic() >= deadline:
                    break
                budget[0] -= 1
                yield child
                yield from self._menu_nodes(child, deadline=deadline, depth=depth + 1, budget=budget)

    def _press(self, element):
        action = self.cf.CFStringCreateWithCString(None, b"AXPress", 0x08000100)
        try:
            return self.ax.AXUIElementPerformAction(element, action) == 0
        finally:
            self.cf.CFRelease(action)

    def _enter_full_screen_menu(self, application):
        """Press an explicit Enter command; never an Exit or generic toggle item."""
        enter = {"enter full screen", "enter fullscreen", "vollbildmodus aktivieren",
                 "vollbild ein", "vollbildmodus ein", "vollbild aktivieren", "in den vollbildmodus wechseln"}
        exit_titles = {"exit full screen", "exit fullscreen", "vollbildmodus beenden",
                       "vollbild aus", "vollbildmodus aus", "vollbild beenden", "vollbildmodus verlassen"}
        with self._attribute(application, "AXMenuBar") as (_, menu):
            if not menu:
                return False
            for item in self._menu_nodes(menu, deadline=time.monotonic() + 1.5):
                with self._attribute(item, "AXTitle") as (_, value):
                    title = (self._text(value) or "").strip().casefold()
                if title in exit_titles:
                    return None
                if title not in enter:
                    continue
                with self._attribute(item, "AXEnabled") as (_, enabled):
                    if enabled is None or self.cf.CFGetTypeID(enabled) != self.cf.CFBooleanGetTypeID():
                        continue
                    if not self.cf.CFBooleanGetValue(enabled):
                        continue
                return self._press(item)
        return False

    def _perform_full_screen_action(self, application, window):
        result = self._enter_full_screen_menu(application)
        if result is not False:
            return bool(result)
        # The dedicated full-screen button is distinct from the zoom button.
        # Do not maximize a window or send keys to whichever app happens to have focus.
        with self._attribute(window, "AXFullScreenButton") as (_, button):
            return bool(button and self._full_screen_state(window) is False and self._press(button))

    def _ensure_full_screen(self, window):
        with self._attribute(window, "AXFullScreen") as (key, current):
            if current is None:
                raise RuntimeError("This app window does not expose a full-screen state.")
            if self.cf.CFGetTypeID(current) != self.cf.CFBooleanGetTypeID():
                raise RuntimeError("The app returned an unsupported full-screen state.")
            if self.cf.CFBooleanGetValue(current):
                return True
            settable = C.c_ubyte()
            error = self.ax.AXUIElementIsAttributeSettable(window, key, C.byref(settable))
            if error != 0 or not settable.value:
                raise RuntimeError("This app window does not support setting full screen.")
            error = self.ax.AXUIElementSetAttributeValue(window, key, self.true)
            if error != 0:
                raise RuntimeError(f"macOS refused to set full screen (Accessibility error {error}).")
            return False

    def make_full_screen(self, path: Path, *, timeout=12.0):
        self._require_access()
        deadline = time.monotonic() + timeout
        application = None
        application_pid = None
        target = None
        requested_at = None
        fallback_sent = False
        last_error = "The app did not provide a full-screen window."
        try:
            while True:
                # Capture once per iteration so the launch and animation budgets
                # remain bounded even when the app replaces its process/window.
                now = time.monotonic()
                if now >= deadline:
                    break
                pid = self.running_pid(path)
                if pid != application_pid:
                    if target:
                        self.cf.CFRelease(target)
                        target = None
                    if application:
                        self.cf.CFRelease(application)
                        application = None
                    application_pid = pid
                    requested_at, fallback_sent = None, False
                if not pid or not self.finished_launching(pid):
                    time.sleep(0.1)
                    continue
                if application is None:
                    application = self.ax.AXUIElementCreateApplication(pid)
                    self.ax.AXUIElementSetMessagingTimeout(application, 0.25)
                with self._windows(application) as windows:
                    windows = [window for window in windows if self._regular_window(window)]
                    if target:
                        current_target = None
                        for index, window in enumerate(windows):
                            matches_target = self.cf.CFEqual(window, target)
                            if index != 0 and not matches_target:
                                continue
                            state = self._full_screen_state(window)
                            if state is True:
                                return
                            if state is False and matches_target:
                                current_target = window
                        if current_target is None:
                            self.cf.CFRelease(target)
                            target = None
                            requested_at, fallback_sent = None, False
                        elif not fallback_sent and now - requested_at >= 0.8:
                            # Some apps acknowledge AXFullScreen without changing
                            # it. Use the app's explicit menu/button once and then
                            # verify fresh window state; never send a global shortcut.
                            fallback_sent = self._perform_full_screen_action(application, current_target)
                            if not fallback_sent:
                                if self._ensure_full_screen(current_target):
                                    return
                                requested_at = now
                        if target:
                            time.sleep(0.1)
                            continue
                    for window in windows:
                        try:
                            already_full = self._ensure_full_screen(window)
                            if already_full:
                                return
                            fallback_sent = False
                        except RuntimeError as error:
                            last_error = str(error)
                            if self._full_screen_state(window) is not False:
                                continue
                            fallback_sent = self._perform_full_screen_action(application, window)
                            if not fallback_sent:
                                continue
                        self.cf.CFRetain(window)
                        target = window
                        requested_at = now
                        last_error = "The requested window did not enter full screen."
                        break
                time.sleep(0.1)
            raise RuntimeError(f"{last_error} Timed out after {timeout:g} seconds.")
        finally:
            if target:
                self.cf.CFRelease(target)
            if application:
                self.cf.CFRelease(application)
