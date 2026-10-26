"""Apply public AppKit panel flags for non-activating voice activity on macOS."""

import ctypes


def configure_panel(native_view):
    """Configure a Cocoa NSPanel before its first show; call only on the UI thread."""
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    number = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    set_number = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(('objc_msgSend', objc))
    is_kind = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
    selector = objc.sel_registerName
    window = pointer(int(native_view), selector(b'window'))
    if not window or not is_kind(window, selector(b'isKindOfClass:'), objc.objc_getClass(b'NSPanel')):
        return False
    # NSWindowStyleMaskNonactivatingPanel prevents application activation.
    style = number(window, selector(b'styleMask')) | (1 << 7)
    set_number(window, selector(b'setStyleMask:'), style)
    # Join desktop Spaces and act as an auxiliary panel in full-screen Spaces.
    behavior = number(window, selector(b'collectionBehavior'))
    behavior &= ~((1 << 1) | (1 << 7) | (1 << 9))
    behavior |= (1 << 0) | (1 << 6) | (1 << 8)
    set_number(window, selector(b'setCollectionBehavior:'), behavior)
    return True
