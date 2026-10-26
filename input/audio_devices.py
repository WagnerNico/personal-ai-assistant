"""Select standby inputs without activating a Bluetooth headset's voice profile."""

import ctypes as C
import platform


def _code(value):
    return int.from_bytes(value.encode("ascii"), "big")


def macos_input_transports():
    """Read device names and transports through CoreAudio; never open a stream."""
    core = C.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    class Address(C.Structure):
        _fields_ = [("selector", C.c_uint32), ("scope", C.c_uint32), ("element", C.c_uint32)]
    get = core.AudioObjectGetPropertyData
    get.argtypes = [C.c_uint32, C.POINTER(Address), C.c_uint32, C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p]
    get.restype = C.c_int32
    cf.CFStringGetCString.argtypes = [C.c_void_p, C.c_char_p, C.c_long, C.c_uint32]
    cf.CFStringGetCString.restype = C.c_bool
    cf.CFRelease.argtypes = [C.c_void_p]
    cf.CFRelease.restype = None
    def read(device, selector, data):
        size = C.c_uint32(C.sizeof(data))
        address = Address(_code(selector), _code("glob"), 0)
        if get(device, C.byref(address), 0, None, C.byref(size), C.byref(data)):
            raise RuntimeError("CoreAudio device information is unavailable.")
        return data
    size = C.c_uint32()
    address = Address(_code("dev#"), _code("glob"), 0)
    get_size = core.AudioObjectGetPropertyDataSize
    get_size.argtypes = [C.c_uint32, C.POINTER(Address), C.c_uint32, C.c_void_p, C.POINTER(C.c_uint32)]
    if get_size(1, C.byref(address), 0, None, C.byref(size)):
        raise RuntimeError("CoreAudio device information is unavailable.")
    devices = read(1, "dev#", (C.c_uint32 * (size.value // 4))())
    result = {}
    for device in devices:
        try:
            name = read(device, "lnam", C.c_void_p()).value
            if not name:
                continue
            try:
                buffer = C.create_string_buffer(1024)
                if not cf.CFStringGetCString(name, buffer, len(buffer), 0x08000100):
                    continue
                label = buffer.value.decode("utf-8")
            finally:
                cf.CFRelease(name)
            transport = read(device, "tran", C.c_uint32()).value.to_bytes(4, "big").decode("ascii", errors="replace")
            result[label] = transport
        except RuntimeError:
            continue
    return result


def standby_input_device(sounddevice, requested=None, *, transports=None):
    """Use a wired/default input or the built-in mic instead of idle Bluetooth.

    Explicit embedding-client selections remain binding. Active question capture
    still uses the caller's chosen/default microphone; only standby is rerouted.
    """
    if requested is not None or platform.system() != "Darwin":
        return requested
    default = sounddevice.query_devices(kind="input")
    try:
        transports = macos_input_transports() if transports is None else transports
    except (OSError, RuntimeError):
        transports = {}
    transport = transports.get(default["name"])
    safe = {"bltn", "usb ", "pci ", "thun", "1394"}
    if transport in safe:
        return None
    devices = list(sounddevice.query_devices())
    candidates = [(index, device) for index, device in enumerate(devices) if device["max_input_channels"] > 0]
    for index, device in candidates:
        if transports.get(device["name"]) == "bltn" or any(
                marker in device["name"].casefold() for marker in ("macbook", "built-in", "internal microphone")):
            return index
    for index, device in candidates:
        if transports.get(device["name"]) in safe:
            return index
    # An uncertain fallback must not silently hold a headset in call mode.
    raise RuntimeError("Wake listening needs a built-in or wired microphone to preserve headphone audio quality.")
