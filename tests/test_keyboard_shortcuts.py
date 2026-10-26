"""Verify bounded global shortcut registration and press/release dispatch."""

import ctypes as C
import unittest
from unittest.mock import Mock, patch

from ui.keyboard_shortcuts import (
    GlobalVoiceShortcuts, HotKeyID, SIGNATURE, PRESSED, RELEASED, NOT_HANDLED,
)


class KeyboardShortcutTests(unittest.TestCase):
    def listener(self):
        listener = GlobalVoiceShortcuts()
        library = Mock()
        library.GetApplicationEventTarget.return_value = 100
        def install(*args):
            C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 200
            return 0
        def register(code, modifiers, key, target, options, reference):
            C.cast(reference, C.POINTER(C.c_void_p))[0] = 300 + key.id
            return 0
        library.InstallEventHandler.side_effect = install
        library.RegisterEventHotKey.side_effect = register
        listener._library = library
        return listener, library

    def start(self, listener, events):
        with patch("ui.keyboard_shortcuts.sys.platform", "darwin"):
            listener.start(events.append)

    def test_registration_keeps_callback_and_two_keys_then_unregisters_everything(self):
        listener, library = self.listener()
        self.start(listener, [])
        self.assertIsNotNone(listener._callback)
        calls = library.RegisterEventHotKey.call_args_list
        self.assertEqual([call.args[0] for call in calls], [0, 49])
        self.assertEqual([call.args[1] for call in calls], [1 << 12] * 2)
        self.assertEqual([call.args[2].signature for call in calls], [SIGNATURE] * 2)
        self.assertEqual([call.args[4] for call in calls], [1, 1])
        listener.stop()
        self.assertEqual([call.args[0] for call in library.UnregisterEventHotKey.call_args_list], [301, 302])
        library.RemoveEventHandler.assert_called_once_with(200)
        self.assertIsNone(listener._callback)
        listener.stop()
        self.assertEqual(library.UnregisterEventHotKey.call_count, 2)

    def test_repeated_presses_do_not_toggle_session_or_send_recording_twice(self):
        listener, _ = self.listener()
        events = []
        self.start(listener, events)
        for key, kind in ((1, PRESSED), (1, PRESSED), (1, RELEASED), (1, PRESSED),
                          (2, PRESSED), (2, PRESSED), (2, RELEASED), (2, RELEASED)):
            listener.dispatch(key, kind)
        self.assertEqual(events, ["toggle_session", "toggle_session", "talk_pressed", "talk_released"])
        listener.stop()
        listener.dispatch(2, PRESSED)
        self.assertEqual(len(events), 4)

    def test_conflict_rolls_back_partial_registration_and_handler(self):
        listener, library = self.listener()
        register = library.RegisterEventHotKey.side_effect
        library.RegisterEventHotKey.side_effect = lambda *args: -9878 if args[2].id == 2 else register(*args)
        with self.assertRaisesRegex(RuntimeError, "Keyboard Shortcuts > Input Sources"):
            self.start(listener, [])
        library.UnregisterEventHotKey.assert_called_once_with(301)
        library.RemoveEventHandler.assert_called_once_with(200)
        self.assertIsNone(listener._callback)
        self.assertEqual(listener._keys, [])

    def test_native_callback_accepts_only_owned_hotkey_ids(self):
        listener, library = self.listener()
        events = []
        self.start(listener, events)
        def parameter(*args):
            C.cast(args[-1], C.POINTER(HotKeyID))[0] = HotKeyID(SIGNATURE, 2)
            return 0
        library.GetEventParameter.side_effect = parameter
        library.GetEventKind.return_value = PRESSED
        self.assertEqual(listener._callback(None, 123, None), 0)
        self.assertEqual(events, ["talk_pressed"])
        library.GetEventParameter.side_effect = lambda *args: 1
        self.assertEqual(listener._callback(None, 123, None), NOT_HANDLED)
        self.assertEqual(events, ["talk_pressed"])
        listener.stop()

    def test_unknown_events_and_unmatched_releases_do_nothing(self):
        listener, _ = self.listener()
        events = []
        self.start(listener, events)
        for key, kind in ((2, RELEASED), (99, PRESSED), (1, 999)):
            listener.dispatch(key, kind)
        self.assertEqual(events, [])
        listener.stop()
