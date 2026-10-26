"""Check voice-specific cue caching without playback or LLM generation."""

import unittest
from unittest.mock import Mock

from ui.wake_workflow import ACKNOWLEDGEMENT, READY_ACKNOWLEDGEMENT, WakeAcknowledgement


class WakeAcknowledgementTests(unittest.TestCase):
    def test_readiness_cue_is_separate_from_wake_cue_and_cached_in_selected_voice(self):
        assistant, player = Mock(), Mock()
        assistant.voice = "mid"
        wake, ready = object(), object()
        assistant.synthesize_speech.side_effect = [wake, ready]
        wake_cue = WakeAcknowledgement(assistant, player)
        ready_cue = WakeAcknowledgement(assistant, player, text=READY_ACKNOWLEDGEMENT)
        wake_cue.play()
        ready_cue.play()
        ready_cue.play()
        self.assertEqual([call.args[0] for call in assistant.synthesize_speech.call_args_list],
                         [ACKNOWLEDGEMENT, READY_ACKNOWLEDGEMENT])
        self.assertEqual([call.args[0] for call in player.call_args_list], [wake, ready, ready])
        assistant.respond.assert_not_called()

    def test_cached_cue_reuses_selected_voice_and_refreshes_after_voice_change(self):
        assistant, player = Mock(), Mock()
        assistant.voice = 'mid'
        first, second = object(), object()
        assistant.synthesize_speech.side_effect = [first, second]
        cue = WakeAcknowledgement(assistant, player)
        cue.prepare()
        cue.play()
        cue.play()
        assistant.synthesize_speech.assert_called_once_with(ACKNOWLEDGEMENT, language='English')
        self.assertEqual([call.args[0] for call in player.call_args_list], [first, first])
        assistant.voice = 'high'
        cue.play()
        self.assertEqual(assistant.synthesize_speech.call_count, 2)
        self.assertIs(player.call_args.args[0], second)
        assistant.respond.assert_not_called()


if __name__ == '__main__':
    unittest.main()
