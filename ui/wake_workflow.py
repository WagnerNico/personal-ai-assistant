"""Fixed interface acknowledgements, independent of LLM replies and history."""


ACKNOWLEDGEMENT = "I'm listening."
READY_ACKNOWLEDGEMENT = "I'm ready."


class WakeAcknowledgement:
    """Cache a short cue in the selected voice for immediate reuse."""

    def __init__(self, assistant, player, *, text=ACKNOWLEDGEMENT):
        self.assistant = assistant
        self.player = player
        self.text = text
        self._voice = None
        self._audio = None

    def prepare(self):
        if self._audio is None or self._voice != self.assistant.voice:
            self._audio = self.assistant.synthesize_speech(self.text, language="English")
            self._voice = self.assistant.voice
        return self._audio

    def play(self):
        self.player(self.prepare())
