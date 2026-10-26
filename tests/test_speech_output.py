"""Integration checks for the speech API without downloading TTS models."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from output import load_prompts
from output.speech_output import _get_default_voice, text_to_speech
from output.voice_design import generate_reference
from output.voice_clone import create_voice_prompt, generate_speech


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SpeechOutputTests(unittest.TestCase):
    def test_default_apple_silicon_api_uses_mlx_in_the_shared_environment(self):
        import io
        import soundfile as sf
        from output.audio import SpeechAudio

        buffer = io.BytesIO()
        sf.write(buffer, [0.0, 0.25, -0.25], 24000, format="WAV", subtype="PCM_16")
        audio = SpeechAudio(buffer.getvalue(), 24000)
        with patch("output.speech_output.platform.system", return_value="Darwin"), patch(
            "output.speech_output.platform.machine", return_value="arm64",
        ), patch("output.mlx_speech.synthesize_speech", return_value=audio) as synthesize:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "reply.wav"
                samples, rate = text_to_speech("Hello.", output_path=path)
                self.assertEqual(samples.tolist(), [0.0, 0.25, -0.25])
                self.assertEqual(rate, 24000)
                self.assertEqual(path.read_bytes(), audio.wav_bytes)
        synthesize.assert_called_once_with("Hello.", language=load_prompts()["language"])

    def test_import_does_not_load_tts_dependencies(self):
        subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; import output.voice_design; import output.voice_clone; "
                "import output.speech_output; "
                "assert not {'torch', 'qwen_tts', 'soundfile', 'numpy'} "
                ".intersection(sys.modules)"
            )],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def test_default_reference_and_prompts_work_from_another_directory(self):
        model = Mock()
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                prompts = load_prompts()
                voice_prompt = create_voice_prompt(model)
            finally:
                os.chdir(original_directory)

        self.assertIs(voice_prompt, model.create_voice_clone_prompt.return_value)
        arguments = model.create_voice_clone_prompt.call_args.kwargs
        self.assertTrue(Path(arguments["ref_audio"]).is_file())
        self.assertEqual(Path(arguments["ref_audio"]).parent.name, "mid")
        self.assertEqual(arguments["ref_text"], prompts["voice_design"]["text"])

    def test_low_profile_keeps_its_original_reference_transcript(self):
        model = Mock()
        reference = PROJECT_ROOT / "output" / "voice_profiles" / "low" / "reference.wav"
        create_voice_prompt(model, reference)
        self.assertEqual(
            model.create_voice_clone_prompt.call_args.kwargs["ref_text"],
            load_prompts()["voice_clone"]["reference_text"],
        )

    def test_missing_reference_fails_before_model_is_called(self):
        model = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                create_voice_prompt(model, Path(directory) / "missing.wav")
        model.create_voice_clone_prompt.assert_not_called()

    def test_reference_text_and_instructions_can_be_overridden(self):
        audio = [0.0, 0.25, -0.25]
        model = Mock()
        model.generate_voice_design.return_value = ([audio], 24000)
        result = generate_reference(
            model, text="Hello.", instruct="A calm voice.",
            language="German", seed=None,
        )
        self.assertIs(result[0], audio)
        self.assertEqual(result[1], 24000)
        model.generate_voice_design.assert_called_once_with(
            text="Hello.", instruct="A calm voice.", language="German",
        )

    def test_repeated_responses_reuse_voice_and_return_audio(self):
        audio = [0.0, 0.25, -0.25]
        model = Mock()
        model.generate_voice_clone.return_value = ([audio], 24000)
        voice_prompt = create_voice_prompt(model)

        for text in ("First response.", "Second response."):
            result = generate_speech(model, text, voice_prompt)
            self.assertIs(result[0], audio)
            self.assertEqual(result[1], 24000)
            self.assertIs(
                model.generate_voice_clone.call_args.kwargs["voice_clone_prompt"],
                voice_prompt,
            )
        model.create_voice_clone_prompt.assert_called_once()
        self.assertEqual(model.generate_voice_clone.call_count, 2)

    def test_optional_output_creates_valid_wav_files(self):
        import soundfile as sf

        audio = [0.0, 0.25, -0.25]
        model = Mock()
        model.generate_voice_clone.return_value = ([audio], 24000)
        model.generate_voice_design.return_value = ([audio], 24000)
        with tempfile.TemporaryDirectory() as directory:
            for filename, generate in (
                ("product.wav", lambda path: text_to_speech(
                    "LLM response.", model=model, voice_prompt=object(), output_path=path,
                )),
                ("speech.wav", lambda path: generate_speech(
                    model, "Hello.", object(), output_path=path,
                )),
                ("reference.wav", lambda path: generate_reference(
                    model, seed=None, output_path=path,
                )),
            ):
                path = Path(directory) / "nested" / filename
                generate(path)
                data, sample_rate = sf.read(path)
                self.assertEqual(sample_rate, 24000)
                self.assertEqual(data.tolist(), audio)
                self.assertEqual(sf.info(path).subtype, "PCM_16")

    def test_product_entry_point_reuses_default_model_and_voice(self):
        model = Mock()
        audio = [0.0, 0.25, -0.25]
        model.generate_voice_clone.return_value = ([audio], 24000)
        _get_default_voice.cache_clear()
        try:
            with patch("output.speech_output.platform.system", return_value="Linux"), patch(
                "output.speech_output.load_clone_model", return_value=model,
            ) as loader:
                for text in ("First LLM response.", "Second LLM response."):
                    result = text_to_speech(text, language="English")
                    self.assertIs(result[0], audio)
                    self.assertEqual(result[1], 24000)
                    arguments = model.generate_voice_clone.call_args.kwargs
                    self.assertEqual(arguments["text"], text)
                    self.assertIs(
                        arguments["voice_clone_prompt"],
                        model.create_voice_clone_prompt.return_value,
                    )
                loader.assert_called_once()
                model.create_voice_clone_prompt.assert_called_once()
        finally:
            _get_default_voice.cache_clear()

    def test_product_entry_point_rejects_invalid_input_before_loading(self):
        with patch("output.speech_output.load_clone_model") as loader:
            for text, error in ((None, TypeError), (123, TypeError), ("", ValueError), (" \n", ValueError)):
                with self.subTest(text=text), self.assertRaises(error):
                    text_to_speech(text)
            for arguments in ({"model": Mock()}, {"voice_prompt": object()}):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    text_to_speech("Hello.", **arguments)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
