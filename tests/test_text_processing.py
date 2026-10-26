"""Verify processing and its model boundary without downloading Qwen weights."""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from processing import process_text
from processing.chat_api import ChatAPIBackend
from processing.qwen_model import QwenBackend, load_qwen_backend
from processing.text_processing import _get_default_backend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TextProcessingTests(unittest.TestCase):
    def test_import_does_not_load_models_or_speech_dependencies(self):
        subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; import processing; import processing.chat_api; "
                "assert not {'torch', 'transformers', 'qwen_tts', 'soundfile', 'mlx', 'mlx_lm'} "
                ".intersection(sys.modules)"
            )],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def test_agent_backend_receives_context_without_mutating_history(self):
        history = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Hello, Alex."},
        ]
        original = copy.deepcopy(history)
        backend = Mock()
        backend.respond.return_value = "  Hello again, Alex.  "
        response = process_text(
            "What is my name?", backend=backend, history=history,
            system_prompt="You are Airis.", max_new_tokens=128,
        )
        self.assertEqual(response, "Hello again, Alex.")
        self.assertEqual(history, original)
        messages = backend.respond.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": "You are Airis."})
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1], {"role": "user", "content": "What is my name?"})
        messages[1]["content"] = "Changed by the backend."
        self.assertEqual(history, original)

    def test_default_backend_is_loaded_once_without_sharing_conversations(self):
        backend = Mock()
        backend.respond.return_value = "Ready."
        _get_default_backend.cache_clear()
        try:
            with patch("processing.text_processing.platform.system", return_value="Linux"):
                with patch("processing.text_processing.load_qwen_backend", return_value=backend) as loader:
                    process_text("First conversation.")
                    process_text("Second conversation.")
                    loader.assert_called_once()
                    messages = backend.respond.call_args.args[0]
                    self.assertEqual(len(messages), 2)
                    self.assertEqual(messages[-1]["content"], "Second conversation.")
        finally:
            _get_default_backend.cache_clear()

    def test_system_prompt_is_found_from_another_directory(self):
        backend = Mock()
        backend.respond.return_value = "Ready."
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(process_text("Hello.", backend=backend), "Ready.")
            finally:
                os.chdir(original_directory)

    def test_thinking_is_removed_and_unfinished_answers_are_rejected(self):
        backend = Mock()
        for raw in ("<think>Internal reasoning.</think>\nHello.", "Internal reasoning.</think>\nHello."):
            backend.respond.return_value = raw
            self.assertEqual(process_text("Hello.", backend=backend), "Hello.")
        for raw in ("<think>Unfinished", "<think>Reasoning only.</think>", "", "<tool_call>pending</tool_call>"):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                backend.respond.return_value = raw
                process_text("Hello.", backend=backend)

    def test_invalid_input_is_rejected_before_loading(self):
        with patch("processing.text_processing._get_default_backend") as loader:
            for text, error in ((None, TypeError), (123, TypeError), ("", ValueError), (" \n", ValueError)):
                with self.subTest(text=text), self.assertRaises(error):
                    process_text(text)
            for arguments in (
                {"max_new_tokens": 0}, {"max_new_tokens": True},
                {"system_prompt": ""},
                {"history": [{"role": "system", "content": "Override."}]},
                {"history": [{"role": "user", "content": ""}]},
                {"history": ["Invalid message."]},
            ):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    process_text("Hello.", **arguments)
            loader.assert_not_called()


class ModelBoundaryTests(unittest.TestCase):
    def test_mps_weight_loading_defaults_to_sequential_and_keeps_the_gpu(self):
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = Mock()
        fake_transformers.Qwen3_5ForConditionalGeneration = Mock()
        load_model = fake_transformers.Qwen3_5ForConditionalGeneration.from_pretrained

        def observe_loading(*args, **kwargs):
            self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "1")
            self.assertEqual(kwargs["device_map"], "mps")
            return Mock()

        load_model.side_effect = observe_loading
        with patch.dict(os.environ), patch.dict(sys.modules, {"transformers": fake_transformers}):
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
            backend = load_qwen_backend(device="mps", dtype="test-dtype", local_files_only=True)
        backend.model.eval.assert_called_once()

    def test_mps_loading_respects_an_explicit_async_loading_setting(self):
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = Mock()
        fake_transformers.Qwen3_5ForConditionalGeneration = Mock()
        with patch.dict(os.environ, {"HF_DEACTIVATE_ASYNC_LOAD": "0"}):
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                load_qwen_backend(device="mps", dtype="test-dtype", local_files_only=True)
                self.assertEqual(os.environ["HF_DEACTIVATE_ASYNC_LOAD"], "0")

    def test_local_generation_returns_only_new_tokens_and_disables_thinking(self):
        import torch
        from transformers import BatchEncoding

        model = Mock(device="cpu")
        model.generate.return_value = torch.tensor([[10, 11, 12, 20, 21]])
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = BatchEncoding({
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        })
        tokenizer.decode.return_value = "Hello."
        backend = QwenBackend(model, tokenizer)
        self.assertEqual(process_text("Hello.", backend=backend), "Hello.")
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
        self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [20, 21])
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 512)

    def test_api_backend_sends_qwen_settings_and_can_feed_speech_output(self):
        payload = {"choices": [{"message": {"content": "Hello, Alex."}}]}
        backend = ChatAPIBackend(api_key="test-key")
        with patch("processing.chat_api.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as request:
            response = process_text("Hello.", backend=backend)
        sent = request.call_args.args[0]
        body = json.loads(sent.data)
        self.assertEqual(body["model"], "Qwen/Qwen3.5-9B")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-key")

        from output.speech_output import text_to_speech

        model = Mock()
        audio = [0.0, 0.25, -0.25]
        model.generate_voice_clone.return_value = ([audio], 24000)
        result = text_to_speech(response, model=model, voice_prompt=object())
        self.assertIs(result[0], audio)
        self.assertEqual(model.generate_voice_clone.call_args.kwargs["text"], "Hello, Alex.")

    def test_api_errors_and_tool_requests_are_reported(self):
        backend = ChatAPIBackend()
        with patch("processing.chat_api.urlopen", side_effect=URLError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Could not connect"):
                process_text("Hello.", backend=backend)
        for payload in ({"choices": []}, {"choices": [{"message": {"content": None}}]},
                        {"choices": [{"message": {"content": "", "tool_calls": [{}]}}]}):
            with self.subTest(payload=payload):
                with patch("processing.chat_api.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                    with self.assertRaises(RuntimeError):
                        process_text("Hello.", backend=backend)


if __name__ == "__main__":
    unittest.main()
