"""Test the Apple Silicon backend without loading Metal or model weights."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from processing import process_text
from processing.mlx_model import load_mlx_backend
from processing.text_processing import _get_default_backend


class MLXProcessingTests(unittest.TestCase):
    def test_apple_silicon_reuses_quantized_model_and_returns_a_final_answer(self):
        fake_mlx_lm = ModuleType("mlx_lm")
        fake_sample_utils = ModuleType("mlx_lm.sample_utils")
        model = object()
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "Formatted conversation."
        fake_mlx_lm.load = Mock(return_value=(model, tokenizer))
        fake_mlx_lm.generate = Mock(return_value="<think>Reasoning.</think>Hello.")
        fake_sample_utils.make_sampler = Mock(return_value=object())
        modules = {"mlx_lm": fake_mlx_lm, "mlx_lm.sample_utils": fake_sample_utils}
        _get_default_backend.cache_clear()
        try:
            with patch.dict(sys.modules, modules):
                with patch("processing.text_processing.platform.system", return_value="Darwin"):
                    with patch("processing.text_processing.platform.machine", return_value="arm64"):
                        for text in ("First request.", "Second request."):
                            self.assertEqual(process_text(text, max_new_tokens=64), "Hello.")
                        fake_mlx_lm.load.assert_called_once_with("mlx-community/Qwen3.5-9B-4bit")
            self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
            self.assertEqual(fake_mlx_lm.generate.call_args.kwargs["max_tokens"], 64)
            self.assertFalse(fake_mlx_lm.generate.call_args.kwargs["verbose"])
        finally:
            _get_default_backend.cache_clear()

    def test_explicit_smaller_model_and_conversation_history_are_supported(self):
        fake_mlx_lm = ModuleType("mlx_lm")
        fake_sample_utils = ModuleType("mlx_lm.sample_utils")
        tokenizer = Mock()
        fake_mlx_lm.load = Mock(return_value=(object(), tokenizer))
        fake_mlx_lm.generate = Mock(return_value="Hello, Alex.")
        fake_sample_utils.make_sampler = Mock()
        with patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm, "mlx_lm.sample_utils": fake_sample_utils}):
            backend = load_mlx_backend(model_id="mlx-community/Qwen3.5-4B-4bit")
            response = process_text(
                "What is my name?", backend=backend,
                history=[{"role": "user", "content": "My name is Alex."}],
            )
        self.assertEqual(response, "Hello, Alex.")
        messages = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[1]["content"], "My name is Alex.")
        fake_mlx_lm.load.assert_called_once_with("mlx-community/Qwen3.5-4B-4bit")

    def test_missing_dependency_has_clear_setup_instructions(self):
        with patch.dict(sys.modules, {"mlx_lm": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-mlx.txt"):
                load_mlx_backend()

    def test_streaming_hides_fragmented_reasoning_and_reports_truncation(self):
        from assistant import Assistant

        module = ModuleType("mlx_lm")
        sample_utils = ModuleType("mlx_lm.sample_utils")
        module.load = Mock(return_value=(object(), Mock()))
        module.stream_generate = Mock(return_value=iter([
            SimpleNamespace(text="<th", finish_reason=None),
            SimpleNamespace(text="ink>Private reasoning.</th", finish_reason=None),
            SimpleNamespace(text="ink>Hello ", finish_reason=None),
            SimpleNamespace(text="world.", finish_reason="length"),
        ]))
        sample_utils.make_sampler = Mock()
        chunks = []
        with patch.dict(sys.modules, {"mlx_lm": module, "mlx_lm.sample_utils": sample_utils}):
            backend = load_mlx_backend()
            response = Assistant(backend=backend).respond("Hello.", max_new_tokens=2048, on_text=chunks.append)
        self.assertEqual("".join(chunks), "Hello world.")
        self.assertEqual(response.text, "Hello world.")
        self.assertTrue(response.truncated)
        self.assertEqual(module.stream_generate.call_args.kwargs["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
