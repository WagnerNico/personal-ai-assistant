"""Offline runtime smoke test using a tiny Qwen3.5 checkpoint, not the 9B weights."""

import tempfile
import unittest

from processing import process_text
from processing.qwen_model import load_qwen_backend


class QwenRuntimeTests(unittest.TestCase):
    def test_local_checkpoint_loads_and_generates_with_real_qwen_architecture(self):
        try:
            import torch
            from tokenizers.pre_tokenizers import ByteLevel
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Tokenizer
        except ImportError:
            self.skipTest("Run with .venv to test the Qwen3.5 runtime.")

        text_config = {
            "vocab_size": 258, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 256, "eos_token_id": None, "pad_token_id": 257,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2],
            },
        }
        vision_config = {
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
        }
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=text_config, vision_config=vision_config,
        ))
        # A zeroed output projection makes greedy output deterministic (token 0).
        with torch.no_grad():
            model.lm_head.weight.zero_()
        vocabulary = {character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))}
        vocabulary.update({"<|endoftext|>": 256, "<|im_end|>": 257})
        tokenizer = Qwen3_5Tokenizer(vocab=vocabulary, merges=[])
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        )

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            backend = load_qwen_backend(
                model_id=directory, device="cpu", local_files_only=True,
            )
            backend.temperature = 0.0
            response = process_text(
                "Hello.", backend=backend, system_prompt="You are Airis.", max_new_tokens=2,
            )
            self.assertEqual(response, tokenizer.decode([0, 0]))
            self.assertFalse(backend.model.training)


if __name__ == "__main__":
    unittest.main()
