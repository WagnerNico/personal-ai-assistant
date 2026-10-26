"""Verify exact-prefix checkpoints without loading native MLX or model weights."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from processing.mlx_model import MLXBackend


class Cache:
    def __init__(self):
        self.tokens = []

    @property
    def state(self):
        return self.tokens

    @property
    def nbytes(self):
        return len(self.tokens) * 4


class Model:
    def __init__(self):
        self.prefills = []

    def make_cache(self):
        return [Cache()]

    def __call__(self, inputs, *, cache):
        tokens = inputs.reshape(-1).tolist()
        self.prefills.append(tokens)
        cache[0].tokens.extend(tokens)


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        tokenizer = Mock()
        tokenizer.bos_token = None
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"] + (
            "X" if kwargs["add_generation_prompt"] else ""
        )
        tokenizer.encode.side_effect = lambda text, **kwargs: [ord(char) for char in text]
        self.backend = MLXBackend(self.model, tokenizer)
        lm, sampling, cache, mx = [ModuleType(name) for name in (
            "mlx_lm", "mlx_lm.sample_utils", "mlx_lm.models.cache", "mlx.core",
        )]
        def generate(model, tokenizer, *, prompt, prompt_cache, **kwargs):
            prompt_cache[0].tokens.extend(prompt)
            yield SimpleNamespace(text="Hello.", finish_reason="stop")
        lm.stream_generate = Mock(side_effect=generate)
        sampling.make_sampler = Mock()
        cache.make_prompt_cache = lambda model: model.make_cache()
        mx.array = np.array
        mx.eval = Mock()
        mlx = ModuleType("mlx")
        mlx.core = mx
        self.generator = lm.stream_generate
        self.modules = {module.__name__: module for module in (lm, sampling, cache, mlx, mx)}

    def stream(self, content):
        return "".join(self.backend.stream([{"role": "user", "content": content}], max_new_tokens=16))

    def test_new_turn_prefills_only_the_uncached_suffix_and_keeps_checkpoint_immutable(self):
        with patch.dict(sys.modules, self.modules):
            self.assertEqual(self.stream("AB"), "Hello.")
            self.assertEqual(self.backend._prompt_cache[0].tokens, [ord("A"), ord("B")])
            self.assertEqual(self.stream("ABCD"), "Hello.")
        self.assertEqual(self.backend.last_reused_tokens, 2)
        self.assertEqual(self.model.prefills, [[ord("A"), ord("B")], [ord("C"), ord("D")]])
        self.assertEqual(self.generator.call_args.kwargs["prompt"], [ord("X")])

    def test_different_instructions_or_reset_invalidate_the_checkpoint(self):
        with patch.dict(sys.modules, self.modules):
            self.stream("AB")
            self.stream("YZ")
            self.assertEqual(self.backend.last_reused_tokens, 0)
            self.backend.reset_cache()
            self.stream("YZCD")
        self.assertEqual(self.backend.last_reused_tokens, 0)
        self.assertEqual(self.model.prefills[-1], list(map(ord, "YZCD")))

    def test_interrupted_generation_discards_uncommitted_cache_state(self):
        with patch.dict(sys.modules, self.modules):
            generator = self.backend.stream([{"role": "user", "content": "AB"}], max_new_tokens=16)
            self.assertEqual(next(generator), "Hello.")
            generator.close()
        self.assertIsNone(self.backend._prompt_cache)


if __name__ == "__main__":
    unittest.main()
