"""Quantized Qwen inference with MLX on Apple Silicon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
from output.telemetry import logger


MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"
FAST_MODEL_ID = "mlx-community/Qwen3.5-4B-4bit"


@dataclass
class MLXBackend:
    """A reusable MLX model and tokenizer implementing the response backend."""

    model: Any
    tokenizer: Any
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    last_finish_reason: str | None = field(default=None, init=False)
    last_reused_tokens: int = field(default=0, init=False)
    last_prompt_tokens: int = field(default=0, init=False)
    _prompt_cache: Any = field(default=None, init=False, repr=False)
    _cache_tokens: list[int] = field(default_factory=list, init=False, repr=False)

    def reset_cache(self):
        """Drop session state when history or response instructions change."""
        self._prompt_cache = None
        self._cache_tokens = []
        self.last_reused_tokens = 0

    def warm_up(self) -> None:
        """Compile the first generation path without creating a conversation turn."""
        from .prompts import build_system_prompt

        messages = [{"role": "system", "content": build_system_prompt(mode="speech")},
                    {"role": "user", "content": "Reply with OK."}]
        for _ in self.stream(messages, max_new_tokens=2):
            pass
        self.reset_cache()

    def stream(self, messages: list[dict[str, str]], *, max_new_tokens: int):
        """Yield generated text chunks and record whether the token limit was reached."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.last_finish_reason = None
        options = {}
        tokens = None
        self.last_reused_tokens = 0
        # Qwen's template removes empty thinking tags from earlier assistant
        # replies. Keep a checkpoint at the end of the user prompt, before the
        # assistant generation prefix, so later history remains an exact prefix.
        if callable(getattr(type(self.model), "make_cache", None)):
            from mlx_lm.models.cache import make_prompt_cache
            import mlx.core as mx

            bos = self.tokenizer.bos_token
            tokens = self.tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))
            stable = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            stable_tokens = self.tokenizer.encode(stable, add_special_tokens=bos is None or not stable.startswith(bos))
            self.last_prompt_tokens = len(tokens)
            count = len(self._cache_tokens)
            if (self._prompt_cache is not None and 0 < count <= len(stable_tokens)
                    and stable_tokens[:count] == self._cache_tokens):
                self.last_reused_tokens = count
                working_cache = deepcopy(self._prompt_cache)
            else:
                self.reset_cache()
                working_cache = make_prompt_cache(self.model)
                count = 0
            if tokens[:len(stable_tokens)] != stable_tokens or len(stable_tokens) >= len(tokens):
                # A different tokenizer/template may merge tokens at the split.
                # Fall back to normal generation rather than alter model input.
                self.reset_cache()
                tokens = None
            else:
                try:
                    for offset in range(count, len(stable_tokens), 512):
                        self.model(mx.array(stable_tokens[offset:offset + 512])[None], cache=working_cache)
                        mx.eval([cache.state for cache in working_cache])
                    if sum(cache.nbytes for cache in working_cache) <= 128 * 1024 * 1024:
                        self._prompt_cache = deepcopy(working_cache)
                        self._cache_tokens = stable_tokens
                    else:
                        self.reset_cache()
                    prompt = tokens[len(stable_tokens):]
                    options["prompt_cache"] = working_cache
                except BaseException:
                    self.reset_cache()
                    raise
        completed = False
        iterator = stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_new_tokens,
            sampler=make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k),
            **options,
        )
        try:
            for response in iterator:
                if response.finish_reason is not None:
                    self.last_finish_reason = response.finish_reason
                if response.text:
                    yield response.text
            completed = True
            if tokens is not None:
                logger.info("Qwen prompt: total_tokens=%d reused_tokens=%d",
                            self.last_prompt_tokens, self.last_reused_tokens)
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()
            if not completed:
                self.reset_cache()

    def respond(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self.last_finish_reason = None

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k),
            verbose=False,
        )


def load_mlx_backend(*, model_id: str = MODEL_ID) -> MLXBackend:
    """Load a quantized Qwen model once and reuse it for subsequent requests.

    model_id can also be a local checkpoint directory. The first uncached
    Hugging Face model load downloads a separate set of quantized weights.
    """
    try:
        from mlx_lm import load
    except ImportError as error:
        raise RuntimeError(
            "Apple Silicon processing requires MLX LM. Install "
            "processing/requirements-mlx.txt in .venv."
        ) from error

    model, tokenizer = load(model_id)
    return MLXBackend(model=model, tokenizer=tokenizer)
