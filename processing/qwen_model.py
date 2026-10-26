"""Local Qwen3.5 inference without loading model dependencies on import."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass
class QwenBackend:
    """A reusable model/tokenizer pair implementing the response backend."""

    model: Any
    tokenizer: Any
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    def respond(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            options.update(
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            )

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **options)

        # Decode only new tokens, excluding the system prompt and user input.
        tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def load_qwen_backend(
    *,
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: Any = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> QwenBackend:
    """Load once and reuse; the first uncached load downloads model weights.

    Install processing/requirements.txt in .venv. Select CUDA, MPS, or CPU
    automatically unless device is provided.
    MPS defaults to sequential loading to avoid concurrent Metal weight-copy
    crashes. An explicit HF_DEACTIVATE_ASYNC_LOAD setting takes precedence.
    """
    try:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5 requires Transformers with Qwen3_5ForConditionalGeneration "
            "support. Install processing/requirements.txt in .venv."
        ) from error

    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

    if device.split(":", 1)[0] == "mps":
        # Transformers 5 can copy weights from multiple worker threads. The
        # Metal shader cache can crash during those concurrent MPS copies.
        # This setting changes loading only; generation still runs on MPS.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    shared_options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **shared_options)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        **shared_options,
    )
    model.eval()
    return QwenBackend(model=model, tokenizer=tokenizer)
