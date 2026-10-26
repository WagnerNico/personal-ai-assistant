"""Text-to-text product entry point, independent of speech input and output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import platform
from typing import Protocol

from .qwen_model import load_qwen_backend
from .mlx_model import load_mlx_backend
from .prompts import SYSTEM_PROMPT_PATH, build_system_prompt


class ResponseBackend(Protocol):
    """A model or future agent that returns a final answer for chat messages."""

    def respond(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        ...


@lru_cache(maxsize=1)
def _get_default_backend() -> ResponseBackend:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return load_mlx_backend()
    return load_qwen_backend()


def _final_answer(response: str) -> str:
    """Remove Qwen thinking blocks before returning text for speech output."""
    if not isinstance(response, str):
        raise TypeError("The response backend must return a string.")
    # A thinking-mode template can place the opening tag in the input prompt.
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    if "<think>" in response:
        raise RuntimeError("The model returned unfinished thinking without a final answer.")
    if "<tool_call>" in response:
        raise RuntimeError("The response requires a tool call; use an agent backend to execute it.")
    response = response.strip()
    if not response:
        raise RuntimeError("The model did not return a final answer.")
    return response


def _visible_stream_text(raw: str) -> str:
    """Hold partial control tags and reasoning blocks out of streamed text."""
    text = raw.lstrip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].lstrip()
    elif "<think>" in text or "<think>".startswith(text):
        return ""
    if "<tool_call>" in text:
        raise RuntimeError("The response requires a tool call; use an agent backend to execute it.")
    for length in range(min(len(text), 11), 0, -1):
        if any(tag.startswith(text[-length:]) for tag in ("<think>", "</think>", "<tool_call>")):
            return text[:-length]
    return text


def validate_text_request(text: str, max_new_tokens: int) -> None:
    """Validate inputs before either inference or a local agent action."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text.strip():
        raise ValueError("text must not be empty or contain only whitespace.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")


def process_text(
    text: str,
    *,
    backend: ResponseBackend | None = None,
    history: Sequence[Mapping[str, str]] | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    on_text: Callable[[str], None] | None = None,
) -> str:
    """Return an answer to text, using Qwen3.5-9B by default.

    Apple Silicon uses 4-bit MLX weights; other platforms use Transformers.
    The default model loads lazily and is reused across calls. History belongs
    to the caller and is never mutated or stored globally. Pass a backend with
    respond(messages, max_new_tokens=...) to substitute a server or future agent.
    """
    validate_text_request(text, max_new_tokens)
    system_prompt = build_system_prompt(system_prompt)

    messages = [{"role": "system", "content": system_prompt}]
    if history is not None:
        for message in history:
            if not isinstance(message, Mapping):
                raise ValueError("History messages must be mappings with role and content.")
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant"):
                raise ValueError("History messages must have a user or assistant role.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("History message content must be a nonempty string.")
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})

    if backend is None:
        backend = _get_default_backend()
    # Streaming is optional; ordinary models and future agent backends keep
    # implementing only respond(). Inspect the class to avoid dynamic adapters
    # fabricating a stream method they do not implement.
    if on_text is not None and callable(getattr(type(backend), "stream", None)):
        raw = ""
        emitted = ""
        for chunk in backend.stream(messages, max_new_tokens=max_new_tokens):
            if not isinstance(chunk, str):
                raise TypeError("Streaming backends must yield strings.")
            raw += chunk
            visible = _visible_stream_text(raw)
            if visible.startswith(emitted) and len(visible) > len(emitted):
                on_text(visible[len(emitted):])
                emitted = visible
        result = _final_answer(raw)
        if result.startswith(emitted) and len(result) > len(emitted):
            on_text(result[len(emitted):])
        return result
    result = _final_answer(backend.respond(messages, max_new_tokens=max_new_tokens))
    if on_text is not None:
        on_text(result)
    return result
