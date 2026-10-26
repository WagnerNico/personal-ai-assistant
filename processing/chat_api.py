"""Chat Completions backend for a separately hosted Qwen model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .qwen_model import MODEL_ID


@dataclass
class ChatAPIBackend:
    """Use a hosted Qwen server through the standard-library HTTP client."""

    base_url: str = "http://localhost:8000/v1"
    model_id: str = MODEL_ID
    api_key: str | None = None
    timeout: float = 120.0

    def respond(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"The model server returned HTTP {error.code}.") from error
        except URLError as error:
            raise RuntimeError("Could not connect to the model server.") from error
        except TimeoutError as error:
            raise RuntimeError("The model server request timed out.") from error
        except json.JSONDecodeError as error:
            raise RuntimeError("The model server returned invalid JSON.") from error
        try:
            message = result["choices"][0]["message"]
            if message.get("tool_calls"):
                raise RuntimeError("The model requested tools; use an agent backend to execute them.")
            content = message["content"]
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError("The model server returned an invalid chat response.") from error
        if not isinstance(content, str):
            raise RuntimeError("The model server did not return final response text.")
        return content
