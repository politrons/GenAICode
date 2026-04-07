"""Ollama local model integration for the chat server."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = (
    "You are an expert coding assistant. "
    "Primary goal: help with programming tasks and bug fixing. "
    "When sharing code, always use fenced code blocks with an explicit language tag. "
    "If the user asks to fix code, provide corrected code first, then a short explanation. "
    "Keep answers technically precise and practical."
)


class OllamaLocalClient:
    """Thin client for a local Ollama server."""

    def __init__(self, *, base_url: str, model: str, timeout_s: int) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, *, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure `ollama serve` is running on localhost:11434."
            ) from exc

        data: dict[str, Any] = json.loads(raw)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from Ollama: {raw}") from exc

    def check_health(self, *, timeout_s: int = 3) -> tuple[bool, str | None]:
        req = Request(
            f"{self.base_url.rstrip('/')}/api/tags",
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                _ = resp.read()
            return True, None
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            return False, f"Ollama HTTP {exc.code}: {details}"
        except URLError as exc:
            return False, f"Cannot connect to Ollama: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)


def build_messages_from_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build valid chat messages from request history and prepend system prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages
