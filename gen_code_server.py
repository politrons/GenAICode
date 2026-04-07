#!/usr/bin/env python3
"""Local Gemma 4 code chat (no MCP, no filesystem tools).

Run:
    python3 gen_code_server.py
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from ollama_local import OllamaLocalClient, build_messages_from_history


BASE_DIR = Path(__file__).resolve().parent
HTML_TEMPLATE_PATH = BASE_DIR / "chat_ui.html"


def render_html_page(model_name: str, request_timeout_ms: int) -> str:
    try:
        template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read HTML template at {HTML_TEMPLATE_PATH}: {exc}") from exc
    html = template.replace("__MODEL__", model_name)
    return html.replace("__REQUEST_TIMEOUT_MS__", str(request_timeout_ms))


class ChatHandler(BaseHTTPRequestHandler):
    model_name = "gemma4:31b"
    ollama_base_url = "http://127.0.0.1:11434"
    ollama_timeout_s = 300
    ollama_client: OllamaLocalClient | None = None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, status: int, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            if self.ollama_client is None:
                self._send_json(500, {"error": "Ollama client not configured"})
                return
            ok, error = self.ollama_client.check_health(timeout_s=3)
            payload: dict[str, Any] = {
                "ok": True,
                "model": self.model_name,
                "ollama_base_url": self.ollama_base_url,
                "ollama_reachable": ok,
                "request_timeout_s": self.ollama_timeout_s,
            }
            if error:
                payload["ollama_error"] = error
            self._send_json(200, payload)
            return

        if self.path != "/":
            self._send_html(404, "<h1>404</h1>")
            return
        try:
            html = render_html_page(
                self.model_name,
                request_timeout_ms=max(1000, int(self.ollama_timeout_s * 1000)),
            )
        except RuntimeError as exc:
            self._send_html(500, f"<h1>500</h1><pre>{exc}</pre>")
            return
        self._send_html(200, html)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        history = payload.get("history", [])
        if not isinstance(history, list):
            self._send_json(400, {"error": "`history` must be a list"})
            return

        messages = build_messages_from_history(history)

        if len(messages) <= 1:
            self._send_json(400, {"error": "No messages to process"})
            return

        if self.ollama_client is None:
            self._send_json(500, {"error": "Ollama client not configured"})
            return

        try:
            reply = self.ollama_client.chat(messages=messages)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"reply": reply})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep console output clean.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Gemma 4 code chat")
    parser.add_argument("--host", default=os.environ.get("CHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHAT_PORT", "8787")))
    parser.add_argument(
        "--model",
        default=os.environ.get("CHAT_MODEL", "gemma4:31b"),
        help="Ollama model tag (default: gemma4:31b)",
    )
    parser.add_argument(
        "--ollama",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CHAT_TIMEOUT", "300")),
        help="Timeout in seconds for each model response (default: 300)",
    )
    args = parser.parse_args()

    ChatHandler.model_name = args.model
    ChatHandler.ollama_base_url = args.ollama
    ChatHandler.ollama_timeout_s = args.timeout
    ChatHandler.ollama_client = OllamaLocalClient(
        base_url=args.ollama,
        model=args.model,
        timeout_s=args.timeout,
    )

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Local Gemma code chat on http://{args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama}")
    print(f"Timeout: {args.timeout}s")
    print("Tip: run `ollama pull gemma4:31b` if you do not have the model yet.")
    server.serve_forever()


if __name__ == "__main__":
    main()
