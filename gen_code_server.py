#!/usr/bin/env python3
"""Local Gemma 4 code chat with local filesystem tool access.

Run:
    python3 gen_code_server.py
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from filesystem_sandbox import FilesystemSandbox
from ollama_local import OllamaLocalClient, build_messages_from_history


BASE_DIR = Path(__file__).resolve().parent
HTML_TEMPLATE_PATH = BASE_DIR / "chat_ui.html"
MAX_TOOL_ROUNDS = 8


def filesystem_tools_schema() -> list[dict[str, Any]]:
    """Return the tool schemas exposed to the model for filesystem actions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fs_set_workdir",
                "description": "Set the active working directory. Absolute path required.",
                "parameters": {
                    "type": "object",
                    "properties": {"root": {"type": "string"}},
                    "required": ["root"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_get_workdir",
                "description": "Get the current active working directory.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_list_dir",
                "description": "List files and directories inside current workdir.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_read_file",
                "description": "Read UTF-8 file content from current workdir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 120000},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_write_file",
                "description": "Write UTF-8 content to a file in current workdir (overwrite).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "create_dirs": {"type": "boolean", "default": True},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fs_append_file",
                "description": "Append UTF-8 content to a file in current workdir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "create_dirs": {"type": "boolean", "default": True},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]


def render_html_page(model_name: str, request_timeout_ms: int) -> str:
    """Load and render the chat HTML template with runtime placeholders."""
    try:
        template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read HTML template at {HTML_TEMPLATE_PATH}: {exc}") from exc
    html = template.replace("__MODEL__", model_name)
    return html.replace("__REQUEST_TIMEOUT_MS__", str(request_timeout_ms))


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for chat UI, health endpoint, and chat API calls."""

    model_name = "gemma4:31b"
    ollama_base_url = "http://127.0.0.1:11434"
    ollama_timeout_s = 300
    ollama_client: OllamaLocalClient | None = None
    fs_sandbox: FilesystemSandbox | None = None
    fs_lock = threading.Lock()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        """Write a JSON HTTP response with the provided status code."""
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, status: int, html: str) -> None:
        """Write an HTML HTTP response with the provided status code."""
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _tool_error(self, text: str) -> str:
        """Normalize tool execution errors into a readable tool-result string."""
        return f"Tool error: {text}"

    def _execute_fs_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one filesystem tool call against the current sandbox root."""
        if self.fs_sandbox is None:
            return self._tool_error("Filesystem sandbox not configured.")

        with self.fs_lock:
            try:
                if name == "fs_set_workdir":
                    root = arguments.get("root")
                    if not isinstance(root, str):
                        return self._tool_error("`root` must be a string.")
                    root_path = Path(root)
                    if not root_path.is_absolute():
                        return self._tool_error("`root` must be an absolute path.")
                    self.fs_sandbox.set_root(root_path)
                    return f"Workdir set to: {self.fs_sandbox.root}"

                if name == "fs_get_workdir":
                    return str(self.fs_sandbox.root)

                if name == "fs_list_dir":
                    rel = arguments.get("path", ".")
                    if not isinstance(rel, str):
                        return self._tool_error("`path` must be a string.")
                    rows = self.fs_sandbox.list_dir(rel)
                    return json.dumps(rows, indent=2)

                if name == "fs_read_file":
                    rel = arguments.get("path")
                    max_chars = arguments.get("max_chars", 120_000)
                    if not isinstance(rel, str):
                        return self._tool_error("`path` must be a string.")
                    if not isinstance(max_chars, int) or max_chars <= 0:
                        return self._tool_error("`max_chars` must be a positive integer.")
                    max_chars = min(max_chars, 200_000)
                    return self.fs_sandbox.read_file(rel, max_chars=max_chars)

                if name == "fs_write_file":
                    rel = arguments.get("path")
                    content = arguments.get("content")
                    create_dirs = arguments.get("create_dirs", True)
                    if not isinstance(rel, str):
                        return self._tool_error("`path` must be a string.")
                    if not isinstance(content, str):
                        return self._tool_error("`content` must be a string.")
                    if not isinstance(create_dirs, bool):
                        return self._tool_error("`create_dirs` must be a boolean.")
                    self.fs_sandbox.write_file(rel, content, create_dirs=create_dirs)
                    return f"Wrote file: {rel}"

                if name == "fs_append_file":
                    rel = arguments.get("path")
                    content = arguments.get("content")
                    create_dirs = arguments.get("create_dirs", True)
                    if not isinstance(rel, str):
                        return self._tool_error("`path` must be a string.")
                    if not isinstance(content, str):
                        return self._tool_error("`content` must be a string.")
                    if not isinstance(create_dirs, bool):
                        return self._tool_error("`create_dirs` must be a boolean.")
                    self.fs_sandbox.append_file(rel, content, create_dirs=create_dirs)
                    return f"Appended file: {rel}"

                return self._tool_error(f"Unknown tool: {name}")
            except Exception as exc:  # noqa: BLE001
                return self._tool_error(str(exc))

    def _chat_with_tools(self, messages: list[dict[str, Any]]) -> str:
        """Run a bounded tool-calling loop until the model returns a final answer."""
        if self.ollama_client is None:
            raise RuntimeError("Ollama client not configured")
        if self.fs_sandbox is None:
            raise RuntimeError("Filesystem sandbox not configured")

        tools = filesystem_tools_schema()
        for _ in range(MAX_TOOL_ROUNDS):
            # Ask the model for the next assistant step, allowing tool calls.
            raw = self.ollama_client.chat_once(messages=messages, tools=tools)
            assistant_msg = raw.get("message")
            if not isinstance(assistant_msg, dict):
                raise RuntimeError(f"Unexpected response from Ollama: {raw}")

            content = assistant_msg.get("content")
            if not isinstance(content, str):
                content = ""
            tool_calls = assistant_msg.get("tool_calls")

            next_assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if isinstance(tool_calls, list) and tool_calls:
                next_assistant["tool_calls"] = tool_calls
            messages.append(next_assistant)

            # No tools requested: return the assistant textual answer as final output.
            if not isinstance(tool_calls, list) or not tool_calls:
                if content.strip():
                    return content
                return "Done."

            # Execute each tool call and feed tool results back to the model.
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue

                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                        arguments = parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                tool_result = self._execute_fs_tool(name, arguments)
                messages.append({"role": "tool", "tool_name": name, "content": tool_result})

        raise RuntimeError("Tool execution loop limit reached.")

    def do_GET(self) -> None:  # noqa: N802
        """Handle health checks and serve the chat HTML page."""
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
            if self.fs_sandbox is not None:
                payload["workdir"] = str(self.fs_sandbox.root)
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
        """Handle chat completion requests from the web client."""
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
        if self.fs_sandbox is None:
            self._send_json(500, {"error": "Filesystem sandbox not configured"})
            return

        # Inject the current workdir into system context for explicit tool guidance.
        with self.fs_lock:
            current_root = str(self.fs_sandbox.root)
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Filesystem tools are available. "
                    f"Current workdir is: {current_root}. "
                    "When user asks to create/update/read files, use tools first. "
                    "After tool calls, respond with what you changed."
                ),
            },
        )

        try:
            reply = self._chat_with_tools(messages)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"reply": reply})

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence default HTTP request logs for a cleaner terminal output."""
        # Keep console output clean.
        return


def main() -> None:
    """Parse CLI args, initialize clients, and start the HTTP server."""
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
    parser.add_argument(
        "--workdir",
        default=os.environ.get("CHAT_WORKDIR", str(Path.cwd())),
        help="Initial working directory for filesystem tools (default: current directory)",
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
    workdir = Path(args.workdir).expanduser()
    if not workdir.is_absolute():
        workdir = (Path.cwd() / workdir).resolve(strict=False)
    ChatHandler.fs_sandbox = FilesystemSandbox(workdir)

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Local Gemma code chat on http://{args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Ollama: {args.ollama}")
    print(f"Timeout: {args.timeout}s")
    print(f"Workdir: {ChatHandler.fs_sandbox.root}")
    print("Tip: run `ollama pull gemma4:31b` if you do not have the model yet.")
    server.serve_forever()


if __name__ == "__main__":
    main()
