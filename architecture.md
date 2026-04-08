# Architecture

This project is split into three Python files:

## 1) `gen_code_server.py`

Responsibilities:

- Runs the HTTP server (`ThreadingHTTPServer`)
- Serves the chat UI (`chat_ui.html`) at `/`
- Exposes API endpoints:
  - `GET /api/health`
  - `POST /api/chat`
- Parses CLI options (`--host`, `--port`, `--model`, `--ollama`, `--timeout`, `--workdir`)
- Exposes filesystem tools to the model via Ollama tool calling
- Creates and injects the Ollama client used by request handlers

In short: this file is the web layer (routing + request/response handling).

## 2) `ollama_local.py`

Responsibilities:

- Defines the coding system prompt (`SYSTEM_PROMPT`)
- Implements `OllamaLocalClient`:
  - `chat(...)` to call Ollama chat completions
  - `check_health(...)` to verify Ollama availability
- Builds normalized chat messages from request history (`build_messages_from_history`)

In short: this file is the model integration layer (Ollama communication + message shaping).

## 3) `filesystem_sandbox.py`

Responsibilities:

- Contains the reusable root-sandbox logic
- Validates all paths stay under configured root
- Implements common filesystem operations:
  - list directory
  - read file
  - write file
  - append file

In short: this file is the filesystem safety layer used by the chat server tool runtime.
