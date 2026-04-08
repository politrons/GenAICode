# Local Code Assistant Chat

## 1) Download the local model

```bash
ollama pull gemma4:31b
```

## 2) Start Ollama

```bash
ollama serve
```

## 3) Start the chat server

```bash
python3 gen_code_server.py
```

Or with a specific initial workdir:

```bash
python3 gen_code_server.py --workdir ./sandbox
```

## 4) Open the chat

Open:

- `http://127.0.0.1:8787`

## How It Works

When you send a message in the chat, the web app sends your conversation to the local server.

The server builds the model context and includes a list of available filesystem tools (for example: read file, write file, list directory, and set working directory). It then sends everything to Ollama in a chat request.

If the model can answer directly, it returns a normal text response.

If the model decides it needs to interact with files, it returns a structured tool call (tool name + arguments) instead of a final answer. The server executes that tool locally through the filesystem sandbox, which restricts all file operations to the configured working directory.

After executing the tool, the server sends the tool result back to the model as part of the conversation. The model can then either request another tool call or return a final user-facing answer.

This loop continues until the model produces the final response shown in the chat.
