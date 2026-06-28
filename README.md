# ZerØ

<p align="center">
  <img src="public/icon.png" alt="ZerØ logo" width="128" />
</p>

A fast, simple **local** agentic AI desktop app. Built for low-end Intel laptops first, scales up to higher-end machines (NVIDIA / AMD / Intel GPUs).

> **Zero cloud.** Everything runs on your machine via llama.cpp — no servers, no egress.
> **Zero billing.** No tokens, no API keys, no subscriptions. Your hardware is the only bill.
> **Zero command line.** A full desktop UI — browse, download, chat, and run agents without a terminal.
> **Zero config.** Auto-detects your hardware and installs the right build for it.
> **Zero telemetry.** Nothing about you or your data ever leaves the device.

- **Runtime:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (bundled `llama-server`, OpenAI-compatible). ZerØ auto-installs the build that matches your hardware:
  - **CUDA** — NVIDIA GPUs
  - **OpenVINO** — Intel CPUs, iGPUs, and Arc dGPUs
  - **HIP/ROCm** — AMD Radeon dGPUs
  - **CPU** — fallback when no supported accelerator is detected
- **Audio:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for speech-to-text and `llama-tts` for text-to-speech
- **Other providers:** ollama or any OpenAI-compatible endpoint (same chat interface, different base URL)
- **UI:** Tauri 2 + React 19 + Tailwind v4 (Fluent Design–inspired)
- **State:** Zustand on the front, SQLite (`sqlx`) on the back
- **Storage root:** `~/.zero/` (same path on Windows, macOS, and Linux)

## Features

- System probe (CPU / RAM / GPU + VRAM) with hardware-aware model recommendations
- llama.cpp lifecycle: install the right variant, load/swap GGUF models per backend
- Hugging Face model browse + multi-file GGUF download
- Streaming chat with conversation persistence
- Multimodal chat: image / document upload (OpenAI vision shape over the chat endpoint)
- User-authored skills (`~/.zero/skills/<id>/SKILL.md`)
- Built-in MCP tools (shell, fs, http, web search/read, clipboard, notify, task.create) plus external MCP servers (HTTP/SSE) listed + smoke-tested from the Tools page
- Knowledge-base documents grounded into the system prompt (Embedding page)
- Short-term + long-term memory
- Speech-to-text (whisper) and text-to-speech (llama-tts)
- Scheduler: cron + interval + manual click-to-run tasks

## Getting started

```bash
pnpm install
pnpm tauri dev
```

### Icons (required for `tauri build`, not for `dev`)

```bash
pnpm tauri icon path/to/your-icon.png
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/` | React frontend (Fluent Design–inspired) |
| `src/stores/` | Zustand stores (one per domain) |
| `src/components/tui/` | Reusable UI primitives |
| `src/pages/` | Top-level views |
| `src-tauri/src/` | Rust backend (one module per domain) |
| `src-tauri/src/commands/` | Tauri IPC commands |

## Storage

```
~/.zero/
├── zero.db                      # SQLite (conversations, tasks, memory, runtime state)
├── system.json                  # Cached hardware probe
├── settings.json
├── runtimes/
│   ├── llama.cpp/<variant>/     # cuda | openvino | hip-radeon | cpu builds
│   │   ├── ov_cache/            # OpenVINO compiled-graph cache
│   │   └── models-preset.ini    # shared router model presets
│   └── whisper.cpp/             # whisper-cli + backend DLLs
├── models/                      # GGUF model cache
│   └── whisper/                 # whisper ggml *.bin
├── attachments/<conv_id>/       # persisted chat uploads (images + docs)
├── skills/<skill_id>/           # SKILL.md + supporting resources
├── documents/                   # knowledge-base files (embedding feature)
└── logs/
```

On Windows that resolves to `C:\Users\<you>\.zero\`. Older installs under
`%LOCALAPPDATA%\zero\zero\data\` are moved into `~/.zero` automatically on
first launch.
