import { create } from "zustand";
import { invoke } from "@/lib/tauri";

export type ProviderKind = "ollama" | "llama.cpp";

/**
 * Where the provider lives. Purely informational today — used by the UI
 * to badge the row and to seed sensible defaults (a `cloud` provider
 * starts with no `base_url`, since each vendor has its own).
 */
export type ProviderLocation = "local" | "cloud";

/**
 * Optional per-layer sampling overrides. Every field is independently
 * optional: a `null` means "don't override at this layer", letting the
 * runner's precedence chain (conversation → provider → per-model
 * profile → upstream default) fall through to the next one.
 *
 * Same shape lives on a conversation row for the per-chat popover —
 * the editor component in the UI is reused for both surfaces.
 */
export interface SamplingConfig {
  /** Sampling temperature override. Typical range 0.0–2.0. */
  temperature: number | null;
  /** Nucleus-sampling cutoff override. Typical range 0.0–1.0. */
  top_p: number | null;
  /** Top-k cutoff override. Sent as an OpenAI-compat extension. */
  top_k: number | null;
}

export const EMPTY_SAMPLING: SamplingConfig = {
  temperature: null,
  top_p: null,
  top_k: null,
};

export interface ProviderConfig {
  id: string;
  kind: ProviderKind;
  /** Local runtime on this machine vs. a remote cloud endpoint. */
  location: ProviderLocation;
  name: string;
  base_url: string;
  enabled: boolean;
  api_key_ref?: string | null;
  /** Sampling defaults applied to every chat that uses this provider. */
  sampling: SamplingConfig;
}

/**
 * External MCP server configuration. Supports three transports:
 *
 * - `http` / `sse` — standard JSON-RPC over HTTP. Uses `url` + `headers`.
 * - `stdio`        — zero spawns `command` with `args` (plus any `env`)
 *                    and speaks JSON-RPC on stdin/stdout per call.
 *
 * Stdio-only fields are optional so existing settings files
 * (HTTP/SSE-only) keep loading without migration.
 */
export interface McpServerConfig {
  id: string;
  name: string;
  /** `http` | `sse` | `stdio`. */
  transport: string;
  /** Endpoint URL for http/sse. Empty for stdio. */
  url: string;
  /** Header tuples forwarded verbatim on every JSON-RPC POST (http/sse). */
  headers: [string, string][];
  /** Executable to spawn (stdio only). */
  command?: string;
  /** argv for `command` (stdio only). */
  args?: string[];
  /** Extra environment variables for the child process (stdio only). */
  env?: [string, string][];
  enabled: boolean;
}

export interface OvmsSettings {
  rest_port: number;
  grpc_port: number;
  /** Free-form OpenVINO device: `CPU`, `GPU`, `NPU`, `AUTO`, `MULTI:CPU,GPU`, ... */
  device: string;

  // ─── advanced server-side knobs (all optional) ───────────────────────
  /** `DEBUG` / `INFO` / `ERROR`. `null` keeps the OVMS default (INFO). */
  log_level: string | null;
  /** Optional path to a log file. */
  log_path: string | null;
  /** Optional model-compilation cache directory. Major startup speedup. */
  cache_dir: string | null;
  /** CORS `Access-Control-Allow-Origin`. */
  allowed_origins: string | null;
  /** Path to a file whose first line holds the API key for `/v3/*`. */
  api_key_file: string | null;
  /**
   * Escape hatch for any flag we don't model explicitly. Each entry is
   * forwarded verbatim as a separate argv token — the Settings UI splits
   * the user's textarea on whitespace so multi-token flags become two
   * entries (`--grpc_workers` then `4`).
   */
  extra_args: string[];
}

/**
 * User-visible knobs for the bundled `llama-server` runtime. Mirrors
 * `LlamaSettings` on the Rust side. Port is determined per-variant (8081
 * for cuda, 8082 for openvino, etc.) — not configurable here.
 */
export interface LlamaSettings {
  /** Interface to bind. `127.0.0.1` keeps the server local-only. */
  host: string;
  /**
   * `--n-gpu-layers` value. `-1` offloads every layer to GPU (recommended
   * for the GPU-enabled builds), `0` forces CPU-only inference.
   */
  n_gpu_layers: number;
  /**
   * `--ctx-size`. `0` keeps the model's training context window.
   */
  ctx_size: number;
  /** `--parallel` — number of concurrent slots. */
  parallel: number;
  /** Escape hatch: extra argv tokens forwarded verbatim to `llama-server`. */
  extra_args: string[];
  /**
   * Experimental. Wire downloaded MTP / speculative-decoding draft models
   * into the router preset at load time. Off by default — MTP drafts can
   * crash or fail to load on some llama.cpp build / GPU combinations.
   */
  mtp_enabled: boolean;
}

/**
 * Speech-to-text + text-to-speech capabilities. Mirrors `AudioSettings`
 * on the Rust side. Off by default — when enabled the chat composer grows
 * a voice-input button and assistant replies grow a read-aloud button.
 *
 * Speech-to-text shells out to `whisper-cli` with a ggml `.bin` model.
 * Text-to-speech shells out to `llama-tts` with an OuteTTS model + the
 * WavTokenizer vocoder. Both are GPU one-shot CLIs.
 */
export interface AudioSettings {
  /** Master switch. Off by default. */
  enabled: boolean;
  /** Whisper ggml model file name (e.g. `ggml-base.en.bin`). */
  stt_model: string | null;
  /**
   * Spoken-language hint for transcription: an ISO code like `"en"`, or
   * `"auto"` to detect. Defaults to English — auto-detect is unreliable
   * on short dictation clips.
   */
  stt_language: string | null;
  /** OuteTTS HuggingFace repo id used for read-aloud. */
  tts_model: string | null;
  /** Retained for back-compat; no longer used. */
  tts_voice: string | null;
}

/**
 * Document-grounding ("embedding") feature. Off by default. When enabled,
 * the text of every enabled knowledge-base document is injected into the
 * system prompt for every chat session.
 *
 * Following the `builtin_tools_disabled` pattern, we track the *disabled*
 * set rather than the enabled set so every newly added document is on by
 * default — the behaviour the Embedding page advertises.
 */
export interface EmbeddingSettings {
  enabled: boolean;
  documents_disabled: string[];
}

export interface Settings {
  active_provider_id: string | null;
  providers: ProviderConfig[];
  hf_token_set: boolean;
  default_model: string | null;
  thinking_enabled: boolean;
  agent_max_iterations: number;
  destructive_tool_confirm: boolean;
  llama: LlamaSettings;
  /** Speech-to-text / text-to-speech capabilities. Off by default. */
  audio: AudioSettings;
  /** Document-grounding ("embedding") feature. Off by default. */
  embedding: EmbeddingSettings;
  /** Auto-install llama.cpp on startup and start the highest-priority variant. */
  auto_provision_llama: boolean;
  /** Folder names under `~/.zero/skills/` to inject into the system prompt. */
  skills_enabled: string[];
  /**
   * Names of built-in tools (e.g. `fs.list`, `shell.exec`) the user has
   * globally disabled from the Tools page. Hidden from every chat's
   * catalog. Per-chat overrides live separately on the conversation row.
   */
  builtin_tools_disabled: string[];
  /**
   * When true, the chat runner ships only the `tools.list` built-in to
   * the LLM on the first round of each turn instead of the full tool
   * catalogue. The model must call `tools.list` to discover what else
   * is available; once it does, the runner expands the catalogue on
   * subsequent rounds in the same turn. Trades one extra round-trip
   * on tool-using turns for a much smaller initial-context cost.
   */
  lazy_tool_discovery: boolean;
  /** External MCP servers (HTTP/SSE) we can list / call tools on. */
  mcp_servers: McpServerConfig[];
  /**
   * Launch zero automatically when the user logs into the OS. Kept in
   * sync with the OS-level autostart entry through the autostart plugin
   * whenever this toggle changes.
   */
  autostart_enabled: boolean;
  /** Start the window minimized instead of focused. */
  minimize_on_startup: boolean;
  /**
   * When true the window's close button minimizes to the taskbar instead
   * of quitting the app.
   */
  close_to_taskbar: boolean;
  /**
   * Absolute path to the active coding workspace (project root), or `null`
   * when none is open. Managed through the dedicated `workspace_*` IPC
   * commands (and the workspace store) rather than the Settings page, but
   * lives here so it round-trips with the rest of the settings file.
   */
  workspace_root: string | null;
}

interface SettingsState extends Settings {
  loaded: boolean;
  load: () => Promise<void>;
  save: (patch: Partial<Settings>) => Promise<void>;
}

const DEFAULTS: Settings = {
  active_provider_id: "local",
  providers: [
    {
      id: "local-llama",
      kind: "llama.cpp",
      location: "local",
      name: "Local (llama.cpp)",
      base_url: "http://127.0.0.1:8081/v1",
      enabled: true,
      sampling: { ...EMPTY_SAMPLING },
    },
  ],
  hf_token_set: false,
  default_model: null,
  thinking_enabled: true,
  agent_max_iterations: 8,
  destructive_tool_confirm: true,
  llama: {
    host: "127.0.0.1",
    n_gpu_layers: -1,
    ctx_size: 0,
    parallel: 1,
    extra_args: [],
    mtp_enabled: false,
  },
  audio: {
    enabled: false,
    stt_model: null,
    stt_language: "en",
    tts_model: null,
    tts_voice: null,
  },
  embedding: {
    enabled: false,
    documents_disabled: [],
  },
  auto_provision_llama: true,
  skills_enabled: [],
  // Default to a conservative built-in tool surface: filesystem mutation
  // (`fs.edit` / `fs.write`) and metadata probing (`fs.stat`), arbitrary
  // outbound HTTP (`http.fetch`), and clipboard I/O are off by default
  // so a fresh install can't surprise the user. They can be re-enabled
  // individually from the Tools page.
  builtin_tools_disabled: [
    "clipboard.read",
    "clipboard.write",
    "http.fetch",
    "fs.edit",
    "fs.write",
    "fs.stat",
  ],
  lazy_tool_discovery: true,
  mcp_servers: [],
  autostart_enabled: false,
  minimize_on_startup: false,
  close_to_taskbar: false,
  workspace_root: null,
};

export const useSettingsStore = create<SettingsState>((set, get) => ({
  ...DEFAULTS,
  loaded: false,
  load: async () => {
    try {
      const s = await invoke<Settings>("settings_load");
      if (s) set({ ...s, loaded: true });
      else set({ loaded: true });
    } catch (e) {
      console.error("settings_load failed", e);
      set({ loaded: true });
    }
  },
  save: async (patch) => {
    const next = { ...get(), ...patch };
    set(patch);
    try {
      await invoke("settings_save", { settings: next });
    } catch (e) {
      console.error("settings_save failed", e);
    }
  },
}));
