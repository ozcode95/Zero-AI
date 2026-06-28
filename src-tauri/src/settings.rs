//! Shared `Settings` model.
//!
//! Lives in its own module (rather than under `commands::settings`) so the
//! agent runner and provider factory can read settings without going through
//! Tauri's IPC layer. The actual `#[tauri::command]` wrappers in
//! `commands::settings` delegate here.
//!
//! Storage is intentionally trivial right now (one JSON blob at
//! `<root>/settings.json`). Schema evolution is `serde(default)`-driven so old
//! files keep working as we add fields.

use crate::paths;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub kind: String,
    /// Where the provider lives: `local` (a runtime on this machine) vs.
    /// `cloud` (a remote hosted endpoint). Free-form string so additional
    /// classifications can be added later without breaking older settings
    /// files. Defaults to `local` for entries written before this field
    /// existed.
    #[serde(default = "default_provider_location")]
    pub location: String,
    pub name: String,
    pub base_url: String,
    pub enabled: bool,
    #[serde(default)]
    pub api_key_ref: Option<String>,
    /// Per-provider sampling override. Empty fields mean "use the
    /// per-model [`crate::chat::runner::ModelProfile`] default". Layered
    /// under any per-conversation override so a single chat can still
    /// diverge from the provider-wide value.
    #[serde(default)]
    pub sampling: SamplingConfig,
}

fn default_provider_location() -> String {
    "local".into()
}

/// Optional sampling overrides. Every field is independently optional:
/// a `None` means "don't override", letting the precedence chain
/// (conversation → provider → model profile → upstream default) fall
/// through to the next layer.
///
/// Stored on [`ProviderConfig`] (global) and on each conversation row
/// (per-chat). Identical shape on the wire so the UI can reuse one
/// editor component for both surfaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    /// Override the sampling temperature. Typical range 0.0–2.0.
    pub temperature: Option<f32>,
    /// Override the nucleus-sampling cutoff. Typical range 0.0–1.0.
    pub top_p: Option<f32>,
    /// Override the top-k cutoff. Forwarded as an OpenAI-compat
    /// extension; ignored by servers that don't understand it.
    pub top_k: Option<u32>,
}

/// User-configured external MCP server. Supports three transports:
///
/// - `http` / `sse` — JSON-RPC over the streamable HTTP transport. `url`
///   + `headers` are used; `command` / `args` / `env` are ignored.
/// - `stdio`        — zero spawns `command` with `args` (and any extra
///   `env`) and speaks JSON-RPC on the child's stdin/stdout per call.
///   `url` and `headers` are ignored.
///
/// All transport-specific fields carry `#[serde(default)]` so older
/// settings files (which only had `url` + `headers`) keep loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub id: String,
    pub name: String,
    /// `http` | `sse` | `stdio`. Free-form so additional transports can
    /// be added without breaking existing settings files.
    #[serde(default = "default_mcp_transport")]
    pub transport: String,
    /// Base URL for the MCP endpoint, e.g. `http://127.0.0.1:9001/mcp`.
    /// Empty for stdio transport.
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    /// For `stdio` transport: the executable to spawn (e.g. `uvx`,
    /// `npx`, an absolute path to a binary). Unused for HTTP/SSE.
    #[serde(default)]
    pub command: String,
    /// For `stdio` transport: argv to pass to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// For `stdio` transport: extra environment variables to set on the
    /// child (merged on top of the zero process env).
    #[serde(default)]
    pub env: Vec<(String, String)>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_mcp_transport() -> String {
    "http".into()
}

/// Legacy OVMS settings preserved for backward compatibility with existing
/// `settings.json` files. No longer functional — OVMS has been removed in
/// favour of the multi-variant llama.cpp orchestrator. The fields are kept
/// so deserialization doesn't break on upgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvmsSettings {
    pub rest_port: u16,
    /// gRPC bind port. `0` disables the gRPC server entirely (OVMS's
    /// documented way to opt out). zero talks to OVMS over REST only, so
    /// the default is `0` to avoid pointlessly grabbing a port (and the
    /// `WSAEADDRINUSE` failures that come with it when something else is
    /// already bound on 9000). Users who need gRPC for other clients can
    /// set this to a free port from the Settings UI.
    pub grpc_port: u16,
    /// One of `CPU`, `GPU`, `NPU`, `AUTO`. Free-form so future OpenVINO
    /// devices (`MULTI:CPU,GPU`, `HETERO:GPU,CPU`, ...) can be entered directly.
    pub device: String,

    // ─── advanced server-side knobs (all optional) ─────────────────────
    /// `DEBUG` / `INFO` / `ERROR`. `None` keeps the OVMS default (INFO).
    /// Mapped to `--log_level`.
    #[serde(default)]
    pub log_level: Option<String>,
    /// Optional path to a log file. Mapped to `--log_path`.
    #[serde(default)]
    pub log_path: Option<String>,
    /// Optional model compilation cache directory. Big startup speedup on
    /// subsequent runs. Mapped to `--cache_dir`.
    #[serde(default)]
    pub cache_dir: Option<String>,
    /// CORS `Access-Control-Allow-Origin`. Mapped to `--allowed_origins`.
    #[serde(default)]
    pub allowed_origins: Option<String>,
    /// Path to a file whose first line holds the API key required on
    /// `/v3/*` endpoints. Mapped to `--api_key_file`.
    #[serde(default)]
    pub api_key_file: Option<String>,
    /// Escape hatch for any flag we don't model explicitly. Each entry is
    /// forwarded verbatim as a separate `argv` token — the UI splits the
    /// user's textarea on newlines so multi-token flags become two entries
    /// (`--grpc_workers` then `4`).
    #[serde(default)]
    pub extra_args: Vec<String>,
}

impl Default for OvmsSettings {
    fn default() -> Self {
        Self {
            rest_port: 8000,
            grpc_port: 0,
            device: "GPU".into(),
            log_level: None,
            log_path: None,
            cache_dir: None,
            allowed_origins: None,
            api_key_file: None,
            extra_args: Vec::new(),
        }
    }
}

impl OvmsSettings {
    // No methods — OVMS has been removed. The struct is kept for
    // backward-compatible deserialization of existing settings.json files.
}

/// User-visible knobs for the bundled `llama-server` runtime.
/// llama-server takes its model as a positional `--model <path>` arg
/// (rather than a multi-model config file), so the only state we need
/// to persist on top of the port is GPU offload + context size and an
/// escape hatch for any flag we don't model explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaSettings {
    /// Which llama.cpp variant is currently active ("cuda", "openvino",
    /// "hip-radeon", "cpu"). `None` means the orchestrator should pick
    /// the highest-priority installed variant automatically.
    #[serde(default)]
    pub active_variant: Option<String>,
    /// Bind address. Defaults to `127.0.0.1` — zero never exposes
    /// llama-server beyond loopback unless the user opts in.
    #[serde(default = "default_llama_host")]
    pub host: String,
    /// Number of model layers to offload to the GPU. `-1` lets
    /// llama-server pick the maximum that fits (its default behaviour
    /// on every GPU-enabled build). `0` keeps inference on CPU even
    /// when a GPU build is installed.
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: i32,
    /// Context size (`-c`). `0` keeps the model's training-time
    /// default, which is what most users want.
    #[serde(default)]
    pub ctx_size: u32,
    /// Parallel sequence slots (`-np`). `1` matches llama-server's
    /// default and is the lowest-VRAM option.
    #[serde(default = "default_parallel")]
    pub parallel: u32,
    /// Optional comma list of additional command-line arguments. Each
    /// entry is forwarded verbatim as a separate argv token so
    /// multi-token flags become two entries (`--mlock` then
    /// `--threads-batch` then `8`).
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// **Experimental.** Wire downloaded MTP / speculative-decoding draft
    /// models into the router preset so they're used at load time. Off by
    /// default: MTP drafts crash or fail to load on some llama.cpp build /
    /// GPU combinations, so enabling this can prevent a model from loading.
    #[serde(default)]
    pub mtp_enabled: bool,
}

fn default_llama_host() -> String {
    "127.0.0.1".into()
}
fn default_n_gpu_layers() -> i32 {
    -1
}
fn default_parallel() -> u32 {
    1
}

impl Default for LlamaSettings {
    fn default() -> Self {
        Self {
            active_variant: None,
            host: default_llama_host(),
            n_gpu_layers: default_n_gpu_layers(),
            ctx_size: 0,
            parallel: default_parallel(),
            extra_args: Vec::new(),
            mtp_enabled: false,
        }
    }
}

impl LlamaSettings {
    /// Canonical OpenAI-compatible base URL for a specific variant.
    /// Each variant listens on a fixed port, so the URL is determined
    /// by the variant + the host setting.
    pub fn base_url_for(&self, variant: crate::llama::variant::LlamaVariant) -> String {
        format!("http://{}:{}/v1", self.host, variant.default_port())
    }

    /// Render the always-on knobs into an `argv` vector ready to hand
    /// to `llama-server.exe`. If a model path is provided, it's appended
    /// last so it can never be shadowed by a user-supplied `extra_args`
    /// entry. If no model is provided, the server starts idle and can
    /// load a model later.
    ///
    /// The port is derived from the variant — each build listens on
    /// its own fixed port so multiple instances can coexist.
    pub fn cli_args_for(&self, variant: crate::llama::variant::LlamaVariant) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "--host".into(),
            self.host.clone(),
            "--port".into(),
            variant.default_port().to_string(),
            "--n-gpu-layers".into(),
            self.n_gpu_layers.to_string(),
            "--parallel".into(),
            self.parallel.to_string(),
            // Cap concurrent requests at the parallel slot count so
            // llama-server queues rather than rejects extras.
            "--cont-batching".into(),
            "--jinja".into(),
        ];

        // Models are loaded via the /models/load HTTP API with absolute
        // paths (no --models-dir needed).

        if self.ctx_size > 0 {
            args.push("--ctx-size".into());
            args.push(self.ctx_size.to_string());
        }
        for raw in &self.extra_args {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                args.push(trimmed.to_string());
            }
        }
        args
    }
}

/// Audio (speech-to-text + text-to-speech) capabilities.
///
/// Off by default. When `enabled` is true the chat composer surfaces a
/// voice-input button (STT) and assistant messages grow a "speak" button
/// (TTS).
///
/// Speech-to-text shells out to `whisper-cli` (whisper.cpp) with a ggml
/// model `.bin`. Text-to-speech shells out to `llama-tts` (a sibling of
/// `llama-server`) with an OuteTTS model + WavTokenizer vocoder. Both run
/// as one-shot GPU CLIs per utterance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioSettings {
    /// Master switch. Off by default so a fresh install ships no audio UI.
    #[serde(default)]
    pub enabled: bool,
    /// Whisper ggml model file name used for transcription (e.g.
    /// `ggml-base.en.bin`). `None` until the user picks / downloads one.
    #[serde(default)]
    pub stt_model: Option<String>,
    /// Spoken-language hint passed to whisper (ISO code like `"en"`, or
    /// `"auto"` to detect). Defaults to English: auto-detection is
    /// unreliable on short dictation clips and often mis-fires (e.g.
    /// transcribing English speech as Japanese). `None` is treated as
    /// `"en"`.
    #[serde(default)]
    pub stt_language: Option<String>,
    /// HuggingFace repo id of the OuteTTS model `llama-tts` reads for
    /// read-aloud (e.g. `OuteAI/OuteTTS-0.2-500M-GGUF`). `None` until the
    /// user picks / downloads one.
    #[serde(default)]
    pub tts_model: Option<String>,
    /// Retained for back-compat; no longer used (read-aloud is synthesized
    /// by `llama-tts`, not the Web Speech API).
    #[serde(default)]
    pub tts_voice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingSettings {
    /// Master switch. Off by default so a fresh install ships no document
    /// grounding. When on, the text of every enabled document is injected
    /// into the system prompt for every chat turn. The Embedding page only
    /// lets the user flip this on once an embedding model is installed.
    #[serde(default)]
    pub enabled: bool,
    /// Ids (stored filenames under `~/.zero/documents/`) of documents the
    /// user has switched *off*. Tracking the disabled set — rather than the
    /// enabled set — means every newly added document is on by default,
    /// which is the behaviour the Embedding page advertises. Ids that no
    /// longer exist on disk are harmlessly ignored.
    #[serde(default)]
    pub documents_disabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub active_provider_id: Option<String>,
    pub providers: Vec<ProviderConfig>,
    pub hf_token_set: bool,
    pub default_model: Option<String>,
    pub thinking_enabled: bool,
    pub agent_max_iterations: u32,
    pub destructive_tool_confirm: bool,
    #[serde(default)]
    pub ovms: OvmsSettings,
    /// User-visible knobs for the bundled `llama-server` runtime.
    /// Defaults so existing `settings.json` files (which predate the
    /// field) keep deserialising without migration.
    #[serde(default)]
    pub llama: LlamaSettings,
    /// Speech-to-text / text-to-speech capabilities. Off by default.
    #[serde(default)]
    pub audio: AudioSettings,
    /// Embedding / document-grounding feature. Off by default; enabled from
    /// the Embedding page once an embedding model is installed.
    #[serde(default)]
    pub embedding: EmbeddingSettings,
    /// Legacy — preserved for backward-compatible deserialization.
    /// No longer functional; OVMS has been removed.
    #[serde(default = "default_true")]
    pub auto_provision_ovms: bool,
    /// When true (the default), startup will install the applicable
    /// llama.cpp variant(s) and start the highest-priority one.
    /// Overridden when a discrete GPU is detected (always installs).
    #[serde(default)]
    pub auto_provision_llama: bool,
    /// Ids of skills (folder names under `~/.zero/skills/`) that should be
    /// injected into the system prompt for every new turn. Skills that no
    /// longer exist on disk are silently ignored.
    #[serde(default)]
    pub skills_enabled: Vec<String>,
    /// Names of built-in tools (e.g. `fs.list`, `shell.exec`) that the user
    /// has globally disabled from the Tools page. The chat catalog hides
    /// these across every conversation; the per-chat tools popover only
    /// disables a tool for one conversation, which is a separate override.
    #[serde(default)]
    pub builtin_tools_disabled: Vec<String>,
    /// When true, the chat runner ships only the `tools.list` built-in to
    /// the LLM on the first round of each turn instead of the full tool
    /// catalogue. The model must call `tools.list` to discover what else
    /// is available; once it does, the runner expands the OpenAI `tools`
    /// array on subsequent rounds in the same turn to expose everything.
    /// Trades a single extra round-trip on tool-using turns for a much
    /// smaller initial-context cost — valuable when many MCP servers are
    /// wired up but most turns don't actually need a tool. Defaults on so
    /// new chats favour smaller initial context; the safety-net check in
    /// the runner still tolerates a hand-edited `false` for users who
    /// want every tool inlined up front.
    #[serde(default = "default_true")]
    pub lazy_tool_discovery: bool,
    /// External MCP server configurations.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Launch zero automatically when the user logs into the OS. The
    /// frontend keeps the OS-level autostart entry in sync via the
    /// autostart plugin whenever this toggle flips.
    #[serde(default)]
    pub autostart_enabled: bool,
    /// Start the main window minimized instead of focused.
    #[serde(default)]
    pub minimize_on_startup: bool,
    /// When true the window close button minimizes to the taskbar instead
    /// of quitting the app.
    #[serde(default)]
    pub close_to_taskbar: bool,
}

fn default_true() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            active_provider_id: Some("local-llama".into()),
            providers: vec![ProviderConfig {
                id: "local-llama".into(),
                kind: "llama.cpp".into(),
                location: "local".into(),
                name: "Local (llama.cpp)".into(),
                base_url: "http://127.0.0.1:8081/v1".into(),
                enabled: true,
                api_key_ref: None,
                sampling: SamplingConfig::default(),
            }],
            hf_token_set: false,
            default_model: None,
            thinking_enabled: true,
            agent_max_iterations: 8,
            destructive_tool_confirm: true,
            ovms: OvmsSettings::default(),
            llama: LlamaSettings::default(),
            audio: AudioSettings::default(),
            embedding: EmbeddingSettings::default(),
            auto_provision_ovms: false,
            auto_provision_llama: true,
            skills_enabled: Vec::new(),
            // Conservative built-in surface for fresh installs: filesystem
            // mutation (`fs.edit` / `fs.write`), metadata probing
            // (`fs.stat`), arbitrary outbound HTTP (`http.fetch`), and
            // clipboard I/O start disabled so the agent can't surprise a
            // brand-new user. Each one can be flipped back on from the
            // Tools page.
            builtin_tools_disabled: vec![
                "clipboard.read".into(),
                "clipboard.write".into(),
                "http.fetch".into(),
                "fs.edit".into(),
                "fs.write".into(),
                "fs.stat".into(),
            ],
            lazy_tool_discovery: true,
            mcp_servers: Vec::new(),
            autostart_enabled: false,
            minimize_on_startup: false,
            close_to_taskbar: false,
        }
    }
}

impl Settings {
    /// Reads the on-disk settings file, falling back to `Default` on a missing
    /// or unparsable file. We deliberately don't error here so a corrupt file
    /// never bricks the app — the user can re-save from the UI to fix it.
    pub async fn load() -> Result<Self> {
        let p = paths::settings_file()?;
        if !p.exists() {
            return Ok(Self::default());
        }
        let bytes = tokio::fs::read(&p).await?;
        let mut s: Self = serde_json::from_slice(&bytes).unwrap_or_else(|_| Self::default());
        s.migrate_legacy_defaults();
        Ok(s)
    }

    /// In-place rewrites for fields whose default changed in a way that
    /// would otherwise leave existing installs stuck on the old value.
    /// Each branch is keyed off the *exact* previous default so users who
    /// explicitly set a non-default value are never overridden.
    fn migrate_legacy_defaults(&mut self) {
        // `builtin_tools_disabled` previously defaulted to an empty list
        // — every built-in tool was advertised on a fresh install. We
        // now ship a conservative subset (clipboard / http.fetch / fs
        // mutation + stat) disabled by default. An exactly-empty list on
        // disk almost certainly came from that old default rather than
        // an explicit user choice ("I want every tool on" is far rarer
        // than "I never touched the Tools page" on a tool already gated
        // behind an opt-in confirm prompt), so upgrade it in place.
        if self.builtin_tools_disabled.is_empty() {
            self.builtin_tools_disabled = Self::default().builtin_tools_disabled;
        }
    }

    pub async fn save(&self) -> Result<()> {
        let p = paths::settings_file()?;
        let bytes = serde_json::to_vec_pretty(self)?;
        tokio::fs::write(&p, bytes).await?;
        Ok(())
    }

    /// Returns the active provider config, falling back to the first enabled
    /// provider if `active_provider_id` is unset or stale.
    pub fn active_provider(&self) -> Option<&ProviderConfig> {
        if let Some(id) = self.active_provider_id.as_deref() {
            if let Some(p) = self.providers.iter().find(|p| p.id == id) {
                return Some(p);
            }
        }
        self.providers.iter().find(|p| p.enabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_legacy_empty_builtin_disabled_to_safe_default() {
        // An exactly-empty list on disk maps to the old "every tool
        // enabled" default; the migration upgrades it to the current
        // conservative set so existing installs match a fresh one.
        let mut s = Settings::default();
        s.builtin_tools_disabled.clear();
        s.migrate_legacy_defaults();
        let want = Settings::default().builtin_tools_disabled;
        assert_eq!(s.builtin_tools_disabled, want);
        assert!(want.iter().any(|n| n == "clipboard.read"));
        assert!(want.iter().any(|n| n == "http.fetch"));
        assert!(want.iter().any(|n| n == "fs.write"));
    }

    #[test]
    fn migrate_preserves_user_chosen_builtin_disabled() {
        // Any non-empty list signals the user has touched the Tools
        // page; we leave their picks alone even if shorter / different
        // from the new default set.
        let mut s = Settings::default();
        s.builtin_tools_disabled = vec!["shell.exec".into()];
        s.migrate_legacy_defaults();
        assert_eq!(s.builtin_tools_disabled, vec!["shell.exec".to_string()]);
    }
}
