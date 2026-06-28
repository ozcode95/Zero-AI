//! Canonical event names emitted from Rust to the frontend.
//! Keep in sync with `src/lib/tauri.ts`.

pub const CHAT_DELTA: &str = "chat://delta";
pub const CHAT_DONE: &str = "chat://done";
pub const CHAT_ERROR: &str = "chat://error";
/// Emitted by the agent loop when it needs to replace the live
/// streaming buffer for an assistant message with a canonical version
/// — used today to clean up legacy ```tool_use``` fenced-JSON blocks
/// that already streamed into the UI before we knew they were a
/// tool-call protocol marker. The frontend handler overwrites the
/// message's `content` rather than appending to it.
pub const CHAT_REWRITE: &str = "chat://rewrite";
/// Emitted when the agent loop wants to invoke a destructive MCP tool
/// while `settings.destructive_tool_confirm` is on. The frontend shows a
/// confirm modal; the user's choice round-trips back via the
/// `chat_tool_confirm` command. The runner blocks the stream waiting for
/// the decision (or chat cancellation).
pub const CHAT_TOOL_CONFIRM: &str = "chat://tool-confirm";
/// Emitted when the model calls the `ask_user_input` built-in. The runner
/// ends the turn and the UI renders the question(s) as tappable options;
/// the user's choice is sent back as an ordinary follow-up user message.
pub const CHAT_ASK_USER_INPUT: &str = "chat://ask-user-input";
/// Emitted when the model calls the `present_files` built-in. The UI
/// renders preview cards for the listed local files so the user can open
/// or download the assistant's deliverables.
pub const CHAT_PRESENT_FILES: &str = "chat://present-files";
pub const MODELS_DOWNLOAD_PROGRESS: &str = "models://download-progress";

pub const LLAMA_LOG: &str = "llama://log";
pub const LLAMA_STATUS: &str = "llama://status";
pub const LLAMA_INSTALL_PROGRESS: &str = "llama://install-progress";
/// Progress frames for the whisper.cpp runtime install and ggml model
/// downloads (speech-to-text). Shape mirrors the llama install progress.
pub const WHISPER_PROGRESS: &str = "whisper://progress";
pub const TASKS_TICK: &str = "tasks://tick";
