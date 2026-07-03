//! zero — local agentic AI built around llama.cpp.
//!
//! Top-level entry point. Wires up Tauri plugins, builds shared app state,
//! initialises the SQLite database, then registers every IPC command.

#![allow(clippy::needless_return)]

pub mod attachments;
pub mod chat;
pub mod commands;
pub mod db;
pub mod documents;
pub mod error;
pub mod events;
pub mod hf;
pub mod llama;
pub mod llm;
pub mod mcp;
pub mod memory;

pub mod paths;
pub mod secrets;
pub mod settings;
pub mod skills;
pub mod state;
pub mod system;
pub mod tasks;
pub mod whisper;
pub mod workspace;

use crate::state::AppState;
use std::sync::Arc;
use tauri::{Manager, RunEvent};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_tracing();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = state::AppState::init(handle).await {
                    tracing::error!("failed to initialise app state: {e:?}");
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // ─── system ──────────────────────────────────────────────
            commands::system::system_probe,
            commands::system::system_recommend_models,
            commands::system::system_recommend_refresh,
            commands::system::system_search_models,
            // ─── settings ─────────────────────────────────────────────
            commands::settings::settings_load,
            commands::settings::settings_save,
            commands::settings::settings_set_hf_token,
            commands::settings::settings_clear_hf_token,
            // ─── workspace (project root) ─────────────────────────────
            commands::workspace::workspace_get,
            commands::workspace::workspace_set,
            commands::workspace::workspace_clear,
            // ─── chat ─────────────────────────────────────────────────
            commands::chat::chat_list_conversations,
            commands::chat::chat_list_messages,
            commands::chat::chat_create_conversation,
            commands::chat::chat_delete_conversation,
            commands::chat::chat_set_model,
            commands::chat::chat_set_title,
            commands::chat::chat_get_disabled_tools,
            commands::chat::chat_set_disabled_tools,
            commands::chat::chat_get_sampling,
            commands::chat::chat_set_sampling,
            commands::chat::chat_send_message,
            commands::chat::chat_cancel,
            commands::chat::chat_tool_confirm,
            commands::chat::chat_retry,
            // ─── attachments ────────────────────────────────────
            commands::attachments::attachments_save,
            commands::attachments::attachments_delete,
            commands::attachments::attachments_purge_conversation,
            // ─── audio ──────────────────────────────────────────
            commands::audio::audio_transcribe,
            commands::audio::audio_speak,
            commands::audio::whisper_install,
            commands::audio::whisper_status,
            commands::audio::whisper_download_model,
            // ─── skills ───────────────────────────────────────
            commands::skills::skills_list,
            commands::skills::skills_create,
            commands::skills::skills_update,
            commands::skills::skills_delete,
            commands::skills::skills_set_enabled,
            commands::skills::skills_read_source,
            // ─── documents (embedding KB) ─────────────────────
            commands::documents::documents_list,
            commands::documents::documents_add,
            commands::documents::documents_delete,
            // ─── mcp ──────────────────────────────────────────
            commands::mcp::mcp_list_servers,
            commands::mcp::mcp_upsert_server,
            commands::mcp::mcp_delete_server,
            commands::mcp::mcp_set_enabled,
            commands::mcp::mcp_list_tools,
            commands::mcp::mcp_call_tool,
            commands::mcp::mcp_list_builtins,
            // ─── models ──────────────────────────────────────────────
            commands::models::models_search,
            commands::models::models_list_local,
            commands::models::models_list_gguf_files,
            commands::models::models_download,
            commands::models::models_download_files,
            commands::models::models_delete,
            commands::models::models_update,
            commands::models::models_cancel,
            // ─── llama.cpp ───────────────────────────────────────
            commands::llama::llama_info,
            commands::llama::llama_install,
            commands::llama::llama_install_variant,
            commands::llama::llama_install_applicable,
            commands::llama::llama_update_variant,
            commands::llama::llama_check_updates,
            commands::llama::llama_start,
            commands::llama::llama_stop,
            commands::llama::llama_load_model,
            commands::llama::llama_unload_model,
            commands::llama::llama_unload_variant,
            commands::llama::llama_switch_variant,
            // ─── tasks ────────────────────────────────────────────────
            commands::tasks::tasks_list,
            commands::tasks::tasks_create,
            commands::tasks::tasks_update,
            commands::tasks::tasks_delete,
            commands::tasks::tasks_run_now,
            commands::tasks::tasks_set_enabled,
            // ─── memory ───────────────────────────────────────────────
            commands::memory::memory_load,
            commands::memory::memory_add,
            commands::memory::memory_replace,
            commands::memory::memory_remove,
            commands::memory::memory_set_raw,
        ])
        .build(tauri::generate_context!())
        .expect("error while building zero");

    // Kill llama-server before the Tauri event loop exits.
    app.run(|handle, event| {
        if matches!(event, RunEvent::Exit) {
            shutdown_llama(handle);
        }
    });
}

/// Stop all running llama.cpp instances on app exit.
fn shutdown_llama<R: tauri::Runtime>(handle: &tauri::AppHandle<R>) {
    let Some(state) = handle.try_state::<Arc<AppState>>() else {
        return;
    };
    let llama = Arc::clone(&state.llama);
    // Stop every running variant. We try each one with a short timeout
    // so a wedged process doesn't keep the UI hanging on exit.
    let res = tauri::async_runtime::block_on(async {
        let info = llama.info().await;
        let mut any_running = false;
        for (slug, instance) in &info.instances {
            if matches!(
                instance.status,
                crate::llama::LlamaStatus::Running | crate::llama::LlamaStatus::Starting
            ) {
                any_running = true;
                if let Some(v) = crate::llama::variant::LlamaVariant::from_slug(slug) {
                    tracing::info!("stopping llama.cpp {slug} on app exit");
                    let _ = llama.stop(v).await;
                }
            }
        }
        if !any_running {
            tracing::info!("no llama.cpp instances running on app exit");
        }
    });
    // If we can't enumerate, just log it.
    let _ = res;
}

/// Wire up tracing for the whole backend.
///
/// Two layers are installed when possible:
///   * stdout — pretty, ANSI-colored, controlled by `ZERO_LOG`
///     (defaults to `info,zero_lib=debug`).
///   * rolling file — written to `<root>/logs/zero.log.<date>`, daily
///     rotation, no ANSI, with the `target` field included so the
///     dedicated `llm::wire` channel (request/response bodies) stands
///     out at a glance.
///
/// The file layer is best-effort: if we can't materialise the logs
/// directory (e.g. read-only home), we fall back to stdout-only rather
/// than panicking. The non-blocking appender's worker guard is leaked on
/// purpose so the background flush thread lives for the entire process —
/// dropping it here would close the channel before `run()` returns.
fn init_tracing() {
    // tokio-console layer. Compiled out entirely unless the `tokio-console`
    // feature is on; `Option<Layer>` is a no-op when `None`, so the registry
    // builder below stays a single code path in both configurations.
    #[cfg(feature = "tokio-console")]
    let console_layer = Some(console_subscriber::spawn());
    #[cfg(not(feature = "tokio-console"))]
    let console_layer: Option<tracing_subscriber::layer::Identity> = None;

    let stdout_filter = EnvFilter::try_from_env("ZERO_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info,zero_lib=debug"));
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_filter(stdout_filter);

    let file_layer = match paths::logs_dir() {
        Ok(dir) => {
            let appender = tracing_appender::rolling::daily(&dir, "zero.log");
            let (nb_writer, guard) = tracing_appender::non_blocking(appender);
            // Keep the flush thread alive for the lifetime of the process.
            // Without this the worker shuts down as soon as the guard goes
            // out of scope and subsequent writes silently disappear.
            Box::leak(Box::new(guard));
            let filter = EnvFilter::try_from_env("ZERO_FILE_LOG").unwrap_or_else(|_| {
                // `llm::wire` and `chat::final` carry the per-turn LLM
                // request/response bodies and the assembled assistant
                // turn. They're logged at `debug` so the terminal stays
                // readable; force them on at `debug` here so the file
                // always captures them even if ZERO_LOG is quieter.
                EnvFilter::new("info,zero_lib=debug,llm::wire=debug,chat::final=debug")
            });
            let layer = tracing_subscriber::fmt::layer()
                .with_writer(nb_writer)
                .with_ansi(false)
                .with_target(true)
                .with_filter(filter);
            Some(layer)
        }
        Err(e) => {
            eprintln!("zero: file logging disabled: {e:?}");
            None
        }
    };

    let _ = tracing_subscriber::registry()
        .with(console_layer)
        .with(stdout_layer)
        .with(file_layer)
        .try_init();
}
