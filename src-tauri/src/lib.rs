//! zero — local agentic AI built around llama.cpp.
//!
//! Top-level entry point. Wires up Tauri plugins, builds shared app state,
//! initialises the SQLite database, then registers every IPC command.

#![allow(clippy::needless_return)]

pub mod agents_md;
pub mod attachments;
pub mod chat;
pub mod commands;
pub mod db;
pub mod documents;
pub mod error;
pub mod events;
pub mod hf;
pub mod hooks;
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
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, RunEvent, WindowEvent,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                // tauri-plugin-log is the *only* logger in the process: it
                // installs the global `log` logger during plugin init. The
                // backend keeps writing through `tracing::` macros, which are
                // bridged into `log` records by tracing's `log` feature (see
                // Cargo.toml) — so everything lands here. Installing a second
                // logger (e.g. tracing-subscriber's LogTracer) would panic
                // with "logger already initialized", so we don't.
                //
                // Targets: terminal (Stdout), devtools (Webview), and a
                // rolling file under `<root>/logs/zero.log`.
                .clear_targets()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Stdout,
                ))
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Webview,
                ))
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Folder {
                        path: paths::logs_dir()
                            .unwrap_or_else(|_| std::path::PathBuf::from("logs")),
                        file_name: Some("zero".into()),
                    },
                ))
                .level(log_level())
                // Keep the backend's own crate chatty (matches the old
                // `zero_lib=debug` filter) while third-party crates stay at
                // the global level.
                .level_for("zero_lib", log::LevelFilter::Debug)
                .build(),
        )
        .setup(|app| {
            // System tray: always present so a hidden window can be
            // restored. The "minimize on close" behaviour lives in
            // `close_to_taskbar` (Settings) — when it's on, the main
            // window's close button hides to tray instead of quitting.
            build_tray(&app.handle())?;
            wire_close_to_tray(&app.handle())?;

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
            // ─── agents.md (context injection) ────────────────────
            commands::agents_md::agents_md_get,
            commands::agents_md::agents_md_set,
            // ─── hooks (lifecycle tool hooks) ────────────────────
            commands::hooks::hooks_get,
            commands::hooks::hooks_set,
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

/// Build the system-tray icon + menu (Show / Quit). Always created at
/// startup; the close-to-tray *behaviour* is gated by `close_to_taskbar`.
fn build_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let show_i = MenuItem::with_id(app, "show", "Show Zero", true, None::<&str>)?;
    let quit_i = MenuItem::with_id(app, "quit", "Quit Zero", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_i, &quit_i])?;
    let _tray = TrayIconBuilder::with_id("zero-tray")
        .tooltip("ZerØ")
        .icon(app.default_window_icon().cloned().unwrap())
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

/// Intercept the main window's close button. When `close_to_taskbar` is
/// on, hide to tray instead of quitting — the tray menu's Show / left-click
/// restores the window. Off (the default) closes normally.
fn wire_close_to_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let win = app.get_webview_window("main").expect("main window");
    let app_for_close = app.clone();
    win.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            let to_tray = tauri::async_runtime::block_on(async {
                crate::settings::Settings::load()
                    .await
                    .map(|s| s.close_to_taskbar)
                    .unwrap_or(false)
            });
            if to_tray {
                api.prevent_close();
                if let Some(w) = app_for_close.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
        }
    });
    Ok(())
}

/// Reveal + focus the main window (called from tray menu / click).
fn show_main_window<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

/// Max log level for tauri-plugin-log. Defaults to `Info`; override with
/// `ZERO_LOG_LEVEL=debug|info|warn|error|trace`.
fn log_level() -> log::LevelFilter {
    std::env::var("ZERO_LOG_LEVEL")
        .ok()
        .and_then(|s| s.parse::<log::LevelFilter>().ok())
        .unwrap_or(log::LevelFilter::Info)
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
