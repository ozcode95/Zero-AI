//! Shared application state.
//!
//! Wraps the SQLite pool, the HTTP client, the per-chat cancellation
//! registry, and any cached system specs. Stored as a Tauri-managed
//! singleton so commands can reach it via `tauri::State<AppState>`
//! (or the `AppStateExt` shortcut below).

use crate::chat::runner::{ChatJobs, ToolConfirms};
use crate::db;
use crate::hf::DownloadJobs;
use crate::llama::LlamaOrchestrator;
use crate::mcp::catalog::McpToolsCache;
use crate::paths;
use crate::system::Specs;
use anyhow::Result;
use sqlx::SqlitePool;
use std::sync::Arc;
use tauri::{AppHandle, Manager};
use tokio::sync::RwLock;

pub struct AppState {
    pub db: SqlitePool,
    pub http: reqwest::Client,
    /// Multi-variant llama.cpp orchestrator. Manages multiple builds
    /// (CUDA, OpenVINO, HIP-Radeon, CPU) that can be installed and run
    /// simultaneously on different ports. The active variant (set by the
    /// user or auto-detected from hardware) determines which instance
    /// the chat runner routes to.
    pub llama: Arc<LlamaOrchestrator>,
    pub chat_jobs: Arc<ChatJobs>,
    /// Pending destructive-tool confirm prompts — see
    /// [`ToolConfirms`] for the flow.
    pub tool_confirms: Arc<ToolConfirms>,
    /// Same pattern as `chat_jobs` — a registry of in-flight HF model
    /// downloads, used both as a per-model concurrency guard and as the
    /// cancellation channel for `models_cancel`.
    pub downloads: Arc<DownloadJobs>,
    /// Cached `tools/list` results, keyed by MCP server id. Lets the chat
    /// runner inject the tool catalog into the system prompt without
    /// re-probing every server on every turn.
    pub mcp_cache: Arc<McpToolsCache>,
    pub specs: RwLock<Option<Specs>>,
}

impl AppState {
    /// Initialise once at startup. Stores the result in the Tauri app handle.
    pub async fn init(app: AppHandle) -> Result<()> {
        // Best-effort one-time migration: if the user has data under the
        // pre-`.zero` layout (`%LOCALAPPDATA%\zero\zero\data\` and
        // friends), move it into `~/.zero` before any code touches the
        // new paths. Failures here are non-fatal — we still come up,
        // just with an empty profile rather than the user's old data.
        if let Err(e) = paths::migrate_legacy_root_if_needed() {
            tracing::warn!("legacy storage migration failed: {e:#}");
        }

        let db = db::open_pool(&paths::db_file()?).await?;
        db::migrate(&db).await?;

        // Best-effort: backfill `verified_files` for rows installed before
        // we tracked it. Failures here are non-fatal — we just lose the
        // verified badge in the UI for legacy entries.
        if let Err(e) = crate::hf::backfill_verified(&db).await {
            tracing::warn!("hf backfill_verified failed: {e:#}");
        }

        let mut builder =
            reqwest::Client::builder().user_agent(concat!("zero/", env!("CARGO_PKG_VERSION")));

        // Load extra CA(s) — corporate MITM root.
        // Path comes from env var so we don't bake site-specific config into the binary.
        if let Ok(path) =
            std::env::var("ZERO_EXTRA_CA_BUNDLE").or_else(|_| std::env::var("SSL_CERT_FILE"))
        {
            match std::fs::read(&path) {
                Ok(pem) => {
                    // A bundle file may contain multiple PEM blocks; add them all.
                    for cert in reqwest::Certificate::from_pem_bundle(&pem).unwrap_or_default() {
                        builder = builder.add_root_certificate(cert);
                    }
                    tracing::info!("loaded extra CA bundle from {path}");
                }
                Err(e) => tracing::warn!("failed reading CA bundle {path}: {e}"),
            }
        }

        let http = builder.build()?;

        let llama = Arc::new(LlamaOrchestrator::new(
            app.clone(),
            db.clone(),
            http.clone(),
        ));
        if let Err(e) = llama.hydrate_from_db().await {
            tracing::warn!("llama hydrate from db failed: {e:#}");
        }

        // Migrate legacy llama_state data to per-variant state.
        if let Err(e) = crate::db::llama_variant_state::migrate_from_legacy(&db, "cuda").await {
            tracing::warn!("llama legacy state migration failed: {e:#}");
        }
        // Migrate legacy runtime_versions row from "llama.cpp" to "llama.cpp-cuda"
        // (or whatever variant was previously installed).
        if let Ok(Some(rv)) = db::runtimes::get(&db, "llama.cpp").await {
            let variant_slug = rv
                .metadata
                .as_ref()
                .and_then(|m| m.get("variant"))
                .and_then(|v| v.as_str())
                .unwrap_or("cuda");
            let new_name = format!("llama.cpp-{variant_slug}");
            // Only migrate if the new row doesn't already exist.
            if db::runtimes::get(&db, &new_name)
                .await
                .ok()
                .flatten()
                .is_none()
            {
                let mut migrated = rv.clone();
                migrated.name = new_name;
                if let Err(e) = db::runtimes::upsert(&db, &migrated).await {
                    tracing::warn!("failed to migrate runtime_versions row: {e:#}");
                } else {
                    tracing::info!(
                        "migrated runtime_versions: llama.cpp → llama.cpp-{variant_slug}"
                    );
                    // Move the install directory if it exists.
                    let old_dir = paths::llama_dir()?;
                    let variant = crate::llama::variant::LlamaVariant::from_slug(variant_slug)
                        .unwrap_or(crate::llama::variant::LlamaVariant::Cuda);
                    let new_dir = paths::llama_variant_dir(variant)?;
                    if old_dir.exists() && !new_dir.exists() {
                        if let Err(e) = std::fs::rename(&old_dir, &new_dir) {
                            tracing::warn!("failed to move llama.cpp dir to variant dir: {e:#}");
                        }
                    }
                    // Delete the old DB row.
                    if let Err(e) = db::runtimes::delete(&db, "llama.cpp").await {
                        tracing::warn!("failed to delete old llama.cpp runtime row: {e:#}");
                    }
                }
            }
        }

        let state = Arc::new(AppState {
            db,
            http,
            llama,
            chat_jobs: Arc::new(ChatJobs::new()),
            tool_confirms: Arc::new(ToolConfirms::new()),
            downloads: DownloadJobs::new(),
            mcp_cache: McpToolsCache::new(),
            specs: RwLock::new(None),
        });

        app.manage(state);
        tracing::info!("app state initialised");

        // ─── hardware-aware runtime auto-provisioning ─────────────────
        //
        // Probe the host once at startup (cache-first). If any discrete
        // GPU is present, force-enable llama.cpp provisioning so the
        // user gets a working local runtime on first launch without
        // manually toggling settings.
        let specs_for_policy = match Self::resolve_specs().await {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(
                    "startup hardware probe failed; falling back to OVMS auto-provision: {e:#}"
                );
                None
            }
        };
        let has_dgpu = specs_for_policy
            .as_ref()
            .map(crate::system::has_discrete_gpu)
            .unwrap_or(false);
        // Cache the probe result on the managed state so the System
        // settings page doesn't have to re-probe on first visit.
        if let Some(s) = specs_for_policy {
            *app.zero().specs.write().await = Some(s);
        }

        tracing::info!(
            "startup: {} — auto-provisioning llama.cpp",
            if has_dgpu {
                "discrete GPU detected"
            } else {
                "no discrete GPU"
            }
        );
        if has_dgpu {
            if let Err(e) = Self::set_first_launch_active_provider("local-llama").await {
                tracing::warn!("could not seed active_provider_id for first launch: {e:#}");
            }
        }
        let llama_provision = Arc::clone(&app.zero().llama);
        tauri::async_runtime::spawn(async move {
            llama_provision.auto_provision(has_dgpu).await;
        });

        // Honour the "start minimized" preference. Reading settings here
        // (rather than in the synchronous `setup`) means the window may
        // flash visible for a frame before minimizing, an acceptable
        // trade for not blocking startup on a disk read.
        if let Ok(settings) = crate::settings::Settings::load().await {
            // Hydrate the process-global workspace cache so the built-in
            // `fs.*` tools resolve relative paths against the user's open
            // project from the very first turn.
            crate::workspace::set(
                settings
                    .workspace_root
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
                    .map(std::path::PathBuf::from),
            );
            if settings.minimize_on_startup {
                if let Some(win) = app.get_webview_window("main") {
                    if let Err(e) = win.minimize() {
                        tracing::warn!("failed to minimize window on startup: {e:#}");
                    }
                }
            }
        }

        // Start the task ticker. It scans the `tasks` table every
        // ~30 s and fires anything whose trigger is due. Lives for
        // the lifetime of the app handle; nothing here needs to await.
        crate::tasks::scheduler::start(app.clone());

        Ok(())
    }

    /// Cache-first hardware probe used by the startup auto-provision
    /// policy. Mirrors the lookup in `commands::system::system_probe`
    /// but doesn't need an `AppHandle` because it runs before the
    /// state has finished initialising.
    async fn resolve_specs() -> Result<crate::system::Specs> {
        if let Some(cached) = crate::system::load_cached() {
            return Ok(cached);
        }
        // Cold probe: WMI on Windows needs a blocking thread.
        let specs = tokio::task::spawn_blocking(crate::system::probe)
            .await
            .map_err(|e| anyhow::anyhow!("hardware probe task panicked: {e}"))??;
        if let Err(e) = crate::system::save_cached(&specs) {
            tracing::warn!("system cache write failed: {e}");
        }
        Ok(specs)
    }

    /// On true first launch (no `settings.json` on disk yet), persist
    /// a `Settings` whose `active_provider_id` matches the hardware
    /// policy. We deliberately only write when the file is missing so
    /// repeated launches with a dGPU don't keep overwriting the user's
    /// explicit later choice.
    async fn set_first_launch_active_provider(provider_id: &str) -> Result<()> {
        let p = paths::settings_file()?;
        if p.exists() {
            return Ok(());
        }
        let mut s = crate::settings::Settings::default();
        s.active_provider_id = Some(provider_id.to_string());
        s.save().await?;
        Ok(())
    }
}

/// Convenience extension: pull the managed `Arc<AppState>` out of a handle.
pub trait AppStateExt {
    fn zero(&self) -> Arc<AppState>;
}

impl<R: tauri::Runtime> AppStateExt for tauri::AppHandle<R> {
    fn zero(&self) -> Arc<AppState> {
        self.state::<Arc<AppState>>().inner().clone()
    }
}
