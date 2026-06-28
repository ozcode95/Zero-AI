//! llama.cpp multi-variant orchestrator.
//!
//! Manages multiple llama.cpp build variants (CUDA, OpenVINO, HIP-Radeon, CPU)
//! that can coexist on disk and run simultaneously on different ports.
//!
//! ## Architecture
//!
//! ```text
//! LlamaOrchestrator
//! ├── instances: HashMap<LlamaVariant, LlamaInstance>
//! │   ├── cuda → LlamaInstance { info, process }
//! │   ├── openvino → LlamaInstance { info, process }
//! │   └── ...
//! └── active_variant: LlamaVariant  ← which variant the chat runner talks to
//! ```
//!
//! Each variant gets:
//! - Its own install directory under `runtimes/llama.cpp/{variant}/`
//! - Its own row in `runtime_versions` keyed by `llama.cpp-{variant}`
//! - Its own DB state in `llama_variant_state` for the loaded model
//! - Its own fixed port (8081=cuda, 8082=openvino, 8083=hip-radeon, 8084=cpu)
//!
//! The **active variant** is the one the chat runner routes to. It defaults
//! to the highest-priority installed variant but can be switched by the
//! user at any time. Multiple instances can run simultaneously on their
//! respective ports.

pub mod github;
pub mod health;
pub mod install;
pub mod preset;
pub mod process;
pub mod tts;
pub mod variant;

use crate::db;
use crate::events;
use crate::llama::variant::LlamaVariant;
use crate::paths;
use crate::settings::Settings;
use process::{ExitReason, LlamaProcess};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};
use tokio::sync::RwLock;

const READY_TIMEOUT: Duration = Duration::from_secs(120);
/// Upper bound for an individual model to finish loading after a
/// `/models/load` request. Large quants on slow disks can take a while.
const LOAD_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a fetched "latest release" tag stays fresh before
/// `check_for_updates` will hit the GitHub API again. Keeps us well under
/// the 60 req/hr unauthenticated rate limit even with frequent UI checks.
const UPDATE_CHECK_TTL: Duration = Duration::from_secs(60 * 60);

// ─── Status & info types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlamaStatus {
    NotInstalled,
    Installing,
    Installed,
    Starting,
    Running,
    Stopping,
    Stopped,
    Error,
}

/// Per-variant status snapshot. Emitted per-variant so the UI can render
/// each instance independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaInstanceInfo {
    /// Which variant this info belongs to.
    pub variant: String,
    pub installed_version: Option<String>,
    /// Latest upstream release tag, once an update check has populated the
    /// orchestrator's cache this session. `None` before the first check.
    pub latest_version: Option<String>,
    /// True when this variant is installed and its `installed_version`
    /// differs from `latest_version`. Always `false` until a check runs.
    pub update_available: bool,
    pub status: LlamaStatus,
    pub pid: Option<u32>,
    /// Base URL for this variant's instance (e.g. `http://127.0.0.1:8081/v1`).
    pub base_url: String,
    pub loaded_model: Option<String>,
    pub loaded_model_path: Option<String>,
    pub last_error: Option<String>,
}

/// Full orchestrator status — the shape the frontend receives via
/// `llama://status` events and the `llama_info` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorInfo {
    /// The variant the chat runner should route to. Determined by
    /// `settings.llama.active_variant` or auto-detected from installed
    /// variants (highest priority wins).
    pub active_variant: String,
    /// Per-variant status. Only variants that are installed or have been
    /// interacted with appear here.
    pub instances: HashMap<String, LlamaInstanceInfo>,
    /// Variant slugs that can actually run on the detected hardware
    /// (accelerator builds the host supports, plus the universal CPU
    /// build). The UI hides/disables variants *not* in this list. Falls
    /// back to every known variant when no hardware probe is cached yet,
    /// so nothing is hidden before the first probe completes.
    pub applicable_variants: Vec<String>,
}

/// Install progress for a single variant. Emitted via
/// `llama://install-progress` with a `variant` field so the UI can show
/// parallel install progress when downloading multiple builds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallProgress {
    pub stage: String,
    pub message: String,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub percent: f64,
    pub variant: String,
}

// ─── Per-variant runtime state ──────────────────────────────────────────

/// In-memory state for a single llama-server instance.
struct LlamaInstance {
    info: LlamaInstanceInfo,
    /// Live subprocess handle, if the instance is running.
    proc: Option<LlamaProcess>,
}

impl LlamaInstance {
    fn new(variant: LlamaVariant, host: &str) -> Self {
        Self {
            info: LlamaInstanceInfo {
                variant: variant.slug().to_string(),
                installed_version: None,
                latest_version: None,
                update_available: false,
                status: LlamaStatus::NotInstalled,
                pid: None,
                base_url: format!("http://{}:{}/v1", host, variant.default_port()),
                loaded_model: None,
                loaded_model_path: None,
                last_error: None,
            },
            proc: None,
        }
    }
}

// ─── Orchestrator ────────────────────────────────────────────────────────

/// Cached result of the most recent upstream "latest release" lookup.
/// One llama.cpp release serves every variant, so a single cached tag is
/// shared across all of them.
#[derive(Clone)]
struct LatestRelease {
    tag: String,
    fetched_at: Instant,
}

pub struct LlamaOrchestrator {
    app: AppHandle,
    db: SqlitePool,
    http: reqwest::Client,
    /// All variant instances, keyed by slug. Only populated for variants
    /// that have been installed or started at least once.
    instances: RwLock<HashMap<String, LlamaInstance>>,
    /// The variant the chat runner routes to. Updated on start/switch.
    active_variant: RwLock<Option<LlamaVariant>>,
    /// Cached latest upstream release tag + when it was fetched. Populated
    /// by `check_for_updates`; read by `info()` to flag available updates.
    latest_release: RwLock<Option<LatestRelease>>,
}

impl LlamaOrchestrator {
    pub fn new(app: AppHandle, db: SqlitePool, http: reqwest::Client) -> Self {
        Self {
            app,
            db,
            http,
            instances: RwLock::new(HashMap::new()),
            active_variant: RwLock::new(None),
            latest_release: RwLock::new(None),
        }
    }

    /// Read the persisted state and populate instance info for every
    /// installed variant. Called once at startup.
    pub async fn hydrate_from_db(&self) -> anyhow::Result<()> {
        let settings = Settings::load().await.unwrap_or_default();
        let host = settings.llama.host.clone();

        // Hydrate instance info for each installed variant.
        for &variant in LlamaVariant::all() {
            let runtime_name = variant.runtime_name();
            if let Some(rv) = db::runtimes::get(&self.db, &runtime_name).await? {
                let mut instances = self.instances.write().await;
                let inst = instances
                    .entry(variant.slug().to_string())
                    .or_insert_with(|| LlamaInstance::new(variant, &host));
                inst.info.installed_version = Some(rv.version);
                inst.info.status = LlamaStatus::Installed;
                inst.info.last_error = None;
            }
        }

        // Restore active variant from settings.
        let active_slug = settings.llama.active_variant.as_deref();
        let mut active = self.active_variant.write().await;
        if let Some(slug) = active_slug {
            *active = LlamaVariant::from_slug(slug);
        } else {
            // Pick the highest-priority installed variant.
            *active = self.highest_priority_installed().await;
        }

        Ok(())
    }

    /// Public snapshot of the full orchestrator state.
    pub async fn info(&self) -> OrchestratorInfo {
        let instances = self.instances.read().await;
        let active = self.active_variant.read().await;
        let active_slug = active.map_or("unset", |v| v.slug());
        let latest = self
            .latest_release
            .read()
            .await
            .as_ref()
            .map(|c| c.tag.clone());

        let mut map = HashMap::new();
        for (slug, inst) in instances.iter() {
            // Derive update availability at read-time from the shared cache
            // rather than storing it per-instance, so a single check keeps
            // every variant's flag in sync.
            let mut info = inst.info.clone();
            info.latest_version = latest.clone();
            info.update_available = match (&info.installed_version, &latest) {
                (Some(installed), Some(latest)) => installed != latest,
                _ => false,
            };
            map.insert(slug.clone(), info);
        }

        OrchestratorInfo {
            active_variant: active_slug.to_string(),
            instances: map,
            applicable_variants: Self::applicable_variant_slugs(),
        }
    }

    /// Variant slugs usable on this host. Reads the cached hardware probe
    /// and maps it through [`variant::usable_variants`]. When no probe is
    /// cached yet we return every known variant so the UI doesn't hide
    /// anything before the first probe lands.
    fn applicable_variant_slugs() -> Vec<String> {
        match crate::system::load_cached() {
            Some(specs) => variant::usable_variants(&specs)
                .into_iter()
                .map(|v| v.slug().to_string())
                .collect(),
            None => LlamaVariant::all()
                .iter()
                .map(|v| v.slug().to_string())
                .collect(),
        }
    }

    /// Fetch the latest llama.cpp release tag and cache it so `info()` can
    /// flag per-variant updates. Honours [`UPDATE_CHECK_TTL`] so repeated
    /// calls stay cheap and rate-safe; pass `force = true` for an explicit
    /// user-triggered check that bypasses the cache. Emits a status update
    /// on success so the UI refreshes its "update available" badges.
    pub async fn check_for_updates(&self, force: bool) -> anyhow::Result<String> {
        if !force {
            let cache = self.latest_release.read().await;
            if let Some(c) = cache.as_ref() {
                if c.fetched_at.elapsed() < UPDATE_CHECK_TTL {
                    return Ok(c.tag.clone());
                }
            }
        }

        let release = github::latest_release(&self.http).await?;
        let tag = release.tag_name;
        tracing::info!("llama.cpp: latest upstream release is {tag}");
        {
            let mut cache = self.latest_release.write().await;
            *cache = Some(LatestRelease {
                tag: tag.clone(),
                fetched_at: Instant::now(),
            });
        }
        self.emit_status().await;
        Ok(tag)
    }

    /// Return the base URL for the currently active variant.
    /// Used by the chat runner to route completions requests.
    pub async fn active_base_url(&self) -> Option<String> {
        let active = *self.active_variant.read().await;
        let active = active?;
        let instances = self.instances.read().await;
        instances
            .get(active.slug())
            .map(|i| i.info.base_url.clone())
    }

    // ─── install / update ────────────────────────────────────────────────

    /// Install a specific variant. If the variant is already installed, this
    /// replaces it (update semantics — downloads the latest release and
    /// overwrites the install directory).
    pub async fn install_variant(&self, variant: LlamaVariant) -> anyhow::Result<()> {
        {
            let mut instances = self.instances.write().await;
            let settings = Settings::load().await.unwrap_or_default();
            let inst = instances
                .entry(variant.slug().to_string())
                .or_insert_with(|| LlamaInstance::new(variant, &settings.llama.host));
            inst.info.status = LlamaStatus::Installing;
            inst.info.last_error = None;
        }
        // Let the UI reflect "Installing …" immediately — set the active
        // variant now so the bottom bar shows progress rather than "No
        // variant selected".
        {
            let mut active = self.active_variant.write().await;
            if active.is_none() || active.map_or(true, |a| a.priority() >= variant.priority()) {
                *active = Some(variant);
            }
        }
        self.emit_status().await;

        match install::install_variant(&self.app, &self.http, &self.db, variant).await {
            Ok(rv) => {
                tracing::info!(
                    "llama: post-install — updating instance status for `{}`",
                    variant.slug()
                );
                {
                    let mut instances = self.instances.write().await;
                    if let Some(inst) = instances.get_mut(variant.slug()) {
                        inst.info.status = LlamaStatus::Installed;
                        inst.info.installed_version = Some(rv.version);
                        inst.info.last_error = None;
                    }
                } // drop write lock before emit_status → info() tries to read
                  // If this is the first variant installed and no active
                  // variant is set, auto-activate it.
                {
                    let mut active = self.active_variant.write().await;
                    if active.is_none()
                        || active.map_or(true, |a| a.priority() >= variant.priority())
                    {
                        *active = Some(variant);
                    }
                }
                tracing::info!(
                    "llama: post-install — emitting status for `{}`",
                    variant.slug()
                );
                self.emit_status().await;
                tracing::info!("llama: install_variant(`{}`) returning Ok", variant.slug());
                Ok(())
            }
            Err(e) => {
                let msg = format!("{e:#}");
                tracing::error!("llama.cpp {} install failed: {msg}", variant.slug());
                {
                    let mut instances = self.instances.write().await;
                    if let Some(inst) = instances.get_mut(variant.slug()) {
                        inst.info.status = LlamaStatus::Error;
                        inst.info.last_error = Some(msg);
                    }
                } // drop write lock before emit_status
                self.emit_status().await;
                Err(e)
            }
        }
    }

    /// Install all variants applicable to the current hardware.
    /// Installs them sequentially to avoid overwhelming the network.
    /// Detect applicable variants for the current hardware and return
    /// them in priority order, **excluding** any that are already installed.
    async fn variants_to_install(&self) -> anyhow::Result<Vec<LlamaVariant>> {
        let specs = match crate::system::load_cached() {
            Some(s) => s,
            None => {
                let s = tokio::task::spawn_blocking(crate::system::probe)
                    .await
                    .map_err(|e| anyhow::anyhow!("hardware probe task panicked: {e}"))??;
                if let Err(e) = crate::system::save_cached(&s) {
                    tracing::warn!("system cache write failed: {e}");
                }
                s
            }
        };

        let all = variant::select_variants(&specs);
        let mut pending = Vec::new();
        for v in all {
            let runtime_name = v.runtime_name();
            if db::runtimes::get(&self.db, &runtime_name).await?.is_some() {
                tracing::info!("llama.cpp {} already installed, skipping", v.slug());
                continue;
            }
            pending.push(v);
        }
        Ok(pending)
    }

    /// Install a single variant by name. Used by the Tauri command.
    pub async fn install_variant_by_slug(self: &Arc<Self>, slug: &str) -> anyhow::Result<()> {
        let variant = LlamaVariant::from_slug(slug)
            .ok_or_else(|| anyhow::anyhow!("unknown llama.cpp variant: {slug}"))?;
        self.install_variant(variant).await
    }

    /// Install all applicable variants sequentially (convenience for
    /// manual/user-triggered installs).
    pub async fn install_applicable_variants(&self) -> anyhow::Result<()> {
        let pending = self.variants_to_install().await?;
        tracing::info!(
            "installing llama.cpp variants: {}",
            pending
                .iter()
                .map(|v| v.slug())
                .collect::<Vec<_>>()
                .join(", ")
        );
        for v in pending {
            if let Err(e) = self.install_variant(v).await {
                tracing::warn!("llama.cpp {} install failed: {e:#}", v.slug());
            }
        }
        Ok(())
    }

    pub async fn update_variant(&self, variant: LlamaVariant) -> anyhow::Result<()> {
        let release = github::latest_release(&self.http).await?;

        let installed = {
            let instances = self.instances.read().await;
            instances
                .get(variant.slug())
                .and_then(|i| i.info.installed_version.clone())
        };

        // Refresh the shared cache from this lookup so the UI's "update
        // available" badge reflects the freshest tag regardless of outcome.
        {
            let mut cache = self.latest_release.write().await;
            *cache = Some(LatestRelease {
                tag: release.tag_name.clone(),
                fetched_at: Instant::now(),
            });
        }

        if installed.as_deref() == Some(release.tag_name.as_str()) {
            tracing::info!(
                "llama.cpp {} already at latest version {}",
                variant.slug(),
                release.tag_name
            );
            self.emit_status().await;
            return Ok(());
        }

        // Windows holds an exclusive lock on a running `llama-server.exe`, so
        // the install step can't overwrite it in place. Stop the instance
        // first (the persisted loaded-model id is kept, so the user can just
        // hit Start again afterwards).
        let running = {
            let instances = self.instances.read().await;
            instances.get(variant.slug()).is_some_and(|i| {
                matches!(
                    i.info.status,
                    LlamaStatus::Running | LlamaStatus::Starting | LlamaStatus::Stopping
                )
            })
        };
        if running {
            tracing::info!(
                "llama.cpp {}: stopping running server before update",
                variant.slug()
            );
            self.stop(variant).await?;
        }

        self.install_variant(variant).await
    }

    // ─── lifecycle ───────────────────────────────────────────────────────

    /// Start (or restart) the server for a variant. If a model_id is
    /// provided, the server is started idle first, then the model is
    /// loaded via the HTTP `/models/load` API with absolute paths.
    /// If None, the server starts idle and stays idle.
    pub async fn start(
        self: &Arc<Self>,
        variant: LlamaVariant,
        model_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let model_id = model_id.map(str::trim).filter(|s| !s.is_empty());
        tracing::info!(
            target: "llama",
            "start({}): model_id={:?}",
            variant.slug(),
            model_id
        );

        // Ensure the instance entry exists.
        {
            let mut instances = self.instances.write().await;
            let settings = Settings::load().await.unwrap_or_default();
            instances
                .entry(variant.slug().to_string())
                .or_insert_with(|| LlamaInstance::new(variant, &settings.llama.host));
        }

        // Fast path: already running with the requested model.
        {
            let instances = self.instances.read().await;
            if let Some(inst) = instances.get(variant.slug()) {
                if matches!(inst.info.status, LlamaStatus::Running)
                    && inst.info.loaded_model.as_deref() == model_id.as_deref()
                {
                    return Ok(());
                }
            }
        }

        // If the server is already running but with a different model,
        // just hot-swap via the HTTP API — no need to restart.
        let already_running = {
            let instances = self.instances.read().await;
            instances
                .get(variant.slug())
                .map(|i| matches!(i.info.status, LlamaStatus::Running))
                .unwrap_or(false)
        };

        if already_running {
            // Server is running — just load the model via HTTP.
            let base_url = {
                let instances = self.instances.read().await;
                instances
                    .get(variant.slug())
                    .map(|i| i.info.base_url.clone())
            };
            if let Some(id) = model_id {
                db::llama_variant_state::set_loaded(&self.db, variant.slug(), Some(id)).await?;
                if let Some(url) = base_url {
                    return self.load_model_via_api(&url, id).await;
                }
            }
            return Ok(());
        }

        // Server is not running — kill any stale process and start fresh.
        {
            let mut instances = self.instances.write().await;
            if let Some(inst) = instances.get_mut(variant.slug()) {
                if let Some(p) = inst.proc.take() {
                    inst.info.status = LlamaStatus::Stopping;
                    drop(instances);
                    self.emit_status().await;
                    if let Err(e) = p.shutdown().await {
                        tracing::warn!(
                            "llama.cpp {} prior process shutdown failed: {e:#}",
                            variant.slug()
                        );
                    }
                    self.instances
                        .write()
                        .await
                        .get_mut(variant.slug())
                        .expect("entry exists")
                        .info
                        .status = LlamaStatus::Stopped;
                }
            }
        }

        // Record which model (if any) we're loading.
        db::llama_variant_state::set_loaded(&self.db, variant.slug(), model_id.as_deref()).await?;

        // Always start the server idle (no --model) — models are loaded
        // via the HTTP API after the server is ready.
        self.spawn_for(variant, model_id.as_deref()).await?;

        // If a model was requested, load it via the HTTP API now that
        // the server is running.
        if let Some(id) = model_id {
            let base_url = {
                let instances = self.instances.read().await;
                instances
                    .get(variant.slug())
                    .map(|i| i.info.base_url.clone())
            };
            if let Some(url) = base_url {
                // Surface load failures to the caller. The server stays up
                // (so other models remain usable / a retry is cheap), but an
                // explicit load needs to report when it didn't work.
                // auto_provision treats start() as best-effort and only logs.
                if let Err(e) = self.load_model_via_api(&url, id).await {
                    tracing::error!(
                        "llama.cpp {} failed to load model '{}' via API: {e:#}",
                        variant.slug(),
                        id
                    );
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    async fn spawn_for(
        self: &Arc<Self>,
        variant: LlamaVariant,
        _model_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let runtime_name = variant.runtime_name();
        let runtime = db::runtimes::get(&self.db, &runtime_name)
            .await?
            .ok_or_else(|| anyhow::anyhow!("llama.cpp {} is not installed yet", variant.slug()))?;
        let executable = PathBuf::from(&runtime.executable);
        if !executable.is_file() {
            return Err(anyhow::anyhow!(
                "llama-server executable missing on disk: {} (re-install required)",
                executable.display()
            ));
        }

        let settings = Settings::load().await.unwrap_or_default();
        let base_url = settings.llama.base_url_for(variant);
        let mut args = settings.llama.cli_args_for(variant);
        let working_dir = paths::llama_variant_state_dir(variant)?;

        // Per-variant environment. For the OpenVINO build this selects the
        // Intel GPU and enables on-disk model caching for faster loads.
        let env = openvino_env(variant)?;

        // The server runs in router mode (no `-m`); models are registered via
        // a `--models-preset` INI and loaded on demand through the HTTP API.
        // Regenerate the preset from the local-model catalogue so the router
        // sees everything currently installed when it boots.
        let preset_path = self.regenerate_preset().await?;
        args.push("--models-preset".into());
        args.push(preset_path.to_string_lossy().into_owned());

        tracing::info!(
            target: "llama",
            "spawn_for({}): executable={}, working_dir={}, args={}, env=[{}]",
            variant.slug(),
            executable.display(),
            working_dir.display(),
            args.join(" "),
            env.iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        {
            let mut instances = self.instances.write().await;
            let inst = instances.get_mut(variant.slug()).expect("entry exists");
            inst.info.status = LlamaStatus::Starting;
            inst.info.base_url = base_url.clone();
            inst.info.last_error = None;
            // Don't set loaded_model yet — the server starts idle.
            // It'll be set after the HTTP load succeeds.
        }
        self.emit_status().await;

        let watcher = Arc::clone(self);
        let variant_for_exit = variant;
        let on_exit = move |reason: ExitReason| {
            let v = variant_for_exit;
            tokio::spawn(async move {
                watcher.handle_child_exit(v, reason).await;
            });
        };

        tracing::info!(
            target: "llama",
            "starting llama-server {} with args: {}",
            executable.display(),
            args.join(" ")
        );
        let proc = LlamaProcess::spawn(&self.app, &executable, &args, &env, &working_dir, on_exit)
            .await
            .map_err(|e| anyhow::anyhow!("spawn llama-server {}: {e}", variant.slug()))?;

        let pid = proc.pid();
        {
            let mut instances = self.instances.write().await;
            let inst = instances.get_mut(variant.slug()).expect("entry exists");
            inst.proc = Some(proc);
            inst.info.pid = Some(pid);
        }

        if let Err(e) = health::wait_ready(&self.http, &base_url, READY_TIMEOUT).await {
            // Try to get the error excerpt from the process, but don't block on it.
            tracing::error!("llama.cpp {} readiness timeout: {e:#}", variant.slug());
            let msg = format!("waiting for llama-server {}: {e:#}", variant.slug());
            let mut instances = self.instances.write().await;
            if let Some(inst) = instances.get_mut(variant.slug()) {
                inst.info.status = LlamaStatus::Error;
                inst.info.pid = None;
                inst.info.last_error = Some(msg);
            }
            // Kill the process if it's still alive.
            if let Some(p) = instances
                .get_mut(variant.slug())
                .and_then(|i| i.proc.take())
            {
                let _ = p.shutdown().await;
            }
            self.emit_status().await;
            return Err(anyhow::anyhow!(
                "llama.cpp {} readiness timeout",
                variant.slug()
            ));
        }

        {
            let mut instances = self.instances.write().await;
            let inst = instances.get_mut(variant.slug()).expect("entry exists");
            inst.info.status = LlamaStatus::Running;
            inst.info.last_error = None;
        }
        self.emit_status().await;
        Ok(())
    }

    /// Stop a specific variant's server but keep the persisted loaded-model
    /// assignment. `auto_provision` will re-stage it on the next start.
    pub async fn stop(&self, variant: LlamaVariant) -> anyhow::Result<()> {
        self.stop_inner(variant, false).await
    }

    /// Stop the variant's server AND clear the persisted loaded-model id.
    pub async fn unload(&self, variant: LlamaVariant) -> anyhow::Result<()> {
        db::llama_variant_state::set_loaded(&self.db, variant.slug(), None).await?;
        self.stop_inner(variant, true).await
    }

    /// Unload the currently-loaded model from the router via
    /// `POST /models/unload`, leaving the server process running so other
    /// models (and a subsequent load) don't pay a full restart.
    ///
    /// If the server isn't running there's nothing to call — we just clear the
    /// persisted/instance loaded-model state.
    pub async fn unload_model(&self, variant: LlamaVariant) -> anyhow::Result<()> {
        let (base_url, loaded, running) = {
            let instances = self.instances.read().await;
            match instances.get(variant.slug()) {
                Some(i) => (
                    i.info.base_url.clone(),
                    i.info.loaded_model.clone(),
                    matches!(i.info.status, LlamaStatus::Running),
                ),
                None => return Ok(()),
            }
        };

        db::llama_variant_state::set_loaded(&self.db, variant.slug(), None).await?;

        if let (true, Some(model_id)) = (running, loaded.as_deref()) {
            let root = base_url
                .strip_suffix("/v1")
                .unwrap_or(&base_url)
                .to_string();
            let url = format!("{root}/models/unload");
            match self
                .http
                .post(&url)
                .json(&serde_json::json!({ "model": model_id }))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        tracing::warn!(
                            "/models/unload {model_id} returned {status}: {}",
                            body.trim()
                        );
                    }
                }
                Err(e) => tracing::warn!("/models/unload {model_id} failed: {e}"),
            }
        }

        {
            let mut instances = self.instances.write().await;
            if let Some(inst) = instances.get_mut(variant.slug()) {
                inst.info.loaded_model = None;
                inst.info.loaded_model_path = None;
            }
        }
        self.emit_status().await;
        Ok(())
    }

    async fn stop_inner(&self, variant: LlamaVariant, clear_loaded: bool) -> anyhow::Result<()> {
        let mut instances = self.instances.write().await;
        if let Some(inst) = instances.get_mut(variant.slug()) {
            if let Some(p) = inst.proc.take() {
                inst.info.status = LlamaStatus::Stopping;
                drop(instances);
                if let Err(e) = p.shutdown().await {
                    tracing::warn!("llama.cpp {} shutdown failed: {e:#}", variant.slug());
                }
                instances = self.instances.write().await;
                let inst = instances.get_mut(variant.slug()).expect("entry exists");
                inst.info.status = LlamaStatus::Stopped;
                inst.info.pid = None;
                if clear_loaded {
                    inst.info.loaded_model = None;
                    inst.info.loaded_model_path = None;
                }
                inst.info.last_error = None;
            } else {
                inst.info.pid = None;
                if clear_loaded {
                    inst.info.loaded_model = None;
                    inst.info.loaded_model_path = None;
                }
                inst.info.last_error = None;
            }
        }
        drop(instances);
        self.emit_status().await;
        Ok(())
    }

    async fn handle_child_exit(&self, variant: LlamaVariant, reason: ExitReason) {
        match reason {
            ExitReason::Expected => {
                // Controller already flipped status — leave it alone.
            }
            ExitReason::Crashed { code } => {
                let msg = match code {
                    Some(c) => format!(
                        "llama-server {} exited unexpectedly with code {c}",
                        variant.slug()
                    ),
                    None => format!("llama-server {} exited unexpectedly", variant.slug()),
                };
                tracing::error!("{msg}");
                let mut instances = self.instances.write().await;
                if let Some(inst) = instances.get_mut(variant.slug()) {
                    inst.info.status = LlamaStatus::Error;
                    inst.info.pid = None;
                    inst.info.last_error = Some(msg);
                }
            }
        }
        self.emit_status().await;
    }

    // ─── variant switching ───────────────────────────────────────────────

    /// Switch the active variant — the instance the chat runner routes to.
    ///
    /// Switching is now a full handover, not just a pointer flip:
    ///
    /// 1. Record + persist the new active variant.
    /// 2. Start the new variant's server, restoring whatever model it had
    ///    loaded last (or any local model) so chat can route to it right
    ///    away — no separate "start" click needed.
    /// 3. Stop every *other* variant's server so only the active one holds
    ///    a port / GPU. This is done *after* the new server is confirmed
    ///    up: if the new variant fails to start (e.g. not installed) we
    ///    leave the previously-running variant alone rather than dropping
    ///    the user to no running server at all.
    pub async fn switch_active_variant(
        self: &Arc<Self>,
        variant: LlamaVariant,
    ) -> anyhow::Result<()> {
        // ── 1. Flip + persist the active pointer ──
        {
            let mut active = self.active_variant.write().await;
            *active = Some(variant);
        }
        let mut settings = Settings::load().await.unwrap_or_default();
        settings.llama.active_variant = Some(variant.slug().to_string());
        settings.save().await?;
        self.emit_status().await;

        // ── 2. Start the newly-active variant (restore its model) ──
        // `start` is a no-op fast path when the variant is already running
        // with the same model, so re-selecting the current active variant
        // won't churn the server.
        let model_id = self.preferred_model_for(variant).await;
        tracing::info!(
            target: "llama",
            "switch_active_variant: starting {} with model {:?}",
            variant.slug(),
            model_id
        );
        self.start(variant, model_id.as_deref()).await?;

        // ── 3. Stop the other variants (only after the new one is up) ──
        let others: Vec<LlamaVariant> = {
            let instances = self.instances.read().await;
            instances
                .iter()
                .filter_map(|(slug, inst)| {
                    let v = LlamaVariant::from_slug(slug)?;
                    if v == variant {
                        return None;
                    }
                    // Only bother with variants that actually hold a process.
                    matches!(
                        inst.info.status,
                        LlamaStatus::Running | LlamaStatus::Starting | LlamaStatus::Stopping
                    )
                    .then_some(v)
                })
                .collect()
        };
        for other in others {
            tracing::info!(
                target: "llama",
                "switch_active_variant: stopping superseded variant {}",
                other.slug()
            );
            // Keep the persisted loaded-model assignment so switching back
            // later re-stages the same model.
            if let Err(e) = self.stop(other).await {
                tracing::warn!(
                    "llama switch: stop superseded variant {} failed: {e:#}",
                    other.slug()
                );
            }
        }

        Ok(())
    }

    /// Return whichever variant is currently active.
    pub async fn active_variant(&self) -> Option<LlamaVariant> {
        *self.active_variant.read().await
    }

    // ─── auto-provisioning ───────────────────────────────────────────────

    /// Auto-provision hook called from app startup. Installs all applicable
    /// variants for the detected hardware and starts the highest-priority
    /// one with its previously-loaded model (if any).
    /// Install all applicable variants for the detected hardware and start
    /// the highest-priority one. If a model was previously loaded for the
    /// variant, re-stage it; otherwise, try to find any locally-downloaded
    /// model and start with that so the server is ready immediately after
    /// install.
    ///
    /// `force = true` bypasses the `Settings::auto_provision_llama` gate,
    /// used when a dGPU is detected so the user doesn't have to opt in.
    pub async fn auto_provision(self: &Arc<Self>, force: bool) {
        let settings = match Settings::load().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("llama auto-provision: settings load failed: {e:#}");
                return;
            }
        };
        if !force && !settings.auto_provision_llama {
            tracing::info!("llama auto-provision disabled by settings");
            return;
        }

        // Figure out which variants need installing.
        let pending = match self.variants_to_install().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("llama auto-provision: failed to detect variants: {e:#}");
                return;
            }
        };

        if pending.is_empty() {
            // Everything already installed — just ensure the server is running.
            tracing::info!("llama auto-provision: all variants already installed");
            let mut active = self.active_variant.write().await;
            if active.is_none() {
                *active = self.highest_priority_installed().await;
            }
            drop(active);

            if let Some(variant) = self.active_variant().await {
                let model_id = self.preferred_model_for(variant).await;
                tracing::info!(
                    "llama auto-provision: starting {} with model {:?}",
                    variant.slug(),
                    model_id
                );
                if let Err(e) = self.start(variant, model_id.as_deref()).await {
                    tracing::warn!(
                        "llama auto-provision: start {:?} on {} failed: {e:#}",
                        model_id,
                        variant.slug()
                    );
                }
            }
            return;
        }

        // ── Phase 1: Install highest-priority variant and start it immediately ──
        let primary = pending[0];
        tracing::info!(
            "llama auto-provision: installing primary variant {}",
            primary.slug()
        );
        if let Err(e) = self.install_variant(primary).await {
            tracing::warn!(
                "llama auto-provision: primary {} install failed: {e:#}",
                primary.slug()
            );
            // Fall through — we still try to install background variants.
        } else {
            tracing::info!(
                "llama auto-provision: primary {} installed successfully",
                primary.slug()
            );
            // Primary installed — set it active and start it.
            {
                let mut active = self.active_variant.write().await;
                if active.is_none() || active.map_or(true, |a| a.priority() >= primary.priority()) {
                    *active = Some(primary);
                }
            }

            let model_id = self.preferred_model_for(primary).await;
            tracing::info!(
                "llama auto-provision: starting primary {} with model {:?}",
                primary.slug(),
                model_id
            );
            if let Err(e) = self.start(primary, model_id.as_deref()).await {
                tracing::warn!(
                    "llama auto-provision: start {:?} on {} failed: {e:#}",
                    model_id,
                    primary.slug()
                );
            }
        }

        // ── Phase 2: Install remaining variants in the background ──
        if pending.len() > 1 {
            let remaining: Vec<LlamaVariant> = pending[1..].to_vec();
            let bg_self = Arc::clone(self);
            tokio::spawn(async move {
                for v in remaining {
                    tracing::info!("llama auto-provision: background-installing {}", v.slug());
                    if let Err(e) = bg_self.install_variant(v).await {
                        tracing::warn!(
                            "llama auto-provision: background {} install failed: {e:#}",
                            v.slug()
                        );
                    }
                }
                tracing::info!("llama auto-provision: background installs complete");
            });
        }
    }

    /// Resolve the best model to load for a given variant:
    /// 1) Previously-loaded model for this variant (re-stage on restart)
    /// 2) Any locally-downloaded model with a .gguf on disk
    /// 3) None (start idle)
    async fn preferred_model_for(&self, variant: LlamaVariant) -> Option<String> {
        let prev = db::llama_variant_state::get_loaded(&self.db, variant.slug())
            .await
            .ok()
            .flatten();
        match prev {
            Some(id) => Some(id),
            None => self.first_local_model().await,
        }
    }

    // ─── helpers ─────────────────────────────────────────────────────────

    /// Return the highest-priority installed variant, or None.
    async fn highest_priority_installed(&self) -> Option<LlamaVariant> {
        let instances = self.instances.read().await;
        let mut best: Option<(LlamaVariant, u8)> = None;
        for (variant_slug, inst) in instances.iter() {
            if matches!(
                inst.info.status,
                LlamaStatus::Installed | LlamaStatus::Stopped | LlamaStatus::Running
            ) {
                if let Some(v) = LlamaVariant::from_slug(&variant_slug) {
                    match best {
                        None => best = Some((v, v.priority())),
                        Some((_, p)) if v.priority() < p => best = Some((v, v.priority())),
                        _ => {}
                    }
                }
            }
        }
        best.map(|(v, _)| v)
    }

    /// Broadcast the current orchestrator state to the frontend.
    async fn emit_status(&self) {
        let info = self.info().await;
        tracing::debug!("llama: emit_status — sending LlamaStatus event");
        let _ = self.app.emit(events::LLAMA_STATUS, &info);
    }

    /// Find the first locally-downloaded model that has a .gguf file on disk.
    /// Returns the model id, or None if no model is available.
    async fn first_local_model(&self) -> Option<String> {
        let rows = sqlx::query("SELECT id, path FROM local_models")
            .fetch_all(&self.db)
            .await
            .ok()?;
        for row in rows {
            let id: String = row.try_get("id").ok()?;
            let path: String = row.try_get("path").ok()?;
            let model_path = PathBuf::from(&path);
            // Use tokio::fs to avoid blocking the async runtime.
            if tokio::fs::metadata(&model_path)
                .await
                .ok()
                .map(|m| m.is_dir())
                .unwrap_or(false)
            {
                if let Ok(mut entries) = tokio::fs::read_dir(&model_path).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        if entry
                            .path()
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                        {
                            return Some(id);
                        }
                    }
                }
            } else if model_path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                return Some(id);
            }
        }
        None
    }

    // ─── HTTP model loading ───────────────────────────────────────────

    /// Scan a downloaded model directory for GGUF files and return the
    /// paths needed for the `/models/load` API call.
    ///
    /// Returns `(model_gguf, mmproj, model_draft)` where `model_gguf` is
    /// the main model file, `mmproj` is the multimodal projector (if
    /// present), and `model_draft` is the speculative-decoding draft model
    /// (MTP, if present).  All paths are absolute filesystem paths.
    ///
    /// Prefers the pre-computed GGUF classification in `.zero_manifest.json`
    /// when it exists; falls back to direct directory scanning for legacy
    /// installs without the classification fields.
    async fn scan_model_files(
        &self,
        model_id: &str,
    ) -> anyhow::Result<(String, Option<String>, Option<String>)> {
        let row = sqlx::query("SELECT path, metadata_json FROM local_models WHERE id = ?")
            .bind(model_id)
            .fetch_optional(&self.db)
            .await?;
        let (model_dir, metadata_json): (String, Option<String>) = match row {
            Some(r) => (r.get("path"), r.try_get("metadata_json").ok()),
            None => {
                return Err(anyhow::anyhow!(
                    "model `{model_id}` is not installed locally"
                ));
            }
        };

        let dir = PathBuf::from(&model_dir);
        if !dir.is_dir() {
            return Err(anyhow::anyhow!(
                "model directory does not exist: {}",
                dir.display()
            ));
        }

        resolve_model_files(&dir, metadata_json.as_deref()).ok_or_else(|| {
            anyhow::anyhow!("no .gguf file found in model directory: {}", dir.display())
        })
    }

    /// Load a model on a running llama-server **router** via the
    /// `POST /models/load` HTTP API.
    ///
    /// In router mode the model is addressed by id (the local-model id, which
    /// is also the preset section id) — not by file path. We make sure the
    /// router knows about the model first (regenerate the preset, then ask it
    /// to reload) so a just-downloaded model is recognised without a restart.
    ///
    /// `base_url` is the variant's base URL (e.g. `http://127.0.0.1:8081/v1`).
    /// `model_id` is the local-model identifier (== router model id).
    async fn load_model_via_api(&self, base_url: &str, model_id: &str) -> anyhow::Result<()> {
        let root = base_url.strip_suffix("/v1").unwrap_or(base_url).to_string();

        // The model may have been downloaded after the router started — refresh
        // the preset on disk and have the router re-read its model sources.
        self.regenerate_preset().await?;
        health::reload_router_models(&self.http, &root).await;

        // Load the primary (chat) model and wait for it to report ready.
        self.post_models_load(&root, model_id).await?;

        // Resolve the main file path for display (best-effort).
        let main_path = self
            .scan_model_files(model_id)
            .await
            .ok()
            .map(|(m, _, _)| m);

        // Update the instance state to reflect the loaded model.
        {
            let mut instances = self.instances.write().await;
            for inst in instances.values_mut() {
                if inst.info.base_url == base_url {
                    inst.info.loaded_model = Some(model_id.to_string());
                    inst.info.loaded_model_path = main_path.clone();
                }
            }
        }
        self.emit_status().await;

        Ok(())
    }

    /// Issue a single `POST /models/load` and wait until the router reports the
    /// model resident. `root` is the server root **without** the `/v1` suffix.
    /// Assumes the preset already lists `model_id` (the caller regenerates it).
    /// Treats an "already running" 400 as success since router mode keeps
    /// multiple models resident at once.
    async fn post_models_load(&self, root: &str, model_id: &str) -> anyhow::Result<()> {
        tracing::info!("loading model '{}' via {}/models/load", model_id, root);

        let load_url = format!("{root}/models/load");
        let resp = self
            .http
            .post(&load_url)
            .json(&serde_json::json!({ "model": model_id }))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("POST /models/load failed: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            // In router mode multiple models can stay resident at once: loading
            // a second model does not evict the first. Re-selecting a model the
            // router still has running comes back as `400 model is already
            // running` — which is precisely the state we want, so treat it as a
            // successful (no-op) load and fall through to the readiness check.
            let already_loaded = status == reqwest::StatusCode::BAD_REQUEST && {
                let b = body.to_ascii_lowercase();
                b.contains("already running") || b.contains("already loaded")
            };
            if already_loaded {
                tracing::info!(
                    "model '{}' is already resident on the router; reusing it",
                    model_id
                );
            } else {
                return Err(anyhow::anyhow!(
                    "/models/load returned {}: {}",
                    status,
                    body.trim()
                ));
            }
        }

        // The load is accepted asynchronously; wait until it actually reports
        // loaded so a load-time crash surfaces as an error (and the UI spinner
        // reflects real readiness).
        health::wait_model_loaded(&self.http, root, model_id, LOAD_TIMEOUT).await?;

        tracing::info!("model '{}' loaded successfully", model_id);
        Ok(())
    }

    /// Resolve the on-disk main GGUF for a local model id (or HF id). Used
    /// to locate the OuteTTS weights and the WavTokenizer vocoder that
    /// `llama-tts` needs. Returns `None` when the model isn't downloaded or
    /// has no resolvable main weight file.
    async fn resolve_local_model_main(&self, id: &str) -> Option<PathBuf> {
        let row =
            sqlx::query("SELECT path, metadata_json FROM local_models WHERE id = ? OR hf_id = ?")
                .bind(id)
                .bind(id)
                .fetch_optional(&self.db)
                .await
                .ok()??;
        let path: String = row.try_get("path").ok()?;
        let metadata_json: Option<String> = row.try_get("metadata_json").ok();
        let dir = PathBuf::from(path);
        let (main, _mmproj, _draft) = resolve_model_files(&dir, metadata_json.as_deref())?;
        Some(PathBuf::from(main))
    }

    /// Locate the `llama-tts` CLI that ships next to the active variant's
    /// `llama-server` binary. It rides in the same release zip, so it's a
    /// sibling on disk — no separate download.
    async fn llama_tts_exe(&self) -> anyhow::Result<PathBuf> {
        let active = self
            .active_variant()
            .await
            .ok_or_else(|| anyhow::anyhow!("no active llama.cpp variant"))?;
        let runtime = db::runtimes::get(&self.db, &active.runtime_name())
            .await?
            .ok_or_else(|| anyhow::anyhow!("llama.cpp {} is not installed", active.slug()))?;
        let server = PathBuf::from(&runtime.executable);
        let dir = server
            .parent()
            .ok_or_else(|| anyhow::anyhow!("llama-server path has no parent dir"))?;
        let name = if cfg!(windows) {
            "llama-tts.exe"
        } else {
            "llama-tts"
        };
        let exe = dir.join(name);
        if !exe.is_file() {
            return Err(anyhow::anyhow!(
                "llama-tts not found next to llama-server at {} \
                 (update or re-install llama.cpp to get the TTS tool)",
                exe.display()
            ));
        }
        Ok(exe)
    }

    /// Synthesize `text` to WAV bytes using the bundled `llama-tts` CLI with
    /// the configured OuteTTS model + WavTokenizer vocoder, offloaded to the
    /// GPU per the user's llama.cpp layer setting. Returns the raw WAV the
    /// renderer plays back.
    pub async fn tts_synthesize(&self, text: &str) -> anyhow::Result<Vec<u8>> {
        let settings = Settings::load().await.unwrap_or_default();
        if !settings.audio.enabled {
            return Err(anyhow::anyhow!("audio is disabled"));
        }
        let oute_id = settings
            .audio
            .tts_model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("no text-to-speech model configured"))?;

        let exe = self.llama_tts_exe().await?;
        let oute = self
            .resolve_local_model_main(oute_id)
            .await
            .ok_or_else(|| {
                anyhow::anyhow!("text-to-speech model '{oute_id}' is not downloaded yet")
            })?;
        let vocoder = self
            .resolve_local_model_main(tts::WAVTOKENIZER_HF_ID)
            .await
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "the WavTokenizer vocoder ('{}') is not downloaded yet",
                    tts::WAVTOKENIZER_HF_ID
                )
            })?;

        tts::synthesize(&exe, &oute, &vocoder, text, settings.llama.n_gpu_layers).await
    }

    /// Build the router preset model list from every locally-installed GGUF
    /// model. Non-GGUF (e.g. legacy OpenVINO-IR) installs resolve to no main
    /// weight file and are skipped.
    async fn build_preset_models(&self) -> Vec<preset::PresetModel> {
        let rows = sqlx::query("SELECT id, path, metadata_json FROM local_models")
            .fetch_all(&self.db)
            .await
            .unwrap_or_default();

        let mut out = Vec::new();
        for row in rows {
            let Ok(id) = row.try_get::<String, _>("id") else {
                continue;
            };
            let Ok(path) = row.try_get::<String, _>("path") else {
                continue;
            };
            let metadata_json: Option<String> = row.try_get("metadata_json").ok();
            let dir = PathBuf::from(&path);
            if !dir.is_dir() {
                continue;
            }
            let Some((model, mmproj, draft)) = resolve_model_files(&dir, metadata_json.as_deref())
            else {
                continue;
            };
            let draft_is_mtp = draft
                .as_deref()
                .map(|d| d.to_lowercase().contains("mtp"))
                .unwrap_or(false);
            out.push(preset::PresetModel {
                id,
                model,
                mmproj,
                draft,
                draft_is_mtp,
            });
        }
        out
    }

    /// Regenerate the router preset INI on disk (atomically) and return its
    /// path. This is the single source the router reads via `--models-preset`
    /// and re-reads on `GET /models?reload=1`.
    async fn regenerate_preset(&self) -> anyhow::Result<PathBuf> {
        let models = self.build_preset_models().await;
        // Experimental MTP / speculative-decoding wiring is opt-in. Read the
        // current preference fresh so toggling it in Settings takes effect on
        // the next model (re)load without an app restart.
        let mtp_enabled = crate::settings::Settings::load()
            .await
            .map(|s| s.llama.mtp_enabled)
            .unwrap_or(false);
        let content = preset::render_preset(&models, mtp_enabled);
        let path = paths::llama_models_preset()?;
        let tmp = path.with_extension("ini.tmp");
        tokio::fs::write(&tmp, content.as_bytes())
            .await
            .map_err(|e| anyhow::anyhow!("write preset {}: {e}", tmp.display()))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| anyhow::anyhow!("persist preset {}: {e}", path.display()))?;
        tracing::info!(
            "llama: regenerated router preset ({} model(s)) at {}",
            models.len(),
            path.display()
        );
        Ok(path)
    }
}

// ── helpers for scan_model_files ─────────────────────────────────────────

/// Build the per-variant environment for the spawned `llama-server`.
///
/// Only the OpenVINO build needs anything today. Its backend is configured
/// entirely through `GGML_OPENVINO_*` environment variables (see the upstream
/// docs/backend/OPENVINO.md), so we set:
///
/// * `GGML_OPENVINO_DEVICE=GPU` — run on the Intel GPU (integrated *or*
///   discrete Arc). We only ship the OpenVINO build to hosts that have an
///   Intel GPU, and OpenVINO transparently falls back to CPU if no usable GPU
///   is found, so defaulting to `GPU` is safe. A user who exports their own
///   `GGML_OPENVINO_DEVICE` (e.g. `CPU`, `NPU`, `GPU.1`) keeps that override.
/// * `GGML_OPENVINO_CACHE_DIR` — persist compiled device graphs on disk so the
///   expensive first-token graph compilation is reused across restarts, giving
///   much faster subsequent loads.
///
/// `GGML_OPENVINO_STATEFUL_EXECUTION` is deliberately left off: upstream only
/// validates it for llama-simple / llama-cli / llama-bench / llama-run, and
/// explicitly notes that `llama-server` (what we run) is not yet supported.
fn openvino_env(variant: LlamaVariant) -> anyhow::Result<Vec<(String, String)>> {
    if variant != LlamaVariant::OpenVino {
        return Ok(Vec::new());
    }

    let device = std::env::var("GGML_OPENVINO_DEVICE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "GPU".to_string());

    let cache_dir = paths::llama_openvino_cache_dir()?;

    Ok(vec![
        ("GGML_OPENVINO_DEVICE".to_string(), device),
        (
            "GGML_OPENVINO_CACHE_DIR".to_string(),
            cache_dir.to_string_lossy().into_owned(),
        ),
    ])
}

/// Resolve the on-disk `(model, mmproj, draft)` for a model directory.
///
/// Prefers the pre-computed GGUF classification in the manifest (written at
/// download time); falls back to a directory scan for legacy installs. All
/// returned paths are absolute. Returns `None` when no main `.gguf` weight
/// file can be found (e.g. a non-GGUF install).
///
/// `metadata_json` is the `local_models.metadata_json` blob; its `bestQuant`
/// field is used to pick the matching draft when several are present.
fn resolve_model_files(
    dir: &Path,
    metadata_json: Option<&str>,
) -> Option<(String, Option<String>, Option<String>)> {
    let resolve = |fname: &str| -> String { dir.join(fname).to_string_lossy().into_owned() };

    let preferred_quant = || {
        metadata_json
            .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
            .and_then(|v| v.get("bestQuant")?.as_str().map(str::to_lowercase))
    };

    // Try the pre-computed manifest catalogue first.
    if let Some(mf) = crate::hf::read_manifest_sync(dir) {
        if let Some(ref model_fname) = mf.model {
            let main = resolve(model_fname);
            let mmproj = mf.mmproj.as_ref().map(|f| resolve(f));
            let draft = if mf.drafts.is_empty() {
                None
            } else {
                let q = preferred_quant();
                let best = if let Some(ref q) = q {
                    mf.drafts
                        .iter()
                        .find(|f| f.to_lowercase().contains(q.as_str()))
                        .or_else(|| mf.drafts.first())
                } else {
                    mf.drafts.first()
                };
                best.map(|f| resolve(f))
            };
            return Some((main, mmproj, draft));
        }
        // Manifest exists but has no model entry; fall through to scan.
    }

    // Fallback: directory scan (legacy installs).
    let mut main_gguf: Option<String> = None;
    let mut main_size: u64 = 0;
    let mut mmproj_f16: Option<String> = None;
    let mut mmproj_fallback: Option<String> = None;
    let mut model_draft: Option<String> = None;
    let q = preferred_quant();

    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        let lower = fname.to_lowercase();
        let abs_path = path.to_string_lossy().into_owned();

        if lower.contains("mmproj") {
            if lower.contains("f16") {
                mmproj_f16 = Some(abs_path);
            } else if mmproj_fallback.is_none() {
                mmproj_fallback = Some(abs_path);
            }
        } else if lower.contains("mtp") || lower.contains("draft") {
            let is_better = match (&model_draft, &q) {
                (None, _) => true,
                (_, None) => false,
                (Some(existing), Some(q)) => {
                    lower.contains(q.as_str()) && !existing.to_lowercase().contains(q.as_str())
                }
            };
            if is_better {
                model_draft = Some(abs_path);
            }
        } else {
            let sz = path.metadata().map(|m| m.len()).unwrap_or(0);
            if sz > main_size {
                main_size = sz;
                main_gguf = Some(abs_path);
            }
        }
    }

    if model_draft.is_none() {
        let _ = scan_draft_recursive(dir, &mut model_draft);
    }

    let main = main_gguf?;
    Some((main, mmproj_f16.or(mmproj_fallback), model_draft))
}

/// Recursively walk `dir` looking for draft (mtp/draft) .gguf files.
/// Stores the first match found and stops as an absolute path.
fn scan_draft_recursive(dir: &Path, best: &mut Option<String>) -> anyhow::Result<()> {
    if best.is_some() {
        return Ok(());
    }
    let entries =
        std::fs::read_dir(dir).map_err(|e| anyhow::anyhow!("read dir {}: {e}", dir.display()))?;
    for entry in entries {
        if best.is_some() {
            return Ok(());
        }
        let entry = entry?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower = fname.to_lowercase();

        if path.is_dir() {
            scan_draft_recursive(&path, best)?;
        } else if path.is_file() && (lower.contains("mtp") || lower.contains("draft")) {
            *best = Some(path.to_string_lossy().into_owned());
            return Ok(());
        }
    }
    Ok(())
}

// ─── Model path lookup (unused since switch to HTTP /models/load) ───

/// Resolve the on-disk `.gguf` file for a model id. The model directory
/// is `$MODELS_DIR/{model_id}/` and we pick the first `.gguf` file found.
#[allow(dead_code)]
async fn lookup_model_gguf(db: &SqlitePool, model_id: &str) -> anyhow::Result<Option<PathBuf>> {
    let row = sqlx::query("SELECT path FROM local_models WHERE id = ?")
        .bind(model_id)
        .fetch_optional(db)
        .await?;

    let model_dir: Option<String> = row.and_then(|r| r.try_get("path").ok());
    let Some(model_dir) = model_dir else {
        return Ok(None);
    };

    let model_path = PathBuf::from(&model_dir);
    if !model_path.is_dir() {
        // Maybe it's already a direct file path.
        if model_path.exists() && model_path.extension().is_some_and(|e| e == "gguf") {
            return Ok(Some(model_path));
        }
        return Ok(None);
    }

    // Scan the model directory for a .gguf file.
    let mut entries = std::fs::read_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("read model dir {}: {e}", model_path.display()))?;
    while let Some(entry) = entries.next() {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            tracing::info!(
                "lookup_model_gguf: resolved model '{}' -> {}",
                model_id,
                path.display()
            );
            return Ok(Some(path));
        }
    }
    tracing::warn!(
        "lookup_model_gguf: no .gguf found in {} for model '{}'",
        model_path.display(),
        model_id
    );
    Ok(None)
}
