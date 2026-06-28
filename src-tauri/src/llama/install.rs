//! llama.cpp install pipeline: fetch release → pick variant → download
//! → extract → record in DB.
//!
//! Each variant gets its own subdirectory under `runtimes/llama.cpp/`:
//! - `runtimes/llama.cpp/cuda/`
//! - `runtimes/llama.cpp/openvino/`
//! - `runtimes/llama.cpp/hip-radeon/`
//! - `runtimes/llama.cpp/cpu/`
//!
//! The install function takes a specific [`LlamaVariant`] so the
//! orchestrator can install multiple variants for the same host.

use crate::db;
use crate::db::runtimes::RuntimeVersion;
use crate::events;
use crate::llama::github;
use crate::llama::variant::LlamaVariant;
use crate::paths;
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use futures_util::StreamExt;
use serde::Serialize;
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallStage {
    FetchRelease,
    Download,
    Extract,
    Verify,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallProgress {
    pub stage: InstallStage,
    pub message: String,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub percent: f64,
    /// Variant slug the install pipeline is targeting. Surfaced on every
    /// progress frame so the UI can show "Installing llama.cpp (cuda)"
    /// without having to round-trip a separate query.
    pub variant: String,
}

fn emit(app: &AppHandle, p: InstallProgress) {
    let _ = app.emit(events::LLAMA_INSTALL_PROGRESS, p);
}

/// Download the latest llama.cpp release for the given variant, extract
/// it into the variant-specific directory, and record the install in
/// `runtime_versions`.
pub async fn install_variant(
    app: &AppHandle,
    http: &reqwest::Client,
    db: &SqlitePool,
    variant: LlamaVariant,
) -> Result<RuntimeVersion> {
    tracing::info!("llama.cpp install: installing variant `{}`", variant.slug());

    // ─── 1. Resolve latest release ──────────────────────────────────
    emit(
        app,
        InstallProgress {
            stage: InstallStage::FetchRelease,
            message: format!(
                "fetching latest llama.cpp release for `{}`…",
                variant.slug()
            ),
            bytes_done: 0,
            bytes_total: None,
            percent: 0.0,
            variant: variant.slug().to_string(),
        },
    );
    let release = github::latest_release(http)
        .await
        .context("fetch latest release")?;

    let asset = release
        .find_asset(|name| variant.matches_asset(name))
        .ok_or_else(|| {
            anyhow!(
                "no matching asset for variant `{}` in release {} (available: {})",
                variant.slug(),
                release.tag_name,
                release
                    .assets
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?
        .clone();

    // The CUDA build needs its companion redistributable.
    let companion = release
        .find_asset(|name| variant.matches_companion_asset(name))
        .cloned();
    if companion.is_none()
        && variant.matches_companion_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")
    {
        tracing::warn!(
            "llama.cpp install: variant `{}` expects a companion redistributable but none was found in release {}",
            variant.slug(),
            release.tag_name
        );
    }

    // ─── 2. Stream-download the zip into a scratch file ─────────────
    let install_dir = paths::llama_variant_dir(variant)?;
    let scratch_dir = std::env::temp_dir().join("zero-llama-install");
    fs::create_dir_all(&scratch_dir).await?;
    let zip_path = scratch_dir.join(&asset.name);

    emit(
        app,
        InstallProgress {
            stage: InstallStage::Download,
            message: format!(
                "downloading {} ({} MB)",
                asset.name,
                asset.size / 1024 / 1024
            ),
            bytes_done: 0,
            bytes_total: Some(asset.size),
            percent: 0.0,
            variant: variant.slug().to_string(),
        },
    );
    download_with_progress(app, http, &asset, &zip_path, variant).await?;

    // ─── 3. Extract into the (wiped) variant-specific install root ─
    emit(
        app,
        InstallProgress {
            stage: InstallStage::Extract,
            message: "extracting archive…".into(),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 0.95,
            variant: variant.slug().to_string(),
        },
    );
    let extract_start = Instant::now();
    let extract_target = install_dir.clone();
    let zip_for_extract = zip_path.clone();
    tokio::task::spawn_blocking(move || {
        wipe_install_dir(&extract_target)?;
        extract_zip(&zip_for_extract, &extract_target)
    })
    .await
    .map_err(|e| anyhow!("extract task panicked: {e}"))??;
    tracing::info!(
        "llama.cpp install: extracted {} in {:.1}s",
        asset.name,
        extract_start.elapsed().as_secs_f64()
    );

    let _ = fs::remove_file(&zip_path).await;

    // ─── 3b. Companion redistributable (CUDA only today) ───────────
    if let Some(companion) = companion {
        let companion_path = scratch_dir.join(&companion.name);
        emit(
            app,
            InstallProgress {
                stage: InstallStage::Download,
                message: format!(
                    "downloading companion {} ({} MB)",
                    companion.name,
                    companion.size / 1024 / 1024
                ),
                bytes_done: 0,
                bytes_total: Some(companion.size),
                percent: 0.0,
                variant: variant.slug().to_string(),
            },
        );
        download_with_progress(app, http, &companion, &companion_path, variant).await?;

        emit(
            app,
            InstallProgress {
                stage: InstallStage::Extract,
                message: format!("extracting companion {}…", companion.name),
                bytes_done: companion.size,
                bytes_total: Some(companion.size),
                percent: 0.97,
                variant: variant.slug().to_string(),
            },
        );
        let companion_extract_start = Instant::now();
        let extract_target = install_dir.clone();
        let zip_for_extract = companion_path.clone();
        tokio::task::spawn_blocking(move || extract_zip(&zip_for_extract, &extract_target))
            .await
            .map_err(|e| anyhow!("companion extract task panicked: {e}"))??;
        tracing::info!(
            "llama.cpp install: extracted companion {} in {:.1}s",
            companion.name,
            companion_extract_start.elapsed().as_secs_f64()
        );
        let _ = fs::remove_file(&companion_path).await;
    }

    let _ = fs::remove_dir(&scratch_dir).await;

    // ─── 4. Verify llama-server.exe is present ──────────────────────
    emit(
        app,
        InstallProgress {
            stage: InstallStage::Verify,
            message: "locating llama-server executable…".into(),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 0.98,
            variant: variant.slug().to_string(),
        },
    );
    let exe = find_executable(&install_dir)
        .ok_or_else(|| anyhow!("llama-server executable not found under {:?}", install_dir))?;

    // ─── 5. Record in DB ────────────────────────────────────────────
    let rv = RuntimeVersion {
        name: variant.runtime_name(),
        version: release.tag_name.clone(),
        install_dir: install_dir.to_string_lossy().into_owned(),
        executable: exe.to_string_lossy().into_owned(),
        installed_at: Utc::now().to_rfc3339(),
        source_url: Some(asset.browser_download_url.clone()),
        metadata: Some(serde_json::json!({
            "asset": asset.name,
            "asset_size": asset.size,
            "release_name": release.name,
            "published_at": release.published_at,
            "variant": variant.slug(),
        })),
    };
    db::runtimes::upsert(db, &rv).await?;

    emit(
        app,
        InstallProgress {
            stage: InstallStage::Done,
            message: format!("installed llama.cpp {} ({})", rv.version, variant.slug()),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 1.0,
            variant: variant.slug().to_string(),
        },
    );
    tracing::info!(
        "llama.cpp install: completed variant `{}` ({})",
        variant.slug(),
        rv.version
    );
    Ok(rv)
}

async fn download_with_progress(
    app: &AppHandle,
    http: &reqwest::Client,
    asset: &github::Asset,
    dest: &PathBuf,
    variant: LlamaVariant,
) -> Result<()> {
    let start = Instant::now();
    let resp = http
        .get(&asset.browser_download_url)
        .send()
        .await
        .context("download request")?;

    let total = resp.content_length().unwrap_or(asset.size);
    let mut file = fs::File::create(dest)
        .await
        .with_context(|| format!("create {dest:?}"))?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("download chunk")?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("write chunk to {dest:?}"))?;
        downloaded += chunk.len() as u64;
        let pct = if total > 0 {
            downloaded as f64 / total as f64
        } else {
            0.0
        };
        emit(
            app,
            InstallProgress {
                stage: InstallStage::Download,
                message: format!("downloading {}…", asset.name),
                bytes_done: downloaded,
                bytes_total: Some(total),
                percent: pct,
                variant: variant.slug().to_string(),
            },
        );
    }
    file.flush()
        .await
        .with_context(|| format!("flush {dest:?}"))?;
    tracing::info!(
        "downloaded {} in {:.1}s ({:.1} MB/s)",
        asset.name,
        start.elapsed().as_secs_f64(),
        (downloaded as f64 / 1_048_576.0) / start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(zip_path).with_context(|| format!("open {zip_path:?}"))?;
    let mut archive = zip::ZipArchive::new(file).with_context(|| format!("unzip {zip_path:?}"))?;

    let prefix = detect_common_prefix(&archive);
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("read entry {i} from {zip_path:?}"))?;
        let entry_path = entry
            .enclosed_name()
            .with_context(|| format!("entry {i} has unsafe path"))?;

        // Strip the common prefix (the top-level build directory) so
        // files land directly in the variant's install directory.
        let relative = match &prefix {
            Some(p) => entry_path.strip_prefix(p).unwrap_or(&entry_path),
            None => &entry_path,
        };

        let out = dest.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&out).with_context(|| format!("create dir {out:?}"))?;
        } else {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create parent dir for {out:?}"))?;
            }
            let mut out_file =
                std::fs::File::create(&out).with_context(|| format!("create {out:?}"))?;
            std::io::copy(&mut entry, &mut out_file)
                .with_context(|| format!("extract {} to {out:?}", entry.name()))?;
        }
    }
    Ok(())
}

fn wipe_install_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {dir:?}"))?;
        return Ok(());
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {dir:?}"))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_name() == ".cache" {
            continue;
        }
        let ft = entry.file_type()?;
        let res = if ft.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if let Err(e) = res {
            tracing::warn!(
                "failed to remove {} during llama.cpp install wipe: {e}",
                path.display()
            );
        }
    }
    Ok(())
}

fn detect_common_prefix(archive: &zip::ZipArchive<std::fs::File>) -> Option<PathBuf> {
    let mut prefix: Option<PathBuf> = None;
    for name in archive.file_names() {
        if name.ends_with('/') {
            continue;
        }
        let rel = Path::new(name);
        let mut comps = rel.components();
        let Some(first) = comps.next() else {
            return None;
        };
        if comps.next().is_none() {
            return None;
        }
        let first_buf: PathBuf = first.as_os_str().into();
        match &prefix {
            None => prefix = Some(first_buf),
            Some(p) if p == &first_buf => {}
            Some(_) => return None,
        }
    }
    prefix
}

/// Walk `root` looking for `llama-server.exe` (Windows) or `llama-server`
/// (other). Checks the most likely locations first; falls back to a
/// shallow walk (max 4 deep — llama.cpp zips sometimes nest the binary
/// under `build/bin/`).
pub fn find_executable(root: &Path) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let candidates = [
        root.join(exe_name),
        root.join("bin").join(exe_name),
        root.join("build").join("bin").join(exe_name),
    ];
    for c in candidates {
        if c.exists() {
            return Some(c);
        }
    }
    walk_for(root, exe_name, 4)
}

fn walk_for(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().and_then(|s| s.to_str()) == Some(name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = walk_for(&path, name, depth - 1) {
                return Some(found);
            }
        }
    }
    None
}
