//! whisper.cpp runtime install: fetch release → pick GPU/CPU asset →
//! download → extract → record in `runtime_versions`.
//!
//! A single install serves the host (unlike llama.cpp's per-variant tree):
//! the only choice is the cuBLAS build on NVIDIA discrete GPUs vs the CPU
//! build everywhere else. Re-installing wipes and replaces the directory.

use crate::db;
use crate::db::runtimes::RuntimeVersion;
use crate::paths;
use crate::system::Specs;
use crate::whisper::github::{self, Asset};
use crate::whisper::{emit, is_nvidia_discrete, WhisperProgress, WhisperStage, RUNTIME_NAME};
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use futures_util::StreamExt;
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use tauri::AppHandle;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Choose the release asset filename predicate for this host. NVIDIA
/// discrete GPUs get the GPU-accelerated cuBLAS 12.4 build; everything else
/// gets the plain CPU x64 build (no Vulkan/HIP Windows build exists).
fn pick_asset(specs: &Specs, release: &github::Release) -> Option<Asset> {
    if is_nvidia_discrete(specs) {
        if let Some(a) = release.find_asset(|n| {
            let n = n.to_lowercase();
            n.contains("cublas-12.4") && n.contains("x64") && n.ends_with(".zip")
        }) {
            return Some(a.clone());
        }
        tracing::warn!("whisper: no cuBLAS 12.4 asset found; falling back to CPU build");
    }
    // CPU build. Match the plain x64 zip, not the blas/Win32 variants.
    release
        .find_asset(|n| n.eq_ignore_ascii_case("whisper-bin-x64.zip"))
        .cloned()
}

/// Download + install the whisper.cpp runtime for this host.
pub async fn install(
    app: &AppHandle,
    http: &reqwest::Client,
    db: &SqlitePool,
    specs: &Specs,
) -> Result<RuntimeVersion> {
    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::FetchRelease,
            message: "fetching latest whisper.cpp release…".into(),
            bytes_done: 0,
            bytes_total: None,
            percent: 0.0,
            target: "runtime".into(),
        },
    );
    let release = github::latest_release(http)
        .await
        .context("fetch latest whisper.cpp release")?;

    let asset = pick_asset(specs, &release).ok_or_else(|| {
        anyhow!(
            "no matching whisper.cpp asset in release {} (available: {})",
            release.tag_name,
            release
                .assets
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let install_dir = paths::whisper_dir()?;
    let scratch_dir = std::env::temp_dir().join("zero-whisper-install");
    fs::create_dir_all(&scratch_dir).await?;
    let zip_path = scratch_dir.join(&asset.name);

    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::Download,
            message: format!(
                "downloading {} ({} MB)",
                asset.name,
                asset.size / 1024 / 1024
            ),
            bytes_done: 0,
            bytes_total: Some(asset.size),
            percent: 0.0,
            target: "runtime".into(),
        },
    );
    download_with_progress(app, http, &asset, &zip_path, "runtime").await?;

    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::Extract,
            message: "extracting archive…".into(),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 0.95,
            target: "runtime".into(),
        },
    );
    let extract_target = install_dir.clone();
    let zip_for_extract = zip_path.clone();
    tokio::task::spawn_blocking(move || {
        wipe_install_dir(&extract_target)?;
        extract_zip(&zip_for_extract, &extract_target)
    })
    .await
    .map_err(|e| anyhow!("extract task panicked: {e}"))??;
    let _ = fs::remove_file(&zip_path).await;
    let _ = fs::remove_dir(&scratch_dir).await;

    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::Verify,
            message: "locating whisper-cli…".into(),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 0.98,
            target: "runtime".into(),
        },
    );
    let exe = find_executable(&install_dir)
        .ok_or_else(|| anyhow!("whisper-cli not found under {}", install_dir.display()))?;

    let rv = RuntimeVersion {
        name: RUNTIME_NAME.to_string(),
        version: release.tag_name.clone(),
        install_dir: install_dir.to_string_lossy().into_owned(),
        executable: exe.to_string_lossy().into_owned(),
        installed_at: Utc::now().to_rfc3339(),
        source_url: Some(asset.browser_download_url.clone()),
        metadata: Some(serde_json::json!({
            "asset": asset.name,
            "asset_size": asset.size,
            "gpu": is_nvidia_discrete(specs),
        })),
    };
    db::runtimes::upsert(db, &rv).await?;

    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::Done,
            message: format!("installed whisper.cpp {}", rv.version),
            bytes_done: asset.size,
            bytes_total: Some(asset.size),
            percent: 1.0,
            target: "runtime".into(),
        },
    );
    tracing::info!("whisper: installed {} ({})", rv.version, asset.name);
    Ok(rv)
}

pub(crate) async fn download_with_progress(
    app: &AppHandle,
    http: &reqwest::Client,
    asset: &Asset,
    dest: &Path,
    target: &str,
) -> Result<()> {
    let resp = http
        .get(&asset.browser_download_url)
        .send()
        .await
        .context("download request")?
        .error_for_status()
        .context("download status")?;

    let total = resp.content_length().unwrap_or(asset.size);
    let mut file = fs::File::create(dest)
        .await
        .with_context(|| format!("create {}", dest.display()))?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("download chunk")?;
        file.write_all(&chunk).await.context("write chunk")?;
        downloaded += chunk.len() as u64;
        let pct = if total > 0 {
            downloaded as f64 / total as f64
        } else {
            0.0
        };
        emit(
            app,
            WhisperProgress {
                stage: WhisperStage::Download,
                message: format!("downloading {}…", asset.name),
                bytes_done: downloaded,
                bytes_total: Some(total),
                percent: pct,
                target: target.to_string(),
            },
        );
    }
    file.flush().await.context("flush download")?;
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
                    .with_context(|| format!("create parent for {out:?}"))?;
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
        let ft = entry.file_type()?;
        let res = if ft.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if let Err(e) = res {
            tracing::warn!(
                "whisper: failed to remove {} during wipe: {e}",
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
        let first = comps.next()?;
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

/// Locate `whisper-cli` under the install root (checks the likely spots,
/// then a shallow walk).
pub fn find_executable(root: &Path) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
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
