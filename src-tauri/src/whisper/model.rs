//! whisper ggml model downloader.
//!
//! Whisper models are plain single-file `.bin` downloads from the canonical
//! `ggerganov/whisper.cpp` HuggingFace repo — *not* multi-file GGUF repos,
//! so they bypass the normal model-download flow and land directly under
//! `models/whisper/`.

use crate::whisper::{emit, WhisperProgress, WhisperStage};
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use std::path::PathBuf;
use tauri::AppHandle;
use tokio::fs;
use tokio::io::AsyncWriteExt;

const HF_REPO: &str = "ggerganov/whisper.cpp";

/// Download a ggml model `.bin` (e.g. `ggml-base.en.bin`) into
/// `models/whisper/`, streaming progress. No-op (returns the existing path)
/// if it's already on disk.
pub async fn download_model(
    app: &AppHandle,
    http: &reqwest::Client,
    file: &str,
) -> Result<PathBuf> {
    let file = file.trim();
    if file.is_empty() || !file.ends_with(".bin") {
        return Err(anyhow!("invalid whisper model file name: {file:?}"));
    }
    // Guard against path traversal — these are bare file names.
    if file.contains('/') || file.contains('\\') || file.contains("..") {
        return Err(anyhow!("unsafe whisper model file name: {file:?}"));
    }

    let dir = crate::paths::whisper_models_dir()?;
    let dest = dir.join(file);
    if dest.is_file() {
        return Ok(dest);
    }

    let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{file}");
    let resp = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("download {file}"))?;

    let total = resp.content_length();
    // Download to a temp sibling then rename, so an interrupted download
    // never leaves a half-written `.bin` that looks complete.
    let tmp = dir.join(format!("{file}.part"));
    let mut out = fs::File::create(&tmp)
        .await
        .with_context(|| format!("create {}", tmp.display()))?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("download chunk")?;
        out.write_all(&chunk).await.context("write chunk")?;
        downloaded += chunk.len() as u64;
        let pct = match total {
            Some(t) if t > 0 => downloaded as f64 / t as f64,
            _ => 0.0,
        };
        emit(
            app,
            WhisperProgress {
                stage: WhisperStage::Download,
                message: format!("downloading {file}…"),
                bytes_done: downloaded,
                bytes_total: total,
                percent: pct,
                target: file.to_string(),
            },
        );
    }
    out.flush().await.context("flush model")?;
    drop(out);
    fs::rename(&tmp, &dest)
        .await
        .with_context(|| format!("finalize {}", dest.display()))?;

    emit(
        app,
        WhisperProgress {
            stage: WhisperStage::Done,
            message: format!("downloaded {file}"),
            bytes_done: downloaded,
            bytes_total: total,
            percent: 1.0,
            target: file.to_string(),
        },
    );
    tracing::info!("whisper: downloaded model {file} ({downloaded} bytes)");
    Ok(dest)
}
