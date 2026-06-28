//! whisper.cpp speech-to-text subsystem.
//!
//! The chat composer's microphone button records an utterance, the
//! renderer converts it to 16 kHz mono WAV, and we transcribe it by
//! shelling out to the `whisper-cli` binary from a whisper.cpp release —
//! GPU-accelerated on NVIDIA (cuBLAS build), CPU elsewhere.
//!
//! Unlike the chat model, whisper does **not** run as a long-lived server.
//! Each utterance spawns a one-shot `whisper-cli` process (cold start is a
//! second or two, which the user has accepted for a local, private STT
//! path that transcribes verbatim instead of "answering" the clip the way
//! an audio chat model does).
//!
//! Two things must be on disk:
//! 1. the runtime (release zip → `runtimes/whisper.cpp/`, recorded in
//!    `runtime_versions` under [`RUNTIME_NAME`]); and
//! 2. a ggml model `.bin` (`models/whisper/ggml-*.bin`).

pub mod github;
pub mod install;
pub mod model;

use crate::db;
use crate::events;
use crate::system::{GpuKind, Specs};
use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};
use tokio::process::Command;

/// `runtime_versions.name` key for the whisper.cpp install.
pub const RUNTIME_NAME: &str = "whisper.cpp";

// ─── Progress reporting (shared by install + model download) ─────────────

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WhisperStage {
    FetchRelease,
    Download,
    Extract,
    Verify,
    Done,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhisperProgress {
    pub stage: WhisperStage,
    pub message: String,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub percent: f64,
    /// What's being fetched: `"runtime"` or the model `.bin` filename. Lets
    /// the Settings UI label the progress without a separate round-trip.
    pub target: String,
}

pub(crate) fn emit(app: &AppHandle, p: WhisperProgress) {
    let _ = app.emit(events::WHISPER_PROGRESS, p);
}

// ─── Status ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct WhisperStatus {
    /// Whether the whisper.cpp runtime binary is installed.
    pub runtime_installed: bool,
    /// Installed runtime version tag, when present.
    pub runtime_version: Option<String>,
    /// `true` when the host has an NVIDIA discrete GPU (so the install will
    /// pull the GPU-accelerated cuBLAS build). Surfaced so the UI can say
    /// "GPU" vs "CPU" before the user commits to a ~680 MB download.
    pub gpu: bool,
    /// Downloaded ggml model `.bin` filenames (just the file names).
    pub models: Vec<String>,
}

/// Snapshot the current whisper install state for the Settings panel.
pub async fn status(db: &SqlitePool) -> WhisperStatus {
    let rv = db::runtimes::get(db, RUNTIME_NAME).await.ok().flatten();
    let runtime_installed = rv
        .as_ref()
        .map(|r| Path::new(&r.executable).is_file())
        .unwrap_or(false);
    let gpu = crate::system::load_cached()
        .map(|s| is_nvidia_discrete(&s))
        .unwrap_or(false);
    WhisperStatus {
        runtime_installed,
        runtime_version: rv.map(|r| r.version),
        gpu,
        models: downloaded_models(),
    }
}

/// `true` when the host has a discrete NVIDIA GPU — the only case where a
/// GPU-accelerated whisper.cpp Windows build exists (cuBLAS). Everything
/// else falls back to the CPU build.
pub fn is_nvidia_discrete(specs: &Specs) -> bool {
    specs
        .gpus
        .iter()
        .any(|g| matches!(g.kind, GpuKind::Discrete) && g.vendor.to_lowercase().contains("nvidia"))
}

/// List the ggml `.bin` model files already downloaded under
/// `models/whisper/`. Returns bare file names (e.g. `ggml-base.en.bin`).
pub fn downloaded_models() -> Vec<String> {
    let Ok(dir) = crate::paths::whisper_models_dir() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("bin"))
            {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

/// Resolve the installed `whisper-cli` executable from the DB, verifying it
/// still exists on disk.
pub async fn installed_exe(db: &SqlitePool) -> Result<Option<PathBuf>> {
    let Some(rv) = db::runtimes::get(db, RUNTIME_NAME).await? else {
        return Ok(None);
    };
    let exe = PathBuf::from(&rv.executable);
    Ok(if exe.is_file() { Some(exe) } else { None })
}

/// Absolute path to a downloaded model `.bin` by file name, if present.
pub fn model_path(file: &str) -> Option<PathBuf> {
    let p = crate::paths::whisper_models_dir().ok()?.join(file);
    p.is_file().then_some(p)
}

// ─── Transcription ───────────────────────────────────────────────────────

/// Transcribe 16 kHz mono WAV `wav` to text using `whisper-cli`.
///
/// Runs one-shot: writes the WAV to a temp file, invokes the CLI with no
/// timestamps / no progress prints, and returns the trimmed transcript from
/// stdout. `lang` is the spoken-language hint (`"auto"` to detect).
pub async fn transcribe(
    exe: &Path,
    model_bin: &Path,
    wav: &[u8],
    lang: Option<&str>,
) -> Result<String> {
    if wav.is_empty() {
        return Err(anyhow!("no audio captured"));
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let wav_path =
        std::env::temp_dir().join(format!("zero-stt-{}-{nanos}.wav", std::process::id()));
    tokio::fs::write(&wav_path, wav)
        .await
        .with_context(|| format!("write temp wav {}", wav_path.display()))?;

    // Inspect what we actually received. A near-silent clip is the usual
    // cause of whisper "hallucinations" (it emits "you" / "Thank you." for
    // silence), so measure peak amplitude + duration and bail early with a
    // clear message rather than handing back a phantom transcript.
    let (duration_s, peak) = analyze_wav(wav);
    tracing::info!(
        target: "whisper",
        "stt input: {} bytes, ~{:.2}s, peak {:.3} of full-scale",
        wav.len(),
        duration_s,
        peak
    );
    // Keep a copy of the most recent capture for debugging.
    if let Ok(logs) = crate::paths::logs_dir() {
        let _ = tokio::fs::write(logs.join("last-stt-input.wav"), wav).await;
    }
    if peak < 0.01 {
        let _ = tokio::fs::remove_file(&wav_path).await;
        return Err(anyhow!(
            "the microphone captured near-silence (peak {:.3}). Check the input device / mic permission and try speaking closer.",
            peak
        ));
    }

    let lang = lang
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("auto");

    let mut cmd = Command::new(exe);
    cmd.arg("-m")
        .arg(model_bin)
        .arg("-f")
        .arg(&wav_path)
        .arg("-l")
        .arg(lang)
        .arg("-nt") // no timestamps
        .arg("-np") // no progress / system prints — keep stdout to the text
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    tracing::info!(
        target: "whisper",
        "transcribing {} bytes via {} (model {})",
        wav.len(),
        exe.display(),
        model_bin.display()
    );

    let output = cmd
        .output()
        .await
        .with_context(|| format!("run {}", exe.display()));
    let _ = tokio::fs::remove_file(&wav_path).await;
    let output = output?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n");
        return Err(anyhow!(
            "whisper-cli exited with {}: {}",
            output.status,
            tail.trim()
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.trim().to_string())
}

/// Parse a 16-bit-PCM mono WAV (our renderer's exact format) into
/// `(duration_seconds, peak_amplitude)` where peak is 0.0..=1.0 of
/// full-scale. Tolerant of a missing/short body; returns zeros then.
fn analyze_wav(wav: &[u8]) -> (f64, f64) {
    // Standard 44-byte header from `encodeWavPcm16`: sample rate @ 24,
    // PCM samples from offset 44.
    if wav.len() < 44 {
        return (0.0, 0.0);
    }
    let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]).max(1) as f64;
    let pcm = &wav[44..];
    let n = pcm.len() / 2;
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut peak: i32 = 0;
    for s in pcm.chunks_exact(2) {
        let v = i16::from_le_bytes([s[0], s[1]]) as i32;
        let a = v.unsigned_abs() as i32;
        if a > peak {
            peak = a;
        }
    }
    let duration = n as f64 / sample_rate;
    (duration, peak as f64 / 32768.0)
}
