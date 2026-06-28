//! Audio commands: speech-to-text (whisper.cpp) and text-to-speech
//! (llama.cpp's `llama-tts`).
//!
//! Both run as GPU-accelerated one-shot CLIs per utterance rather than
//! long-lived servers — llama.cpp's HTTP server exposes neither a
//! transcription nor a speech endpoint.
//!
//! * **Speech → text** shells out to `whisper-cli` from a whisper.cpp
//!   release (cuBLAS build on NVIDIA, CPU otherwise). The renderer records
//!   the mic, converts the clip to 16 kHz mono WAV, and ships the bytes
//!   here; we hand them to `whisper-cli` and return the verbatim transcript.
//! * **Text → speech** shells out to `llama-tts` (a sibling of
//!   `llama-server` in the same release) with the configured OuteTTS model
//!   + WavTokenizer vocoder, returning WAV bytes the renderer plays back.

use crate::error::IpcResult;
use crate::state::AppStateExt;
use crate::whisper;
use tauri::AppHandle;

/// Resolve cached system specs, probing once if necessary. Used by the
/// whisper install command to pick the GPU vs CPU build.
async fn resolve_specs() -> Result<crate::system::Specs, String> {
    if let Some(s) = crate::system::load_cached() {
        return Ok(s);
    }
    let specs = tokio::task::spawn_blocking(crate::system::probe)
        .await
        .map_err(|e| format!("hardware probe task panicked: {e}"))?
        .map_err(|e| format!("hardware probe failed: {e}"))?;
    let _ = crate::system::save_cached(&specs);
    Ok(specs)
}

/// Transcribe a recorded utterance with the configured whisper model.
///
/// `audio` is 16 kHz mono WAV (the renderer converts its `MediaRecorder`
/// blob first). The whisper runtime + model must already be installed;
/// returns a clear error otherwise so the UI can prompt a download.
#[tauri::command]
pub async fn audio_transcribe(
    app: AppHandle,
    audio: Vec<u8>,
    lang: Option<String>,
) -> IpcResult<String> {
    let state = app.zero();

    let settings = crate::settings::Settings::load().await.unwrap_or_default();
    if !settings.audio.enabled {
        return Err("audio is disabled".into());
    }
    let model_file = settings
        .audio
        .stt_model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or("no speech-to-text model configured")?;

    let exe = whisper::installed_exe(&state.db)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("the whisper.cpp runtime is not installed yet")?;
    let model_bin = whisper::model_path(model_file)
        .ok_or_else(|| format!("speech-to-text model '{model_file}' is not downloaded yet"))?;

    // Prefer the per-call hint, fall back to the configured language, and
    // default to English. Whisper's auto-detect is unreliable on short
    // clips, so we never silently fall through to it.
    let lang = lang
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            settings
                .audio
                .stt_language
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "en".to_string());

    let text = whisper::transcribe(&exe, &model_bin, &audio, Some(&lang))
        .await
        .map_err(|e| e.to_string())?;
    Ok(text)
}

/// Synthesize `text` to WAV bytes via `llama-tts`. Returns the raw WAV the
/// renderer wraps in a `Blob` and plays.
#[tauri::command]
pub async fn audio_speak(app: AppHandle, text: String) -> IpcResult<Vec<u8>> {
    app.zero()
        .llama
        .tts_synthesize(&text)
        .await
        .map_err(|e| e.to_string().into())
}

/// Install (or update) the whisper.cpp runtime for this host. Streams
/// progress via the `whisper://progress` event.
#[tauri::command]
pub async fn whisper_install(app: AppHandle) -> IpcResult<()> {
    let state = app.zero();
    let specs = resolve_specs().await?;
    whisper::install::install(&app, &state.http, &state.db, &specs)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string().into())
}

/// Current whisper install state (runtime + downloaded models + GPU flag).
#[tauri::command]
pub async fn whisper_status(app: AppHandle) -> IpcResult<whisper::WhisperStatus> {
    Ok(whisper::status(&app.zero().db).await)
}

/// Download a whisper ggml model `.bin` into `models/whisper/`. Streams
/// progress via the `whisper://progress` event.
#[tauri::command]
pub async fn whisper_download_model(app: AppHandle, file: String) -> IpcResult<()> {
    let state = app.zero();
    whisper::model::download_model(&app, &state.http, &file)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string().into())
}
