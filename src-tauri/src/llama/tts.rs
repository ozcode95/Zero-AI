//! Text-to-speech via the bundled `llama-tts` CLI.
//!
//! llama.cpp's HTTP server exposes no TTS endpoint, but the official
//! release ships a `llama-tts` executable right next to `llama-server`.
//! It runs an OuteTTS model + a WavTokenizer vocoder on the GPU (`-ngl`)
//! and writes a `output.wav` file to its working directory.
//!
//! We invoke it as a one-shot CLI per utterance (it has no persistent
//! server mode), capturing the WAV bytes to hand back to the renderer for
//! playback. Each call uses its own temp working dir so concurrent
//! "speak" clicks don't clobber each other's `output.wav`.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// HuggingFace repo id of the WavTokenizer vocoder `llama-tts` needs
/// alongside the OuteTTS model. Downloaded through the normal model flow
/// and resolved out of `local_models` at synthesis time.
pub const WAVTOKENIZER_HF_ID: &str = "ggml-org/WavTokenizer";
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

/// Synthesize `text` to WAV bytes using `llama-tts`.
///
/// `llama_tts_exe` is the CLI binary; `oute_model` is the OuteTTS GGUF and
/// `vocoder_model` the WavTokenizer GGUF. `n_gpu_layers` mirrors the chat
/// runtime's offload setting (`-1`/all is mapped to a large number so the
/// whole tiny model rides on the GPU).
pub async fn synthesize(
    llama_tts_exe: &Path,
    oute_model: &Path,
    vocoder_model: &Path,
    text: &str,
    n_gpu_layers: i32,
) -> Result<Vec<u8>> {
    let text = text.trim();
    if text.is_empty() {
        bail!("nothing to speak (empty text)");
    }

    // Unique temp working dir — llama-tts writes `output.wav` into the
    // current directory, so isolating per call keeps concurrent requests
    // from racing on the same filename.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let work = std::env::temp_dir().join(format!("zero-tts-{}-{nanos}", std::process::id()));
    tokio::fs::create_dir_all(&work)
        .await
        .with_context(|| format!("create tts work dir {}", work.display()))?;
    let out_wav = work.join("output.wav");

    // `-1` (the settings sentinel for "offload everything") isn't a valid
    // llama.cpp `-ngl` value; map it to a number larger than any layer
    // count so the (small) TTS model fully offloads to the GPU.
    let ngl = if n_gpu_layers < 0 { 99 } else { n_gpu_layers };

    let mut cmd = Command::new(llama_tts_exe);
    cmd.arg("-m")
        .arg(oute_model)
        .arg("-mv")
        .arg(vocoder_model)
        .arg("-ngl")
        .arg(ngl.to_string())
        .arg("-p")
        .arg(text)
        .current_dir(&work)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    tracing::info!(
        target: "llama::tts",
        "synthesizing {} chars via {}",
        text.len(),
        llama_tts_exe.display()
    );

    let output = cmd
        .output()
        .await
        .with_context(|| format!("run {}", llama_tts_exe.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n");
        let _ = tokio::fs::remove_dir_all(&work).await;
        bail!("llama-tts exited with {}: {}", output.status, tail.trim());
    }

    let bytes = tokio::fs::read(&out_wav).await.with_context(|| {
        format!(
            "read llama-tts output {} (the tool exited 0 but wrote no audio)",
            out_wav.display()
        )
    });
    let _ = tokio::fs::remove_dir_all(&work).await;
    let bytes = bytes?;
    if bytes.is_empty() {
        bail!("llama-tts produced an empty WAV file");
    }
    Ok(bytes)
}
