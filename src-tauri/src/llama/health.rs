//! llama-server readiness probe.
//!
//! llama-server signals "ready to serve" via its `/health` endpoint
//! (200 once model weights are loaded; 503 while still loading). The
//! OpenAI-compat `/v1/models` endpoint also flips to 200 around the
//! same time, so we hit `/health` first and fall back to `/v1/models`
//! if the health endpoint is unavailable on older builds.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::time::{Duration, Instant};

/// Block until llama-server reports healthy or `timeout` elapses.
pub async fn wait_ready(http: &reqwest::Client, base_url: &str, timeout: Duration) -> Result<()> {
    let trimmed = base_url.trim_end_matches('/');
    let health = format!("{trimmed}/health");
    let models = format!("{trimmed}/v1/models");
    let started = Instant::now();
    let mut delay = Duration::from_millis(250);
    let mut warn_at = Duration::from_secs(10);

    tracing::info!(
        "llama health: waiting for server to become ready at {base_url} (timeout={:?})",
        timeout
    );

    loop {
        if started.elapsed() >= timeout {
            return Err(anyhow!(
                "llama-server did not become ready within {:?} (last probes: {} | {})",
                timeout,
                health,
                models
            ));
        }

        // Emit a visible warning periodically so the user knows the app
        // hasn't frozen — the health probe is just taking a while.
        if started.elapsed() >= warn_at {
            tracing::warn!(
                "llama health: still waiting for server at {base_url} after {:.0}s",
                started.elapsed().as_secs_f64()
            );
            warn_at += Duration::from_secs(30);
        }

        // Try /health first. It's the canonical readiness signal and
        // distinguishes "still loading the model" (503) from "wedged
        // and refusing connections" (transport error).
        match http
            .get(&health)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => {
                tracing::trace!("llama health: {} {} (still waiting)", resp.status(), health);
            }
            Err(e) => {
                tracing::trace!("llama health: {e} (still waiting)");
                // Fallback for builds that don't ship /health: any 2xx on
                // /v1/models also means we're done loading.
                if let Ok(resp) = http
                    .get(&models)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                {
                    if resp.status().is_success() {
                        return Ok(());
                    }
                }
            }
        }

        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(Duration::from_secs(2));
    }
}

#[derive(Debug, Deserialize)]
struct ModelStatus {
    value: String,
    #[serde(default)]
    failed: bool,
    #[serde(default)]
    exit_code: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct RouterModel {
    id: String,
    #[serde(default)]
    status: Option<ModelStatus>,
}

#[derive(Debug, Deserialize)]
struct RouterModelsResponse {
    data: Vec<RouterModel>,
}

/// Ask the router to re-read its model sources (preset / models-dir / cache).
///
/// `root_url` is the server root **without** the `/v1` suffix. This is how a
/// freshly-downloaded model becomes visible to an already-running router
/// without restarting it (the preset file must have been regenerated first).
pub async fn reload_router_models(http: &reqwest::Client, root_url: &str) {
    let url = format!("{}/models?reload=1", root_url.trim_end_matches('/'));
    match http.get(&url).timeout(Duration::from_secs(15)).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("llama router reloaded model list");
        }
        Ok(resp) => {
            tracing::warn!("router /models?reload=1 returned {}", resp.status());
        }
        Err(e) => {
            tracing::warn!("router /models?reload=1 failed: {e}");
        }
    }
}

/// Poll the router's `GET /models` until `model_id` reports `loaded`
/// (or `sleeping`), fails, or `timeout` elapses.
///
/// `POST /models/load` returns as soon as the load is *accepted* — the model
/// is then loaded asynchronously, so we have to watch the status to surface a
/// crash (e.g. an unsupported kernel) as a real error instead of a silent
/// background failure. `root_url` is the server root without the `/v1` suffix.
pub async fn wait_model_loaded(
    http: &reqwest::Client,
    root_url: &str,
    model_id: &str,
    timeout: Duration,
) -> Result<()> {
    let url = format!("{}/models", root_url.trim_end_matches('/'));
    let started = Instant::now();
    let mut warn_at = Duration::from_secs(15);

    loop {
        if started.elapsed() >= timeout {
            return Err(anyhow!(
                "model `{model_id}` did not finish loading within {timeout:?}"
            ));
        }
        if started.elapsed() >= warn_at {
            tracing::info!(
                "llama: still loading `{model_id}` after {:.0}s",
                started.elapsed().as_secs_f64()
            );
            warn_at += Duration::from_secs(30);
        }

        if let Ok(resp) = http.get(&url).timeout(Duration::from_secs(5)).send().await {
            if resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                if let Ok(body) = serde_json::from_str::<RouterModelsResponse>(&text) {
                    if let Some(m) = body.data.iter().find(|m| m.id == model_id) {
                        if let Some(st) = &m.status {
                            if st.failed {
                                return Err(anyhow!(
                                    "model `{model_id}` failed to load{}",
                                    st.exit_code
                                        .map(|c| format!(" (exit code {c})"))
                                        .unwrap_or_default()
                                ));
                            }
                            match st.value.as_str() {
                                "loaded" | "sleeping" => return Ok(()),
                                // loading / downloading / unloaded → keep waiting
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
