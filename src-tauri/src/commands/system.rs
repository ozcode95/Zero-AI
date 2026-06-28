use crate::error::IpcResult;
use crate::state::AppStateExt;
use crate::system::{self, recommend, Specs};
use tauri::AppHandle;

/// Hardware/OS probe. Cached to `system.json`; `force = true` re-probes.
#[tauri::command]
pub async fn system_probe(app: AppHandle, force: Option<bool>) -> IpcResult<Specs> {
    let force = force.unwrap_or(false);

    // Try the in-memory cache first.
    let state = app.zero();
    if !force {
        if let Some(cached) = state.specs.read().await.clone() {
            return Ok(cached);
        }
        // Then the on-disk cache.
        if let Some(disk) = system::load_cached() {
            *state.specs.write().await = Some(disk.clone());
            return Ok(disk);
        }
    }

    // Fresh probe (sysinfo + WMI on Windows). Run on a blocking pool so we
    // don't block the Tokio runtime with COM/WMI calls.
    let specs = tokio::task::spawn_blocking(system::probe)
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    if let Err(e) = system::save_cached(&specs) {
        tracing::warn!("system cache write failed: {e}");
    }
    *state.specs.write().await = Some(specs.clone());
    Ok(specs)
}

/// Recommend models based on the user's hardware.
///
/// Uses llmfit-core's `ModelDatabase` and `ModelFit` scoring to find the
/// best-fitting models for the running machine.  Results are cached for 24 h,
/// keyed by `mode` (`"gpu"` / `"ram"`) and `quant` (e.g. `"Q4_K_M"`).
///
/// * `mode` — `"gpu"` scores against the discrete GPU (VRAM), `"ram"` against
///   CPU + iGPU + system RAM. Defaults to `"gpu"`.
/// * `quant` — quantization to score against. Defaults to `"Q4_K_M"`.
///
/// Returns a ranked list of [`recommend::RecommendedModel`], sorted by
/// estimated tokens-per-second (fastest first).
#[tauri::command]
pub async fn system_recommend_models(
    mode: Option<String>,
    quant: Option<String>,
) -> IpcResult<Vec<recommend::RecommendedModel>> {
    let hw_mode = recommend::HwMode::from_opt(mode.as_deref());
    let quant = recommend::normalize_quant(quant.as_deref());
    // `recommend_all` is CPU-bound (sysinfo probe + model scoring).
    // Run on the blocking pool so we don't stall the Tokio runtime.
    let models = tokio::task::spawn_blocking(move || recommend::recommend_all(hw_mode, &quant))
        .await
        .map_err(|e| e.to_string())?;
    Ok(models)
}

/// Search the llmfit model database for models matching a query, scored
/// against the current hardware.  Returns up to 5 best‑fitting results.
#[tauri::command]
pub async fn system_search_models(query: String) -> IpcResult<Vec<recommend::RecommendedModel>> {
    let q = query.clone();
    let results = tokio::task::spawn_blocking(move || recommend::search_models(&q))
        .await
        .map_err(|e| e.to_string())?;
    Ok(results)
}

/// Force-refresh recommendations by clearing caches, re-fetching the
/// online model catalogue, and re-scoring against current hardware.
///
/// Accepts the same `mode` / `quant` parameters as [`system_recommend_models`].
#[tauri::command]
pub async fn system_recommend_refresh(
    mode: Option<String>,
    quant: Option<String>,
) -> IpcResult<Vec<recommend::RecommendedModel>> {
    let hw_mode = recommend::HwMode::from_opt(mode.as_deref());
    let quant = recommend::normalize_quant(quant.as_deref());
    let models = tokio::task::spawn_blocking(move || recommend::recommend_refresh(hw_mode, &quant))
        .await
        .map_err(|e| e.to_string())?;
    Ok(models)
}
