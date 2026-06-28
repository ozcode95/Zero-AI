//! Tauri commands for the llama.cpp multi-variant orchestrator.
//!
//! Every command that takes a variant accepts the variant slug string
//! ("cuda", "openvino", "hip-radeon", "cpu") so the frontend can control
//! each instance independently.

use crate::error::IpcResult;
use crate::llama::variant::LlamaVariant;
use crate::llama::OrchestratorInfo;
use crate::state::AppStateExt;
use tauri::AppHandle;

/// Return the full orchestrator status: active variant + per-variant info.
#[tauri::command]
pub async fn llama_info(app: AppHandle) -> IpcResult<OrchestratorInfo> {
    Ok(app.zero().llama.info().await)
}

/// Install a specific llama.cpp variant. Also used for updates.
#[tauri::command]
pub async fn llama_install_variant(app: AppHandle, variant: String) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .install_variant(v)
        .await
        .map_err(|e| e.to_string().into())
}

/// Install all variants applicable to the current hardware.
#[tauri::command]
pub async fn llama_install_applicable(app: AppHandle) -> IpcResult<()> {
    app.zero()
        .llama
        .install_applicable_variants()
        .await
        .map_err(|e| e.to_string().into())
}

/// Update (re-install) a specific variant to the latest release.
#[tauri::command]
pub async fn llama_update_variant(app: AppHandle, variant: String) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .update_variant(v)
        .await
        .map_err(|e| e.to_string().into())
}

/// Check GitHub for the latest llama.cpp release and return the refreshed
/// orchestrator status (each variant's `latest_version` / `update_available`
/// reflect the result). Best-effort: a network failure still returns the
/// current status so the UI never hard-errors on a connectivity blip.
/// `force` bypasses the in-memory TTL cache for an explicit user check.
#[tauri::command]
pub async fn llama_check_updates(
    app: AppHandle,
    force: Option<bool>,
) -> IpcResult<OrchestratorInfo> {
    let llama = &app.zero().llama;
    if let Err(e) = llama.check_for_updates(force.unwrap_or(false)).await {
        tracing::warn!("llama.cpp update check failed: {e:#}");
    }
    Ok(llama.info().await)
}

/// Start (or restart) a variant's server. If `model_id` is provided,
/// loads that model; if null/empty, starts the server idle so a model
/// can be loaded later.
#[tauri::command]
pub async fn llama_start(
    app: AppHandle,
    variant: String,
    model_id: Option<String>,
) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .start(v, model_id.as_deref())
        .await
        .map_err(|e| e.to_string().into())
}

/// Stop a variant's server but keep the persisted loaded-model assignment
/// so `auto_provision` can replay it on the next start.
#[tauri::command]
pub async fn llama_stop(app: AppHandle, variant: String) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .stop(v)
        .await
        .map_err(|e| e.to_string().into())
}

/// Convenience: equivalent to `llama_start(active_variant, model_id)`.
/// Starts the model on whichever variant is currently active.
/// If model_id is null/empty, starts the server idle.
#[tauri::command]
pub async fn llama_load_model(app: AppHandle, model_id: Option<String>) -> IpcResult<()> {
    let llama = &app.zero().llama;
    let active = llama
        .active_variant()
        .await
        .ok_or_else(|| "no active llama.cpp variant".to_string())?;
    llama
        .start(active, model_id.as_deref())
        .await
        .map_err(|e| e.to_string().into())
}

/// Unload the model on the active variant, keeping the router running so a
/// different model can be loaded without a full server restart.
#[tauri::command]
pub async fn llama_unload_model(app: AppHandle) -> IpcResult<()> {
    let llama = &app.zero().llama;
    let active = llama
        .active_variant()
        .await
        .ok_or_else(|| "no active llama.cpp variant".to_string())?;
    llama
        .unload_model(active)
        .await
        .map_err(|e| e.to_string().into())
}

/// Stop a variant's server and clear the persisted loaded-model id.
#[tauri::command]
pub async fn llama_unload_variant(app: AppHandle, variant: String) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .unload(v)
        .await
        .map_err(|e| e.to_string().into())
}

/// Switch the active variant — the one the chat runner routes to.
#[tauri::command]
pub async fn llama_switch_variant(app: AppHandle, variant: String) -> IpcResult<()> {
    let v =
        LlamaVariant::from_slug(&variant).ok_or_else(|| format!("unknown variant: {variant}"))?;
    app.zero()
        .llama
        .switch_active_variant(v)
        .await
        .map_err(|e| e.to_string().into())
}

/// Backward-compatible install: installs all applicable variants for
/// the current hardware. This preserves the existing `llama_install`
/// command name for the frontend while the new multi-variant system
/// takes over.
#[tauri::command]
pub async fn llama_install(app: AppHandle) -> IpcResult<()> {
    app.zero()
        .llama
        .install_applicable_variants()
        .await
        .map_err(|e| e.to_string().into())
}
