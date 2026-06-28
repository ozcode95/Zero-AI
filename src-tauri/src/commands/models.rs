use crate::error::IpcResult;
use crate::hf::{self, Cancelled, DownloadJobError, HfModelSummary, LocalModel};
use crate::state::AppStateExt;
use sqlx::Row;
use tauri::AppHandle;

#[tauri::command]
pub async fn models_search(app: AppHandle, query: String) -> IpcResult<Vec<HfModelSummary>> {
    let state = app.zero();
    hf::search(&state.http, &query)
        .await
        .map_err(|e| e.to_string().into())
}

#[tauri::command]
pub async fn models_list_local(app: AppHandle) -> IpcResult<Vec<LocalModel>> {
    let state = app.zero();
    let rows = sqlx::query(
        "SELECT id, hf_id, path, bytes, added_at, revision, files, verified_files, pipeline_tag, metadata_json
         FROM local_models
         ORDER BY added_at DESC",
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|r| {
            let path: String = r.get("path");
            // Same nullable-column pattern as `hf_id` above: `.ok()`
            // collapses both SQL NULL and "column not present in the
            // row" to `None`, which keeps the read path resilient if the
            // baseline schema ever lags a deployed db.
            let stored: Option<String> = r.try_get("pipeline_tag").ok();
            // Backfill: pre-`pipeline_tag` rows store NULL, and the HF
            // upstream sometimes leaves the field unset too. Fall back to
            // `text-generation` — the implicit default for legacy
            // text-LM installs.
            let pipeline_tag = stored.or_else(|| Some("text_generation".to_string()));
            LocalModel {
                id: r.get("id"),
                hf_id: r.try_get("hf_id").ok(),
                path,
                bytes: r.get::<i64, _>("bytes") as u64,
                added_at: r.get("added_at"),
                revision: r.try_get("revision").ok(),
                files: r.try_get::<i64, _>("files").ok().map(|n| n.max(0) as u64),
                verified_files: r
                    .try_get::<i64, _>("verified_files")
                    .ok()
                    .map(|n| n.max(0) as u64),
                pipeline_tag,
                metadata_json: r.try_get("metadata_json").ok(),
            }
        })
        .collect())
}

/// Initial install of a HF model. Idempotent — calling it on an already
/// up-to-date model is a fast no-op (manifest read + DB upsert).
#[tauri::command]
pub async fn models_download(
    app: AppHandle,
    model_id: String,
    metadata_json: Option<String>,
) -> IpcResult<LocalModel> {
    install_or_update(app, model_id, metadata_json, None).await
}

/// A single GGUF file in a HuggingFace repo, surfaced for the manual
/// download picker on the Models page.
#[derive(serde::Serialize)]
pub struct GgufFileInfo {
    /// Repo-relative path (forward slashes), e.g. `model-Q4_K_M.gguf`.
    pub name: String,
    /// Size in bytes (LFS-aware). `0` when upstream didn't advertise one.
    pub size: u64,
    /// Canonical quant token when recognizable (e.g. `Q4_K_M`).
    pub quant: Option<String>,
    /// `main` | `mmproj` | `draft` — lets the UI group/badge files.
    pub kind: String,
}

/// List the `.gguf` files in a HuggingFace repo so the user can hand-pick
/// which ones to download (manual download flow). Returns an error if the
/// repo can't be reached or has no GGUF files.
#[tauri::command]
pub async fn models_list_gguf_files(
    app: AppHandle,
    model_id: String,
) -> IpcResult<Vec<GgufFileInfo>> {
    let state = app.zero();
    let info = hf::model_info(&state.http, &model_id, None)
        .await
        .map_err(|e| e.to_string())?;

    let mut files: Vec<GgufFileInfo> = info
        .siblings
        .iter()
        .filter(|s| s.rfilename.to_lowercase().ends_with(".gguf"))
        .map(|s| {
            let size = s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0);
            let lower = s.rfilename.to_lowercase();
            let kind = if lower.contains("mmproj") {
                "mmproj"
            } else if lower.contains("mtp") || lower.contains("draft") {
                "draft"
            } else {
                "main"
            };
            GgufFileInfo {
                quant: hf::select::extract_quant(&s.rfilename),
                kind: kind.to_string(),
                name: s.rfilename.clone(),
                size,
            }
        })
        .collect();

    if files.is_empty() {
        return Err(format!("{model_id} has no .gguf files").into());
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(files)
}

/// Manual download: pull exactly the GGUF files the user selected from the
/// repo (plus the repo's support files). Bypasses the automatic quant picker.
#[tauri::command]
pub async fn models_download_files(
    app: AppHandle,
    model_id: String,
    files: Vec<String>,
) -> IpcResult<LocalModel> {
    if files.is_empty() {
        return Err("no files selected for download".to_string().into());
    }
    install_or_update(app, model_id, None, Some(files)).await
}

/// Same code path as `models_download` — `install_or_update` already diffs
/// the upstream revision against the local manifest and re-pulls only the
/// changed files.
#[tauri::command]
pub async fn models_update(app: AppHandle, model_id: String) -> IpcResult<LocalModel> {
    install_or_update(app, model_id, None, None).await
}

async fn install_or_update(
    app: AppHandle,
    model_id: String,
    metadata_json: Option<String>,
    selected_gguf: Option<Vec<String>>,
) -> IpcResult<LocalModel> {
    let state = app.zero();

    // Reserve the per-model slot before we do anything else so a double-
    // click in the UI surfaces as a clear error instead of two concurrent
    // writers stomping on the same directory.
    let token = match state.downloads.start(&model_id).await {
        Ok(t) => t,
        Err(DownloadJobError::AlreadyRunning(_)) => {
            return Err(format!("a download for `{model_id}` is already in progress").into());
        }
    };
    let cancel = token.handle();

    let result = hf::install_or_update(
        &app,
        &state.http,
        &state.db,
        &model_id,
        cancel,
        metadata_json.as_deref(),
        selected_gguf.as_deref(),
    )
    .await;
    // `token` stays alive for the whole call above; dropping it here releases
    // the slot for the next attempt (success, cancel, or error).
    drop(token);

    match result {
        Ok(m) => Ok(m),
        Err(e) => {
            // Distinguish a user-initiated cancel from a real failure so the
            // UI can render them differently.
            let (state_kind, msg) = if e.downcast_ref::<Cancelled>().is_some() {
                (hf::DownloadState::Cancelled, "cancelled".to_string())
            } else {
                (hf::DownloadState::Error, format!("{e:#}"))
            };
            let _ = tauri::Emitter::emit(
                &app,
                crate::events::MODELS_DOWNLOAD_PROGRESS,
                hf::DownloadProgress {
                    model_id: model_id.clone(),
                    bytes_done: 0,
                    bytes_total: None,
                    files_done: 0,
                    files_total: 0,
                    state: state_kind,
                    error: Some(msg.clone()),
                },
            );
            Err(msg.into())
        }
    }
}

/// Request cancellation of an in-flight `models_download` / `models_update`.
///
/// Returns `true` when a job was found and signalled, `false` when there was
/// no active download for `model_id`. The frontend can treat both the same
/// way (the runner will emit a terminal `cancelled` progress event when the
/// abort actually lands).
#[tauri::command]
pub async fn models_cancel(app: AppHandle, model_id: String) -> IpcResult<bool> {
    Ok(app.zero().downloads.cancel(&model_id).await)
}

#[tauri::command]
pub async fn models_delete(app: AppHandle, model_id: String) -> IpcResult<()> {
    let state = app.zero();
    hf::delete_model(&state.db, &model_id)
        .await
        .map_err(|e| e.to_string().into())
}
