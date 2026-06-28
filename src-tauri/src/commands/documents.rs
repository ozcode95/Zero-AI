//! Tauri commands for embedding knowledge-base documents.

use crate::documents::{self, Document};
use crate::error::IpcResult;
use std::path::PathBuf;

/// List every document in the knowledge base.
#[tauri::command]
pub async fn documents_list() -> IpcResult<Vec<Document>> {
    documents::list().await.map_err(|e| e.to_string().into())
}

/// Copy a file from `source_path` (an OS-picker path) into the knowledge
/// base and return the persisted metadata. New documents are enabled by
/// default — they only count as *disabled* once their id lands in
/// `settings.embedding.documents_disabled`.
#[tauri::command]
pub async fn documents_add(source_path: String) -> IpcResult<Document> {
    let p = PathBuf::from(&source_path);
    documents::add(&p).await.map_err(|e| e.to_string().into())
}

/// Remove a document from the knowledge base. Idempotent.
#[tauri::command]
pub async fn documents_delete(id: String) -> IpcResult<()> {
    documents::delete(&id)
        .await
        .map_err(|e| e.to_string().into())
}
