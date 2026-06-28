//! Hugging Face integration — search, download, update, delete.
//!
//! Layout on disk (under `<app-local>/zero/models/`):
//!
//! ```text
//! models/
//!   <org>/<repo>/
//!     <file_a>
//!     <file_b>
//!     .zero_manifest.json   ← revision SHA + per-file size/sha + GGUF classification
//! ```
//!
//! The directory layout is intentionally flat (no per-revision sub-dir). The
//! manifest carries enough information to update only changed files and to
//! reconcile against the SQLite `local_models` row. It also stores a
//! pre-computed GGUF classification (`model`, `mmproj`, `drafts`) so the
//! llama.cpp loader doesn't need to re-scan the directory.
//!
//! Cancellation + concurrency are coordinated through [`DownloadJobs`]:
//! attempting to start a download for a model that already has an in-flight
//! job returns [`DownloadJobError::AlreadyRunning`] instead of racing against
//! itself on the same directory.

pub mod api;
pub mod download;
pub mod jobs;
pub mod select;

pub use api::model_info;
pub use api::search;
pub use download::{
    backfill_verified, classify_gguf_files, delete as delete_model, install_or_update,
    read_manifest_sync, Cancelled, Manifest,
};
pub use jobs::{CancelHandle, DownloadJobError, DownloadJobs, JobToken};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelSummary {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub updated_at: String,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub has_openvino_ir: bool,
    #[serde(default)]
    pub has_gguf: bool,
    pub total_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub id: String,
    pub path: String,
    pub bytes: u64,
    pub added_at: String,
    pub hf_id: Option<String>,
    pub revision: Option<String>,
    pub files: Option<u64>,
    /// How many of `files` carry a recorded sha256 — i.e. were verified
    /// against an HF-published LFS digest at download time. `None` when the
    /// manifest pre-dates verification tracking; a follow-up install will
    /// repopulate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified_files: Option<u64>,
    /// Upstream HuggingFace `pipeline_tag` captured at download time —
    /// e.g. `text-generation`, `image-text-to-text`, `feature-extraction`.
    /// Used by the Models page to render a per-model category badge.
    /// `None` on rows that pre-date the column; readers fall back to
    /// `text_generation`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_tag: Option<String>,
    /// JSON blob with llmfit recommendation metadata (use case, fit level,
    /// score, best quant, capabilities, etc.).  `None` for models installed
    /// before this field existed or installed outside the recommendation flow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Pending,
    Downloading,
    Verifying,
    Done,
    Cancelled,
    Error,
}

/// Progress event for `models://download-progress`. The shape matches the
/// frontend `DownloadProgress` type in `src/stores/models.ts`.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub files_done: u64,
    pub files_total: u64,
    pub state: DownloadState,
    pub error: Option<String>,
}

/// Reads the HF token via [`crate::secrets`] (OS keychain, with a plaintext
/// fallback for environments without one). Returns `None` when nothing is
/// stored or the storage backend errored — callers treat the absence the
/// same way as a deliberate unset.
pub fn load_token() -> Option<String> {
    match crate::secrets::get("hf_token") {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("hf: load_token failed ({e:#})");
            None
        }
    }
}
