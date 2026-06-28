//! Minimal GitHub releases client for `ggml-org/whisper.cpp`.
//!
//! Reuses the [`Release`]/[`Asset`] shapes from the llama.cpp client since
//! the GitHub releases JSON is identical; only the owner/repo differ.

pub use crate::llama::github::{Asset, Release};
use anyhow::{anyhow, Result};

const OWNER: &str = "ggml-org";
const REPO: &str = "whisper.cpp";

pub async fn latest_release(http: &reqwest::Client) -> Result<Release> {
    let url = format!("https://api.github.com/repos/{OWNER}/{REPO}/releases/latest");
    let resp = http
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await?
        .error_for_status()?;
    let release: Release = resp.json().await?;
    if release.assets.is_empty() {
        return Err(anyhow!(
            "github: whisper.cpp release {} has no assets",
            release.tag_name
        ));
    }
    Ok(release)
}
