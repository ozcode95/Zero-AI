//! Minimal GitHub releases API client for `ggml-org/llama.cpp`.
//!
//! Mirrors [`crate::ovms::github`] in spirit but is intentionally kept
//! separate: the OVMS upstream lives at `openvinotoolkit/model_server`
//! and ships a single Windows zip per release, while llama.cpp ships
//! many flavour-specific zips per release and we need to pick by
//! detected GPU at install time (see [`crate::llama::variant`]).

use anyhow::{anyhow, Result};
use serde::Deserialize;

const UPSTREAM_OWNER: &str = "ggml-org";
const UPSTREAM_REPO: &str = "llama.cpp";

#[derive(Debug, Deserialize, Clone)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Asset {
    pub name: String,
    pub size: u64,
    pub browser_download_url: String,
}

impl Release {
    /// First asset whose name passes `predicate`. Used by the install
    /// path to pick the variant-specific zip without hard-coding the
    /// full filename (the build number changes every release).
    pub fn find_asset<F: Fn(&str) -> bool>(&self, predicate: F) -> Option<&Asset> {
        self.assets.iter().find(|a| predicate(&a.name))
    }
}

pub async fn latest_release(http: &reqwest::Client) -> Result<Release> {
    let url =
        format!("https://api.github.com/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/releases/latest");
    let resp = http
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await?
        .error_for_status()?;
    let release: Release = resp.json().await?;
    if release.assets.is_empty() {
        return Err(anyhow!(
            "github: llama.cpp release {} has no assets",
            release.tag_name
        ));
    }
    Ok(release)
}
